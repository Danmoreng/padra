#include "server/response_store.h"
#include "server/secure_id.h"
#include "server/session_store.h"
#include <algorithm>
#include <stdexcept>
#include <vector>
namespace mellum16::server {
ResponseReservation::~ResponseReservation() {
  if (store)
    store->Release(id, owner, committed);
}
ResponseStore::ResponseStore(ResponseStoreLimits limits, Now now)
    : limits_(limits), now_(std::move(now)) {
  if (!limits.entries || !limits.bytes || !limits.per_record || limits.per_record > limits.bytes ||
      limits.ttl.count() < 1 || !now_)
    throw std::invalid_argument("invalid response-store limits");
}
Status ResponseStore::Missing() {
  return {StatusCode::kNotFound, "response absent/expired or outside credential namespace"};
}
void ResponseStore::Prune(Retired &retired, SessionStore *sessions) {
  retired.reserve(retired.size() + records_.size());
  const auto now = now_();
  for (auto it = records_.begin(); it != records_.end();) {
    auto &r = it->second;
    if (sessions && r.record && r.record->root) {
      const auto &root = r.record->root;
      auto session = sessions->Find(root->session_id, root->owner);
      if (session && session->active) {
        ++it;
        continue;
      }
    }
    if (!r.pending && (!r.record || r.record.use_count() == 1) && (r.deleted || now >= r.expires)) {
      retired.push_back(std::move(r.record));
      bytes_ -= r.bytes;
      it = records_.erase(it);
    } else
      ++it;
  }
}
Result<std::shared_ptr<ResponseReservation>>
ResponseStore::Reserve(std::string id, std::string owner, std::size_t predicted,
                       std::shared_ptr<SessionStore> sessions, bool create_root,
                       std::function<void(const std::string &, const std::string &)> discard,
                       AdmissionDecision *decision) {
  if (decision)
    *decision = {};
  const auto reject = [&](AdmissionCause cause,
                          Status status) -> Result<std::shared_ptr<ResponseReservation>> {
    if (decision)
      decision->cause = cause;
    return status;
  };
  if (id.size() != 37 || !id.starts_with("resp_") ||
      id.find_first_not_of("0123456789abcdef", 5) != std::string::npos || owner.empty() ||
      owner.size() > 128 || (create_root && !sessions))
    return reject(AdmissionCause::InvalidRequest,
                  Status(StatusCode::kInvalidArgument, "invalid response/root identity/owner"));
  if (!predicted || predicted > limits_.per_record)
    return reject(
        AdmissionCause::ResponseStoreFull,
        Status(StatusCode::kResourceExhausted, "bounded response-store admission exhausted"));

  // Allocate the new nodes/callback before committing any eviction. Inserting
  // these same-allocator map nodes cannot allocate after planning succeeds.
  auto ticket = std::make_shared<ResponseReservation>();
  ticket->id = id;
  ticket->owner = owner;
  ticket->budget = predicted;
  decltype(records_) new_records;
  Entry entry;
  entry.owner = owner;
  entry.bytes = predicted;
  new_records.emplace(id, std::move(entry));
  auto response_node = new_records.extract(id);
  std::map<std::string, std::shared_ptr<SessionRecord>, std::less<>> new_sessions;
  std::function<void()> on_discard;
  if (create_root) {
    auto generated = MakeSecureId("sess-");
    if (!generated.ok())
      return reject(AdmissionCause::Internal, generated.status());
    ticket->root = std::make_shared<ResponseRoot>();
    ticket->root->session_id = generated.value();
    ticket->root->owner = owner;
    const auto root_id = generated.value();
    on_discard = [sessions, discard = std::move(discard), root_id, owner] {
      (void)sessions->Delete(root_id, owner);
      if (discard)
        discard(root_id, owner);
    };
    auto record = std::make_shared<SessionRecord>();
    record->id = root_id;
    record->owner = owner;
    record->bytes = 2;
    new_sessions.emplace(root_id, std::move(record));
  }
  auto self = shared_from_this();
  // Retired record/root destructors may call into the coordinator/session store.
  // Declare them before the locks so callbacks run only after both are released.
  Retired retired;
  std::unique_lock response_lock(mutex_, std::defer_lock);
  std::unique_lock<std::mutex> session_lock;
  if (sessions) {
    session_lock = std::unique_lock(sessions->mutex_, std::defer_lock);
    std::lock(response_lock, session_lock);
    sessions->Prune();
  } else
    response_lock.lock();
  Prune(retired, sessions.get());
  if (records_.contains(id))
    return reject(AdmissionCause::InvalidRequest,
                  Status(StatusCode::kInvalidArgument, "duplicate response identity"));
  if (create_root &&
      (sessions->records_.contains(ticket->root->session_id) || sessions->limits_.per_record < 2))
    return reject(AdmissionCause::RootCapacity,
                  Status(StatusCode::kResourceExhausted, "root identity/budget unavailable"));

  struct RootGroup {
    std::size_t references = 0, removed = 0;
    SessionRecord *record = nullptr;
    bool evicted = false;
  };
  std::map<const ResponseRoot *, RootGroup> groups;
  for (const auto &e : retired)
    if (sessions && e && e->root) {
      const auto &root = e->root;
      auto record = sessions->Find(root->session_id, root->owner);
      auto [it, inserted] = groups.try_emplace(
          root.get(), RootGroup{static_cast<std::size_t>(root.use_count()), 0, record.get()});
      (void)inserted;
      ++it->second.removed;
    }
  std::vector<decltype(records_)::iterator> eligible;
  eligible.reserve(records_.size());
  for (auto it = records_.begin(); it != records_.end(); ++it) {
    const auto &e = it->second;
    if (e.pending || e.record.use_count() != 1)
      continue;
    if (sessions && e.record->root) {
      const auto &root = e.record->root;
      auto record = sessions->Find(root->session_id, root->owner);
      if (record && record->active)
        continue;
      groups.try_emplace(root.get(),
                         RootGroup{static_cast<std::size_t>(root.use_count()), 0, record.get()});
    }
    eligible.push_back(it);
  }
  std::sort(eligible.begin(), eligible.end(),
            [](auto a, auto b) { return a->second.expires < b->second.expires; });
  auto count = records_.size(), bytes = bytes_;
  std::size_t retire = 0, free_roots = 0, free_session_bytes = 0;
  for (const auto &[root, group] : groups)
    if (group.removed == group.references && group.record) {
      (void)root;
      ++free_roots;
      free_session_bytes += group.record->bytes;
    }
  const auto root_fits = [&] {
    return !create_root || (sessions->records_.size() - free_roots < sessions->limits_.entries &&
                            sessions->bytes_ + sessions->reserved_ - free_session_bytes <=
                                sessions->limits_.bytes - 2);
  };
  const auto fits = [&] {
    return count < limits_.entries && predicted <= limits_.bytes - bytes && root_fits();
  };
  while (!fits() && retire < eligible.size()) {
    const auto &e = eligible[retire++]->second;
    --count;
    bytes -= e.bytes;
    if (sessions && e.record->root) {
      auto &group = groups.at(e.record->root.get());
      group.evicted = true;
      if (++group.removed == group.references && group.record) {
        ++free_roots;
        free_session_bytes += group.record->bytes;
      }
    }
  }
  if (!fits())
    return reject(root_fits() ? AdmissionCause::ResponseStoreFull : AdmissionCause::RootCapacity,
                  Status(StatusCode::kResourceExhausted,
                         root_fits()
                             ? "bounded response-store admission exhausted"
                             : "protected or referenced root sessions exhaust the session budget"));

  // Commit the proven plan under both locks. Pending/pinned records and active
  // or manually created sessions cannot be selected. Shared roots count once.
  retired.reserve(retired.size() + retire);
  const auto response_expiry = now_() + limits_.ttl;
  const auto session_expiry =
      sessions ? sessions->now_() + sessions->limits_.ttl : Clock::time_point{};
  for (auto &[root, group] : groups) {
    if (group.record && group.removed == group.references) {
      group.record->deleted.store(true);
      sessions->bytes_ -= group.record->bytes;
      sessions->records_.erase(root->session_id);
      if (group.evicted)
        ++evicted_roots_;
    }
  }
  for (std::size_t i = 0; i < retire; ++i) {
    retired.push_back(std::move(eligible[i]->second.record));
    bytes_ -= eligible[i]->second.bytes;
    records_.erase(eligible[i]);
    ++evicted_records_;
  }
  response_node.mapped().expires = response_expiry;
  records_.insert(std::move(response_node));
  bytes_ += predicted;
  if (create_root) {
    auto node = new_sessions.extract(ticket->root->session_id);
    node.mapped()->expires = session_expiry;
    sessions->records_.insert(std::move(node));
    sessions->bytes_ += 2;
    ticket->root->discard = std::move(on_discard);
  }
  ticket->store = std::move(self);
  return ticket;
}
Status ResponseStore::Commit(const std::shared_ptr<ResponseReservation> &ticket,
                             std::string envelope, std::string history,
                             std::shared_ptr<ResponseRoot> root, bool continuable) {
  if (!ticket || ticket->store.get() != this)
    return {StatusCode::kInvalidArgument, "foreign response reservation"};
  std::lock_guard lock(mutex_);
  auto it = records_.find(ticket->id);
  if (it == records_.end() || it->second.owner != ticket->owner || it->second.deleted)
    return Missing();
  auto &e = it->second;
  if (ticket->committed || !e.pending)
    return {StatusCode::kInvalidArgument, "response already committed"};
  // Both copies plus fixed record identity are admitted; no GPU KV lives here.
  constexpr std::size_t overhead = 512;
  if (history.size() > ticket->budget || envelope.size() > ticket->budget - history.size() ||
      overhead > ticket->budget - history.size() - envelope.size())
    return {StatusCode::kResourceExhausted, "response exceeds admitted reservation"};
  const auto size = history.size() + envelope.size() + overhead;
  auto record = std::make_shared<StoredResponse>();
  record->id = ticket->id;
  record->owner = ticket->owner;
  record->envelope = std::move(envelope);
  record->history = std::move(history);
  record->continuable = continuable;
  record->root = std::move(root);
  bytes_ -= e.bytes - size;
  e.bytes = size;
  e.record = std::move(record);
  e.pending = false;
  e.expires = now_() + limits_.ttl;
  ticket->committed = true;
  return {};
}
Result<std::shared_ptr<const StoredResponse>> ResponseStore::Get(const std::string &id,
                                                                 const std::string &owner) {
  Retired retired;
  std::lock_guard lock(mutex_);
  Prune(retired);
  auto it = records_.find(id);
  if (it == records_.end() || it->second.owner != owner || it->second.deleted ||
      it->second.pending || now_() >= it->second.expires)
    return Missing();
  it->second.expires = now_() + limits_.ttl;
  return it->second.record;
}
Status ResponseStore::Delete(const std::string &id, const std::string &owner) {
  Retired retired;
  std::lock_guard lock(mutex_);
  Prune(retired);
  auto it = records_.find(id);
  if (it == records_.end() || it->second.owner != owner || it->second.deleted ||
      it->second.pending || now_() >= it->second.expires)
    return Missing();
  it->second.deleted = true;
  Prune(retired);
  return {};
}
void ResponseStore::Release(const std::string &id, const std::string &owner, bool committed) {
  Retired retired;
  std::lock_guard lock(mutex_);
  auto it = records_.find(id);
  if (it != records_.end() && it->second.owner == owner && !committed && it->second.pending) {
    bytes_ -= it->second.bytes;
    records_.erase(it);
  }
  Prune(retired);
}
std::pair<std::size_t, std::size_t> ResponseStore::Stats() {
  Retired retired;
  std::lock_guard lock(mutex_);
  Prune(retired);
  return {records_.size(), bytes_};
}
std::pair<std::size_t, std::size_t> ResponseStore::Evictions() {
  std::lock_guard lock(mutex_);
  return {evicted_records_, evicted_roots_};
}
} // namespace mellum16::server
