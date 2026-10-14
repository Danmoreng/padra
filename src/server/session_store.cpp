#include "server/session_store.h"
#include "server/secure_id.h"
#include <stdexcept>
namespace mellum16::server {
SessionLease::~SessionLease() {
  if (store)
    store->Release(record, reservation);
}
SessionStore::SessionStore(SessionLimits limits, Now now) : limits_(limits), now_(std::move(now)) {
  if (!limits.entries || !limits.bytes || !limits.per_record || limits.per_record > limits.bytes ||
      limits.ttl.count() < 1 || !now_)
    throw std::invalid_argument("invalid bounded session-store limits");
}
Status SessionStore::Missing() {
  return {StatusCode::kNotFound, "session absent/expired or not in this credential namespace"};
}
void SessionStore::Prune() {
  const auto now = now_();
  for (auto it = records_.begin(); it != records_.end();) {
    auto &r = *it->second;
    if (!r.active && (r.deleted.load() || now >= r.expires)) {
      bytes_ -= r.bytes;
      it = records_.erase(it);
    } else
      ++it;
  }
}
std::shared_ptr<SessionRecord> SessionStore::Find(const std::string &id, const std::string &owner) {
  auto it = records_.find(id);
  if (it == records_.end() || it->second->owner != owner || it->second->deleted.load())
    return {};
  return it->second;
}
Result<SessionSnapshot> SessionStore::Create(const std::string &owner) {
  if (owner.empty() || owner.size() > 128)
    return Status(StatusCode::kInvalidArgument, "invalid session credential namespace");
  auto id = MakeSecureId("sess-");
  if (!id.ok())
    return id.status();
  std::lock_guard lock(mutex_);
  Prune();
  if (records_.size() >= limits_.entries)
    return Status(StatusCode::kResourceExhausted, "bounded logical session count exhausted");
  if (limits_.per_record < 2 || bytes_ + reserved_ > limits_.bytes - 2)
    return Status(StatusCode::kResourceExhausted, "bounded logical history store exhausted");
  auto r = std::make_shared<SessionRecord>();
  r->id = id.value();
  r->owner = owner;
  r->expires = now_() + limits_.ttl;
  r->bytes = 2;
  if (!records_.emplace(r->id, r).second)
    return Status(StatusCode::kInternal, "secure session identity collision");
  bytes_ += r->bytes;
  return SessionSnapshot{r->id, r->history, false, r->bytes};
}
Result<SessionSnapshot> SessionStore::Get(const std::string &id, const std::string &owner) {
  std::lock_guard lock(mutex_);
  Prune();
  auto r = Find(id, owner);
  if (!r)
    return Missing();
  return SessionSnapshot{r->id, r->history, r->active, r->bytes};
}
Status SessionStore::Delete(const std::string &id, const std::string &owner) {
  std::lock_guard lock(mutex_);
  Prune();
  auto r = Find(id, owner);
  if (!r)
    return Missing();
  r->deleted.store(true);
  Prune();
  return {};
}
Result<std::shared_ptr<SessionLease>> SessionStore::Acquire(const std::string &id,
                                                            const std::string &owner,
                                                            std::size_t predicted,
                                                            AdmissionDecision *decision) {
  if (decision)
    *decision = {};
  const auto reject = [&](AdmissionCause cause,
                          Status status) -> Result<std::shared_ptr<SessionLease>> {
    if (decision)
      decision->cause = cause;
    return status;
  };
  std::lock_guard lock(mutex_);
  Prune();
  auto r = Find(id, owner);
  if (!r)
    return reject(AdmissionCause::SessionMissing, Missing());
  if (r->active)
    return reject(
        AdmissionCause::SessionBusy,
        Status(StatusCode::kResourceExhausted, "session already has a queued/running request"));
  const auto extra = predicted > r->bytes ? predicted - r->bytes : 0;
  if (predicted > limits_.per_record || extra > limits_.bytes - bytes_ - reserved_)
    return reject(
        AdmissionCause::SessionBudget,
        Status(StatusCode::kResourceExhausted, "bounded logical session history budget exhausted"));
  auto lease = std::make_shared<SessionLease>();
  lease->record = r;
  lease->store = shared_from_this();
  lease->reservation = extra;
  lease->budget = predicted;
  r->active = true;
  r->expires = now_() + limits_.ttl;
  reserved_ += extra;
  return lease;
}
Status SessionStore::Commit(const std::shared_ptr<SessionLease> &lease, std::string history) {
  std::lock_guard lock(mutex_);
  auto r = lease->record;
  if (r->deleted.load())
    return Missing();
  if (lease->committed || !r->active || lease->store.get() != this ||
      history.size() > lease->budget || history.size() > limits_.per_record)
    return {StatusCode::kResourceExhausted,
            "session history exceeds admitted reservation/already committed"};
  const auto next = bytes_ - r->bytes + history.size();
  if (next > limits_.bytes)
    return {StatusCode::kResourceExhausted, "session history total budget exceeded"};
  // Convert the reservation to actual retained bytes exactly once; no CPU KV.
  reserved_ -= lease->reservation;
  lease->reservation = 0;
  lease->committed = true;
  bytes_ = next;
  r->history = std::move(history);
  r->bytes = r->history.size();
  r->expires = now_() + limits_.ttl;
  return {};
}
void SessionStore::Release(const std::shared_ptr<SessionRecord> &r, std::size_t reservation) {
  std::lock_guard lock(mutex_);
  reserved_ -= reservation;
  r->active = false;
  r->expires = now_() + limits_.ttl;
  Prune();
}
std::pair<std::size_t, std::size_t> SessionStore::Stats() {
  std::lock_guard lock(mutex_);
  Prune();
  return {records_.size(), bytes_};
}
} // namespace mellum16::server
