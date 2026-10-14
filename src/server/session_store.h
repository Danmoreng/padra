#pragma once
#include "mellum16/status.h"
#include "server/admission.h"
#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
namespace mellum16::server {
struct SessionRecord {
  std::string id, owner, history = "[]";
  std::atomic<bool> deleted{false};
  bool active = false;
  std::size_t bytes = 0;
  std::chrono::steady_clock::time_point expires;
};
class SessionStore;
struct SessionLease {
  std::shared_ptr<SessionRecord> record;
  std::shared_ptr<SessionStore> store;
  std::size_t reservation = 0, budget = 0;
  bool committed = false;
  ~SessionLease();
};
struct SessionLimits {
  std::size_t entries = 32, bytes = 32 * 1024 * 1024, per_record = 8 * 1024 * 1024;
  std::chrono::milliseconds ttl{1800000};
};
struct SessionSnapshot {
  std::string id, history;
  bool active = false;
  std::size_t bytes = 0;
};
class SessionStore : public std::enable_shared_from_this<SessionStore> {
public:
  using Clock = std::chrono::steady_clock;
  using Now = std::function<Clock::time_point()>;
  explicit SessionStore(SessionLimits limits = {}, Now now = [] { return Clock::now(); });
  Result<SessionSnapshot> Create(const std::string &owner);
  Result<SessionSnapshot> Get(const std::string &id, const std::string &owner);
  Status Delete(const std::string &id, const std::string &owner);
  Result<std::shared_ptr<SessionLease>> Acquire(const std::string &id, const std::string &owner,
                                                std::size_t predicted,
                                                AdmissionDecision * = nullptr);
  Status Commit(const std::shared_ptr<SessionLease> &, std::string history);
  void Release(const std::shared_ptr<SessionRecord> &, std::size_t reservation);
  std::pair<std::size_t, std::size_t> Stats();
  std::int64_t ttl_seconds() const {
    return limits_.ttl.count() / 1000;
  }

private:
  // Joint response/root admission takes both store locks; no callbacks run there.
  friend class ResponseStore;
  void Prune();
  std::shared_ptr<SessionRecord> Find(const std::string &, const std::string &);
  static Status Missing();
  SessionLimits limits_;
  Now now_;
  std::mutex mutex_;
  std::map<std::string, std::shared_ptr<SessionRecord>, std::less<>> records_;
  std::size_t bytes_ = 0, reserved_ = 0;
};
} // namespace mellum16::server
