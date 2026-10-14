#pragma once
#include "mellum16/status.h"
#include "server/admission.h"
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
namespace mellum16::server {
struct ResponseRoot {
  std::string session_id, owner;
  std::function<void()> discard;
  ~ResponseRoot() {
    if (discard) {
      try {
        discard();
      } catch (...) {
      }
    }
  }
};
struct StoredResponse {
  std::string id, owner, envelope, history;
  // A retrievable incomplete envelope need not contain valid continuation output.
  bool continuable = true;
  std::shared_ptr<ResponseRoot> root;
};
struct ResponseStoreLimits {
  std::size_t entries = 32, bytes = 32 * 1024 * 1024, per_record = 8 * 1024 * 1024;
  std::chrono::milliseconds ttl{1800000};
};
class ResponseStore;
class SessionStore;
struct ResponseReservation {
  std::shared_ptr<ResponseStore> store;
  std::string id, owner;
  std::size_t budget = 0;
  bool committed = false;
  std::shared_ptr<ResponseRoot> root;
  ~ResponseReservation();
};
class ResponseStore : public std::enable_shared_from_this<ResponseStore> {
public:
  using Clock = std::chrono::steady_clock;
  using Now = std::function<Clock::time_point()>;
  explicit ResponseStore(ResponseStoreLimits = {}, Now = [] { return Clock::now(); });
  Result<std::shared_ptr<ResponseReservation>>
  Reserve(std::string id, std::string owner, std::size_t predicted,
          std::shared_ptr<SessionStore> sessions = {}, bool create_root = false,
          std::function<void(const std::string &, const std::string &)> discard = {},
          AdmissionDecision *decision = nullptr);
  Status Commit(const std::shared_ptr<ResponseReservation> &, std::string envelope,
                std::string history, std::shared_ptr<ResponseRoot> root, bool continuable = true);
  Result<std::shared_ptr<const StoredResponse>> Get(const std::string &id,
                                                    const std::string &owner);
  Status Delete(const std::string &id, const std::string &owner);
  void Release(const std::string &id, const std::string &owner, bool committed);
  std::pair<std::size_t, std::size_t> Stats();
  std::pair<std::size_t, std::size_t> Evictions();

private:
  struct Entry {
    std::shared_ptr<const StoredResponse> record;
    std::string owner;
    std::size_t bytes = 0;
    bool pending = true, deleted = false;
    Clock::time_point expires;
  };
  using Retired = std::vector<std::shared_ptr<const StoredResponse>>;
  void Prune(Retired &, SessionStore *sessions = nullptr);
  static Status Missing();
  ResponseStoreLimits limits_;
  Now now_;
  std::mutex mutex_;
  std::map<std::string, Entry, std::less<>> records_;
  std::size_t bytes_ = 0;
  std::size_t evicted_records_ = 0, evicted_roots_ = 0;
};
} // namespace mellum16::server
