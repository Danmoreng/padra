#pragma once
#include <optional>
#include <span>
#include "mellum16/status.h"
namespace mellum16 {
enum class SessionPhase { Ready, Prepared, Running, Completed, Cancelled, Poisoned };
class SessionState {
public:
  explicit SessionState(int capacity) : capacity_(capacity) {}
  Status Prepare(std::span<const int> ids, int outputs) {
    if (phase_ != SessionPhase::Ready)
      return Fail("prepare requires Reset/Ready");
    if (capacity_ < 1 || ids.empty() || outputs < 0 ||
        ids.size() > static_cast<std::size_t>(capacity_) ||
        static_cast<std::size_t>(outputs) > static_cast<std::size_t>(capacity_) - ids.size())
      return Fail("prompt/output budget exceeds session capacity");
    for (int id : ids)
      if (id < 0 || id >= 98304)
        return Fail("token ID outside vocabulary");
    prompt_ = static_cast<int>(ids.size());
    requested_ = outputs;
    phase_ = SessionPhase::Prepared;
    return {};
  }
  Status PrepareAppend(std::span<const int> ids, int outputs, int base) {
    if (phase_ != SessionPhase::Completed && phase_ != SessionPhase::Cancelled)
      return Fail("append requires a completed/cancelled session");
    if (base != committed_ || ids.size() <= static_cast<std::size_t>(base) ||
        ids.size() > static_cast<std::size_t>(capacity_) || outputs < 0 ||
        static_cast<std::size_t>(outputs) > static_cast<std::size_t>(capacity_) - ids.size())
      return Fail("invalid append budget/base");
    for (int id : ids)
      if (id < 0 || id >= 98304)
        return Fail("invalid append token");
    prompt_ = static_cast<int>(ids.size());
    requested_ = outputs;
    emitted_ = 0;
    phase_ = SessionPhase::Prepared;
    return {};
  }
  // This transition alone provides no KV proof: the worker must first validate
  // a completed snapshot identity/prefix, then restore its local rings before
  // returning Prepared. Ordinary append's no-rollback contract is unchanged.
  Status PrepareCheckpointRestore(std::span<const int> ids, int outputs, int base) {
    if (phase_ != SessionPhase::Completed && phase_ != SessionPhase::Cancelled)
      return Fail("checkpoint restore requires completed/cancelled session");
    if (base < 1 || base > committed_ || ids.size() <= static_cast<std::size_t>(base) ||
        ids.size() > static_cast<std::size_t>(capacity_) || outputs < 0 ||
        static_cast<std::size_t>(outputs) > static_cast<std::size_t>(capacity_) - ids.size())
      return Fail("invalid checkpoint restore budget/base");
    for (int id : ids)
      if (id < 0 || id >= 98304)
        return Fail("invalid checkpoint restore token");
    prompt_ = static_cast<int>(ids.size());
    requested_ = outputs;
    emitted_ = 0;
    committed_ = base;
    pending_.reset();
    phase_ = SessionPhase::Prepared;
    return {};
  }
  Status CommitPrefix(int count) {
    if (phase_ != SessionPhase::Running || count < committed_ || count > prompt_)
      return Fail("invalid partial prefix commit");
    if (count > committed_)
      pending_.reset();
    committed_ = count;
    return {};
  }
  Status Begin(bool cancelled = false) {
    if (phase_ != SessionPhase::Prepared)
      return Fail("Generate requires Prepared");
    phase_ = cancelled ? SessionPhase::Cancelled : SessionPhase::Running;
    return {};
  }
  Status CommitPrefill() {
    if (phase_ != SessionPhase::Running)
      return Fail("prefill outside Running");
    committed_ = prompt_;
    return {};
  }
  Status Emit(int id) {
    if (phase_ != SessionPhase::Running || id < 0 || id >= 98304 || emitted_ >= requested_)
      return Fail("invalid token emission");
    if (emitted_ > 0)
      ++committed_;
    pending_ = id;
    ++emitted_;
    if (id == 28 || emitted_ == requested_)
      phase_ = SessionPhase::Completed;
    return {};
  }
  void Finish() {
    if (phase_ == SessionPhase::Running)
      phase_ = SessionPhase::Completed;
  }
  void Cancel() {
    if (phase_ == SessionPhase::Running || phase_ == SessionPhase::Prepared ||
        phase_ == SessionPhase::Completed)
      phase_ = SessionPhase::Cancelled;
  }
  void Poison() {
    phase_ = SessionPhase::Poisoned;
  }
  Status Reset() {
    if (phase_ == SessionPhase::Running || phase_ == SessionPhase::Poisoned)
      return Fail("running or poisoned session cannot Reset");
    phase_ = SessionPhase::Ready;
    prompt_ = requested_ = emitted_ = committed_ = 0;
    pending_.reset();
    return {};
  }
  SessionPhase phase() const {
    return phase_;
  }
  int committed() const {
    return committed_;
  }
  int emitted() const {
    return emitted_;
  }
  std::optional<int> pending() const {
    return pending_;
  }

private:
  static Status Fail(const char *message) {
    return {StatusCode::kInvalidArgument, message};
  }
  int capacity_, prompt_ = 0, requested_ = 0, emitted_ = 0, committed_ = 0;
  std::optional<int> pending_;
  SessionPhase phase_ = SessionPhase::Ready;
};
} // namespace mellum16
