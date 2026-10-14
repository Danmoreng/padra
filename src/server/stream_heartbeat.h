#pragma once
#include <chrono>
namespace mellum16::server {
// Time is supplied by the caller so deadline boundaries can be tested without
// sleeps. Only a successful wire write resets the idle interval.
class StreamHeartbeat {
public:
  using Clock = std::chrono::steady_clock;
  StreamHeartbeat(std::chrono::milliseconds interval, Clock::time_point now)
      : interval_(interval), last_(now) {}
  bool Due(Clock::time_point now) const {
    return now - last_ >= interval_;
  }
  void Wrote(Clock::time_point now) {
    last_ = now;
  }

private:
  std::chrono::milliseconds interval_;
  Clock::time_point last_;
};
} // namespace mellum16::server
