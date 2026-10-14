#pragma once
#include "mellum16/status.h"
#include <atomic>
#include <httplib.h>
namespace mellum16::server::http_detail {
class Preparation {
public:
  explicit Preparation(std::atomic<unsigned> &count, unsigned maximum = 4) : count_(count) {
    auto current = count_.load();
    while (current < maximum)
      if (count_.compare_exchange_weak(current, current + 1)) {
        held_ = true;
        break;
      }
  }
  ~Preparation() {
    Release();
  }
  bool acquired() const {
    return held_;
  }
  void Release() {
    if (held_) {
      --count_;
      held_ = false;
    }
  }

private:
  std::atomic<unsigned> &count_;
  bool held_ = false;
};
Result<std::string> SessionOwner(const httplib::Request &);
} // namespace mellum16::server::http_detail
