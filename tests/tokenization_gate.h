#pragma once
#include <condition_variable>
#include <memory>
#include <mutex>
namespace mellum16::server::test {
// Test-only synchronization at the real tokenizer call boundary. Production
// tokenizer sources and libraries have no knowledge of this fixture.
struct TokenizationBarrier {
  std::mutex mutex;
  std::condition_variable changed;
  unsigned entered = 0;
  bool released = false;
  void Wait() {
    std::unique_lock lock(mutex);
    ++entered;
    changed.notify_all();
    changed.wait(lock, [&] { return released; });
  }
  void Release() {
    std::lock_guard lock(mutex);
    released = true;
    changed.notify_all();
  }
};
void SetTokenizationBarrier(std::shared_ptr<TokenizationBarrier>);
struct ReleaseTokenization {
  std::shared_ptr<TokenizationBarrier> barrier;
  ~ReleaseTokenization() {
    barrier->Release();
    SetTokenizationBarrier({});
  }
};
} // namespace mellum16::server::test
