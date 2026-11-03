#pragma once
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <compare>
#include <tuple>

namespace mellum16 {
// Only verified persistent artifacts may be registered. The head conversion is
// part of the device representation and therefore part of its identity.
struct SharedWeightKey {
  int device;
  std::string manifest, weights, shared_head, diagnostic_attention;
  bool q8_head;
  auto operator<=>(const SharedWeightKey &) const = default;
};

template <class T> class SharedWeightRegistry {
public:
  using Lock = std::unique_lock<std::mutex>;
  Lock Acquire() {
    return Lock(mutex_);
  }
  std::shared_ptr<T> Find(const SharedWeightKey &key, const Lock &lock) {
    if (!lock.owns_lock() || lock.mutex() != &mutex_)
      throw std::logic_error("weight registry lock required");
    auto it = entries_.find(key);
    if (it == entries_.end())
      return {};
    auto result = it->second.lock();
    if (!result)
      entries_.erase(it);
    return result;
  }
  void Publish(const SharedWeightKey &key, const std::shared_ptr<T> &value, const Lock &lock) {
    if (!lock.owns_lock() || lock.mutex() != &mutex_ || !value)
      throw std::logic_error("weight registry lock and value required");
    entries_[key] = value;
  }

private:
  std::mutex mutex_;
  std::map<SharedWeightKey, std::weak_ptr<T>> entries_;
};
} // namespace mellum16
