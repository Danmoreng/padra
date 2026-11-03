#pragma once
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <compare>
#include <tuple>
#include <cstddef>
#include <limits>

namespace mellum16 {
// Only verified persistent artifacts may be registered. The head conversion is
// part of the device representation and therefore part of its identity.
struct SharedWeightKey {
  int device;
  std::string manifest, weights, shared_head, diagnostic_attention;
  bool q8_head;
  auto operator<=>(const SharedWeightKey &) const = default;
};

struct SharedDraftWeightKey {
  int device;
  std::string manifest, weights;
  bool fp8_experts, exl3_experts, q8_head;
  auto operator<=>(const SharedDraftWeightKey &) const = default;
};

// planned includes each logical arena once; only newly allocated bytes need
// admission against currently free device memory.
inline std::size_t NewRuntimeAllocationBytes(std::size_t planned, std::size_t shared_target,
                                             std::size_t shared_draft) {
  if (shared_target > planned || shared_draft > planned - shared_target)
    throw std::overflow_error("shared allocations exceed runtime memory plan");
  return planned - shared_target - shared_draft;
}

template <class T, class Key = SharedWeightKey> class SharedWeightRegistry {
public:
  using Lock = std::unique_lock<std::mutex>;
  Lock Acquire() {
    return Lock(mutex_);
  }
  std::shared_ptr<T> Find(const Key &key, const Lock &lock) {
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
  void Publish(const Key &key, const std::shared_ptr<T> &value, const Lock &lock) {
    if (!lock.owns_lock() || lock.mutex() != &mutex_ || !value)
      throw std::logic_error("weight registry lock and value required");
    entries_[key] = value;
  }

private:
  std::mutex mutex_;
  std::map<Key, std::weak_ptr<T>> entries_;
};
} // namespace mellum16
