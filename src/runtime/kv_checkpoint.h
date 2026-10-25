#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>
#include "mellum16/status.h"
#include "runtime/kv_cache_layout.h"

namespace mellum16 {
// A restored prefix resumes on the original chunk grid. Leave at least four
// rows in a partial first chunk so it uses the same expert arithmetic as the
// original full chunk. Retaining up to three fewer tokens avoids a T<4 switch.
inline int CheckpointCaptureAnchor(int anchor, int chunk, bool whole_chunks = false) {
  if (anchor < 0 || chunk < 4 || chunk > 2048)
    throw std::invalid_argument("invalid checkpoint capture anchor/chunk");
  const int offset = anchor % chunk;
  // SIMT sparse/dense expert dispatch depends on all routes in the chunk.
  // Retain complete chunks so reuse cannot change the arithmetic of a suffix.
  // A prefix shorter than one chunk deliberately produces no snapshot.
  if (whole_chunks)
    return anchor - offset;
  return offset > chunk - 4 ? anchor - (offset - (chunk - 4)) : anchor;
}
// Keep the original chat-template tokens. Only choose how much of a proven
// stable prefix to cache; leaving four rows avoids an unnecessary T<4 expert
// arithmetic branch when replaying the generation-only header.
inline Result<int> PlanCheckpointAnchor(std::span<const int> canonical,
                                        std::span<const int> stable) {
  if (canonical.empty() || canonical.size() > 131072 || stable.size() > canonical.size())
    return Status(StatusCode::kInvalidArgument, "invalid canonical/stable checkpoint extent");
  for (int id : canonical)
    if (id < 0 || id >= 98304)
      return Status(StatusCode::kInvalidArgument, "invalid canonical checkpoint-planning token");
  if (!std::equal(stable.begin(), stable.end(), canonical.begin()))
    return Status(StatusCode::kInvalidArgument,
                  "rendered stable history is not a canonical prefix");
  if (canonical.size() <= 4)
    return 0;
  return static_cast<int>(std::min(stable.size(), canonical.size() - 4));
}
// CPU identity ledger only. Capture must be called AFTER the GPU has completed
// all21 local-window copies; this object neither copies KV nor authorizes HTTP
// access. Fresh cache writes/Reset invalidate it before changing global rows.
struct KVCheckpointHandle {
  std::uint64_t runtime_binding = 0, generation = 0;
  int capacity = 0, anchor = 0;
  KVCacheFormat format = KVCacheFormat::BF16;
  bool operator==(const KVCheckpointHandle &) const = default;
};
class KVCheckpointLedger {
public:
  static constexpr std::size_t kLocalSnapshotBytes = 21ULL * 1024 * 2048;
  static std::size_t LocalSnapshotBytes(KVCacheFormat format) {
    return 21 * KVCacheLayout::Make(format, 1024).bytes;
  }
  KVCheckpointLedger(int capacity, std::uint64_t binding,
                     KVCacheFormat format = KVCacheFormat::BF16)
      : capacity_(capacity), binding_(binding), format_(format) {
    if (capacity < 1024 || capacity > 131072 || !binding)
      throw std::invalid_argument("invalid checkpoint binding/capacity");
    (void)LocalSnapshotBytes(format);
    prefix_.reserve(capacity);
  }
  Status Capture(std::span<const int> canonical, int anchor) {
    if (anchor < 1 || canonical.size() > static_cast<std::size_t>(capacity_) ||
        static_cast<std::size_t>(anchor) > canonical.size())
      return Invalid("invalid completed checkpoint prefix extent");
    for (int id : canonical)
      if (id < 0 || id >= 98304)
        return Invalid("invalid canonical checkpoint token");
    if (generation_ == std::numeric_limits<std::uint64_t>::max())
      return {StatusCode::kResourceExhausted, "checkpoint generation exhausted; recreate runtime"};
    prefix_.assign(canonical.begin(), canonical.begin() + anchor);
    ++generation_;
    handle_ = KVCheckpointHandle{binding_, generation_, capacity_, anchor, format_};
    return {};
  }
  Result<int> RestoreBase(std::span<const int> canonical, const KVCheckpointHandle &handle) const {
    if (!handle_ || handle != *handle_)
      return Invalid("checkpoint is absent/stale or belongs to another binding/profile");
    if (canonical.size() <= prefix_.size() ||
        canonical.size() > static_cast<std::size_t>(capacity_))
      return Invalid("checkpoint restore needs a nonempty suffix within capacity");
    for (int id : canonical)
      if (id < 0 || id >= 98304)
        return Invalid("invalid canonical restore token");
    if (!std::equal(prefix_.begin(), prefix_.end(), canonical.begin()))
      return Invalid("canonical history changed before retained checkpoint");
    return handle.anchor;
  }
  void Invalidate() {
    handle_.reset();
    prefix_.clear();
  }
  std::optional<KVCheckpointHandle> handle() const {
    return handle_;
  }

private:
  static Status Invalid(const char *message) {
    return {StatusCode::kInvalidArgument, message};
  }
  int capacity_;
  std::uint64_t binding_, generation_ = 0;
  KVCacheFormat format_;
  std::vector<int> prefix_;
  std::optional<KVCheckpointHandle> handle_;
};
} // namespace mellum16
