#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>
#include "mellum16/status.h"
namespace mellum16 {
struct ChunkGraphKey {
  std::string checkpoint_sha256, execution_sha256, build_identity;
  std::string kv_format = "bf16";
  int capacity = 8192, workspace_chunk = 2048, active_tokens = 2048;
  std::uint32_t binding_epoch = 1;
  bool operator==(const ChunkGraphKey &) const = default;
};
// Explicit startup/hot-maintenance admission only. Ordinary Prepare must look
// up an existing shape or declare same-body eager tail; it cannot add entries.
class ChunkShapeSet {
public:
  static constexpr std::size_t kMaxShapes = 8;
  Status Admit(ChunkGraphKey key, std::size_t nodes, std::size_t bytes) {
    if (key.active_tokens < 1 || key.active_tokens > key.workspace_chunk ||
        key.workspace_chunk > 2048 || key.capacity < 1 || key.capacity > 131072 ||
        key.binding_epoch == 0 ||
        (key.kv_format != "bf16" && key.kv_format != "q8_token_head128_f32scale_v1" &&
         key.kv_format != "q8_0_token512_block32_f16scale_rot_k128_v64_v2") ||
        key.checkpoint_sha256.empty() || key.execution_sha256.empty() || key.build_identity.empty())
      return {StatusCode::kInvalidArgument, "invalid chunk graph identity"};
    if (Find(key))
      return {};
    if (count_ == kMaxShapes || nodes > 4096 || bytes > 16 * 1024 * 1024 ||
        bytes > 64 * 1024 * 1024 - resident_)
      return {StatusCode::kResourceExhausted, "bounded graph shape/node/memory budget exceeded"};
    keys_[count_++] = std::move(key);
    resident_ += bytes;
    return {};
  }
  bool Find(const ChunkGraphKey &key) const {
    for (std::size_t i = 0; i < count_; ++i)
      if (keys_[i] == key)
        return true;
    return false;
  }
  void Invalidate() {
    count_ = 0;
    resident_ = 0;
    for (auto &key : keys_)
      key = ChunkGraphKey{};
  }
  std::size_t size() const {
    return count_;
  }
  std::size_t resident() const {
    return resident_;
  }

private:
  std::array<ChunkGraphKey, kMaxShapes> keys_{};
  std::size_t count_ = 0, resident_ = 0;
};
// Canonical token history, independent of rendering. Ring eviction permits
// continuation, not rollback. Pending output is separate and is never counted
// as already committed until a subsequent actual forward consumes it.
class SessionPrefix {
public:
  explicit SessionPrefix(std::size_t capacity) : capacity_(capacity) {
    committed_.reserve(capacity);
  }
  Status Set(std::span<const int> ids, std::optional<int> pending = {}) {
    if (ids.size() > capacity_)
      return {StatusCode::kInvalidArgument, "committed prefix exceeds capacity"};
    for (int token : ids)
      if (token < 0 || token >= 98304)
        return {StatusCode::kInvalidArgument, "invalid committed token"};
    if (pending && (*pending < 0 || *pending >= 98304))
      return {StatusCode::kInvalidArgument, "invalid pending token"};
    committed_.assign(ids.begin(), ids.end());
    pending_ = pending;
    return {};
  }
  Status Capture(std::span<const int> input, std::span<const int> output, std::size_t count,
                 std::optional<int> pending) {
    if (count > capacity_ || count > input.size() + output.size())
      return {StatusCode::kInvalidArgument, "invalid recorded committed prefix"};
    const auto promptCount = std::min(count, input.size());
    committed_.assign(input.begin(), input.begin() + promptCount);
    if (count > promptCount)
      committed_.insert(committed_.end(), output.begin(), output.begin() + count - promptCount);
    pending_ = pending;
    return {};
  }
  Result<std::size_t> AppendBase(std::span<const int> canonical) const {
    if (canonical.size() > capacity_ || canonical.size() <= committed_.size())
      return Status(StatusCode::kUnsupported,
                    "append requires a nonempty suffix; no rollback or zero-tail reuse");
    if (!std::equal(committed_.begin(), committed_.end(), canonical.begin()))
      return Status(StatusCode::kInvalidArgument,
                    "canonical prompt does not match committed prefix; Reset required");
    for (int token : canonical)
      if (token < 0 || token >= 98304)
        return Status(StatusCode::kInvalidArgument, "invalid canonical token");
    return committed_.size();
  }
  void Reset() {
    committed_.clear();
    pending_.reset();
  }
  std::span<const int> committed() const {
    return committed_;
  }
  std::optional<int> pending() const {
    return pending_;
  }

private:
  std::size_t capacity_;
  std::vector<int> committed_;
  std::optional<int> pending_;
};
} // namespace mellum16
