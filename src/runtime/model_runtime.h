#pragma once
#include "runtime/inference.h"
#include "runtime/gpu_build_config.h"
#include "runtime/context_capacity.h"
#include "runtime/server_profile.h"
#include "runtime/session_state.h"
#include "runtime/kv_checkpoint.h"
#include <functional>
#include <memory>
#include <span>
namespace mellum16 {
inline constexpr const char *kServerProfile = "exl3-k4-all-rows-wide-vector-8192-v1";
inline constexpr const char *kServerChunkProfile = "exl3-k4-chunk-graph-eager-tail-8192-v1";
inline std::string ServerExecutionProfile(int capacity, bool chunks,
                                          KVCacheFormat kv_format = KVCacheFormat::BF16,
                                          int chunk = 2048, int budget_gib = 0,
                                          bool checkpoint = false, int expert_rate = 4) {
#if PADRA_BUILD_SIMT
  (void)chunks;
  const auto base = std::string("exl3-k") + std::to_string(expert_rate) +
                    "-simt-sm" + kBuildArchitecture + "-layer-batch256-staged-kv-";
#else
  const auto base = std::string("exl3-k") + std::to_string(expert_rate) +
                    (chunks ? "-chunk-graph-eager-tail-" : "-all-rows-wide-vector-");
#endif
  std::string math;
#if PADRA_BUILD_SIMT
  if (std::string(kBuildArchitecture) == "61") math = "-pascal-fp32-v2-attention-tiered-";
#endif
  return base + math +
         std::to_string(capacity) + "-v1" +
         (kv_format == KVCacheFormat::Q8TokenHead128F32ScaleV1 ? "-q8-token-head128-f32scale-v1"
          : kv_format == KVCacheFormat::Q8Token512Block32F16ScaleRotatedV2
              ? "-q8_0-token512-block32-f16scale-rot-k128-v64-v2"
              : "") +
         (chunk == 2048 ? "" : "-chunk" + std::to_string(chunk)) +
         (budget_gib ? "-budget" + std::to_string(budget_gib) + "GiB" : "") +
         (checkpoint ? (kv_format == KVCacheFormat::BF16 ? "-local-checkpoint-bf16-v1"
                                                         : "-local-checkpoint-encoded-v1")
                     : "");
}
struct RuntimeStats {
  int mtp_depth = 0;
  std::uint64_t mtp_resident_bytes = 0, mtp_groups = 0, mtp_draft_calls = 0, mtp_proposed = 0,
                mtp_accepted = 0, mtp_ordinary_tails = 0;
  int capacity = 8192, prefill_chunk = 2048, device_budget_gib = 0, expert_rate = 4;
  std::uint64_t device_total_bytes = 0, device_free_before_bytes = 0, admission_available_bytes = 0,
                required_bytes = 0, post_graph_available_bytes = 0;
  KVCacheFormat kv_format = KVCacheFormat::BF16;
  std::uint64_t model_loads = 0, weight_uploads = 0, static_initializations = 0, graph_captures = 0,
                binding_epoch = 1;
  std::uint64_t resident_bytes = 0, requests = 0;
  std::string math_profile, kernel_profile, weights_sha256, manifest_sha256;
  bool controlled_chunks = false;
  std::uint64_t prefill_chunks = 0, eager_tail_chunks = 0;
  bool checkpoint_enabled = false, checkpoint_available = false;
  std::uint64_t checkpoint_bytes = 0, checkpoint_captures = 0, checkpoint_restores = 0;
  int cached_tokens = 0;
  int committed = 0;
  std::optional<int> pending;
  SessionPhase phase = SessionPhase::Ready;
};
using TokenCallback = std::function<bool(int)>;
using PrefillCallback = std::function<bool(int)>;
class RuntimeBackend {
public:
  virtual ~RuntimeBackend() = default;
  virtual Status Prepare(std::span<const int>, int, bool, SamplingOptions = {}) = 0;
  virtual Status Append(std::span<const int>, int, bool, SamplingOptions = {}) = 0;
  virtual Status PrepareCheckpoint(std::span<const int>, int, int, bool, SamplingOptions = {}) {
    return {StatusCode::kUnsupported, "native checkpoint capture not enabled"};
  }
  virtual Status RestoreCheckpoint(std::span<const int>, int, const KVCheckpointHandle &, int, bool,
                                   SamplingOptions = {}) {
    return {StatusCode::kUnsupported, "native checkpoint restore not enabled"};
  }
  virtual std::optional<KVCheckpointHandle> Checkpoint() const {
    return {};
  }
  virtual Result<InferenceResult> Generate(TokenCallback, PrefillCallback) = 0;
  virtual Status Reset() = 0;
  virtual void Cancel() = 0;
  virtual RuntimeStats Stats() const = 0;
};
// One preallocated session per runtime. Concurrent Prepare/Generate/Reset are
// rejected; Cancel and Stats may be called concurrently with Generate.
class ModelRuntime {
public:
  static Result<std::unique_ptr<ModelRuntime>>
  Load(const std::filesystem::path &, const std::string &tokenizer_sha256,
       bool controlled_chunks = false,
       NativeContextCapacity capacity = NativeContextCapacity::C8192,
       KVCacheFormat kv_format = KVCacheFormat::BF16, int diagnostic_decode_token = -1,
       int prefill_chunk = 2048, int device_budget_gib = 0, bool enable_checkpoint = false,
       MtpOptions mtp = {}, std::optional<bool> q8_head = std::nullopt);
  explicit ModelRuntime(std::unique_ptr<RuntimeBackend> backend) : backend_(std::move(backend)) {}
  Status PrepareRequestShape(std::span<const int> ids, int outputs, bool capture_logits = false,
                             SamplingOptions sampling = {}) {
    return backend_->Prepare(ids, outputs, capture_logits, sampling);
  }
  Status PrepareAppend(std::span<const int> ids, int outputs, bool capture_logits = false,
                       SamplingOptions sampling = {}) {
    return backend_->Append(ids, outputs, capture_logits, sampling);
  }
  Status PrepareWithCheckpoint(std::span<const int> ids, int outputs, int anchor,
                               bool capture_logits = false, SamplingOptions sampling = {}) {
    return backend_->PrepareCheckpoint(ids, outputs, anchor, capture_logits, sampling);
  }
  Status PrepareCheckpointRestore(std::span<const int> ids, int outputs,
                                  const KVCheckpointHandle &handle, int next_anchor = 0,
                                  bool capture_logits = false, SamplingOptions sampling = {}) {
    return backend_->RestoreCheckpoint(ids, outputs, handle, next_anchor, capture_logits, sampling);
  }
  std::optional<KVCheckpointHandle> Checkpoint() const {
    return backend_->Checkpoint();
  }
  Result<InferenceResult> Generate(TokenCallback callback = {}, PrefillCallback progress = {}) {
    return backend_->Generate(std::move(callback), std::move(progress));
  }
  Status Reset() {
    return backend_->Reset();
  }
  void Cancel() {
    backend_->Cancel();
  }
  RuntimeStats Stats() const {
    return backend_->Stats();
  }
  ModelRuntime(const ModelRuntime &) = delete;
  ModelRuntime &operator=(const ModelRuntime &) = delete;

private:
  std::unique_ptr<RuntimeBackend> backend_;
};
InferenceOptions SelectedServerOptions(const std::filesystem::path &, const std::string &,
                                       int expert_rate = 4);
} // namespace mellum16
