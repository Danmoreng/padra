#pragma once
#include <filesystem>
#include <iosfwd>
#include <vector>
#include <atomic>
#include "mellum16/status.h"
#include "runtime/prefill_state.h"
#include "runtime/sampling.h"
#include "runtime/kv_cache_layout.h"
namespace mellum16 {
inline constexpr int kLegacyContextCapacity = 4096;
inline constexpr int kW4A8FContextCapacity = 8192;
inline constexpr int kW4A8FMaxPrefillChunk = 2048;
inline constexpr const char *kCompensatedMathProfile = "mellum16-bf16-compensated-rnexp-v1";
inline constexpr const char *kW4A8FProfile = "mellum16-w4a8f-v1";
struct MtpOptions {
  std::filesystem::path artifact;
  int depth = 1; // An empty artifact selects ordinary execution.
  bool fused_experts = true;
  bool packed_qkv = true;
  bool packed_o = false;     // Explicit research override; selected path keeps paired ordinary O.
  bool batched_prime = true; // Exact prompt-to-draft KV rows in dead prefill arenas.
  bool paired_attention = true; // Exact adjacent-row attention with saved ring history.
  bool batched_rows = true;     // Reuse verifier arenas for batched postnorm/router.
  bool compact_experts =
      true; // Qualified graph-static active-expert dispatch; full remains selectable.
  bool exl3_experts = false; // Separate pinned EXL3-K4 draft experts, nonexperts unchanged.
  bool fp8_experts = false;  // Separate pinned draft-only FP8 artifact; explicit opt-in.
};
struct InferenceOptions {
  MtpOptions mtp;
  bool q8_head = false; // Explicit shared target/draft Q8-block32 head; no BF16 GPU shadow.
  SamplingOptions sampling;
  KVCacheFormat kv_cache_format = KVCacheFormat::BF16;
  std::filesystem::path artifact;
  std::vector<int> input_ids;
  std::string tokenizer_sha256;
  int capacity = 256;
  int generate = 16;
  bool capture_logits = false;
  bool capture_hidden = false; // Last post-final-norm row, for independent draft qualification.
  bool capture_trace = false;
  bool warp_projections = true;
  int prefill_chunk = 128;
  int server_device_budget_gib = 0;  // Explicit ceiling; never reduces context/chunk/precision.
  bool server_kv_checkpoint = false; // Explicit one42MiB local-ring snapshot; no default promotion.
  bool native_prefill = false;
  bool accurate_native_prefill = false;
  bool fp8_expert_activations = false;
  bool native_decode = false;
  bool tc_prefill_attention = false;
  bool online_prefill_attention = false;
  bool prune_last_prefill_ffn =
      false; // G4 normal last-logit-only output; diagnostics retain all rows.
  bool gqa8_prefill_attention = false; // G3 register-Q 8-position/8-head BF16 pipeline.
  bool packed_bf16_prefill_attention =
      false; // Exact temporary BF16 current chunk,16-byte async loads.
  bool padded_bf16_prefill_attention = false; // Exact shared136/singleK dataflow.
  bool quad_router_prefill =
      false; // Four independent compensated BF16 rows share normalized input.
  bool pair64_qknorm_prefill = false; // T>=128 pair64, short original128; same math.
  bool exact_norm_prefill = false;    // Same norm/FP8 bytes; cached RN encoder used on batches.
  bool ordinary_fp8_projections = false;
  bool ordinary_fp8_simt = false;
  bool ordinary_rms_norm = false;
  bool compact_experts = false;
  bool ordinary_block_experts = false;
  bool staged_experts = false, staged_mma_decode = false;
  std::filesystem::path diagnostic_attention; // X0 offline BF16 source overlay; explicit only.
  bool diagnostic_h128_fp32 = false, diagnostic_routes = false;
  int diagnostic_exl3_layer = -1;    // X1 single-chunk BF16 input/route fixture only.
  bool exl3_decode_dual = false;     // Same per-projection math,paired distinct gate/up grid.
  bool wide_rms_dataflow = false;    // First128 products/compensated tree,256 independent outputs.
  bool wide_attention_merge = false; // Same N4a math,512 workers.
  bool paired_attention_o = false;   // Same ordinary O math,two rows share E4 decode.
  bool exact_router_rank = false;    // Same normalized-probability top8 contract,decode only.
  bool attention_fragment_validation = false; // Same BF16 operands and arithmetic.
  bool exact_fp8_encoding = false;            // Same E4 bytes,exhaustive boundary correction.
  bool exl3_decode_lane = false;              // Exact transposed MMA/one-shuffle K4 fragment.
  bool exl3_prefill_warp = false;             // Exact warp H128 input transport.
  bool exl3_wide_input = false; // Exact Gate/Up vector transport/eight independent warps.
  bool exl3_packed_activations =
      false; // Exact BF16 physical prefill storage plus SiLU/down-input consumer.
  bool exl3_silu_table = false;       // Exact finite BF16 SiLU table; prebuilt resident.
  bool exl3_bf16_postnorm = false;    // Exact norm with unused FP8 encoder removed.
  bool attention_fp8_mma = false;     // Explicit decode-only FP8-MMA override,quantizer unchanged.
  bool exl3_decode_dot = false;       // Exact UDP4A byte-sum with same M1 operands/order.
  bool exl3_decode_fused = false;     // Exact fused H128/M1-N128 dataflow.
  bool exl3_decode_mma = false;       // Explicit overnight N64/F16-MMA, BF16 input cast.
  bool exl3_decode_n32 = false;       // R-D1 N32/four warps/two K partitions,full H128.
  bool exl3_decode_selected = false;  // Explicit X2 K4 ordinary FP32 FMA selected vector.
  bool exl3_prefill_dot = false;      // Exact UDP4A codebook/register-B.
  bool exl3_prefill_register = false; // Exact register-B transport, maskedM32 math.
  bool exl3_prefill_masked = false;   // R-P1 masked M32, explicit BF16->FP16/MMA tails.
  bool exl3_prefill_fp16 = false;     // Explicit X1 research, frozen SIMT decode/tails.
  int exl3_rate = 0;
  bool exl3_experts = false;          // Explicit K4/H128 BF16-input represented-math research.
  bool scaled_bf16_prefill = false;   // E3 full64 scale-folded BF16, exact M16 tails.
  bool ordinary_prefill_silu = false; // N1a changes only the winning prefill exponential.
  bool packed_expert_decode = false;  // G2 E4M3 pair-dot/common-SF/two-accumulator FP32 research.
  bool q8_expert_decode = false; // N2b direct BF16 -> block32 Q8/FP16 scale; prefill remains E4M3.
  bool reciprocal_fused_quantization =
      false; // N4c only selected O/postnorm encoders, finite FP32 inverse.
  bool capture_quantization_rows = false; // Fixed32KiB layer0/G2 diagnostic, never a benchmark.
  bool fused_residual_norm =
      false; // N4b attention residual/postnorm/expert encoding; BF16 router retained.
  bool fused_attention_o = false; // N4a full-row merge and O-input encoding; same direct-div math.
  bool fp8_kv_cache = false; // N3b E4M3 per-token/head K,V with FP32 scales; no BF16 cache shadow.
  bool gqa8_decode_attention = false; // N3a BF16 cache, grouped BF16 QK/PV MMA, FP32 softmax/fixup.
  bool block_expert_prefill = false;  // Explicit phase override, never a silent fallback.
  bool device_decode = false, decode_graphs = false, full_decode_graph = false;
  bool controlled_chunk_graphs =
      false; // S2 explicit persistent chunk_graph_with_eager_tail policy.
  bool prefill_graphs =
      false; // Explicit fixed-prompt whole-prefill graph, prepared/uploaded before timers.
  bool fast_w4a8 = false;
  // Diagnostic one-token teacher forcing, not used in performance/generation.
  int diagnostic_decode_token = -1;
  int diagnostic_cancel_after = -1;
  const std::atomic<bool> *cancellation_requested = nullptr;
  int snapshot_layer = -1, snapshot_position = 0;
};
struct SnapshotRegion {
  std::string name;
  std::vector<int> shape;
  std::size_t offset = 0, count = 0;
};
std::vector<SnapshotRegion> OperatorSnapshotLayout(int layer, int position);
struct InferenceResult {
  std::vector<unsigned char> diagnostic_exl3_fixture;
  std::vector<unsigned> diagnostic_h128_ranges;
  std::vector<int> diagnostic_route_ids;
  std::string actual_attention_projection, actual_expert_dispatch, diagnostic_attention_sha256;
  std::vector<int> generated_ids;
  std::vector<float> logits;
  std::vector<float> hidden;
  std::vector<float> trace;
  std::vector<unsigned char> quantization_rows;
  std::vector<float> operator_snapshot;
  std::vector<SnapshotRegion> snapshot_regions;
  std::string weights_sha256;
  std::string kernel_profile;
  std::string math_profile = kCompensatedMathProfile;
  std::string execution_profile = "existing";
  int prefill_chunk = 1;
  double initialization_seconds = 0, inference_seconds = 0;
  double prefill_seconds = 0, decode_seconds = 0, decode_setup_seconds = 0,
         prefill_setup_seconds = 0;
  std::string decode_execution = "host";
  std::string prefill_execution = "stream";
  bool cancelled = false;
  int prefill_tokens = 0, decode_steps = 0;
  int mtp_depth = 0;
  std::uint64_t mtp_resident_bytes = 0, mtp_groups = 0, mtp_draft_calls = 0, mtp_proposed = 0,
                mtp_accepted = 0, mtp_ordinary_tails = 0;
  // CUDA stream elapsed time through the final prompt logits, before selection.
  // Kept separate from wall-clock TTFT/prefill and N-1 decode measurements.
  double prompt_forward_stream_seconds = 0;
  std::uint64_t weights_bytes = 0, kv_bytes = 0, resident_buffer_bytes = 0;
  std::uint64_t device_free_before = 0, device_free_after_initialization = 0,
                device_free_after_execution = 0;
};
Result<InferenceResult> RunInference(const InferenceOptions &options);
} // namespace mellum16

namespace mellum16 {
Status ProbeHeadBF16Prefetch(std::ostream &, const std::filesystem::path &,
                             const std::filesystem::path &, bool = false);
}

namespace mellum16 {
Status ProbeLegacyBF16Partition(std::ostream &, const std::filesystem::path &,
                                const std::filesystem::path &, bool = false);
}

namespace mellum16 {
Status ProbeHeadBF16MMA(std::ostream &, const std::filesystem::path &,
                        const std::filesystem::path &, bool = false);
}

namespace mellum16 {
Status ProbeNormFP8Exact(std::ostream &);
}

namespace mellum16 {
Status ProbeNormFP8ExactProduction(std::ostream &, const std::filesystem::path &,
                                   const std::filesystem::path &, bool = false);
}

namespace mellum16 {
Status ProbeCombineNormExact(std::ostream &, const std::filesystem::path &,
                             const std::filesystem::path &, int, int, bool = false);
}

namespace mellum16 {
Status ProbePositiveNorm(std::ostream &);
Status ProbePositiveNormProduction(std::ostream &, const std::filesystem::path &,
                                   const std::filesystem::path &, bool = false);
} // namespace mellum16

namespace mellum16 {
Status ProbeRouterWarp32(std::ostream &, const std::filesystem::path &,
                         const std::filesystem::path &, bool = false);
}

namespace mellum16 {
Status ProbeRmsTailNorm(std::ostream &);
Status ProbeRmsTailNormProduction(std::ostream &, const std::filesystem::path &,
                                  const std::filesystem::path &, bool = false);
} // namespace mellum16

namespace mellum16 {
Status ProbeWideNorm(std::ostream &);
Status ProbeWideNormProduction(std::ostream &, const std::filesystem::path &,
                               const std::filesystem::path &, bool = false);
} // namespace mellum16
