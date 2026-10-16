#include "runtime/inference.h"
#include <algorithm>
namespace mellum16 {
#if MELLUM16_HAS_CUDA
Result<InferenceResult> RunCudaInference(const InferenceOptions &);
#endif
std::vector<SnapshotRegion> OperatorSnapshotLayout(int layer, int position) {
  std::vector<SnapshotRegion> regions;
  std::size_t offset = 0;
  const auto add = [&](const char *name, std::vector<int> shape) {
    std::size_t count = 1;
    for (int d : shape)
      count *= d;
    regions.push_back({name, std::move(shape), offset, count});
    offset += count;
  };
  add("input", {2304});
  add("input_norm", {2304});
  add("q_proj", {32, 128});
  add("k_proj", {4, 128});
  add("v_proj", {4, 128});
  add("q_norm", {32, 128});
  add("k_norm", {4, 128});
  add("q_rope", {32, 128});
  add("k_rope", {4, 128});
  const int visible = layer % 4 == 3 ? position + 1 : std::min(position + 1, 1024);
  add("key_cache", {visible, 4, 128});
  add("value_cache", {visible, 4, 128});
  add("attention", {32, 128});
  add("o_proj", {2304});
  add("after_attention", {2304});
  add("post_norm", {2304});
  add("router_logits", {64});
  add("router_ids", {8});
  add("routing_weights", {8});
  add("expert_gate", {8, 896});
  add("expert_up", {8, 896});
  add("expert_activation", {8, 896});
  add("expert_down", {8, 2304});
  add("output", {2304});
  add("input_norm_stats", {2});
  add("post_norm_stats", {2});
  add("attention_scores", {32, visible});
  add("attention_probabilities", {32, visible});
  return regions;
}
Result<InferenceResult> RunInference(const InferenceOptions &o) {
  if ((o.cancellation_requested || o.diagnostic_cancel_after >= 0) && !o.device_decode)
    return Status(StatusCode::kUnsupported,
                  "token-boundary cancellation requires device decode state");
  if (o.diagnostic_cancel_after < -1 ||
      (o.diagnostic_cancel_after >= 0 &&
       (o.diagnostic_cancel_after < 1 || o.diagnostic_cancel_after >= o.generate)))
    return Status(StatusCode::kUnsupported, "diagnostic cancellation requires1 <= cancel-after < "
                                            "requested outputs; never use in speed comparisons");
  if (o.diagnostic_decode_token < -1 || o.diagnostic_decode_token >= 98304 ||
      (o.diagnostic_decode_token >= 0 && (!o.device_decode || o.generate != 2)))
    return Status(StatusCode::kUnsupported, "diagnostic decode token requires device state and "
                                            "exactly two outputs; not a generation benchmark");
  if (o.fast_w4a8 &&
      (!o.fp8_expert_activations || !o.native_decode || !o.device_decode || o.snapshot_layer >= 0))
    return Status(StatusCode::kUnsupported,
                  "W4A8F F1a requires native-w4a8, decode same, device/graph state and no operator "
                  "snapshots; no fallback");
  if (o.packed_expert_decode &&
      (!o.fast_w4a8 || !o.staged_experts || o.staged_mma_decode || !o.block_expert_prefill ||
       !o.ordinary_prefill_silu || !o.ordinary_fp8_simt || !o.online_prefill_attention ||
       !o.gqa8_decode_attention || !o.fused_attention_o || !o.fused_residual_norm ||
       o.q8_expert_decode || o.fp8_kv_cache || o.ordinary_rms_norm ||
       o.reciprocal_fused_quantization))
    return Status(StatusCode::kUnsupported,
                  "packed4 expert reduction requires selected N4b BF16-KV/E4M3 direct-div "
                  "block-prefill/vector-decode profile; no fallback");
  if (o.scaled_bf16_prefill &&
      (!o.gqa8_prefill_attention || !o.fp8_expert_activations || !o.compact_experts ||
       !o.block_expert_prefill || !o.ordinary_prefill_silu || o.prefill_chunk < 2 ||
       o.q8_expert_decode || o.fp8_kv_cache))
    return Status(StatusCode::kUnsupported, "scaled-BF16 hybrid prefill requires selected G4/G3 "
                                            "BF16-cache direct-div E4M3 profile; no fallback");
  if (o.prune_last_prefill_ffn && (!o.gqa8_prefill_attention || o.capture_trace ||
                                   o.snapshot_layer >= 0 || o.diagnostic_decode_token >= 0))
    return Status(StatusCode::kUnsupported,
                  "last-logit FFN pruning requires selected G3 normal token-ID inference without "
                  "all-row/teacher diagnostics; explicitly choose all-rows for diagnostics");
  if (o.exact_norm_prefill &&
      (!o.exact_fp8_encoding || !o.exl3_experts || (o.exl3_rate != 4 && o.exl3_rate != 6) ||
       o.prefill_chunk < 2 || o.ordinary_rms_norm))
    return Status(
        StatusCode::kUnsupported,
        "exact batch FP8 norm requires explicit K4/K6/corrected encoder/compensated batch profile");
  if (o.pair64_qknorm_prefill && (!o.exl3_experts || (o.exl3_rate != 4 && o.exl3_rate != 6) ||
                                  !o.fast_w4a8 || o.prefill_chunk < 2 || o.ordinary_rms_norm))
    return Status(StatusCode::kUnsupported,
                  "pair64 QK prefill requires explicit K4/K6/W4A8F/compensated batch");
  if (o.quad_router_prefill && (!o.exl3_experts || (o.exl3_rate != 4 && o.exl3_rate != 6) ||
                                !o.fast_w4a8 || o.prefill_chunk < 2 || o.ordinary_rms_norm))
    return Status(
        StatusCode::kUnsupported,
        "quad router prefill requires explicit native K4/K6/W4A8F/compensated normalization/batch");
  if (o.exl3_wide_input && (!o.exl3_packed_activations || !o.exl3_experts ||
                            (o.exl3_rate != 4 && o.exl3_rate != 6) || o.prefill_chunk < 2))
    return Status(StatusCode::kUnsupported, "wide-vector H128 requires explicit K4/K6/BF16-storage "
                                            "batch profile; T<4 unchanged declared SIMT");
  if (o.exl3_packed_activations &&
      (!o.exl3_experts || (o.exl3_rate != 4 && o.exl3_rate != 6) || !o.exl3_prefill_warp ||
       !o.exl3_silu_table || o.capture_trace || o.snapshot_layer >= 0 ||
       o.diagnostic_exl3_layer >= 0 || o.diagnostic_routes || o.capture_quantization_rows))
    return Status(
        StatusCode::kUnsupported,
        "BF16 expert storage requires K4/K6/warp-M32/exact table; no trace/operator/route "
        "captures; T<4 keeps declared widened SIMT");
  if (o.padded_bf16_prefill_attention && !o.packed_bf16_prefill_attention)
    return Status(StatusCode::kUnsupported,
                  "padded single-K requires explicit packed BF16 prefill");
  if (o.packed_bf16_prefill_attention && (!o.gqa8_prefill_attention || !o.exl3_experts ||
                                          (o.exl3_rate != 4 && o.exl3_rate != 6) || o.fp8_kv_cache))
    return Status(StatusCode::kUnsupported,
                  "packed BF16 chunk requires explicit native K4/K6/BF16-cache research profile");
  if (o.gqa8_prefill_attention &&
      (!o.fast_w4a8 || !o.staged_experts || o.staged_mma_decode || !o.block_expert_prefill ||
       !o.ordinary_prefill_silu || !o.ordinary_fp8_simt || !o.online_prefill_attention ||
       !o.gqa8_decode_attention || !o.fused_attention_o || !o.fused_residual_norm ||
       o.q8_expert_decode || o.fp8_kv_cache || o.ordinary_rms_norm ||
       o.reciprocal_fused_quantization || o.packed_expert_decode))
    return Status(StatusCode::kUnsupported,
                  "G3 prefill pipeline requires selected N4b BF16-KV/E4M3 direct-div "
                  "block-prefill/vector-decode profile; no fallback");
  if (o.q8_expert_decode &&
      (!o.fast_w4a8 || !o.staged_experts || o.staged_mma_decode || !o.block_expert_prefill ||
       !o.ordinary_prefill_silu || !o.ordinary_fp8_simt || !o.online_prefill_attention ||
       o.ordinary_rms_norm))
    return Status(StatusCode::kUnsupported,
                  "Q8 expert decode requires explicit N1a online/block/vector W4A8F expf with "
                  "compensated norms; prefill stays E4M3, no fallback");
  if (o.gqa8_decode_attention &&
      (!o.fast_w4a8 || !o.staged_experts || o.staged_mma_decode || !o.block_expert_prefill ||
       !o.ordinary_prefill_silu || !o.ordinary_fp8_simt || !o.online_prefill_attention ||
       o.ordinary_rms_norm || o.q8_expert_decode))
    return Status(StatusCode::kUnsupported,
                  "GQA8 BF16 attention requires explicit N1a online/block/vector E4M3 W4A8F expf "
                  "with compensated norms; Q8 composition is separate, no fallback");
  if (o.reciprocal_fused_quantization && (!o.fused_attention_o || !o.fused_residual_norm))
    return Status(StatusCode::kUnsupported, "row-reciprocal quantization requires both selected N4 "
                                            "fused BF16/E4M3 pipelines; no fallback");
  if (o.capture_quantization_rows &&
      (!o.fused_attention_o || !o.fused_residual_norm || !o.full_decode_graph || o.generate != 2 ||
       o.diagnostic_cancel_after >= 0 || o.cancellation_requested))
    return Status(StatusCode::kUnsupported,
                  "quantization-row capture requires both N4 pipelines, full graph and exactly G2 "
                  "without cancellation; not for benchmarks");
  if (o.fused_residual_norm &&
      (!o.gqa8_decode_attention || o.fp8_kv_cache || o.q8_expert_decode || o.ordinary_rms_norm))
    return Status(StatusCode::kUnsupported, "residual-norm fusion requires explicit N3a BF16 "
                                            "KV/E4M3 compensated profile; no composition fallback");
  if (o.fused_attention_o && (!o.gqa8_decode_attention || o.fp8_kv_cache || o.q8_expert_decode))
    return Status(
        StatusCode::kUnsupported,
        "attention-o fusion requires explicit N3a BF16 KV/E4M3 profile; no composition fallback");
  if (o.fp8_kv_cache &&
      (!o.gqa8_decode_attention || !o.online_prefill_attention || o.prefill_chunk < 2))
    return Status(StatusCode::kUnsupported,
                  "E4M3 KV requires explicit N3a GQA8 online/block/vector E4M3 profile and "
                  "chunk>=2; no cache/precision fallback");
  if ((o.accurate_native_prefill || o.fp8_expert_activations) && !o.native_prefill)
    return Status(StatusCode::kInvalidArgument,
                  "quantized activation profile requires native prefill");
  if (o.accurate_native_prefill && o.fp8_expert_activations)
    return Status(StatusCode::kInvalidArgument, "choose one native activation profile");
  if (o.native_decode && !o.native_prefill)
    return Status(StatusCode::kInvalidArgument,
                  "quantized decode requires an explicit native prefill profile");
  if (o.ordinary_prefill_silu &&
      (!o.block_expert_prefill || !o.staged_experts || o.staged_mma_decode))
    return Status(
        StatusCode::kUnsupported,
        "expf prefill SiLU requires explicit block-prefill/vector-decode W4A8F; no fallback");
  if (o.block_expert_prefill && !o.staged_experts)
    return Status(StatusCode::kUnsupported,
                  "block expert prefill override requires staged decode profile; no fallback");
  if (o.staged_experts && (!o.compact_experts || o.ordinary_block_experts))
    return Status(StatusCode::kUnsupported, "staged BF16 experts require compact W4A8F and one "
                                            "declared expert arithmetic; no fallback");
  if (o.staged_mma_decode && !o.staged_experts)
    return Status(StatusCode::kUnsupported, "staged M1 MMA requires staged expert prefill");
  if (o.ordinary_block_experts && !o.compact_experts)
    return Status(StatusCode::kUnsupported,
                  "ordinary block experts require compact W4A8F profile; no fallback");
  if (o.compact_experts && (!o.fast_w4a8 || !o.ordinary_fp8_projections))
    return Status(StatusCode::kUnsupported,
                  "compact experts require explicit W4A8F ordinary projections; no fallback");
  if (o.ordinary_rms_norm && (!o.fast_w4a8 || !o.ordinary_fp8_projections))
    return Status(StatusCode::kUnsupported,
                  "ordinary RMS requires explicit W4A8F and ordinary FP8 projections; no fallback");
  if (o.ordinary_fp8_simt && !o.ordinary_fp8_projections)
    return Status(StatusCode::kUnsupported,
                  "ordinary vector decode requires ordinary FP8 projection profile");
  if (o.ordinary_fp8_projections && !o.fast_w4a8)
    return Status(StatusCode::kUnsupported,
                  "ordinary FP8 projections require explicit W4A8F; no fallback");
  if (o.online_prefill_attention &&
      (!o.fast_w4a8 || o.tc_prefill_attention || o.prefill_chunk < 2 || o.snapshot_layer >= 0))
    return Status(StatusCode::kUnsupported,
                  "online BF16 prefill requires W4A8F, chunk > 1, no operator snapshots and no "
                  "simultaneous corrected TC path; no fallback");
  if (o.tc_prefill_attention && (!(o.accurate_native_prefill || o.fp8_expert_activations) ||
                                 o.prefill_chunk < 2 || o.snapshot_layer >= 0))
    return Status(StatusCode::kUnsupported, "BF16 TC prefill attention requires corrected native "
                                            "A4/A8, chunk > 1, no operator snapshots");
  if (o.prefill_graphs &&
      (!o.exl3_experts || (o.exl3_rate != 4 && o.exl3_rate != 6) || !o.full_decode_graph ||
       o.prefill_chunk < 2 || o.capture_trace || o.snapshot_layer >= 0 ||
       o.diagnostic_exl3_layer >= 0 || o.diagnostic_routes || o.capture_quantization_rows))
    return Status(StatusCode::kUnsupported,
                  "whole-prefill graph requires explicit K4/K6/full decode "
                  "graph/batch without operator captures; no fallback");
  if (o.full_decode_graph && (!o.decode_graphs || !o.fast_w4a8))
    return Status(StatusCode::kUnsupported,
                  "full-step graph requires explicit W4A8F graph profile; no fallback");
  if (o.decode_graphs && !o.device_decode)
    return Status(StatusCode::kInvalidArgument, "graphs require device decode state");
  if (o.device_decode &&
      (!o.native_decode || !(o.accurate_native_prefill || o.fp8_expert_activations) ||
       o.prefill_chunk < 2))
    return Status(StatusCode::kUnsupported,
                  "device/graph decode requires explicit corrected A4 or A8 native prefill/decode");
  if (o.device_decode && o.snapshot_layer >= 0)
    return Status(StatusCode::kUnsupported,
                  "device/graph decode does not support operator snapshots; use full layer trace");
  if (o.prefill_chunk < 1 || o.prefill_chunk > kW4A8FMaxPrefillChunk)
    return Status(StatusCode::kInvalidArgument, "prefill chunk must be 1..2048");
  if (o.prefill_chunk > 512 &&
      (!o.fast_w4a8 || !o.online_prefill_attention || !o.ordinary_fp8_simt || !o.compact_experts ||
       !o.block_expert_prefill || !o.staged_experts || o.staged_mma_decode ||
       !o.ordinary_prefill_silu))
    return Status(StatusCode::kUnsupported,
                  "chunks above512 require explicit N1a ordinary block-prefill/vector-decode "
                  "online W4A8F expf profile; no fallback");
  if (o.native_prefill && o.prefill_chunk == 1)
    return Status(StatusCode::kInvalidArgument,
                  "native prefill requires chunk > 1; decode remains explicit W4A16");
  if (o.prefill_chunk > 1 && o.snapshot_layer >= 0 && o.input_ids.size() != 1)
    return Status(StatusCode::kUnsupported, "batched operator snapshots currently require a "
                                            "one-token prompt; use batched layer trace otherwise");
  if (o.input_ids.empty() || o.capacity < 1 || o.capacity > kW4A8FContextCapacity ||
      o.generate < 0 || o.generate > kW4A8FContextCapacity ||
      o.input_ids.size() + static_cast<std::size_t>(o.generate) >
          static_cast<std::size_t>(o.capacity))
    return Status(StatusCode::kInvalidArgument,
                  "require 1..8192 capacity, nonempty prompt and prompt+generation <= capacity");
  if (o.capacity > kLegacyContextCapacity && (!o.fast_w4a8 || !o.online_prefill_attention))
    return Status(
        StatusCode::kUnsupported,
        "capacity above4096 requires explicit W4A8F online prefill; no precision/context fallback");
  if (o.snapshot_layer < -1 || o.snapshot_layer > 27 ||
      (o.snapshot_layer >= 0 &&
       (o.snapshot_position < 0 ||
        static_cast<std::size_t>(o.snapshot_position) >= o.input_ids.size())))
    return Status(StatusCode::kInvalidArgument,
                  "operator snapshot requires layer 0..27 and a prompt position");
  for (auto id : o.input_ids)
    if (id < 0 || id >= 98304)
      return Status(StatusCode::kInvalidArgument, "token outside Mellum vocabulary");
#if MELLUM16_HAS_CUDA
  return RunCudaInference(o);
#else
  return Status(StatusCode::kUnsupported, "native inference requires the CUDA build");
#endif
}
} // namespace mellum16
