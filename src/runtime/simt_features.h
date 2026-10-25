#pragma once
#include "runtime/inference.h"

namespace mellum16 {
// Admission shared by the SIMT runtime and checkpoint-free host tests.
inline Status ValidateSimtFeatures(const InferenceOptions &options) {
  if (!options.mtp.artifact.empty()) {
    if (!options.mtp.exl3_experts || options.mtp.fp8_experts || !options.q8_head)
      return {StatusCode::kUnsupported, "SIMT MTP requires the EXL3-K4 draft and stored shared Q8 head"};
    if (options.mtp.depth < 1 || options.mtp.depth > 4)
      return {StatusCode::kInvalidArgument, "SIMT MTP depth must be between 1 and 4"};
    if (options.simt_reference_math || options.simt_reference_prefill || !options.simt_teacher_tokens.empty() || !options.simt_teacher_sequences.empty())
      return {StatusCode::kUnsupported, "SIMT MTP requires batched prefill and no teacher replay"};
  }
  if ((options.kv_cache_format != KVCacheFormat::BF16 && options.kv_cache_format != KVCacheFormat::Q8Token512Block32F16ScaleRotatedV2) || options.fp8_kv_cache)
    return {StatusCode::kUnsupported, "SIMT backend requires BF16 or native Q8_0 KV"};
  if (!options.exl3_experts || (options.exl3_rate != 3 && options.exl3_rate != 4))
    return {StatusCode::kUnsupported, "SIMT backend requires the EXL3 K3 or K4 artifact"};
  return {};
}
} // namespace mellum16
