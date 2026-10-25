#pragma once
#include "runtime/inference.h"

namespace mellum16 {
// Admission shared by the SIMT runtime and checkpoint-free host tests.
inline Status ValidateSimtFeatures(const InferenceOptions &options) {
  if (!options.mtp.artifact.empty())
    return {StatusCode::kUnsupported, "MTP decoding is not implemented in the SIMT backend"};
  if (options.q8_head)
    return {StatusCode::kUnsupported, "the shared Q8 head is not implemented in the SIMT backend"};
  if (options.capture_hidden)
    return {StatusCode::kUnsupported, "final hidden-state capture is not implemented in the SIMT backend"};
  if (options.kv_cache_format != KVCacheFormat::BF16 || options.fp8_kv_cache)
    return {StatusCode::kUnsupported, "SIMT backend currently requires BF16 KV"};
  if (!options.exl3_experts || options.exl3_rate != 4)
    return {StatusCode::kUnsupported, "SIMT backend requires the existing EXL3 K4 artifact"};
  return {};
}
} // namespace mellum16
