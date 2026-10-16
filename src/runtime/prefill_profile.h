#pragma once
#include <algorithm>
#include <cstddef>
#ifndef PADRA_PREFILL_SPLIT_KV
#define PADRA_PREFILL_SPLIT_KV 0
#endif
#ifndef PADRA_PREFILL_MATRIX_LOAD
#define PADRA_PREFILL_MATRIX_LOAD 0
#endif
namespace mellum16 {
inline constexpr int kPrefillSplitThreshold = 8192;
inline constexpr std::size_t PrefillPartialBytes(std::size_t chunk) {
  return PADRA_PREFILL_SPLIT_KV ? std::min(chunk, std::size_t(512)) * 32 * 4 * 130 * sizeof(float)
                                : 0;
}
} // namespace mellum16
