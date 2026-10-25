#pragma once
#include "runtime/prefill_state.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace kv_checkpoint {
constexpr int Window = 1024, Features = 512, Threads = 256, Blocks = 128;
// Copy represented rows, including scale bits, from the old ring or the
// containing chunk. The layout is identical to the live cache; no decoding or
// requantization occurs. Unwritten pre-token-zero slots are never read.
template <int RowBytes>
__device__ void CopyEncodedPlane(const void *oldRows, const void *currentRows, void *savedRows,
                                 int base, int anchor) {
  constexpr int Words = RowBytes / sizeof(std::uint32_t);
  const auto *old = static_cast<const std::uint32_t *>(oldRows);
  const auto *current = static_cast<const std::uint32_t *>(currentRows);
  auto *saved = static_cast<std::uint32_t *>(savedRows);
  for (int i = blockIdx.x * Threads + threadIdx.x; i < Window * Words; i += Blocks * Threads) {
    const int slot = i / Words, word = i % Words;
    const int position = anchor - 1 - ((anchor - 1 - slot + Window) % Window);
    saved[i] = position >= base ? current[(position - base) * Words + word]
               : position >= 0  ? old[i]
                                : 0;
  }
}
template <bool Native>
__global__ void CaptureEncoded(const void *oldK, const void *oldV, const void *oldScales,
                               const void *currentK, const void *currentV,
                               const void *currentScales, void *saved,
                               const mellum16::PrefillState *state, int tokens, int anchor,
                               int *bad, int maximum = 131072) {
  if (!oldK || !oldV || !currentK || !currentV || !saved || !state || !bad ||
      (Native ? (oldScales || currentScales) : (!oldScales || !currentScales)) ||
      !mellum16::ValidPrefillState(*state, tokens, state->capacity, 1, maximum) ||
      state->capacity < Window || anchor <= state->base || anchor > state->base + tokens ||
      saved == oldK || saved == oldV || saved == currentK || saved == currentV ||
      saved == oldScales || saved == currentScales || blockDim.x != Threads || blockDim.y != 1 ||
      blockDim.z != 1 || gridDim.x != Blocks || gridDim.y != 1 || gridDim.z != 1) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  if (*bad)
    return;
  constexpr int RowBytes = Native ? 544 : 512;
  auto *bytes = static_cast<unsigned char *>(saved);
  CopyEncodedPlane<RowBytes>(oldK, currentK, bytes, state->base, anchor);
  CopyEncodedPlane<RowBytes>(oldV, currentV, bytes + Window * RowBytes, state->base, anchor);
  if constexpr (!Native)
    CopyEncodedPlane<32>(oldScales, currentScales, bytes + Window * RowBytes * 2, state->base,
                         anchor);
}
// Called before the local suffix commit, after the containing chunk's current
// K/V have already been represented as BF16. Copies bits only; no new rounding
// and no change to expert grouping, chunk partition or forward arithmetic.
__global__ void CaptureLocal(const std::uint16_t *oldK, const std::uint16_t *oldV,
                             const std::uint16_t *currentK, const std::uint16_t *currentV,
                             std::uint16_t *savedK, std::uint16_t *savedV,
                             const mellum16::PrefillState *state, int tokens, int anchor, int *bad,
                             int maximum = 131072) {
  if (!oldK || !oldV || !currentK || !currentV || !savedK || !savedV || !state || !bad ||
      !mellum16::ValidPrefillState(*state, tokens, state->capacity, 1, maximum) ||
      state->capacity < Window || anchor <= state->base || anchor > state->base + tokens ||
      savedK == savedV || savedK == oldK || savedK == oldV || savedK == currentK ||
      savedK == currentV || savedV == oldK || savedV == oldV || savedV == currentK ||
      savedV == currentV || blockDim.x != Threads || blockDim.y != 1 || blockDim.z != 1 ||
      gridDim.x != Blocks || gridDim.y != 1 || gridDim.z != 1) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  if (*bad)
    return;
  for (int i = blockIdx.x * Threads + threadIdx.x; i < Window * Features; i += Blocks * Threads) {
    const int slot = i / Features, feature = i % Features;
    const int position = anchor - 1 - ((anchor - 1 - slot + Window) % Window);
    std::uint16_t k = 0, v = 0;
    if (position >= state->base) {
      const int source = (position - state->base) * Features + feature;
      k = currentK[source];
      v = currentV[source];
    } else if (position >= 0) {
      k = oldK[i];
      v = oldV[i];
    }
    // Before token0 no KV row exists. Do not read uninitialized/future slots.
    savedK[i] = k;
    savedV[i] = v;
  }
}
} // namespace kv_checkpoint
