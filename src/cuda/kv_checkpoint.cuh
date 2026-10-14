#pragma once
#include "runtime/prefill_state.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace kv_checkpoint {
constexpr int Window = 1024, Features = 512, Threads = 256, Blocks = 128;
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
