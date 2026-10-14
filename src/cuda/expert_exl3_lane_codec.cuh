// Exact four-fragment K4 decode for fixed MMA lane ownership.
// p0=lane*8; four adjacent pair windows end at8/16/24/32 bits.
// One previous-word shuffle replaces eight generic pair shuffles.
#pragma once
#include "expert_exl3_dot_codec.cuh"
namespace exl3_lane_codec {
__device__ __forceinline__ unsigned PairBits(unsigned window) {
  return exl3_dot_codec::Bits(exl3_dot_codec::FromStates((window >> 4) & 65535U, window & 65535U));
}
__device__ __forceinline__ void Four(unsigned word, unsigned (&a)[4]) {
  const unsigned previous = __shfl_sync(0xffffffffU, word, ((threadIdx.x & 31) + 31) & 31);
  a[0] = PairBits(__funnelshift_r(word, previous, 24));
  a[1] = PairBits(__funnelshift_r(word, word, 8));
  a[2] = PairBits(__funnelshift_r(word, previous, 16));
  a[3] = PairBits(word);
}
} // namespace exl3_lane_codec
