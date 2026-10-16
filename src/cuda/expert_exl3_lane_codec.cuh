// Exact four-fragment K4 decode for fixed MMA lane ownership.
// p0=lane*8; four adjacent pair windows end at8/16/24/32 bits.
// One previous-word shuffle replaces eight generic pair shuffles.
#pragma once
#include "expert_exl3_dot_codec.cuh"
#include <type_traits>
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
struct Word6 {
  unsigned first, last;
};
template <int Rate> using Word = std::conditional_t<Rate == 4, unsigned, Word6>;
template <int Rate>
__device__ __forceinline__ Word<Rate> Load(const unsigned char *tile, unsigned lane) {
  static_assert(Rate == 4 || Rate == 6);
  if constexpr (Rate == 4) {
    return reinterpret_cast<const unsigned *>(tile)[lane];
  } else {
    // Each lane owns eight consecutive states: 48 bits, including word crossings.
    // The native stream swaps adjacent U16 words; XOR restores logical MSB order.
    const auto *words = reinterpret_cast<const unsigned short *>(tile);
    const unsigned i = lane * 3;
    return {(unsigned(words[i ^ 1]) << 16) | words[(i + 1) ^ 1], words[(i + 2) ^ 1]};
  }
}
__device__ __forceinline__ unsigned PairBits6(unsigned window) {
  return exl3_dot_codec::Bits(exl3_dot_codec::FromStates((window >> 6) & 65535U, window & 65535U));
}
template <int Rate> __device__ __forceinline__ void Four(Word<Rate> word, unsigned (&a)[4]) {
  if constexpr (Rate == 4) {
    Four(word, a);
  } else {
    const unsigned previous = __shfl_sync(0xffffffffU, word.last, ((threadIdx.x & 31) + 31) & 31);
    a[0] = PairBits6(__funnelshift_r(word.first, previous, 20));
    a[1] = PairBits6((word.first << 4) | (word.last >> 12));
    a[2] = PairBits6(__funnelshift_r(word.first, previous, 8));
    a[3] = PairBits6((word.first << 16) | word.last);
  }
}
} // namespace exl3_lane_codec
