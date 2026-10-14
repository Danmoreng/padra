// Integer K4 standard tile and procedural mul1 FP16 codec. Scheduling/warp
// cooperation follows Gem16 c0a807a; no E4M3 rounding. See PROVENANCE.md.
#pragma once
namespace exl3_fp16 {
__device__ __forceinline__ unsigned Permutation(unsigned input, unsigned output) {
  return ((output & 7) * 4 + ((input & 7) >> 1)) * 8 + (output >= 8 ? 4 : 0) +
         (input >= 8 ? 2 : 0) + (input & 1);
}
__device__ __forceinline__ __half Mul1(unsigned state) {
  const unsigned h = state * 0x83DCD12DU;
  const unsigned s = (h & 255) + ((h >> 8) & 255) + ((h >> 16) & 255) + (h >> 24);
  return __hfma(__ushort_as_half(0x6400 + s), __ushort_as_half(0x1eee), __ushort_as_half(0xc931));
}
// Every lane loads one word; every shuffle is executed by the entire warp.
__device__ __forceinline__ __half DecodeK4(unsigned word, unsigned input, unsigned output) {
  const int end = int(Permutation(input, output)) * 4 + 4 + 1024;
  const int begin = end - 16, last = (end - 1) / 32, first = begin / 32,
            shift = (last + 1) * 32 - end;
  const unsigned a = __shfl_sync(0xffffffffU, word, last & 31);
  const unsigned b = __shfl_sync(0xffffffffU, word, first & 31);
  return Mul1(__funnelshift_r(a, b, shift) & 65535U);
}
// Same operation order as InputHad/OutputHad. One normalization exactly once.
__device__ __forceinline__ void H128(float (&v)[4]) {
  const float a = __fadd_rn(v[0], v[1]), b = __fsub_rn(v[0], v[1]);
  const float c = __fadd_rn(v[2], v[3]), d = __fsub_rn(v[2], v[3]);
  v[0] = __fadd_rn(a, c);
  v[1] = __fadd_rn(b, d);
  v[2] = __fsub_rn(a, c);
  v[3] = __fsub_rn(b, d);
  const unsigned lane = threadIdx.x & 31;
#pragma unroll
  for (unsigned mask = 1; mask < 32; mask *= 2) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float other = __shfl_xor_sync(0xffffffffU, v[i], mask);
      v[i] = (lane & mask) ? __fsub_rn(other, v[i]) : __fadd_rn(v[i], other);
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
    v[i] = __fmul_rn(v[i], 0.08838834764831845f);
}
} // namespace exl3_fp16
