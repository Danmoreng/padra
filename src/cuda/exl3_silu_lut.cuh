// Exact discrete BF16 gate -> RN-exp/division/BF16 SiLU function.
// Built once on GPU before inference; generic math table,not weight repacking.
#pragma once
namespace exl3_silu_lut {
__device__ __forceinline__ float Rounded(float g) {
  const float denominator = __fadd_rn(1.f, __double2float_rn(exp(double(-g))));
  return B(__fdiv_rn(g, denominator));
}
__global__ void Build(unsigned short *table) {
  const unsigned i = blockIdx.x * 256 + threadIdx.x;
  if (i < 65536)
    table[i] =
        static_cast<unsigned short>(__float_as_uint(Rounded(__uint_as_float(i << 16))) >> 16);
}
__global__ void Activate(float *gate, const float *up, const unsigned short *table, int count,
                         int *bad) {
  const int i = blockIdx.x * 128 + threadIdx.x;
  if (i >= count)
    return;
  const float g = gate[i], u = up[i];
  const unsigned bits = __float_as_uint(g);
  if (!table || !isfinite(g) || !isfinite(u) || (bits & 65535U)) {
    atomicExch(bad, 1);
    return;
  }
  const float s = __uint_as_float(unsigned(table[bits >> 16]) << 16);
  const float value = B(__fmul_rn(s, u));
  if (!isfinite(value))
    atomicExch(bad, 1);
  gate[i] = value;
}
} // namespace exl3_silu_lut
