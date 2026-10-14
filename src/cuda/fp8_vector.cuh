// Packed-load mechanism follows read-only gem16 FP8 sm120.cu, Apache-2.0.
// Mellum SIMT adaptation: exact pair decode, compensated four-value work partition.
__device__ __forceinline__ float4 DecodeFP8Four(unsigned packed) {
  const auto lo = __half22float2(static_cast<__half2>(
      __nv_cvt_fp8x2_to_halfraw2(static_cast<unsigned short>(packed), __NV_E4M3)));
  const auto hi = __half22float2(static_cast<__half2>(
      __nv_cvt_fp8x2_to_halfraw2(static_cast<unsigned short>(packed >> 16), __NV_E4M3)));
  return make_float4(lo.x, lo.y, hi.x, hi.y);
}
__device__ __forceinline__ void ProjectFP8VectorRow(DeviceMatrix m, const unsigned char *x,
                                                    const float *scales, float *y, int *bad, int t,
                                                    int r) {
  if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 || t != 0 || m.format != 1 ||
      (m.cols != 2304 && m.cols != 4096) || m.rows <= 0 || m.rows > 4096 || !m.data || !m.scales ||
      !x || !scales || !y || (reinterpret_cast<std::uintptr_t>(m.data) & 3) ||
      (reinterpret_cast<std::uintptr_t>(x) & 3)) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  const unsigned lane = threadIdx.x & 31;
  if (r >= m.rows)
    return;
  float sum = 0, error = 0;
  for (int c = lane * 4; c < m.cols; c += 128) {
    const float4 w = DecodeFP8Four(Load32(m.data + r * m.cols + c)),
                 a = DecodeFP8Four(Load32(x + c));
    Accumulate(__fmul_rn(w.x, a.x), sum, error);
    Accumulate(__fmul_rn(w.y, a.y), sum, error);
    Accumulate(__fmul_rn(w.z, a.z), sum, error);
    Accumulate(__fmul_rn(w.w, a.w), sum, error);
  }
  sum = WarpAccurateReduce(sum, error);
  if (lane == 0) {
    const float value = B(__fmul_rn(__fmul_rn(sum, scales[0]), BF(m.scales, r)));
    y[r] = value;
    Finite(value, bad);
  }
}
__global__ void ProjectFP8Vector(DeviceMatrix m, const unsigned char *x, const float *scales,
                                 float *y, int *bad) {
  if (gridDim.y != 1 || gridDim.z != 1 || gridDim.x != static_cast<unsigned>((m.rows + 3) / 4)) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  ProjectFP8VectorRow(m, x, scales, y, bad, 0, blockIdx.x * 4 + threadIdx.x / 32);
}
