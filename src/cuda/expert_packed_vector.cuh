// G2 research: current E4M3 codes/native NVFP4 weights, new FP32 reduction.
// Packed row/load mechanism retains the Gem16-derived SelectedVector control.
// Paired independent dot4 products share a K16 SF; two temporal accumulators.
namespace fast_expert {
__device__ __forceinline__ float PackedDot4RN(float4 a, float4 w) {
  const float p0 = __fmaf_rn(a.y, w.y, __fmul_rn(a.x, w.x));
  const float p1 = __fmaf_rn(a.w, w.w, __fmul_rn(a.z, w.z));
  return __fadd_rn(p0, p1);
}
template <bool Fused>
__global__ void SelectedPacked4(const DeviceMatrix *bindings, int layer, const int *ids,
                                const unsigned char *activation, const float *rowScales,
                                float *output, int *bad) {
  constexpr int rows = Fused ? 896 : 2304, cols = Fused ? 2304 : 896;
  if (!bindings || !ids || !activation || !rowScales || !output || !bad || layer < 0 ||
      layer >= 28 || blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
      gridDim.x != rows / 4 || gridDim.y != 8 || gridDim.z != 1 ||
      (reinterpret_cast<std::uintptr_t>(activation) & 3)) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  const int slot = blockIdx.y, expert = ids[slot];
  if (expert < 0 || expert >= 64) {
    if (threadIdx.x == 0)
      atomicExch(bad, 1);
    return;
  }
  const DeviceMatrix m = bindings[(layer * 64 + expert) * 3 + (Fused ? 0 : 2)],
                     u = bindings[(layer * 64 + expert) * 3 + 1];
  if (m.format != 2 || m.rows != rows || m.cols != cols || !m.data || !m.scales || !m.divisor ||
      (Fused &&
       (u.format != 2 || u.rows != rows || u.cols != cols || !u.data || !u.scales || !u.divisor))) {
    if (threadIdx.x == 0)
      atomicExch(bad, 1);
    return;
  }
  const int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned char *x = activation + (Fused ? 0 : slot * cols);
  float sum0 = 0, sum1 = 0, up0 = 0, up1 = 0;
  for (int c = lane * 4, iteration = 0; c < cols; c += 128, ++iteration) {
    const auto a = DecodeFP8Four(Load32(x + c));
    const auto w = DecodeFP8Four(gem_reuse::BasisFP8(m.data + row * (cols / 2) + c / 2));
    const float sf = DecodeFP8(m.scales[row * (cols / 16) + c / 16]);
    const float dot = PackedDot4RN(a, w);
    if (iteration & 1)
      sum1 = __fmaf_rn(dot, sf, sum1);
    else
      sum0 = __fmaf_rn(dot, sf, sum0);
    if constexpr (Fused) {
      const auto v = DecodeFP8Four(gem_reuse::BasisFP8(u.data + row * (cols / 2) + c / 2));
      const float uf = DecodeFP8(u.scales[row * (cols / 16) + c / 16]);
      const float other = PackedDot4RN(a, v);
      if (iteration & 1)
        up1 = __fmaf_rn(other, uf, up1);
      else
        up0 = __fmaf_rn(other, uf, up0);
    }
  }
  float sum = __fadd_rn(sum0, sum1), up = __fadd_rn(up0, up1);
  for (int d = 16; d; d /= 2) {
    sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, d));
    if constexpr (Fused)
      up = __fadd_rn(up, __shfl_down_sync(0xffffffff, up, d));
  }
  if (lane == 0) {
    const float sf = rowScales[Fused ? 0 : slot];
    float value = B((sum * sf) / (*m.divisor));
    if constexpr (Fused) {
      const float v = B((up * sf) / (*u.divisor));
      value = B(B(value / (1.f + expf(-value))) * v);
    }
    output[slot * rows + row] = value;
    Finite(value, bad);
  }
}
} // namespace fast_expert
