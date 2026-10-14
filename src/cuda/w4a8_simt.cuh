// Mellum W4A8 M1 SIMT candidate; Gem16 packed-load/row-warp mechanics only.
// Exact E4M3 fixed units (1/512) and doubled E2M1 weights; K16 int32 dot/1024.
namespace gem_reuse {
__device__ __forceinline__ int W4A8FixedE4M3(unsigned code) {
  const unsigned exponent = (code >> 3) & 15, mantissa = code & 7;
  const int value =
      exponent ? static_cast<int>((8 + mantissa) << (exponent - 1)) : static_cast<int>(mantissa);
  return code & 128 ? -value : value;
}
__device__ __forceinline__ float W4A8Dot16(const unsigned char *weight,
                                           const unsigned char *activation) {
  int dot = 0;
#pragma unroll
  for (unsigned group = 0; group < 4; ++group) {
    const unsigned w = W4A4SignedBasis4(static_cast<unsigned>(weight[group * 2]) |
                                        (static_cast<unsigned>(weight[group * 2 + 1]) << 8));
    const unsigned a = LoadU32(activation + group * 4);
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      const unsigned code = (a >> (j * 8)) & 255;
      if ((code & 127) == 127)
        return __int_as_float(0x7fc00000);
      const int v = static_cast<int>(w << (24 - j * 8)) >> 24;
      dot += v * W4A8FixedE4M3(code);
    }
  }
  // <=16*12*229376=44040192, no int32 overflow. Preserve RN K16 F32 boundary.
  return __fmul_rn(__int2float_rn(dot), 1.f / 1024.f);
}
__global__ void ExpertW4A8SIMT(const DeviceMatrix *bindings, int layer, const int *ids,
                               const unsigned char *activation, const float *activationScales,
                               float *gate, float *up, int *bad) {
  constexpr unsigned rows = 896, cols = 2304;
  if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 || gridDim.x != 224 ||
      gridDim.y != 8 || gridDim.z != 1 || layer < 0 || layer >= 28 || !bindings || !ids ||
      !activation || !activationScales || !gate || !up ||
      (reinterpret_cast<std::uintptr_t>(activation) & 3)) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  const unsigned expert = static_cast<unsigned>(ids[blockIdx.y]);
  if (expert >= 64) {
    if (threadIdx.x == 0)
      atomicExch(bad, 1);
    return;
  }
  const unsigned logicalRow = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x & 31;
  const bool isUp = logicalRow >= rows;
  const unsigned row = logicalRow - (isUp ? rows : 0);
  const auto m = bindings[(layer * 64 + expert) * 3 + (isUp ? 1 : 0)];
  if (m.rows != rows || m.cols != cols || m.format != 2 || !m.data || !m.scales || !m.divisor ||
      (reinterpret_cast<std::uintptr_t>(m.data) & 3)) {
    if (lane == 0)
      atomicExch(bad, 1);
    return;
  }
  const float divisor = *m.divisor, activationScale = activationScales[0];
  if (!isfinite(divisor) || divisor <= 0 || !isfinite(activationScale) || activationScale <= 0) {
    if (lane == 0)
      atomicExch(bad, 1);
    return;
  }
  float sum = 0, error = 0;
  const auto *w = m.data + row * (cols / 2);
  const auto *sf = m.scales + row * (cols / 16);
  for (unsigned block = lane; block < cols / 16; block += 32) {
    const float dot = W4A8Dot16(w + block * 8, activation + block * 16);
    AccumulateScaledFP8RN(dot, DecodeFP8(sf[block]), sum, error);
  }
  sum = WarpAccurateReduce(sum, error);
  if (lane == 0) {
    const float value = B(__fdiv_rn(__fmul_rn(sum, activationScale), divisor));
    (isUp ? up : gate)[blockIdx.y * rows + row] = value;
    Finite(value, bad);
  }
}
} // namespace gem_reuse
