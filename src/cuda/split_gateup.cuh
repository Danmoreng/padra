// Scheduling adapted from Danmoreng/gem16 sm120.cu, Apache-2.0.
// SelectedSplitGateUp: separate gate/up warps, direct replicated M1 registers.
// Mellum row-linear descriptors and compensated arithmetic are retained.
namespace gem_reuse {
template <bool A8, bool Compensated = true>
__global__ void ExpertSplitGateUp(const DeviceMatrix *bindings, int layer, const int *ids,
                                  const unsigned char *activation, const unsigned char *scales,
                                  const float *activationDivisors, float *gate, float *up,
                                  int *bad) {
  if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 8 || gridDim.z != 1 ||
      layer < 0 || layer >= 28 || !bindings || !ids || !activation || !activationDivisors ||
      !gate || !up || (!A8 && !scales)) {
    if (threadIdx.x == 0)
      atomicExch(bad, 1);
    return;
  }
  const unsigned expert = static_cast<unsigned>(ids[blockIdx.y]);
  if (expert >= 64) {
    if (threadIdx.x == 0)
      atomicExch(bad, 1);
    return;
  }
  const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32, quarter = lane % 4,
                 group = lane / 4;
  const unsigned logicalWarp = blockIdx.x * 8 + warp;
  constexpr unsigned rows = 896, cols = 2304, rowTiles = rows / 8;
  if (logicalWarp >= 2 * rowTiles)
    return;
  const bool isUp = logicalWarp >= rowTiles;
  const unsigned firstRow = (logicalWarp - (isUp ? rowTiles : 0)) * 8;
  const auto m = bindings[(layer * 64 + expert) * 3 + (isUp ? 1 : 0)];
  if (m.rows != rows || m.cols != cols || m.format != 2 || !m.data || !m.scales || !m.divisor) {
    if (lane == 0)
      atomicExch(bad, 1);
    return;
  }
  const float divisor = *m.divisor, actDivisor = activationDivisors[0];
  if (!isfinite(divisor) || divisor <= 0 || !isfinite(actDivisor) || actDivisor <= 0) {
    if (lane == 0)
      atomicExch(bad, 1);
    return;
  }
  float sum[4] = {}, error[4] = {};
  for (unsigned kb = 0; kb < cols / 64; ++kb) {
    if constexpr (A8) {
#pragma unroll
      for (unsigned q = 0; q < 4; ++q) {
        const unsigned offset = (firstRow + group) * (cols / 2) + kb * 32 + q * 8 + quarter * 2;
        unsigned b[2] = {};
        b[q & 1] = BasisFP8(m.data + offset);
        const unsigned ai = kb * 64 + (q / 2) * 32 + quarter * 4;
        const unsigned aLow = LoadU32(activation + ai), aHigh = LoadU32(activation + ai + 16);
        unsigned a[4] = {aLow, aLow, aHigh, aHigh};
        a[(q & 1) ? 0 : 2] = 0;
        a[(q & 1) ? 1 : 3] = 0;
        float part[4] = {};
        MmaFP8(a, b, part);
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
          const unsigned col = firstRow + quarter * 2 + (j & 1);
          AccumulateScaledFP8RN(part[j], DecodeFP8(m.scales[col * (cols / 16) + kb * 4 + q]),
                                sum[j], error[j]);
        }
      }
    } else {
      const unsigned ai = kb * 32 + quarter * 4, wi = (firstRow + group) * (cols / 2) + ai;
      const unsigned a0 = LoadU32(activation + ai), a2 = LoadU32(activation + ai + 16);
      const unsigned sa = LoadU32(scales + kb * 4),
                     sb = LoadU32(m.scales + (firstRow + group) * (cols / 16) + kb * 4);
      MmaReused<Compensated>(sum[0], sum[1], sum[2], sum[3], a0, a0, a2, a2, LoadU32(m.data + wi),
                             LoadU32(m.data + wi + 16), sa, sb, error);
    }
  }
  if (lane < 4) {
    float *output = (isUp ? up : gate) + blockIdx.y * rows + firstRow + lane * 2;
#pragma unroll
    for (unsigned j = 0; j < 2; ++j) {
      const float total = __fadd_rn(sum[j], error[j]);
      const float value = A8 ? B(__fdiv_rn(__fmul_rn(total, actDivisor), divisor))
                             : B(__fdiv_rn(total, __fmul_rn(divisor, actDivisor)));
      output[j] = value;
      Finite(value, bad);
    }
  }
}
} // namespace gem_reuse
