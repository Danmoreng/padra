// Verifier rows sharing an expert share its decoded weight tiles. The K16 MMA
// order and BF16/H128 boundaries match exl3_mma_dual::Projection exactly.
#pragma once
namespace mtp_experts {
inline constexpr int kRows = 5;
template <bool Compact = false>
__global__ void BuildUnion(const int *ids, unsigned *masks, int *slots, int rows, int *bad,
                           int *activeExperts = nullptr, int *activeCount = nullptr) {
  const int expert = threadIdx.x;
  unsigned mask = 0;
  for (int r = 0; r < rows; ++r) {
    int found = -1;
    for (int s = 0; s < 8; ++s) {
      const int id = ids[r * 8 + s];
      if (id < 0 || id >= 64)
        atomicExch(bad, 1);
      if (id == expert)
        found = s;
    }
    slots[expert * kRows + r] = found;
    if (found >= 0)
      mask |= 1U << r;
  }
  masks[expert] = mask;
  if constexpr (Compact) {
    const unsigned active = __ballot_sync(0xffffffffU, mask != 0);
    __shared__ unsigned warpMasks[2];
    const int lane = expert & 31;
    if (!lane)
      warpMasks[expert / 32] = active;
    __syncthreads();
    // Prefix rank in increasing expert order, independent of route overlap.
    const unsigned preceding = lane ? ((1U << lane) - 1U) : 0U;
    const int rank = __popc(active & preceding) + (expert >= 32 ? __popc(warpMasks[0]) : 0);
    if (mask)
      activeExperts[rank] = expert;
    if (!expert)
      *activeCount = __popc(warpMasks[0]) + __popc(warpMasks[1]);
  }
}
template <int Rows, bool Compact = false, int Rate = 4>
__global__ void Projection(const DeviceMatrix *matrices, int layer, int requested,
                           const unsigned *masks, const int *slots, int rows, const float *input,
                           float *output, float *otherOutput, int *bad,
                           const int *activeExperts = nullptr, const int *activeCount = nullptr) {
  static_assert(Rate == 3 || Rate == 4 || Rate == 6);
  int expert = blockIdx.y;
  if constexpr (Compact) {
    const int count = *activeCount;
    if (count < 0 || count > 8 * Rows) {
      if (!threadIdx.x)
        atomicExch(bad, 1);
      return;
    }
    // Uniform rejection precedes all descriptors, shared setup and barriers.
    if (expert >= count)
      return;
    expert = activeExperts[expert];
    if (expert < 0 || expert >= 64) {
      if (!threadIdx.x)
        atomicExch(bad, 1);
      return;
    }
  }
  const int part = requested == 0 ? int(blockIdx.z) : requested;
  if (part == 1)
    output = otherOutput;
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const auto m = matrices[(layer * 64 + expert) * 3 + part];
  const int base = blockIdx.x * 128, group = lane / 4, ki = (lane % 4) * 2;
  __shared__ unsigned mask;
  __shared__ int abort;
  __shared__ int slot[Rows];
  if (!threadIdx.x) {
    mask = masks[expert];
    abort = *bad || rows != Rows ||
            m.format != (Rate == 3   ? 6
                         : Rate == 6 ? 5
                                     : 3) ||
            !output || base + 128 > m.rows || m.cols < 128 || m.cols > 2304 || m.cols % 128;
    for (int r = 0; r < Rows; ++r) {
      slot[r] = slots[expert * kRows + r];
      if ((mask & (1U << r)) && (slot[r] < 0 || slot[r] >= 8))
        abort = 1;
    }
    if (abort)
      atomicExch(bad, 1);
  }
  __syncthreads();
  if (abort || !mask)
    return;
  __shared__ __align__(16) __half transformed[Rows * 2304];
  __shared__ float raw[Rows * 128];
  for (int r = 0; r < Rows; ++r) {
    if (!(mask & (1U << r)))
      continue;
    for (int h = warp; h < m.cols / 128; h += 8) {
      float v[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int k = h * 128 + lane * 4 + j;
        const std::uint64_t row = part == 2 ? r * 8 + slot[r] : r;
        v[j] = __fmul_rn(input[row * m.cols + k], __half2float(__ushort_as_half(m.suh[k])));
      }
      exl3_fp16::H128(v);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float value = B(v[j]);
        const auto half = __float2half_rn(value);
        if (!isfinite(value) || (__half_as_ushort(half) & 0x7c00) == 0x7c00)
          atomicExch(bad, 1);
        transformed[r * 2304 + h * 128 + lane * 4 + j] = half;
      }
    }
  }
  __syncthreads();
  exl3_lane_codec::Word<Rate> prefetched[4];
  const int iterations = m.cols / 16;
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    const std::uint64_t t = std::uint64_t(d) * (m.rows / 16) + base / 16 + warp;
    prefetched[d] = exl3_lane_codec::Load<Rate>(m.data + t * (32 * Rate), lane);
  }
  float accum[4] = {};
  for (int at = 0; at < iterations; at += 4) {
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      const int iteration = at + d, k = iteration * 16;
      const auto word = prefetched[d];
      if (iteration + 4 < iterations) {
        const std::uint64_t t = std::uint64_t(iteration + 4) * (m.rows / 16) + base / 16 + warp;
        prefetched[d] = exl3_lane_codec::Load<Rate>(m.data + t * (32 * Rate), lane);
      }
      unsigned b[2] = {};
      if (group < Rows && (mask & (1U << group))) {
        b[0] = exl3_mma_dot::PackHalf(transformed + group * 2304 + k + ki);
        b[1] = exl3_mma_dot::PackHalf(transformed + group * 2304 + k + ki + 8);
      }
      unsigned a[4];
      exl3_lane_codec::Four<Rate>(word, a);
      exl3_masked::Mma(a, b, accum);
    }
  }
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    const int r = (lane % 4) * 2 + c;
    if (r < Rows && (mask & (1U << r))) {
      raw[r * 128 + warp * 16 + group] = accum[c];
      raw[r * 128 + warp * 16 + group + 8] = accum[2 + c];
    }
  }
  __syncthreads();
  if (warp < Rows && (mask & (1U << warp))) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      v[j] = raw[warp * 128 + lane * 4 + j];
    exl3_fp16::H128(v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int n = base + lane * 4 + j;
      const float value = B(__fmul_rn(v[j], __half2float(__ushort_as_half(m.svh[n]))));
      Finite(value, bad);
      output[(warp * 8 + slot[warp]) * m.rows + n] = value;
    }
  }
}
template <int Rate>
inline void LaunchRate(const DeviceMatrix *matrices, int layer, int part, const unsigned *masks,
                       const int *slots, int rows, const float *input, float *output, float *up,
                       int *bad, cudaStream_t stream, const int *activeExperts = nullptr,
                       const int *activeCount = nullptr) {
  if (bool(activeExperts) != bool(activeCount))
    throw std::runtime_error("MTP compact expert binding mismatch");
  const dim3 grid((part == 2 ? 2304 : 896) / 128, activeExperts ? std::min(64, 8 * rows) : 64,
                  part == 0 ? 2 : 1);
  if (activeExperts) {
    switch (rows) {
    case 2:
      Projection<2, true, Rate><<<grid, 256, 0, stream>>>(matrices, layer, part, masks, slots, rows,
                                                          input, output, up, bad, activeExperts,
                                                          activeCount);
      break;
    case 3:
      Projection<3, true, Rate><<<grid, 256, 0, stream>>>(matrices, layer, part, masks, slots, rows,
                                                          input, output, up, bad, activeExperts,
                                                          activeCount);
      break;
    case 4:
      Projection<4, true, Rate><<<grid, 256, 0, stream>>>(matrices, layer, part, masks, slots, rows,
                                                          input, output, up, bad, activeExperts,
                                                          activeCount);
      break;
    case 5:
      Projection<5, true, Rate><<<grid, 256, 0, stream>>>(matrices, layer, part, masks, slots, rows,
                                                          input, output, up, bad, activeExperts,
                                                          activeCount);
      break;
    default:
      throw std::runtime_error("MTP expert row count unsupported");
    }
    return;
  }
  switch (rows) {
  case 2:
    Projection<2, false, Rate><<<grid, 256, 0, stream>>>(matrices, layer, part, masks, slots, rows,
                                                         input, output, up, bad);
    break;
  case 3:
    Projection<3, false, Rate><<<grid, 256, 0, stream>>>(matrices, layer, part, masks, slots, rows,
                                                         input, output, up, bad);
    break;
  case 4:
    Projection<4, false, Rate><<<grid, 256, 0, stream>>>(matrices, layer, part, masks, slots, rows,
                                                         input, output, up, bad);
    break;
  case 5:
    Projection<5, false, Rate><<<grid, 256, 0, stream>>>(matrices, layer, part, masks, slots, rows,
                                                         input, output, up, bad);
    break;
  default:
    throw std::runtime_error("MTP expert row count unsupported");
  }
}
inline void Launch(const DeviceMatrix *matrices, int layer, int part, const unsigned *masks,
                   const int *slots, int rows, const float *input, float *output, float *up,
                   int *bad, cudaStream_t stream, const int *activeExperts = nullptr,
                   const int *activeCount = nullptr, int rate = 4) {
  switch (rate) {
  case 3:
    LaunchRate<3>(matrices, layer, part, masks, slots, rows, input, output, up, bad, stream,
                  activeExperts, activeCount);
    break;
  case 4:
    LaunchRate<4>(matrices, layer, part, masks, slots, rows, input, output, up, bad, stream,
                  activeExperts, activeCount);
    break;
  case 6:
    LaunchRate<6>(matrices, layer, part, masks, slots, rows, input, output, up, bad, stream,
                  activeExperts, activeCount);
    break;
  default:
    throw std::runtime_error("MTP expert rate unsupported");
  }
}
} // namespace mtp_experts
