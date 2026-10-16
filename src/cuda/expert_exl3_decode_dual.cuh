// Exact paired gate/up grid;each plane retains its own matrix,suh,H128 and MMA.
// Transpose the M1 MMA ownership: exact weights in A, replicated vector in B.
// One m16n8 product covers16 outputs,halving duplicate MMA instructions.
// Requires actual production bit-equality proof; new research orientation.
// N128/eight warps: warp H128 input, per-expert exact weight decode/MMA,
// and complete H128 output in one kernel. Gate/Up remain distinct.
#pragma once
#include "expert_exl3_decode_lane.cuh"
#include "expert_exl3_lane_codec.cuh"
namespace exl3_mma_dual {
__device__ __forceinline__ void Admission(float v, __half h, unsigned *stats) {
  const unsigned bits = __float_as_uint(v), half = __half_as_ushort(h);
  atomicMax(stats + (bits >> 31), bits & 0x7fffffffU);
  atomicAdd(stats + 2, 1U);
  if ((bits & 0x7fffffffU) == 0) {
    atomicAdd(stats + 3, 1U);
    if (bits >> 31)
      atomicAdd(stats + 4, 1U);
  }
  if ((half & 0x7c00) == 0 && (half & 1023))
    atomicAdd(stats + 5, 1U);
  if ((half & 0x7fff) == 0 && v != 0)
    atomicAdd(stats + 6, 1U);
  if ((half & 0x7c00) == 0x7c00)
    atomicAdd(stats + 7, 1U);
  if (__half2float(h) != v)
    atomicAdd(stats + 8, 1U);
}
template <int Rate>
__global__ void Projection(const DeviceMatrix *matrices, int layer, int requested, const int *ids,
                           const float *input, float *output, float *otherOutput, int pairs,
                           int *bad, unsigned *ranges = nullptr) {
  const int part = requested == 0 ? int(blockIdx.z) : requested;
  if (requested == 0 && blockIdx.z == 1) {
    if (!otherOutput) {
      if (!threadIdx.x)
        atomicExch(bad, 1);
      return;
    }
    output = otherOutput;
    if (ranges)
      ranges += 9;
  }
  const int pair = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  if (pair >= pairs)
    return;
  const int expert = ids[pair];
  if (expert < 0 || expert >= 64) {
    if (!threadIdx.x)
      atomicExch(bad, 1);
    return;
  }
  const auto m = exl3_expert::Matrix(matrices, layer, part, ids, pair, bad);
  const int base = blockIdx.x * 128, group = lane / 4, ki = (lane % 4) * 2;
  if (m.format != (Rate == 4 ? 3 : 5) || base + 128 > m.rows || m.cols < 128 || m.cols % 128 ||
      m.cols > 2304 || blockDim.x != 256) {
    if (!threadIdx.x)
      atomicExch(bad, 1);
    return;
  }
  __shared__ __align__(16) __half transformed[2304];
  __shared__ float raw[128];
  for (int h = warp; h < m.cols / 128; h += 8) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int k = h * 128 + lane * 4 + j;
      v[j] = __fmul_rn(input[std::uint64_t(part == 2 ? pair : pair / 8) * m.cols + k],
                       __half2float(__ushort_as_half(m.suh[k])));
    }
    exl3_fp16::H128(v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float x = B(v[j]);
      const auto value = __float2half_rn(x);
      if (!isfinite(x) || (__half_as_ushort(value) & 0x7c00) == 0x7c00)
        atomicExch(bad, 1);
      transformed[h * 128 + lane * 4 + j] = value;
      if (ranges && blockIdx.x == 0)
        Admission(x, value, ranges);
    }
  }
  __syncthreads();
  exl3_lane_codec::Word<Rate> prefetched[4];
  const int iterations = m.cols / 16;
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    const std::uint64_t t = std::uint64_t(d) * std::uint64_t(m.rows / 16) + base / 16 + warp;
    prefetched[d] = exl3_lane_codec::Load<Rate>(m.data + t * (32 * Rate), lane);
  }
  float accum[4] = {};
  for (int at = 0; at < iterations; at += 4) {
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      const int iteration = at + d, k = iteration * 16;
      const auto word = prefetched[d];
      if (iteration + 4 < iterations) {
        const std::uint64_t t =
            std::uint64_t(iteration + 4) * std::uint64_t(m.rows / 16) + base / 16 + warp;
        prefetched[d] = exl3_lane_codec::Load<Rate>(m.data + t * (32 * Rate), lane);
      }
      const unsigned b[2] = {exl3_mma_dot::PackHalf(transformed + k + ki),
                             exl3_mma_dot::PackHalf(transformed + k + ki + 8)};
      unsigned a[4];
      exl3_lane_codec::Four<Rate>(word, a);
      exl3_masked::Mma(a, b, accum);
    }
  }
  if ((lane % 4) == 0) {
    raw[warp * 16 + group] = accum[0];
    raw[warp * 16 + group + 8] = accum[2];
  }
  __syncthreads();
  if (warp == 0) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      v[j] = raw[lane * 4 + j];
    exl3_fp16::H128(v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int n = base + lane * 4 + j;
      const float value = B(__fmul_rn(v[j], __half2float(__ushort_as_half(m.svh[n]))));
      if (!isfinite(value))
        atomicExch(bad, 1);
      output[std::uint64_t(pair) * m.rows + n] = value;
    }
  }
}
inline void Launch(const DeviceMatrix *matrices, int layer, int part, const int *ids,
                   const float *input, float *output, float *scratch, int pairs, int *bad,
                   cudaStream_t stream, unsigned *ranges = nullptr, float *otherOutput = nullptr,
                   int rate = 4) {
  const int rows = part == 2 ? 2304 : 896;
  if (rate == 6)
    Projection<6><<<dim3(rows / 128, pairs, part == 0 ? 2 : 1), 256, 0, stream>>>(
        matrices, layer, part, ids, input, output, otherOutput, pairs, bad, ranges);
  else
    Projection<4><<<dim3(rows / 128, pairs, part == 0 ? 2 : 1), 256, 0, stream>>>(
        matrices, layer, part, ids, input, output, otherOutput, pairs, bad, ranges);
}
} // namespace exl3_mma_dual
