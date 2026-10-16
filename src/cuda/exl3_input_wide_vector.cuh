// Exact native warp-H128;8 independent warps/vector source-sidecar/output transport.
#pragma once
#include "expert_exl3_prefill_warp.cuh"
namespace exl3_input_wide_vector {
using exl3_fp16::H128;
__global__ void Input(const DeviceMatrix *matrices, int layer, int part, const int *ids,
                      const float *input, __half *transformed, int pairs, int *bad,
                      unsigned *stats = nullptr) {
  const int lane = threadIdx.x & 31, globalWarp = blockIdx.x * 8 + threadIdx.x / 32;
  const int cols = part == 2 ? 896 : 2304, blocks = cols / 128, pair = globalWarp / blocks,
            h = globalWarp % blocks;
  if (pair >= pairs)
    return;
  const auto m = exl3_expert::Matrix(matrices, layer, part, ids, pair, bad);
  if ((m.format != 3 && m.format != 5 && m.format != 6) || m.cols != cols || blockDim.x != 256) {
    if (!lane)
      atomicExch(bad, 1);
    return;
  }
  const int i = h * 128 + lane * 4;
  const float4 source = *reinterpret_cast<const float4 *>(
      input + std::uint64_t(part == 2 ? pair : pair / 8) * cols + i);
  const ushort4 signs = *reinterpret_cast<const ushort4 *>(m.suh + i);
  float v[4] = {__fmul_rn(source.x, __half2float(__ushort_as_half(signs.x))),
                __fmul_rn(source.y, __half2float(__ushort_as_half(signs.y))),
                __fmul_rn(source.z, __half2float(__ushort_as_half(signs.z))),
                __fmul_rn(source.w, __half2float(__ushort_as_half(signs.w)))};
  H128(v);
  ushort4 result;
  unsigned short *fields = reinterpret_cast<unsigned short *>(&result);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float value = B(v[j]);
    const auto a = __float2half_rn(value);
    const unsigned bits = __half_as_ushort(a);
    if (!isfinite(value) || (bits & 0x7c00) == 0x7c00)
      atomicExch(bad, 1);
    fields[j] = __half_as_ushort(a);
    if (stats) {
      atomicMax(stats, __float_as_uint(fabsf(value)));
      atomicAdd(stats + 1, 1U);
      if ((bits & 0x7c00) == 0 && (bits & 1023))
        atomicAdd(stats + 2, 1U);
      if ((bits & 0x7fff) == 0 && value != 0)
        atomicAdd(stats + 3, 1U);
      if ((bits & 0x7c00) == 0x7c00)
        atomicAdd(stats + 4, 1U);
      if (__half2float(a) != value)
        atomicAdd(stats + 5, 1U);
    }
  }
  *reinterpret_cast<ushort4 *>(transformed + std::uint64_t(pair) * cols + h * 128 + lane * 4) =
      result;
}
} // namespace exl3_input_wide_vector
