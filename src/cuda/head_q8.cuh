#pragma once
// One Q8 block32 matrix shared by target and draft. Per-row activation scratch.
namespace head_q8 {
inline constexpr int kRows = 98304, kCols = 2304, kBlocks = kCols / 32;
inline constexpr std::size_t kInputBytes = kCols + kBlocks * 2;
inline constexpr std::size_t kWorkspaceBytes = 5 * kInputBytes;
static_assert(kWorkspaceBytes == kHeadQ8InputBytes);
__device__ __forceinline__ float Maximum(float x) {
  for (int d = 16; d; d /= 2)
    x = fmaxf(x, __shfl_xor_sync(0xffffffffU, x, d));
  return x;
}
__global__ void QuantizeInput(const float *input, unsigned char *workspace, int rows, int *bad) {
  if (!input || !workspace || !bad || rows < 1 || rows > 5 || blockDim.x != 128 ||
      blockDim.y != 1 || blockDim.z != 1 || gridDim.x != 18 || gridDim.y != rows ||
      gridDim.z != 1) {
    if (!threadIdx.x && bad)
      atomicExch(bad, 1);
    return;
  }
  const int row = blockIdx.y, c = blockIdx.x * 128 + threadIdx.x;
  auto *codes = reinterpret_cast<signed char *>(workspace + row * kInputBytes);
  auto *scales = reinterpret_cast<__nv_bfloat16 *>(codes + kCols);
  const float value = input[row * kCols + c];
  if (!isfinite(value) || (__float_as_uint(value) & 65535U))
    atomicExch(bad, 1);
  const float maximum = Maximum(fabsf(value));
  float scale = maximum == 0.f ? 1.f : B(__fdiv_rn(maximum, 127.f));
  if (!isfinite(scale) || scale <= 0.f) {
    atomicExch(bad, 1);
    scale = 1.f;
  }
  if (!(threadIdx.x & 31))
    scales[c / 32] = __float2bfloat16_rn(scale);
  codes[c] = static_cast<signed char>(max(-127, min(127, __float2int_rn(__fdiv_rn(value, scale)))));
}
template <int Rows>
__global__ void Projection(DeviceMatrix weights, const unsigned char *workspace, float *output,
                           int *bad) {
  if (weights.format != 7 || weights.rows != kRows || weights.cols != kCols || !weights.data ||
      !weights.scales || !workspace || !output || !bad ||
      (reinterpret_cast<std::uintptr_t>(weights.data) & 3) ||
      (reinterpret_cast<std::uintptr_t>(workspace) & 3) || blockDim.x != 128 || blockDim.y != 1 ||
      blockDim.z != 1 || gridDim.x != kRows / 4 || gridDim.y != 1 || gridDim.z != 1) {
    if (!threadIdx.x && bad)
      atomicExch(bad, 1);
    return;
  }
  const int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x & 31;
  const auto *weightScales = reinterpret_cast<const __nv_bfloat16 *>(weights.scales);
  float sum[Rows] = {};
#pragma unroll
  for (int i = 0; i < 18; ++i) {
    const int c = i * 128 + lane * 4, block = c / 32;
    const int packed = reinterpret_cast<const int *>(weights.data)[(row * kCols + c) / 4];
    const float ws = __bfloat162float(weightScales[row * kBlocks + block]);
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
      const auto *input = workspace + r * kInputBytes;
      const auto *scales = reinterpret_cast<const __nv_bfloat16 *>(input + kCols);
      const int dot = __dp4a(packed, reinterpret_cast<const int *>(input)[c / 4], 0);
      const float scale = __fmul_rn(ws, __bfloat162float(scales[block]));
      sum[r] = __fmaf_rn(float(dot), scale, sum[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < Rows; ++r) {
    for (int d = 16; d; d /= 2)
      sum[r] = __fadd_rn(sum[r], __shfl_down_sync(0xffffffffU, sum[r], d));
    if (!lane) {
      output[r * kRows + row] = B(sum[r]);
      Finite(output[r * kRows + row], bad);
    }
  }
}
inline void Launch(int rows, DeviceMatrix weights, const float *hidden, unsigned char *workspace,
                   float *out, int *bad, cudaStream_t stream) {
  if (rows < 1 || rows > 5)
    throw std::runtime_error("shared Q8 head row count unsupported");
  QuantizeInput<<<dim3(18, rows), 128, 0, stream>>>(hidden, workspace, rows, bad);
  switch (rows) {
  case 1:
    Projection<1><<<24576, 128, 0, stream>>>(weights, workspace, out, bad);
    break;
  case 2:
    Projection<2><<<24576, 128, 0, stream>>>(weights, workspace, out, bad);
    break;
  case 3:
    Projection<3><<<24576, 128, 0, stream>>>(weights, workspace, out, bad);
    break;
  case 4:
    Projection<4><<<24576, 128, 0, stream>>>(weights, workspace, out, bad);
    break;
  case 5:
    Projection<5><<<24576, 128, 0, stream>>>(weights, workspace, out, bad);
    break;
  }
}
} // namespace head_q8
