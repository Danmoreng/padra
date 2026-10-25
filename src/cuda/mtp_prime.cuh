#pragma once
// Included inside mtp. Independent prompt rows retain the single-row arithmetic.
namespace prime {
__global__ void NormRows(const float *x, float *out, const float *weights, int dim, int inputStride,
                         int outputStride, int rows, int *bad) {
  const int heads = dim == 128 ? 4 : 1;
  if (!x || !out || !weights || !bad || rows < 1 || rows > kWindow ||
      (dim != 128 && dim != kHidden) || inputStride < heads * dim || outputStride < heads * dim ||
      blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 || gridDim.x != unsigned(heads) ||
      gridDim.y != unsigned(rows) || gridDim.z != 1) {
    if (!threadIdx.x && bad)
      atomicExch(bad, 1);
    return;
  }
  x += blockIdx.y * inputStride;
  out += blockIdx.y * outputStride;
  __shared__ float scratch[8], inverse;
  const int h = blockIdx.x;
  float sum = 0, error = 0;
  for (int i = threadIdx.x; i < dim; i += 128)
    Accumulate(__fmul_rn(x[h * dim + i], x[h * dim + i]), sum, error);
  sum = AccurateReduce(sum, error, scratch);
  if (threadIdx.x == 0)
    inverse = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, float(dim)), 1e-6f)));
  __syncthreads();
  for (int i = threadIdx.x; i < dim; i += 128)
    out[h * dim + i] = B(__fmul_rn(B(__fmul_rn(x[h * dim + i], inverse)), weights[i]));
}
template <int Tokens = 4>
__global__ void Projection(DeviceMatrix m, const float *x, float *y, int rows, int *bad) {
  if (m.format != 0 || !m.data || !x || !y || x == y || !bad || rows < 1 || rows > kWindow ||
      !((m.rows == 2304 && m.cols == 4608) || (m.rows == 512 && m.cols == 2304)) ||
      blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
      gridDim.x != unsigned(m.rows / 4) || gridDim.y != unsigned((rows + Tokens - 1) / Tokens) ||
      gridDim.z != 1) {
    if (!threadIdx.x && bad)
      atomicExch(bad, 1);
    return;
  }
  const int lane = threadIdx.x & 31, r = blockIdx.x * 4 + threadIdx.x / 32;
  float sum[Tokens] = {}, error[Tokens] = {};
  for (int c = lane; c < m.cols; c += 32) {
    const float weight = BF(m.data, r * m.cols + c);
#pragma unroll
    for (int t = 0; t < Tokens; ++t) {
      const int row = blockIdx.y * Tokens + t;
      if (row < rows)
        Accumulate(__fmul_rn(weight, x[row * m.cols + c]), sum[t], error[t]);
    }
  }
#pragma unroll
  for (int t = 0; t < Tokens; ++t) {
    const int row = blockIdx.y * Tokens + t;
    if (row < rows) {
      const float value = B(WarpAccurateReduce(sum[t], error[t]));
      if (lane == 0) {
        y[row * m.rows + r] = value;
        Finite(value, bad);
      }
    }
  }
}
__global__ void RopeStore(const float *kn, const float *v, const float *rope, __nv_bfloat16 *keys,
                          __nv_bfloat16 *values, int first, int rows, int capacity, int *bad) {
  if (!kn || !v || !rope || !keys || !values || !bad || first < 0 || rows < 1 || rows > kWindow ||
      first >= capacity || rows > capacity - first || blockDim.x != 128 || blockDim.y != 1 ||
      blockDim.z != 1 || gridDim.x != 4 || gridDim.y != unsigned(rows) || gridDim.z != 1) {
    if (!threadIdx.x && bad)
      atomicExch(bad, 1);
    return;
  }
  const int h = blockIdx.x, row = blockIdx.y, i = threadIdx.x, position = first + row;
  __shared__ float old[128];
  old[i] = kn[row * 512 + h * 128 + i];
  __syncthreads();
  const float c = rope[position * 128 + i % 64], s = rope[position * 128 + 64 + i % 64];
  const float rotated = i < 64 ? -old[i + 64] : old[i - 64];
  const float key = B(__fadd_rn(B(__fmul_rn(old[i], c)), B(__fmul_rn(rotated, s))));
  const float value = v[row * 512 + h * 128 + i];
  Finite(key, bad);
  Finite(value, bad);
  keys[(position % kWindow) * 512 + h * 128 + i] = __float2bfloat16_rn(key);
  values[(position % kWindow) * 512 + h * 128 + i] = __float2bfloat16_rn(value);
}
} // namespace prime
