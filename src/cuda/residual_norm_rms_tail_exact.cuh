// Exact N4b BF16 residual/compensated postnorm; omit unused expert FP8 encoding.
#pragma once
#include "rms_tail_reduce_exact.cuh"
__global__ void ResidualPostNormRmsTailExact(float *x, const float *residual, float *normalized,
                                             DeviceMatrix weight, int *bad,
                                             float *statistics = nullptr) {
  if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 || gridDim.x != 1 || gridDim.y != 1 ||
      gridDim.z != 1 || !x || !residual || !normalized || x == normalized || !weight.data || !bad ||
      weight.format != 0 || weight.rows != 2304 || weight.cols != 1) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  __shared__ float sums[8], inv;
  float row[18], sum = 0.f, error = 0.f;
#pragma unroll
  for (int j = 0; j < 18; ++j) {
    const int i = threadIdx.x + j * 128;
    const float value = B(x[i] + residual[i]);
    x[i] = value;
    row[j] = value;
    Finite(value, bad);
    Accumulate(__fmul_rn(value, value), sum, error);
  }
  sum = RmsTailReduceExact(sum, error, sums);
  if (threadIdx.x == 0) {
    const float variance = __fdiv_rn(sum, 2304.f);
    inv = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(variance, 1e-6f)));
    if (statistics) {
      statistics[0] = variance;
      statistics[1] = inv;
    }
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 18; ++j) {
    const int i = threadIdx.x + j * 128;
    const float value = B(B(row[j] * inv) * BF(weight.data, i));
    normalized[i] = value;
    Finite(value, bad);
  }
}
