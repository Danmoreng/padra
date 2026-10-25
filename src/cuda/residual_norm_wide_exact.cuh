// Keep the default profile's exact128 reduction. The opt-in K64 profile
// composes its short-prefill repair with the qualified D2 register-row norm.
#pragma once
#include "rms_wide_exact.cuh"
#if PADRA_EXL3_PREFILL_ASYNC64
__global__ void ResidualPostNormWideExact(float *x, const float *residual, float *normalized,
                                          DeviceMatrix weight, int *bad,
                                          float *statistics = nullptr, int rows = 1) {
  if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 || rows < 1 || rows > 5 ||
      gridDim.x != unsigned(rows) || gridDim.y != 1 || gridDim.z != 1 || !x || !residual ||
      !normalized || x == normalized || !weight.data || !bad || weight.format != 0 ||
      weight.rows != 2304 || weight.cols != 1) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  x += blockIdx.x * 2304;
  residual += blockIdx.x * 2304;
  normalized += blockIdx.x * 2304;
  if (statistics)
    statistics += blockIdx.x * 2;
  __shared__ float sums[8], errors[8], inv;
  float values[9], sum = 0.f, error = 0.f;
#pragma unroll
  for (int j = 0; j < 9; ++j) {
    const int i = threadIdx.x + j * 256;
    const float v = B(__fadd_rn(x[i], residual[i]));
    values[j] = v;
    x[i] = v;
    Finite(v, bad);
    Accumulate(__fmul_rn(v, v), sum, error);
  }
#pragma unroll
  for (int d = 16; d; d /= 2) {
    const float other = __shfl_down_sync(0xffffffff, sum, d),
                low = __shfl_down_sync(0xffffffff, error, d);
    if ((threadIdx.x & 31) + d < 32) {
      Accumulate(other, sum, error);
      Accumulate(low, sum, error);
    }
  }
  if ((threadIdx.x & 31) == 0) {
    sums[threadIdx.x / 32] = sum;
    errors[threadIdx.x / 32] = error;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < 8 ? sums[threadIdx.x] : 0.f;
    error = threadIdx.x < 8 ? errors[threadIdx.x] : 0.f;
#pragma unroll
    for (int d = 16; d; d /= 2) {
      const float other = __shfl_down_sync(0xffffffff, sum, d),
                  low = __shfl_down_sync(0xffffffff, error, d);
      if ((threadIdx.x & 31) + d < 32) {
        Accumulate(other, sum, error);
        Accumulate(low, sum, error);
      }
    }
    if (threadIdx.x == 0) {
      sum = __fadd_rn(sum, error);
      Finite(sum, bad);
      const float variance = __fdiv_rn(sum, 2304.f);
      const float regularized = __fadd_rn(variance, 1e-6f);
      inv = __fdiv_rn(1.f, __fsqrt_rn(regularized));
      if (statistics) {
        statistics[0] = variance;
        statistics[1] = inv;
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 9; ++j) {
    int i = threadIdx.x + j * 256;
    float v = B(__fmul_rn(B(__fmul_rn(values[j], inv)), BF(weight.data, i)));
    normalized[i] = v;
    Finite(v, bad);
  }
}

#else
__global__ void ResidualPostNormWideExact(float *x, const float *residual, float *normalized,
                                          DeviceMatrix weight, int *bad,
                                          float *statistics = nullptr, int rows = 1) {
  if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 || rows < 1 || rows > 5 ||
      gridDim.x != unsigned(rows) || gridDim.y != 1 || gridDim.z != 1 || !x || !residual ||
      !normalized || x == normalized || !weight.data || !bad || weight.format != 0 ||
      weight.rows != 2304 || weight.cols != 1) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  x += blockIdx.x * 2304;
  residual += blockIdx.x * 2304;
  normalized += blockIdx.x * 2304;
  if (statistics)
    statistics += blockIdx.x * 2;
  __shared__ float sums[8], inv;
  __shared__ float row[2304];
  float sum = 0.f, error = 0.f;
  if (threadIdx.x < 128) {
#pragma unroll
    for (int j = 0; j < 18; ++j) {
      const int i = threadIdx.x + j * 128;
      const float value = B(x[i] + residual[i]);
      x[i] = value;
      row[i] = value;
      Finite(value, bad);
      Accumulate(__fmul_rn(value, value), sum, error);
    }
  }
  sum = Rms128In256Exact(sum, error, sums);
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
  for (int j = 0; j < 9; ++j) {
    const int i = threadIdx.x + j * 256;
    const float value = B(B(row[i] * inv) * BF(weight.data, i));
    normalized[i] = value;
    Finite(value, bad);
  }
}

#endif
