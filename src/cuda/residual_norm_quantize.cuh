// N4b Gem16 c01789a2 row staging (Apache-2.0), via existing NormFP8.
// Mellum Add BF16 boundary, compensated Norm and direct E4 coding retained.
// No Gemma branch postnorm, 1+weight, approximate rsqrt or shared experts.
template <bool Reciprocal = false>
__global__ void ResidualPostNormFP8(float *x, const float *residual, float *normalized,
                                    DeviceMatrix weight, unsigned char *packed, float *scales,
                                    int *bad, float *statistics = nullptr) {
  if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 || gridDim.x != 1 || gridDim.y != 1 ||
      gridDim.z != 1 || !x || !residual || !normalized || x == normalized || !weight.data ||
      !packed || !scales || !bad || weight.format != 0 || weight.rows != 2304 || weight.cols != 1) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  __shared__ float sums[8], maxima[4], inv, scale, inverse;
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
  sum = AccurateReduce(sum, error, sums);
  if (threadIdx.x == 0) {
    const float variance = __fdiv_rn(sum, 2304.f);
    inv = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(variance, 1e-6f)));
    if (statistics) {
      statistics[0] = variance;
      statistics[1] = inv;
    }
  }
  __syncthreads();
  float maximum = 0.f;
#pragma unroll
  for (int j = 0; j < 18; ++j) {
    const int i = threadIdx.x + j * 128;
    const float value = B(B(row[j] * inv) * BF(weight.data, i));
    row[j] = value;
    normalized[i] = value;
    Finite(value, bad);
    maximum = fmaxf(maximum, fabsf(value));
  }
  maximum = MaxReduce(maximum, maxima);
  if (threadIdx.x == 0) {
    scale = maximum == 0.f ? 1.f : __fdiv_rn(maximum, 448.f);
    scales[0] = scale;
    inverse = row_quantization::Inverse<Reciprocal>(scale);
  }
  __syncthreads();
  if constexpr (Reciprocal) {
    if (!isfinite(inverse)) {
      if (threadIdx.x == 0)
        atomicExch(bad, 1);
      for (int j = 0; j < 18; ++j)
        packed[threadIdx.x + j * 128] = 0x7f;
      return;
    }
  }
#pragma unroll
  for (int j = 0; j < 18; ++j)
    packed[threadIdx.x + j * 128] = row_quantization::Encode<Reciprocal>(row[j], scale, inverse);
}
