// Ordinary RMS reduction/fusion mechanism follows pinned Gem16 norm/reference.cu.
// Mellum direct learned weights, epsilon1e-6 and two BF16 casts, never1+weight.
// Donor: gem16 norm/reference.cu:56-77, Apache-2.0; warp reduction adapted.
namespace fast_norm {
__device__ __forceinline__ float Sum128(float value, float *scratch) {
  for (int d = 16; d; d /= 2)
    value += __shfl_down_sync(0xffffffff, value, d);
  if ((threadIdx.x & 31) == 0)
    scratch[threadIdx.x / 32] = value;
  __syncthreads();
  value = threadIdx.x < 4 ? scratch[threadIdx.x] : 0.f;
  if (threadIdx.x < 32)
    for (int d = 16; d; d /= 2)
      value += __shfl_down_sync(0xffffffff, value, d);
  return value;
}
template <bool FP8>
__global__ void Rms2304(const float *input, float *output, DeviceMatrix weight,
                        unsigned char *packed, float *scales, int *bad) {
  if (!input || !output || !weight.data || !bad || weight.format != 0 || weight.rows != 2304 ||
      weight.cols != 1 || (FP8 && (!packed || !scales)) || blockDim.x != 128 || blockDim.y != 1 ||
      blockDim.z != 1 || gridDim.x > 512 || gridDim.y != 1 || gridDim.z != 1) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  constexpr int dim = 2304;
  const int row = blockIdx.x;
  input += row * dim;
  output += row * dim;
  __shared__ float scratch[4], inverse, scale;
  float sum = 0;
  for (int i = threadIdx.x; i < dim; i += 128)
    sum = fmaf(input[i], input[i], sum);
  sum = Sum128(sum, scratch);
  if (threadIdx.x == 0) {
    Finite(sum, bad);
    inverse = rsqrtf(sum / static_cast<float>(dim) + 1e-6f);
    Finite(inverse, bad);
  }
  __syncthreads();
  float maximum = 0;
  for (int i = threadIdx.x; i < dim; i += 128) {
    const float value = B(B(input[i] * inverse) * BF(weight.data, i));
    output[i] = value;
    Finite(value, bad);
    if constexpr (FP8)
      maximum = fmaxf(maximum, fabsf(value));
  }
  if constexpr (FP8) {
    maximum = MaxReduce(maximum, scratch);
    if (threadIdx.x == 0) {
      scale = maximum == 0 ? 1.f : maximum / 448.f;
      scales[row] = scale;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < dim; i += 128)
      packed[row * dim + i] = __nv_cvt_float_to_fp8(output[i] / scale, __NV_SATFINITE, __NV_E4M3);
  }
}
} // namespace fast_norm
