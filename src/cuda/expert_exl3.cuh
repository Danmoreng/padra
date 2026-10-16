// EXL3 K4/K6/K8 mul1/H128 represented-math path, adapted from pinned Gem16 codec.
// See NOTICE. No reconstructed full weight matrix or runtime repack.
namespace exl3_expert {
__device__ unsigned Index(unsigned row, unsigned col) {
  return ((col & 7) * 4 + ((row & 7) >> 1)) * 8 + (col >= 8 ? 4 : 0) + (row >= 8 ? 2 : 0) +
         (row & 1);
}
__device__ float Weight(DeviceMatrix m, int input, int output) {
  const int rate = m.format == 6 ? 3 : m.format == 4 ? 8 : m.format == 5 ? 6 : 4;
  const auto *tile = reinterpret_cast<const unsigned *>(
      m.data + ((input / 16) * (m.rows / 16) + output / 16) * (32 * rate));
  const unsigned p = Index(input & 15, output & 15);
  const int end = p * rate + rate + 256 * rate;
  const int begin = end - 16, first = begin / 32, last = (end - 1) / 32,
            shift = (last + 1) * 32 - end;
  const unsigned state =
      __funnelshift_r(tile[last % (8 * rate)], tile[first % (8 * rate)], shift) & 65535;
  const unsigned h = state * 0x83DCD12DU;
  const unsigned sum = (h & 255) + ((h >> 8) & 255) + ((h >> 16) & 255) + (h >> 24);
  return __half2float(
      __hfma(__ushort_as_half(0x6400 + sum), __ushort_as_half(0x1eee), __ushort_as_half(0xc931)));
}
__device__ DeviceMatrix Matrix(const DeviceMatrix *matrices, int layer, int part, const int *ids,
                               int pair, int *bad) {
  const int e = ids[pair];
  if (e < 0 || e >= 64) {
    atomicExch(bad, 1);
    return matrices[layer * 192 + part];
  }
  return matrices[(layer * 64 + e) * 3 + part];
}
__global__ void InputHad(const DeviceMatrix *matrices, int layer, int part, const int *ids,
                         const float *input, float *transformed, int pairs, int *bad,
                         bool retain_fp32 = false) {
  const int pair = blockIdx.y, i = blockIdx.x * 128 + threadIdx.x;
  if (pair >= pairs)
    return;
  const auto m = Matrix(matrices, layer, part, ids, pair, bad);
  if ((m.format != 3 && m.format != 4 && m.format != 5 && m.format != 6) || i >= m.cols) {
    atomicExch(bad, 1);
    return;
  }
  __shared__ float x[128];
  const int t = threadIdx.x;
  x[t] = __fmul_rn(input[(part == 2 ? pair : pair / 8) * m.cols + i],
                   __half2float(__ushort_as_half(m.suh[i])));
  __syncthreads();
  for (int d = 1; d < 128; d *= 2) {
    float a = x[t], b = x[t ^ d];
    __syncthreads();
    x[t] = (t & d) ? __fsub_rn(b, a) : __fadd_rn(a, b);
    __syncthreads();
  }
  const float raw = __fmul_rn(x[t], 0.08838834764831845f);
  float value = retain_fp32 ? raw : B(raw);
  if (!isfinite(value))
    atomicExch(bad, 1);
  transformed[pair * m.cols + i] = value;
}
__global__ void Dot(const DeviceMatrix *matrices, int layer, int part, const int *ids,
                    const float *transformed, float *out, int pairs, int *bad) {
  const int pair = blockIdx.y, row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (pair >= pairs)
    return;
  const auto m = Matrix(matrices, layer, part, ids, pair, bad);
  if ((m.format != 3 && m.format != 4 && m.format != 5 && m.format != 6) || row >= m.rows) {
    atomicExch(bad, 1);
    return;
  }
  float sum = 0, error = 0;
  for (int k = lane; k < m.cols; k += 32)
    Accumulate(__fmul_rn(transformed[pair * m.cols + k], Weight(m, k, row)), sum, error);
  for (int d = 16; d; d /= 2) {
    const float other = __shfl_down_sync(0xffffffff, sum, d),
                low = __shfl_down_sync(0xffffffff, error, d);
    if (lane + d < 32) {
      Accumulate(other, sum, error);
      Accumulate(low, sum, error);
    }
  }
  if (!lane) {
    const float v = __fadd_rn(sum, error);
    if (!isfinite(v))
      atomicExch(bad, 1);
    out[pair * m.rows + row] = v;
  }
}
__global__ void OutputHad(const DeviceMatrix *matrices, int layer, int part, const int *ids,
                          const float *transformed, float *out, int pairs, int *bad) {
  const int pair = blockIdx.y, i = blockIdx.x * 128 + threadIdx.x;
  if (pair >= pairs)
    return;
  const auto m = Matrix(matrices, layer, part, ids, pair, bad);
  if ((m.format != 3 && m.format != 4 && m.format != 5 && m.format != 6) || i >= m.rows) {
    atomicExch(bad, 1);
    return;
  }
  __shared__ float x[128];
  const int t = threadIdx.x;
  x[t] = transformed[pair * m.rows + i];
  __syncthreads();
  for (int d = 1; d < 128; d *= 2) {
    float a = x[t], b = x[t ^ d];
    __syncthreads();
    x[t] = (t & d) ? __fsub_rn(b, a) : __fadd_rn(a, b);
    __syncthreads();
  }
  const float v =
      B(__fmul_rn(__fmul_rn(x[t], 0.08838834764831845f), __half2float(__ushort_as_half(m.svh[i]))));
  if (!isfinite(v))
    atomicExch(bad, 1);
  out[pair * m.rows + i] = v;
}
__global__ void Range(const float *x, int count, unsigned *stats) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float v = x[i];
  const unsigned bits = __float_as_uint(v);
  atomicMax(stats + (bits >> 31), bits & 0x7fffffffU);
  atomicAdd(stats + 2, 1U);
  if ((bits & 0x7fffffffU) == 0) {
    atomicAdd(stats + 3, 1U);
    if (bits >> 31)
      atomicAdd(stats + 4, 1U);
  }
  const auto h = __half_as_ushort(__float2half_rn(v));
  if ((h & 0x7c00) == 0 && (h & 1023))
    atomicAdd(stats + 5, 1U);
  if ((h & 0x7fff) == 0 && v != 0)
    atomicAdd(stats + 6, 1U);
  if ((h & 0x7c00) == 0x7c00)
    atomicAdd(stats + 7, 1U);
  if (__half2float(__ushort_as_half(h)) != v)
    atomicAdd(stats + 8, 1U);
}
inline void Launch(const DeviceMatrix *matrices, int layer, int part, const int *ids,
                   const float *input, float *output, float *inScratch, float *outScratch,
                   int tokens, int *bad, cudaStream_t stream, bool fp32 = false,
                   unsigned *ranges = nullptr) {
  const int rows = part == 2 ? 2304 : 896, cols = part == 2 ? 896 : 2304, pairs = tokens * 8;
  InputHad<<<dim3(cols / 128, pairs), 128, 0, stream>>>(matrices, layer, part, ids, input,
                                                        inScratch, pairs, bad, fp32);
  if (ranges)
    Range<<<(cols * pairs + 127) / 128, 128, 0, stream>>>(inScratch, cols * pairs,
                                                          ranges + (layer * 3 + part) * 9);
  Dot<<<dim3(rows / 4, pairs), 128, 0, stream>>>(matrices, layer, part, ids, inScratch, outScratch,
                                                 pairs, bad);
  OutputHad<<<dim3(rows / 128, pairs), 128, 0, stream>>>(matrices, layer, part, ids, outScratch,
                                                         output, pairs, bad);
}

} // namespace exl3_expert
