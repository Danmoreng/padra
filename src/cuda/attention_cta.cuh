// Mellum compensated block reductions extended to8 warps; softmax uses original4 warp inputs + zero
// suffix.
__device__ float AttentionAccurateReduce8(float sum, float error, float *scratch) {
  for (int d = 16; d; d /= 2) {
    const float other = __shfl_down_sync(0xffffffff, sum, d),
                low = __shfl_down_sync(0xffffffff, error, d);
    if ((threadIdx.x & 31) + d < 32) {
      Accumulate(other, sum, error);
      Accumulate(low, sum, error);
    }
  }
  if ((threadIdx.x & 31) == 0) {
    scratch[threadIdx.x / 32] = sum;
    scratch[8 + threadIdx.x / 32] = error;
  }
  __syncthreads();
  sum = threadIdx.x < 8 ? scratch[threadIdx.x] : 0;
  error = threadIdx.x < 8 ? scratch[8 + threadIdx.x] : 0;
  if (threadIdx.x < 32)
    for (int d = 16; d; d /= 2) {
      const float other = __shfl_down_sync(0xffffffff, sum, d),
                  low = __shfl_down_sync(0xffffffff, error, d);
      if (threadIdx.x + d < 32) {
        Accumulate(other, sum, error);
        Accumulate(low, sum, error);
      }
    }
  return __fadd_rn(sum, error);
}
__device__ float AttentionMaxReduce8(float x, float *scratch) {
  for (int d = 16; d; d /= 2)
    x = fmaxf(x, __shfl_down_sync(0xffffffff, x, d));
  if ((threadIdx.x & 31) == 0)
    scratch[threadIdx.x / 32] = x;
  __syncthreads();
  x = threadIdx.x < 8 ? scratch[threadIdx.x] : -INFINITY;
  if (threadIdx.x < 32)
    for (int d = 16; d; d /= 2)
      x = fmaxf(x, __shfl_down_sync(0xffffffff, x, d));
  return x;
}
// Gem16 warp/token scheduling only; Mellum compensated scores/softmax/PV intact.
template <bool Local>
__global__ void AttendDecodeCTA256(const float *q, const __nv_bfloat16 *kc, const __nv_bfloat16 *vc,
                                   float *out, int pos, int cap, int *bad,
                                   float *captured_scores = nullptr,
                                   float *captured_probabilities = nullptr) {
  if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 || gridDim.x != 32 ||
      gridDim.y != 1 || gridDim.z != 1 || !q || !kc || !vc || !out || !bad || pos < 0 ||
      pos >= 4096 || cap < 1 || cap > 4096 || (Local ? (cap < min(pos + 1, 1024)) : (pos >= cap))) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  extern __shared__ float scores[];
  __shared__ float scratch[16];
  __shared__ float maxv, total;
  const int h = blockIdx.x, kvh = h / 8, start = Local ? max(0, pos - 1023) : 0,
            n = pos - start + 1;
  const int lane = threadIdx.x & 7, group = threadIdx.x / 8;
  const unsigned mask = 0xffu << (threadIdx.x & 24);
  float query[16];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    query[i] = q[h * 128 + lane + i * 8];
  float mx = -INFINITY;
  for (int j = group; j < n; j += 32) {
    float dot = 0, error = 0;
    const int offset = ((start + j) % cap) * 512 + kvh * 128 + lane;
#pragma unroll
    for (int i = 0; i < 16; ++i)
      Accumulate(__fmul_rn(query[i], __bfloat162float(kc[offset + i * 8])), dot, error);
#pragma unroll
    for (int d = 4; d; d /= 2) {
      const float other = __shfl_down_sync(mask, dot, d, 8),
                  low = __shfl_down_sync(mask, error, d, 8);
      if (lane + d < 8) {
        Accumulate(other, dot, error);
        Accumulate(low, dot, error);
      }
    }
    dot = __fadd_rn(dot, error);
    if (lane == 0) {
      const float score = B(B(dot) * .08838834764831845f);
      scores[j] = score;
      if (captured_scores)
        captured_scores[h * n + j] = score;
      mx = fmaxf(mx, score);
    }
  }
  mx = AttentionMaxReduce8(mx, scratch);
  if (threadIdx.x == 0)
    maxv = mx;
  __syncthreads();
  float sum = 0, sum_error = 0;
  if (threadIdx.x < 128)
    for (int j = threadIdx.x; j < n; j += 128) {
      scores[j] = ExpRN(__fsub_rn(scores[j], maxv));
      Accumulate(scores[j], sum, sum_error);
    }
  sum = AttentionAccurateReduce8(sum, sum_error, scratch);
  if (threadIdx.x == 0)
    total = sum;
  __syncthreads();
  if (threadIdx.x < 128)
    for (int j = threadIdx.x; j < n; j += 128) {
      scores[j] = B(__fdiv_rn(scores[j], total));
      if (captured_probabilities)
        captured_probabilities[h * n + j] = scores[j];
    }
  __syncthreads();
  if (threadIdx.x < 128) {
    float value = 0, error = 0;
    for (int j = 0; j < n; ++j)
      Accumulate(__fmul_rn(scores[j], __bfloat162float(
                                          vc[((start + j) % cap) * 512 + kvh * 128 + threadIdx.x])),
                 value, error);
    out[h * 128 + threadIdx.x] = B(__fadd_rn(value, error));
    Finite(value, bad);
  }
}
