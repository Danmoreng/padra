// Online QK/softmax/PV recurrence adapted from Gem16 c01789a2 (Apache-2.0),
// attention/prefill_local_sm120.cu:267-679 and prefill_global_sm120.cu:399-844.
// Mellum: D128/Q32/KV4, BF16 cached and float-backed BF16 chunk operands,
// scale once, per-query local1024 mask; no Gemma vision/scaling/cache codec.
namespace fast_prefill_attention {
constexpr int kThreads = 256, kRows = 32, kHeads = 4, kKeys = 32, kDim = 128;
__device__ __forceinline__ float GroupMax(float x) {
  x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1, 4));
  return fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2, 4));
}
__device__ __forceinline__ float GroupSum(float x) {
  x += __shfl_xor_sync(0xffffffff, x, 1, 4);
  return x + __shfl_xor_sync(0xffffffff, x, 2, 4);
}
__device__ __forceinline__ unsigned Pack(const __nv_bfloat16 *p) {
  return static_cast<unsigned>(__bfloat16_as_ushort(p[0])) |
         (static_cast<unsigned>(__bfloat16_as_ushort(p[1])) << 16);
}
__device__ __forceinline__ float KV(const __nv_bfloat16 *cache, const float *chunk, int key,
                                    int first, int last, int base, int cap, int feature) {
  // Skip keys outside the entire CTA's visibility before reading a ring slot.
  // In particular, alignment padding must not load evicted or poisoned entries.
  if (key < first || key > last)
    return 0.f;
  return key >= base ? chunk[(key - base) * 512 + feature]
                     : __bfloat162float(cache[(key % cap) * 512 + feature]);
}
template <bool Local>
__launch_bounds__(kThreads, 1) __global__
    void OnlineBF16(const float *q, const __nv_bfloat16 *kc, const __nv_bfloat16 *vc,
                    const float *chunkK, const float *chunkV, float *output, int base, int tokens,
                    int cap, int *bad) {
  if (!q || !kc || !vc || !chunkK || !chunkV || !output || !bad || base < 0 || tokens < 1 ||
      tokens > kW4A8FMaxPrefillChunk || base > kW4A8FContextCapacity - tokens || cap < 1 ||
      cap > kW4A8FContextCapacity ||
      (Local ? cap < min(1024, base + tokens) : cap < base + tokens) || blockDim.x != kThreads ||
      blockDim.y != 1 || blockDim.z != 1 ||
      blockIdx.x >= static_cast<unsigned>((tokens + kRows - 1) / kRows) || blockIdx.y >= 8 ||
      blockIdx.z != 0) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  __shared__ __align__(16) __nv_bfloat16 qs[kHeads * kRows * kDim];
  __shared__ __align__(16) __nv_bfloat16 operand[kKeys * kDim];
  const int headBase = blockIdx.y * kHeads, kvhead = headBase / 8, rowBase = blockIdx.x * kRows;
  const int lane = threadIdx.x % 32, headIn = threadIdx.x / 64,
            warpRow = ((threadIdx.x / 32) % 2) * 16;
  const int group = lane / 4, tid = lane % 4, head = headBase + headIn;
  const int row0 = rowBase + warpRow + group, row1 = row0 + 8;
  const int first = Local ? max(0, base + rowBase - 1023) : 0;
  const int last = base + min(rowBase + kRows, tokens) - 1;
  for (int i = threadIdx.x; i < kHeads * kRows * kDim; i += kThreads) {
    const int d = i % kDim, row = (i / kDim) % kRows, h = i / (kDim * kRows);
    qs[i] = __float2bfloat16_rn(
        rowBase + row < tokens ? q[((rowBase + row) * 32 + headBase + h) * kDim + d] : 0.f);
  }
  __syncthreads();
  float accum[16][4] = {};
  float maximum[2] = {-INFINITY, -INFINITY}, sum[2] = {0, 0};
  for (int keyBase = (first / kKeys) * kKeys; keyBase <= last; keyBase += kKeys) {
    for (int i = threadIdx.x; i < kKeys * kDim; i += kThreads)
      operand[i] = __float2bfloat16_rn(
          KV(kc, chunkK, keyBase + i / kDim, first, last, base, cap, kvhead * kDim + i % kDim));
    __syncthreads();
    float scores[4][4] = {};
#pragma unroll
    for (int k = 0; k < kDim; k += 16) {
      unsigned a[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        a[j] = Pack(qs + (headIn * kRows + warpRow + group + (j % 2) * 8) * kDim + k + tid * 2 +
                    (j / 2) * 8);
#pragma unroll
      for (int tile = 0; tile < 4; ++tile) {
        unsigned b[2];
#pragma unroll
        for (int j = 0; j < 2; ++j)
          b[j] = Pack(operand + (tile * 8 + group) * kDim + k + tid * 2 + j * 8);
        AttentionMmaBF16(a, b, scores[tile]);
      }
    }
    float blockMaximum[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int row = j / 2 ? row1 : row0, key = keyBase + tile * 8 + tid * 2 + j % 2,
                  pos = base + row;
        const bool valid = row < tokens && key <= pos && key >= (Local ? max(0, pos - 1023) : 0);
        scores[tile][j] = valid ? scores[tile][j] * .08838834764831845f : -INFINITY;
        if (valid)
          Finite(scores[tile][j], bad);
        blockMaximum[j / 2] = fmaxf(blockMaximum[j / 2], scores[tile][j]);
      }
    }
    float alpha[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float next = fmaxf(maximum[r], GroupMax(blockMaximum[r]));
      alpha[r] = isfinite(maximum[r]) ? expf(maximum[r] - next) : 0.f;
      maximum[r] = next;
    }
    unsigned probability[2][4];
    float blockSum[2] = {0, 0};
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
      float p[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        p[j] = isfinite(scores[tile][j]) ? expf(scores[tile][j] - maximum[j / 2]) : 0.f;
        blockSum[j / 2] += p[j];
      }
      probability[tile / 2][(tile % 2) * 2] = AttentionPackBF16(p[0], p[1]);
      probability[tile / 2][(tile % 2) * 2 + 1] = AttentionPackBF16(p[2], p[3]);
    }
#pragma unroll
    for (int r = 0; r < 2; ++r)
      sum[r] = fmaf(sum[r], alpha[r], blockSum[r]);
#pragma unroll
    for (int tile = 0; tile < 16; ++tile) {
#pragma unroll
      for (int j = 0; j < 4; ++j)
        accum[tile][j] *= alpha[j / 2];
    }
    __syncthreads(); // all QK reads complete before K storage becomes V
    for (int i = threadIdx.x; i < kKeys * kDim; i += kThreads)
      operand[i] = __float2bfloat16_rn(
          KV(vc, chunkV, keyBase + i / kDim, first, last, base, cap, kvhead * kDim + i % kDim));
    __syncthreads();
#pragma unroll
    for (int step = 0; step < 2; ++step) {
#pragma unroll
      for (int tile = 0; tile < 16; ++tile) {
        unsigned b[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int key = step * 16 + tid * 2 + j * 8, col = tile * 8 + group;
          b[j] =
              static_cast<unsigned>(__bfloat16_as_ushort(operand[key * kDim + col])) |
              (static_cast<unsigned>(__bfloat16_as_ushort(operand[(key + 1) * kDim + col])) << 16);
        }
        AttentionMmaBF16(probability[step], b, accum[tile]);
      }
    }
    __syncthreads(); // all PV reads complete before next K tile
  }
  const float denominator[2] = {GroupSum(sum[0]), GroupSum(sum[1])};
#pragma unroll
  for (int tile = 0; tile < 16; ++tile) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int row = j / 2 ? row1 : row0;
      if (row < tokens) {
        const float v = B(accum[tile][j] / denominator[j / 2]);
        output[(row * 32 + head) * kDim + tile * 8 + tid * 2 + j % 2] = v;
        Finite(v, bad);
      }
    }
  }
}
void Launch(const float *q, const __nv_bfloat16 *kc, const __nv_bfloat16 *vc, const float *chunkK,
            const float *chunkV, float *output, int base, int tokens, int cap, bool local, int *bad,
            cudaStream_t stream) {
  if (!q || !kc || !vc || !chunkK || !chunkV || !output || !bad || base < 0 || tokens < 1 ||
      tokens > kW4A8FMaxPrefillChunk || base > kW4A8FContextCapacity - tokens || cap < 1 ||
      cap > kW4A8FContextCapacity ||
      (local ? cap < std::min(1024, base + tokens) : cap < base + tokens))
    throw std::runtime_error("online BF16 prefill attention shape/binding unsupported");
  const dim3 grid((tokens + kRows - 1) / kRows, 8);
  if (local)
    OnlineBF16<true>
        <<<grid, kThreads, 0, stream>>>(q, kc, vc, chunkK, chunkV, output, base, tokens, cap, bad);
  else
    OnlineBF16<false>
        <<<grid, kThreads, 0, stream>>>(q, kc, vc, chunkK, chunkV, output, base, tokens, cap, bad);
}
} // namespace fast_prefill_attention
