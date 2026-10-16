// Owner10K exact G3 arithmetic;shared stride136 removes operand bank conflicts.
// One K buffer:prefetch begins only after all currentQK reads finish and CTA sync.
// Native N4b BF16 MMA/FP32 softmax/rounded-P/output boundaries retained;
// head-position mapping, register Q and K/V prefetch adapted from pinned llama.
#ifndef PADRA_PREFILL_GLOBAL_DATAFLOW
#define PADRA_PREFILL_GLOBAL_DATAFLOW 0
#endif
namespace padded_single_prefill_attention {
constexpr int kThreads = 128, kRows = 8, kHeads = 8, kKeys = 32, kDim = 128, kStride = 136;
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
// Scheduling reference: llama.cpp MIT, pin10a60cf fattn-mma-f16.cuh
// register-resident Q + asynchronous K/V stages; ported to BF16 with explicit
// old-cache versus float-backed current chunk and pre-load visibility checks.
template <bool Local, bool DirectGlobalAddress>
__device__ __forceinline__ void Stage(__nv_bfloat16 *target, const __nv_bfloat16 *cache,
                                      const __nv_bfloat16 *chunk, int keyBase, int first, int last,
                                      int base, int cap, int kvhead) {
  for (int i = threadIdx.x; i < kKeys * kDim / 8; i += kThreads) {
    const int key = keyBase + i / (kDim / 8), d = (i % (kDim / 8)) * 8;
    const bool valid = key >= first && key <= last;
    const int slot = DirectGlobalAddress && !Local ? key : key % cap;
    const auto src = valid ? (key < base ? cache + slot * 512 + kvhead * kDim + d
                                         : chunk + (key - base) * 512 + kvhead * kDim + d)
                           : cache;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(
                     unsigned(__cvta_generic_to_shared(target + (i / (kDim / 8)) * kStride + d))),
                 "l"(src), "r"(valid ? 16 : 0)
                 : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void Wait() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}
template <bool Local, bool RoundOutput = true, bool DirectGlobalAddress = false,
          bool SkipUnitAlpha = false>
__launch_bounds__(kThreads, 2) __global__
    void PipelineBF16(const float *q, const __nv_bfloat16 *kc, const __nv_bfloat16 *vc,
                      const __nv_bfloat16 *chunkK, const __nv_bfloat16 *chunkV, float *output,
                      int base, int tokens, int cap, int *bad,
                      const PrefillState *prefill = nullptr,
                      int context_limit = kW4A8FContextCapacity) {
  if (prefill) {
    if (!ValidPrefillState(*prefill, tokens, prefill->capacity, 1, context_limit) ||
        cap != (Local ? min(1024, prefill->capacity) : prefill->capacity)) {
      if (threadIdx.x == 0 && bad)
        atomicExch(bad, 1);
      return;
    }
    base = prefill->base;
  }
  if (!q || !kc || !vc || !chunkK || !chunkV || !output || !bad || base < 0 || tokens < 1 ||
      tokens > kW4A8FMaxPrefillChunk || base > context_limit - tokens || cap < 1 ||
      cap > context_limit || (Local ? cap < min(1024, base + tokens) : cap < base + tokens) ||
      blockDim.x != kThreads || blockDim.y != 1 || blockDim.z != 1 ||
      blockIdx.x >= static_cast<unsigned>((tokens + kRows - 1) / kRows) || blockIdx.y >= 4 ||
      blockIdx.z != 0) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  // One K stage and one V stage, padded BF16. Q fragments stay in registers.
  __shared__ __align__(16) __nv_bfloat16 keys[1][kKeys * kStride];
  __shared__ __align__(16) __nv_bfloat16 values[kKeys * kStride];
  const int headBase = blockIdx.y * kHeads, kvhead = blockIdx.y, rowBase = blockIdx.x * kRows;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int group = lane / 4, tid = lane % 4, head = headBase + group;
  const int row0 = rowBase + warp * 2, row1 = row0 + 1;
  const int first = Local ? max(0, base + rowBase - 1023) : 0;
  const int last = base + min(rowBase + kRows, tokens) - 1;
  unsigned query[8][4];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int row = j % 2 ? row1 : row0, d = k * 16 + tid * 2 + (j / 2) * 8;
      query[k][j] = row < tokens ? AttentionPackBF16(q[(row * 32 + head) * kDim + d],
                                                     q[(row * 32 + head) * kDim + d + 1])
                                 : 0;
    }
  }
  const int initial = (first / kKeys) * kKeys;
  Stage<Local, DirectGlobalAddress>(keys[0], kc, chunkK, initial, first, last, base, cap, kvhead);
  Wait();
  __syncthreads();
  float accum[16][4] = {};
  float maximum[2] = {-INFINITY, -INFINITY}, sum[2] = {0, 0};
  for (int keyBase = initial, stage = 0; keyBase <= last; keyBase += kKeys) {
    Stage<Local, DirectGlobalAddress>(values, vc, chunkV, keyBase, first, last, base, cap, kvhead);
    float scores[4][4] = {};
#pragma unroll
    for (int k = 0; k < kDim; k += 16) {
#pragma unroll
      for (int tile = 0; tile < 4; ++tile) {
        unsigned b[2];
#pragma unroll
        for (int j = 0; j < 2; ++j)
          b[j] = Pack(keys[0] + (tile * 8 + group) * kStride + k + tid * 2 + j * 8);
        AttentionMmaBF16(query[k / 16], b, scores[tile]);
      }
    }
    // Complete V load and all old K reads before overwriting the same K buffer.
    Wait();
    __syncthreads();
    if (keyBase + kKeys <= last)
      Stage<Local, DirectGlobalAddress>(keys[0], kc, chunkK, keyBase + kKeys, first, last, base,
                                        cap, kvhead);
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
    bool rescale = true;
    if constexpr (SkipUnitAlpha)
      rescale = !__all_sync(0xffffffff, alpha[0] == 1.f && alpha[1] == 1.f);
    if (rescale) {
#pragma unroll
      for (int tile = 0; tile < 16; ++tile) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
          accum[tile][j] *= alpha[j / 2];
      }
    }
#pragma unroll
    for (int step = 0; step < 2; ++step) {
#pragma unroll
      for (int tile = 0; tile < 16; ++tile) {
        unsigned b[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int key = step * 16 + tid * 2 + j * 8, col = tile * 8 + group;
          b[j] = static_cast<unsigned>(__bfloat16_as_ushort(values[key * kStride + col])) |
                 (static_cast<unsigned>(__bfloat16_as_ushort(values[(key + 1) * kStride + col]))
                  << 16);
        }
        AttentionMmaBF16(probability[step], b, accum[tile]);
      }
    }
    Wait();
    __syncthreads(); // next K ready; all V reads complete before overwrite
  }
  const float denominator[2] = {GroupSum(sum[0]), GroupSum(sum[1])};
#pragma unroll
  for (int tile = 0; tile < 16; ++tile) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int row = j / 2 ? row1 : row0;
      if (row < tokens) {
        const float v = RoundOutput ? B(accum[tile][j] / denominator[j / 2])
                                    : accum[tile][j] / denominator[j / 2];
        output[(row * 32 + head) * kDim + tile * 8 + tid * 2 + j % 2] = v;
        Finite(v, bad);
      }
    }
  }
}
__global__ void EncodeChunk(const float *k, const float *v, __nv_bfloat16 *packedK,
                            __nv_bfloat16 *packedV, int tokens) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i < tokens * 512) {
    packedK[i] = __float2bfloat16_rn(k[i]);
    packedV[i] = __float2bfloat16_rn(v[i]);
  }
}
template <bool RoundOutput = true>
void Launch(const float *q, const __nv_bfloat16 *kc, const __nv_bfloat16 *vc, const float *chunkK,
            const float *chunkV, float *output, int base, int tokens, int cap, bool local, int *bad,
            cudaStream_t stream, __nv_bfloat16 *packedK, __nv_bfloat16 *packedV,
            const PrefillState *prefill = nullptr, int context_limit = kW4A8FContextCapacity) {
  if (!q || !kc || !vc || !chunkK || !chunkV || !output || !bad || base < 0 || tokens < 1 ||
      tokens > kW4A8FMaxPrefillChunk || base > context_limit - tokens || cap < 1 ||
      cap > context_limit || (local ? cap < std::min(1024, base + tokens) : cap < base + tokens))
    throw std::runtime_error("G3 pipeline BF16 prefill attention shape/binding unsupported");
  if (!packedK || !packedV)
    throw std::runtime_error("packed BF16 chunk workspace missing");
  EncodeChunk<<<(tokens * 512 + 255) / 256, 256, 0, stream>>>(chunkK, chunkV, packedK, packedV,
                                                              tokens);
  const dim3 grid((tokens + kRows - 1) / kRows, 4);
  if (local)
    PipelineBF16<true, RoundOutput><<<grid, kThreads, 0, stream>>>(
        q, kc, vc, packedK, packedV, output, base, tokens, cap, bad, prefill, context_limit);
  else
    PipelineBF16<false, RoundOutput, PADRA_PREFILL_GLOBAL_DATAFLOW != 0,
                 PADRA_PREFILL_GLOBAL_DATAFLOW != 0><<<grid, kThreads, 0, stream>>>(
        q, kc, vc, packedK, packedV, output, base, tokens, cap, bad, prefill, context_limit);
}
} // namespace padded_single_prefill_attention
