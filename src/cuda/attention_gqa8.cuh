#include "context_admission.cuh"
// N3a eight useful Q rows per KV head. llama.cpp10a60cf3 (MIT) grouped
// fattn-mma-f16 tile/online/fixup mechanics; Gem16c01789a2 (Apache2) BF16 MMA
// and async packed loads. Native Mellum BF16 operands, D128/Q32/KV4,
// local1024 absolute ring chronology and score scale exactly once.
namespace gqa8_attention {
constexpr int kThreads = 128, kKeys = 32, kStride = 136, kProbStride = 40, kHeads = 32, kDim = 128;
constexpr int kLocalSplits = 16, kGlobalSplits = 32;
constexpr float kScoreScale = .08838834764831845f;
int MaxSplits(int cap, bool local, int context_limit = kW4A8FContextCapacity) {
  if (cap < 1 || cap > context_limit || context_limit < 1 || context_limit > 131072)
    throw std::runtime_error("GQA8 BF16 capacity unsupported");
  return std::min((cap + kKeys - 1) / kKeys, local ? kLocalSplits : kGlobalSplits);
}
std::size_t WorkspaceElements(int cap, int context_limit = kW4A8FContextCapacity) {
  return static_cast<std::size_t>(MaxSplits(cap, false, context_limit)) * kHeads * (kDim + 1);
}
__device__ __forceinline__ unsigned Pack(const __nv_bfloat16 *p) {
  return unsigned(__bfloat16_as_ushort(p[0])) | (unsigned(__bfloat16_as_ushort(p[1])) << 16);
}
__device__ __forceinline__ unsigned Pair(__nv_bfloat16 a, __nv_bfloat16 b) {
  return unsigned(__bfloat16_as_ushort(a)) | (unsigned(__bfloat16_as_ushort(b)) << 16);
}
__device__ __forceinline__ void Mma(const unsigned *a, const unsigned *b, float *c) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void Copy16(void *dst, const void *src, int bytes) {
  const auto address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(address), "l"(src),
               "r"(bytes));
}
__device__ __forceinline__ void Commit() {
  asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void Wait() {
  asm volatile("cp.async.wait_group 0;\n");
}
__device__ __forceinline__ float WarpMax(float x) {
  for (int d = 16; d; d /= 2)
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, d));
  return x;
}
__device__ __forceinline__ float WarpSum(float x) {
  for (int d = 16; d; d /= 2)
    x += __shfl_xor_sync(0xffffffff, x, d);
  return x;
}
template <bool Local>
__device__ __forceinline__ bool Admitted(const DecodeState *s, int cap, int splits,
                                         int context_limit = kW4A8FContextCapacity) {
  return DecodeReadyAtLimit(s, context_limit) && cap > 0 && cap <= s->capacity &&
         cap <= context_limit && (Local ? cap == min(1024, s->capacity) : cap == s->capacity) &&
         splits == min((cap + kKeys - 1) / kKeys, Local ? kLocalSplits : kGlobalSplits);
}
// Invalid tail rows use cp.async zero fill, and never address a future/evicted
// ring slot. Token rows are256B, dimension chunks16B; padded shared stride272B.
__device__ __forceinline__ void Stage(const __nv_bfloat16 *kc, const __nv_bfloat16 *vc,
                                      __nv_bfloat16 *ks, __nv_bfloat16 *vs, int first, int tile,
                                      int n, int cap, int kv) {
  for (int i = threadIdx.x; i < kKeys * 16; i += kThreads) {
    const int row = i / 16, d = (i % 16) * 8, offset = tile * kKeys + row;
    const bool valid = offset < n;
    const int source = valid ? ((first + offset) % cap) * 512 + kv * kDim + d : 0;
    Copy16(ks + row * kStride + d, kc + source, valid ? 16 : 0);
    Copy16(vs + row * kStride + d, vc + source, valid ? 16 : 0);
  }
  Commit();
}
template <bool Local>
__launch_bounds__(kThreads, 2) __global__
    void SplitBF16(const float *q, const __nv_bfloat16 *kc, const __nv_bfloat16 *vc, float *partial,
                   float *lse, const DecodeState *state, int cap, int max_splits, int *bad,
                   int context_limit = kW4A8FContextCapacity) {
  if (!Admitted<Local>(state, cap, max_splits, context_limit) || !q || !kc || !vc || !partial ||
      !lse || !bad || blockDim.x != kThreads || blockDim.y != 1 || blockDim.z != 1 ||
      gridDim.x != unsigned(4 * max_splits) || gridDim.y != 1 || gridDim.z != 1) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  const int n = Local ? min(state->position + 1, 1024) : state->position + 1,
            first = state->position + 1 - n;
  const int tiles = (n + kKeys - 1) / kKeys, active = min(max_splits, tiles),
            split = blockIdx.x / 4, kv = blockIdx.x % 4;
  if (split >= active)
    return; // Merge uses exactly active; poisoned unused scratch is unread.
  const int begin = tiles * split / active, end = tiles * (split + 1) / active;
  __shared__ __align__(16) __nv_bfloat16 qs[16 * kStride], ks[2][kKeys * kStride],
      vs[2][kKeys * kStride], p[16 * kProbStride];
  __shared__ float scores[8 * kKeys], maximum[8], denominator[8], alpha[8];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, group = lane / 4, tid = lane % 4;
  for (int i = threadIdx.x; i < 16 * kStride; i += kThreads) {
    const int row = i / kStride, d = i % kStride;
    float x = row < 8 && d < kDim ? q[(kv * 8 + row) * kDim + d] : 0.f;
    if (!isfinite(x) || (__float_as_uint(x) & 0xffffu) != 0)
      atomicExch(bad, 1);
    qs[i] = __float2bfloat16_rn(x);
  }
  for (int i = threadIdx.x; i < 16 * kProbStride; i += kThreads)
    p[i] = __float2bfloat16_rn(0.f);
  if (threadIdx.x < 8) {
    maximum[threadIdx.x] = -INFINITY;
    denominator[threadIdx.x] = 0.f;
  }
  Stage(kc, vc, ks[0], vs[0], first, begin, n, cap, kv);
  __syncthreads();
  float accum[4][4] = {};
  for (int tile = begin; tile < end; ++tile) {
    const int slot = (tile - begin) % 2;
    Wait();
    __syncthreads();
    // Prefetch the next tile while this tile is consumed; no buffer is reused
    // until all warps have finished PV at the end-of-tile barrier.
    if (tile + 1 < end)
      Stage(kc, vc, ks[1 - slot], vs[1 - slot], first, tile + 1, n, cap, kv);
    for (int i = threadIdx.x; i < kKeys * kDim; i += kThreads) {
      const int index = (i / kDim) * kStride + i % kDim;
      Finite(__bfloat162float(ks[slot][index]), bad);
      Finite(__bfloat162float(vs[slot][index]), bad);
    }
    float score[4] = {};
#pragma unroll
    for (int k = 0; k < kDim; k += 16) {
      unsigned a[4], b[2];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        a[j] = Pack(qs + (group + (j % 2) * 8) * kStride + k + tid * 2 + (j / 2) * 8);
#pragma unroll
      for (int j = 0; j < 2; ++j)
        b[j] = Pack(ks[slot] + (warp * 8 + group) * kStride + k + tid * 2 + j * 8);
      Mma(a, b, score);
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int key = warp * 8 + tid * 2 + j;
      const float value = score[j] * kScoreScale;
      if (tile * kKeys + key < n)
        Finite(value, bad);
      scores[group * kKeys + key] = tile * kKeys + key < n ? value : -INFINITY;
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int row = warp * 2 + r;
      const float s = scores[row * kKeys + lane];
      const float next = fmaxf(maximum[row], WarpMax(s)),
                  a = isfinite(maximum[row]) ? expf(maximum[row] - next) : 0.f;
      const float prob = isfinite(s) ? expf(s - next) : 0.f, total = WarpSum(prob);
      p[row * kProbStride + lane] = __float2bfloat16_rn(prob);
      if (lane == 0) {
        alpha[row] = a;
        maximum[row] = next;
        denominator[row] = fmaf(denominator[row], a, total);
      }
    }
    __syncthreads();
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      accum[c][0] *= alpha[group];
      accum[c][1] *= alpha[group];
    }
#pragma unroll
    for (int k = 0; k < kKeys; k += 16) {
      unsigned a[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        a[j] = Pack(p + (group + (j % 2) * 8) * kProbStride + k + tid * 2 + (j / 2) * 8);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        unsigned b[2];
        const int d = warp * 32 + c * 8 + group;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int row = k + tid * 2 + j * 8;
          b[j] = Pair(vs[slot][row * kStride + d], vs[slot][(row + 1) * kStride + d]);
        }
        Mma(a, b, accum[c]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int c = 0; c < 4; ++c) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int d = warp * 32 + c * 8 + tid * 2 + j;
      const float value = accum[c][j] / denominator[group];
      partial[(split * kHeads + kv * 8 + group) * kDim + d] = value;
      Finite(value, bad);
    }
  }
  if (threadIdx.x < 8) {
    const int row = threadIdx.x;
    const float value = maximum[row] + logf(denominator[row]);
    lse[split * kHeads + kv * 8 + row] = value;
    Finite(value, bad);
  }
}
template <bool Local>
__launch_bounds__(kThreads, 2) __global__
    void MergeBF16(const float *partial, const float *lse, float *out, const DecodeState *state,
                   int cap, int max_splits, int *bad, int context_limit = kW4A8FContextCapacity) {
  if (!Admitted<Local>(state, cap, max_splits, context_limit) || !partial || !lse || !out || !bad ||
      blockDim.x != kThreads || blockDim.y != 1 || blockDim.z != 1 || gridDim.x != 4 ||
      gridDim.y != 1 || gridDim.z != 1) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  const int n = Local ? min(state->position + 1, 1024) : state->position + 1,
            active = min(max_splits, (n + kKeys - 1) / kKeys);
  const int kv = blockIdx.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  __shared__ float weights[8 * kGlobalSplits];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int row = warp * 2 + r, head = kv * 8 + row;
    const float v = lane < active ? lse[lane * kHeads + head] : -INFINITY;
    const float mx = WarpMax(v), w = lane < active ? expf(v - mx) : 0.f, den = WarpSum(w);
    if (lane < active) {
      const float value = w / den;
      weights[row * kGlobalSplits + lane] = value;
      Finite(value, bad);
    }
  }
  __syncthreads();
#pragma unroll
  for (int row = 0; row < 8; ++row) {
    float sum = 0.f;
    const int head = kv * 8 + row, d = threadIdx.x;
    for (int s = 0; s < active; ++s)
      sum = fmaf(weights[row * kGlobalSplits + s], partial[(s * kHeads + head) * kDim + d], sum);
    const float value = B(sum);
    out[head * kDim + d] = value;
    Finite(value, bad);
  }
}
void Launch(const float *q, const __nv_bfloat16 *kc, const __nv_bfloat16 *vc, float *out,
            const DecodeState *state, int cap, bool local, float *scratch, int *bad,
            cudaStream_t stream) {
  const int splits = MaxSplits(cap, local);
  auto *lse = scratch + splits * kHeads * kDim;
  if (local) {
    SplitBF16<true>
        <<<4 * splits, kThreads, 0, stream>>>(q, kc, vc, scratch, lse, state, cap, splits, bad);
    MergeBF16<true><<<4, kThreads, 0, stream>>>(scratch, lse, out, state, cap, splits, bad);
  } else {
    SplitBF16<false>
        <<<4 * splits, kThreads, 0, stream>>>(q, kc, vc, scratch, lse, state, cap, splits, bad);
    MergeBF16<false><<<4, kThreads, 0, stream>>>(scratch, lse, out, state, cap, splits, bad);
  }
}
} // namespace gqa8_attention
