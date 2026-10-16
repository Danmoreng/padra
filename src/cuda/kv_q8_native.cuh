#pragma once
// Q8_0 block layout, rotated KV and integer QK inspired by ggml-org/llama.cpp
// 10a60cf303566e10d6a7a2774c17d2085503d87b (MIT), ggml/src/ggml-cuda/fattn-common.cuh and
// fattn-vec.cuh. Native Padra implementation: four query heads share aligned, staged KV rows.
#include "context_admission.cuh"
namespace q8_native {
// One token:16 FP16 scales followed by512 codes; each head starts on a
// 32-byte boundary. Logical32-element block coefficients are unchanged.
struct alignas(16) CacheRow {
  __half d[16];
  signed char qs[512];
};
struct QueryBlock {
  float d;
  int qs[8];
};
static_assert(sizeof(CacheRow) == 544 && alignof(CacheRow) == 16);
static_assert(sizeof(QueryBlock) == 36 && alignof(QueryBlock) == 4);
constexpr int Features = 512, Heads = 32, Dim = 128, Threads = 128, Tile = 128;

__device__ __forceinline__ float WarpMax(float v) {
#pragma unroll
  for (int d = 16; d; d /= 2)
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, d));
  return v;
}
__device__ __forceinline__ float WarpSum(float v) {
#pragma unroll
  for (int d = 16; d; d /= 2)
    v += __shfl_xor_sync(0xffffffff, v, d);
  return v;
}
template <int Width> __device__ __forceinline__ void Rotate(float (&x)[4]) {
  const int lane = threadIdx.x % 32;
#pragma unroll
  for (int d = 1; d < 32; d *= 2) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const float other = __shfl_xor_sync(0xffffffff, x[r], d);
      x[r] = lane & d ? other - x[r] : x[r] + other;
    }
  }
#pragma unroll
  for (int d = 1; d < Width / 32; d *= 2) {
    float old[4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
      old[r] = x[r];
#pragma unroll
    for (int r = 0; r < 4; ++r)
      x[r] = r & d ? old[r ^ d] - old[r] : old[r] + old[r ^ d];
  }
#pragma unroll
  for (int r = 0; r < 4; ++r)
    x[r] *= Width == 128 ? .08838834764831845f : .125f;
}
__device__ __forceinline__ signed char Code(float v, float inverse) {
  return static_cast<signed char>(max(-127, min(127, static_cast<int>(roundf(v * inverse)))));
}
__device__ __forceinline__ int Load4(const signed char *p) {
  return __ldg(reinterpret_cast<const int *>(p));
}
template <bool Controlled = false>
__global__ void EncodeKV(const float *k, const float *v, CacheRow *kc, CacheRow *vc, int tokens,
                         int *bad, const DecodeState *state = nullptr, int cap = 0,
                         int maximum = 8192) {
  if (!k || !v || !kc || !vc || !bad || tokens < 1 || tokens > 2048 || blockDim.x != Threads ||
      blockDim.y != 1 || blockDim.z != 1 || gridDim.x != unsigned(tokens) || gridDim.y != 1 ||
      gridDim.z != 1 ||
      (Controlled && (tokens != 1 || !DecodeReadyAtLimit(state, maximum) ||
                      (cap != state->capacity && cap != min(1024, state->capacity))))) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  const int row = blockIdx.x, lane = threadIdx.x % 32, head = threadIdx.x / 32;
  const int target = Controlled ? state->position % cap : row;
  float a[4], b[4];
  bool invalid = false;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    a[r] = k[row * Features + head * Dim + lane + r * 32];
    b[r] = v[row * Features + head * Dim + lane + r * 32];
    invalid |= !isfinite(a[r]) || !isfinite(b[r]);
  }
  if (__any_sync(0xffffffff, invalid)) {
    if (lane == 0)
      atomicExch(bad, 1);
    return;
  }
  Rotate<128>(a);
  Rotate<64>(b);
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const float am = WarpMax(fabsf(a[r])), bm = WarpMax(fabsf(b[r]));
    const float ad = am / 127.f, bd = bm / 127.f;
    const __half ah = __float2half_rn(ad), bh = __float2half_rn(bd);
    if (!isfinite(am) || !isfinite(bm) || !isfinite(__half2float(ah)) ||
        !isfinite(__half2float(bh))) {
      if (!lane)
        atomicExch(bad, 1);
      return;
    }
    const int index = target;
    if (!lane) {
      kc[index].d[head * 4 + r] = ah;
      vc[index].d[head * 4 + r] = bh;
    }
    kc[index].qs[head * 128 + r * 32 + lane] = Code(a[r], ad == 0.f ? 0.f : 1.f / ad);
    vc[index].qs[head * 128 + r * 32 + lane] = Code(b[r], bd == 0.f ? 0.f : 1.f / bd);
  }
}

__global__ void EncodeQuery(const float *q, QueryBlock *out, int *bad) {
  if (!q || !out || !bad || blockDim.x != Threads || blockDim.y != 1 || blockDim.z != 1 ||
      gridDim.x != 8 || gridDim.y != 1 || gridDim.z != 1) {
    if (!threadIdx.x && bad)
      atomicExch(bad, 1);
    return;
  }
  const int head = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
  float x[4];
  bool invalid = false;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    x[r] = q[head * 128 + r * 32 + lane];
    invalid |= !isfinite(x[r]);
  }
  if (__any_sync(0xffffffff, invalid)) {
    if (!lane)
      atomicExch(bad, 1);
    return;
  }
  Rotate<128>(x);
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const float d = WarpMax(fabsf(x[r])) / 127.f;
    if (!isfinite(d)) {
      if (!lane)
        atomicExch(bad, 1);
      return;
    }
    const int c = static_cast<unsigned char>(Code(x[r], d == 0 ? 0 : 1.f / d));
    const int b0 = __shfl_sync(0xffffffff, c, (lane / 4) * 4),
              b1 = __shfl_sync(0xffffffff, c, (lane / 4) * 4 + 1),
              b2 = __shfl_sync(0xffffffff, c, (lane / 4) * 4 + 2),
              b3 = __shfl_sync(0xffffffff, c, (lane / 4) * 4 + 3);
    if (!lane)
      out[head * 4 + r].d = d;
    if (lane % 4 == 0)
      out[head * 4 + r].qs[lane / 4] = static_cast<unsigned>(b0) | static_cast<unsigned>(b1) << 8 |
                                       static_cast<unsigned>(b2) << 16 |
                                       static_cast<unsigned>(b3) << 24;
  }
}

// Four query-head warps share one KV head. Each retains the prior four
// independent32-key online states and the exact previous merge order.
// Double-buffered cp.async staging overlaps next-tile loads with current QK/PV.
// Shared row padding prevents bank conflicts for both QK subgroup and PV reads.

__device__ __forceinline__ int SharedIndex(int r) {
  return r * 32 + (r / 4) * 4;
}
__device__ __forceinline__ void CopyAsync16(void *dst, const void *src, int bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(
                   unsigned(__cvta_generic_to_shared(dst))),
               "l"(src), "r"(bytes)
               : "memory");
}
template <bool Local>
__device__ __forceinline__ void StageAsync(const CacheRow *kc, const CacheRow *vc, int *keys,
                                           unsigned *values, float *ks, float *vs, int chunk, int n,
                                           int first, int kv) {
  const int r = threadIdx.x / 4, b = threadIdx.x % 4, offset = chunk * 32 + r,
            row = Local ? ((first + offset) & 1023) : first + offset;
  ks[r * 5 + b] = offset < n ? __half2float(kc[row].d[kv * 4 + b]) : 0.f;
  vs[r * 4 + b] = offset < n ? __half2float(vc[row].d[kv * 4 + b]) : 0.f;
#pragma unroll
  for (int i = threadIdx.x; i < 256; i += 128) {
    const int r = i / 8, c = (i % 8) * 4, offset = chunk * 32 + r,
              row = Local ? ((first + offset) & 1023) : first + offset;
    const bool valid = offset < n;
    const void *k = valid ? static_cast<const void *>(kc[row].qs + kv * 128 + c * 4)
                          : static_cast<const void *>(kc);
    const void *v = valid ? static_cast<const void *>(vc[row].qs + kv * 128 + c * 4)
                          : static_cast<const void *>(vc);
    CopyAsync16(keys + SharedIndex(r) + c, k, valid ? 16 : 0);
    CopyAsync16(values + SharedIndex(r) + c, v, valid ? 16 : 0);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <bool Local>
__launch_bounds__(128, 5) __global__
    void Split(const QueryBlock *__restrict__ q, const CacheRow *__restrict__ kc,
               const CacheRow *__restrict__ vc, float *__restrict__ partial,
               float *__restrict__ lse, const DecodeState *state, int cap, int splits, int *bad,
               int maximum) {
  if (!DecodeReadyAtLimit(state, maximum) ||
      (Local ? cap != 1024 || state->capacity < 1024 : cap != state->capacity) || !q || !kc ||
      !vc || !partial || !lse || !bad || splits < 1 || splits > 32 || blockDim.x != 128 ||
      blockDim.y != 1 || blockDim.z != 1 || gridDim.x != 8 || gridDim.y != unsigned(splits) ||
      gridDim.z != 1) {
    if (!threadIdx.x && bad)
      atomicExch(bad, 1);
    return;
  }
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, sub = lane % 4,
            head = blockIdx.x * 4 + warp, kv = blockIdx.x / 2;
  const int n = Local ? min(state->position + 1, 1024) : state->position + 1,
            first = state->position + 1 - n, tiles = (n + 127) / 128, active = min(splits, tiles),
            split = blockIdx.y;
  if (split >= active)
    return;
  const int begin = tiles * split / active, end = tiles * (split + 1) / active;
  int query[4][2];
  float qs[4];
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const auto &qb = q[head * 4 + b];
    query[b][0] = qb.qs[sub];
    query[b][1] = qb.qs[sub + 4];
    qs[b] = qb.d;
  }
  __shared__ int keys[2][32 * 33];
  __shared__ float ks[2][32 * 5], vs[2][32 * 4];
  __shared__ __align__(16) unsigned values[2][32 * 33];
  float accum[4][4] = {}, mx[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}, den[4] = {};
  StageAsync<Local>(kc, vc, keys[0], values[0], ks[0], vs[0], begin * 4, n, first, kv);
  for (int tile = begin; tile < end; ++tile) {
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {

      const int slot = slice % 2;
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
      if (tile * 4 + slice + 1 < end * 4)
        StageAsync<Local>(kc, vc, keys[1 - slot], values[1 - slot], ks[1 - slot], vs[1 - slot],
                          tile * 4 + slice + 1, n, first, kv);
      float own = -INFINITY;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int key = (lane & ~3) + j;
        float dot = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          int sum = __dp4a(keys[slot][SharedIndex(key) + b * 8 + sub], query[b][0], 0);
          sum = __dp4a(keys[slot][SharedIndex(key) + b * 8 + sub + 4], query[b][1], sum);
          dot = fmaf(float(sum), ks[slot][key * 5 + b] * qs[b], dot);
        }
        dot += __shfl_xor_sync(0xffffffff, dot, 1, 4);
        dot += __shfl_xor_sync(0xffffffff, dot, 2, 4);
        if (sub == j)
          own = tile * 128 + slice * 32 + key < n ? dot * .08838834764831845f : -INFINITY;
      }
      const float next = fmaxf(mx[slice], WarpMax(own)),
                  alpha = isfinite(mx[slice]) ? expf(mx[slice] - next) : 0.f,
                  p = isfinite(own) ? expf(own - next) : 0.f;
      mx[slice] = next;
      den[slice] = fmaf(den[slice], alpha, p);
#pragma unroll
      for (int d = 0; d < 4; ++d)
        accum[slice][d] *= alpha;
#pragma unroll 4
      for (int key = 0; key < 32; ++key) {
        float prob = __shfl_sync(0xffffffff, p, key);
        float factor = prob * vs[slot][key * 4 + lane / 8];
        const unsigned codes = values[slot][SharedIndex(key) + lane];
#pragma unroll
        for (int d = 0; d < 4; ++d)
          accum[slice][d] =
              fmaf(factor, float(static_cast<signed char>(codes >> (8 * d))), accum[slice][d]);
      }

      __syncthreads();
    }
  }
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    den[s] = WarpSum(den[s]);
    mx[s] = den[s] > 0 ? mx[s] + logf(den[s]) : -INFINITY;
#pragma unroll
    for (int d = 0; d < 4; ++d)
      accum[s][d] = den[s] > 0 ? accum[s][d] / den[s] : 0.f;
  }
  float maximumLse = fmaxf(fmaxf(mx[0], mx[1]), fmaxf(mx[2], mx[3])), total = 0, sum[4] = {};
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    float w = isfinite(mx[s]) ? expf(mx[s] - maximumLse) : 0.f;
    total += w;
#pragma unroll
    for (int d = 0; d < 4; ++d)
      sum[d] = fmaf(w, accum[s][d], sum[d]);
  }
  if (!(total > 0.f) || !isfinite(total)) {
    if (!lane)
      atomicExch(bad, 1);
    return;
  }
#pragma unroll
  for (int d = 0; d < 4; ++d)
    partial[(split * 32 + head) * 128 + lane * 4 + d] = sum[d] / total;
  if (!lane)
    lse[split * 32 + head] = maximumLse + logf(total);
}

template <bool Local, class Output = __nv_bfloat16>
__global__ void Merge(const float *partial, const float *lse, Output *out, const DecodeState *state,
                      int splits, int *bad, int maximum) {
  if (!DecodeReadyAtLimit(state, maximum) || !partial || !lse || !out || !bad || splits < 1 ||
      splits > 32 || blockDim.x != Threads || blockDim.y != 1 || blockDim.z != 1 ||
      gridDim.x != 8 || gridDim.y != 1 || gridDim.z != 1) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  const int head = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = Local ? min(state->position + 1, 1024) : state->position + 1;
  const int active = min(splits, (n + Tile - 1) / Tile);
  const float s = lane < active ? lse[lane * 32 + head] : -INFINITY;
  const float mx = WarpMax(s), p = lane < active ? expf(s - mx) : 0.f;
  const float den = WarpSum(p), weight = p / den;
  float x[4] = {};
  for (int split = 0; split < active; ++split) {
    const float w = __shfl_sync(0xffffffff, weight, split);
#pragma unroll
    for (int r = 0; r < 4; ++r)
      x[r] = fmaf(w, partial[(split * 32 + head) * 128 + r * 32 + lane], x[r]);
  }
  Rotate<64>(x);
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    if (!isfinite(x[r]))
      atomicExch(bad, 1);
    const auto value = __float2bfloat16_rn(x[r]);
    if constexpr (sizeof(Output) == sizeof(float))
      out[head * 128 + r * 32 + lane] = __bfloat162float(value);
    else
      out[head * 128 + r * 32 + lane] = value;
  }
}
// One per-layer temporary prefix workspace, reused across all28 layers.
// A fixed grid reads the current device base, so captured chunk graphs remain
// valid when replayed at later absolute positions.
template <class Output>
__global__ void Dequantize(const CacheRow *k, const CacheRow *v, Output *outK, Output *outV,
                           int rows, int *bad) {
  if (!k || !v || !outK || !outV || !bad || rows < 1 || rows > 2048 || blockDim.x != 256 ||
      blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1) {
    if (!threadIdx.x && bad)
      atomicExch(bad, 1);
    return;
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * Features;
       i += blockDim.x * gridDim.x) {
    const auto &kb = k[i / 512], &vb = v[i / 512];
    const float kd = __half2float(kb.d[(i % 512) / 32]), vd = __half2float(vb.d[(i % 512) / 32]);
    if ((i % 32 == 0 && (!isfinite(kd) || !isfinite(vd) || kd < 0 || vd < 0)) ||
        kb.qs[i % 512] == -128 || vb.qs[i % 512] == -128)
      atomicExch(bad, 1);
    const auto a = __float2bfloat16_rn(static_cast<float>(kb.qs[i % 512]) * kd);
    const auto b = __float2bfloat16_rn(static_cast<float>(vb.qs[i % 512]) * vd);
    if constexpr (sizeof(Output) == sizeof(float)) {
      outK[i] = __bfloat162float(a);
      outV[i] = __bfloat162float(b);
    } else {
      outK[i] = a;
      outV[i] = b;
    }
  }
}
template <class State>
__global__ void DequantizePrefix(const CacheRow *k, const CacheRow *v, __nv_bfloat16 *outK,
                                 __nv_bfloat16 *outV, const State *state, int tokens, int cap,
                                 int *bad, int maximum) {
  if (!state || !k || !v || !outK || !outV || !bad ||
      !ValidPrefillState(*state, tokens, state->capacity, 1, maximum) || tokens > 2048 ||
      (cap != state->capacity && cap != min(1024, state->capacity))) {
    if (!threadIdx.x && bad)
      atomicExch(bad, 1);
    return;
  }
  const int rows = min(state->base, cap);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * Features;
       i += blockDim.x * gridDim.x) {
    const auto &kb = k[i / 512], &vb = v[i / 512];
    const float kd = __half2float(kb.d[(i % 512) / 32]), vd = __half2float(vb.d[(i % 512) / 32]);
    if ((i % 32 == 0 && (!isfinite(kd) || !isfinite(vd) || kd < 0 || vd < 0)) ||
        kb.qs[i % 512] == -128 || vb.qs[i % 512] == -128)
      atomicExch(bad, 1);
    outK[i] = __float2bfloat16_rn(static_cast<float>(kb.qs[i % 512]) * kd);
    outV[i] = __float2bfloat16_rn(static_cast<float>(vb.qs[i % 512]) * vd);
  }
}
template <bool InverseValue> __global__ void RotateRows(float *data, int tokens, int *bad) {
  if (!data || !bad || tokens < 1 || tokens > 2048 || blockDim.x != Threads || blockDim.y != 1 ||
      blockDim.z != 1 || gridDim.x != unsigned(tokens * 8) || gridDim.y != 1 || gridDim.z != 1) {
    if (!threadIdx.x && bad)
      atomicExch(bad, 1);
    return;
  }
  const int headRow = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (headRow >= tokens * 32)
    return;
  float x[4];
#pragma unroll
  for (int r = 0; r < 4; ++r)
    x[r] = data[headRow * 128 + r * 32 + lane];
  if constexpr (InverseValue)
    Rotate<64>(x);
  else
    Rotate<128>(x);
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    if (!isfinite(x[r]))
      atomicExch(bad, 1);
    data[headRow * 128 + r * 32 + lane] = __bfloat162float(__float2bfloat16_rn(x[r]));
  }
}
template <class State>
__global__ void Commit(const CacheRow *k, const CacheRow *v, CacheRow *kc, CacheRow *vc,
                       const State *state, int tokens, int cap, int first, int *bad, int maximum) {
  if (!state || !k || !v || !kc || !vc || !bad ||
      !ValidPrefillState(*state, tokens, state->capacity, 1, maximum) || tokens > 2048 ||
      (cap != state->capacity && cap != min(1024, state->capacity)) ||
      first != (cap == state->capacity ? 0 : max(0, tokens - cap)) || first < 0 ||
      first >= tokens || tokens - first > cap || blockDim.x != Threads || blockDim.y != 1 ||
      blockDim.z != 1 || gridDim.x != unsigned(tokens - first) || gridDim.y != 1 ||
      gridDim.z != 1) {
    if (!threadIdx.x && bad)
      atomicExch(bad, 1);
    return;
  }
  if (*bad)
    return;
  const int row = first + blockIdx.x, target = (state->base + row) % cap;
  for (int word = threadIdx.x; word < 136; word += blockDim.x) {
    reinterpret_cast<unsigned *>(kc + target)[word] =
        reinterpret_cast<const unsigned *>(k + row)[word];
    reinterpret_cast<unsigned *>(vc + target)[word] =
        reinterpret_cast<const unsigned *>(v + row)[word];
  }
}
} // namespace q8_native
