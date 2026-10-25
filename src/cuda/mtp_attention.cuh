#pragma once
// Exact D1 attention. Equal tile partitions share the 16-row MMA tile; other
// positions retain independent per-row partitions in one batched launch.
namespace mtp_attention {
using namespace gqa8_fragment_validation;
template <bool Local>
__device__ __forceinline__ void
StageRows(const __nv_bfloat16 *kc, const __nv_bfloat16 *vc, __nv_bfloat16 *ks, __nv_bfloat16 *vs,
          int first, int tile, int n, int cap, int kv, int token, const DecodeState *state,
          const __nv_bfloat16 *oldSecondRow) {
  for (int i = threadIdx.x; i < kKeys * 16; i += kThreads) {
    const int row = i / 16, d = (i % 16) * 8, offset = tile * kKeys + row;
    const bool valid = offset < n;
    const int absolute = first + offset;
    const int source = valid ? (absolute % cap) * 512 + kv * kDim + d : 0;
    const bool saved = Local && valid && token == 0 && state[1].position >= cap &&
                       absolute == state[1].position - cap;
    const auto *key = saved ? oldSecondRow + kv * kDim + d : kc + source;
    const auto *value = saved ? oldSecondRow + 512 + kv * kDim + d : vc + source;
    Copy16(ks + row * kStride + d, key, valid ? 16 : 0);
    Copy16(vs + row * kStride + d, value, valid ? 16 : 0);
  }
  Commit();
}
template <bool Local>
__launch_bounds__(kThreads, 2) __global__
    void SplitRows(const float *q, const __nv_bfloat16 *kc, const __nv_bfloat16 *vc, float *partial,
                   float *lse, const DecodeState *state, int cap, int max_splits, int *bad,
                   const __nv_bfloat16 *oldSecondRow, int context_limit = kW4A8FContextCapacity) {
  if (!Admitted<Local>(state, cap, max_splits, context_limit) ||
      !Admitted<Local>(state + 1, cap, max_splits, context_limit) ||
      state[1].position != state[0].position + 1 || (Local && !oldSecondRow) || !q || !kc || !vc ||
      !partial || !lse || !bad || blockDim.x != kThreads || blockDim.y != 1 || blockDim.z != 1 ||
      gridDim.x != unsigned(4 * max_splits) || gridDim.y != 1 || gridDim.z != 2) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  const int n0 = Local ? min(state[0].position + 1, 1024) : state[0].position + 1;
  const int n1 = Local ? min(state[1].position + 1, 1024) : state[1].position + 1;
  const int first0 = state[0].position + 1 - n0, first1 = state[1].position + 1 - n1;
  const bool paired = first0 == first1 && (n0 + 31) / 32 == (n1 + 31) / 32;
  if (paired && blockIdx.z)
    return;
  const int token = blockIdx.z;
  const int n = paired || token ? n1 : n0, first = paired || !token ? first0 : first1;
  const int tiles = (n + 31) / 32, active = min(max_splits, tiles);
  const int split = blockIdx.x / 4, kv = blockIdx.x % 4;
  const int workspace = max_splits * 32 * 129;
  if (split >= active)
    return;
  const int begin = tiles * split / active, end = tiles * (split + 1) / active;
  __shared__ __align__(16) __nv_bfloat16 qs[16 * kStride], ks[2][kKeys * kStride],
      vs[2][kKeys * kStride], p[16 * kProbStride];
  __shared__ float scores[16 * kKeys], maximum[16], denominator[16], alpha[16];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, group = lane / 4, tid = lane % 4;
  for (int i = threadIdx.x; i < 16 * kStride; i += kThreads) {
    const int row = i / kStride, d = i % kStride;
    const int sourceRow = paired ? row / 8 : token;
    float x =
        (paired || row < 8) && d < kDim ? q[sourceRow * 4096 + (kv * 8 + row % 8) * kDim + d] : 0.f;
    if (!isfinite(x) || (__float_as_uint(x) & 0xffffu) != 0)
      atomicExch(bad, 1);
    qs[i] = __float2bfloat16_rn(x);
  }
  for (int i = threadIdx.x; i < 16 * kProbStride; i += kThreads)
    p[i] = __float2bfloat16_rn(0.f);
  if (threadIdx.x < 16) {
    maximum[threadIdx.x] = -INFINITY;
    denominator[threadIdx.x] = 0.f;
  }
  StageRows<Local>(kc, vc, ks[0], vs[0], first, begin, n, cap, kv, token, state, oldSecondRow);
  __syncthreads();
  float accum[4][4] = {};
  for (int tile = begin; tile < end; ++tile) {
    const int slot = (tile - begin) % 2;
    Wait();
    __syncthreads();
    // Prefetch the next tile while this tile is consumed; no buffer is reused
    // until all warps have finished PV at the end-of-tile barrier.
    if (tile + 1 < end)
      StageRows<Local>(kc, vc, ks[1 - slot], vs[1 - slot], first, tile + 1, n, cap, kv, token,
                       state, oldSecondRow);
    unsigned nonfinite = 0;
    float score[4] = {};
#pragma unroll
    for (int k = 0; k < kDim; k += 16) {
      unsigned a[4], b[2];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        a[j] = Pack(qs + (group + (j % 2) * 8) * kStride + k + tid * 2 + (j / 2) * 8);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        b[j] = Pack(ks[slot] + (warp * 8 + group) * kStride + k + tid * 2 + j * 8);
        nonfinite |= NonfinitePair(b[j]);
      }
      Mma(a, b, score);
    }
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      if (t && !paired)
        continue;
      const int visible = paired ? (t ? n1 : n0) : n;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int key = warp * 8 + tid * 2 + j;
        const float value = score[j + t * 2] * kScoreScale;
        if (tile * kKeys + key < visible)
          Finite(value, bad);
        scores[(group + t * 8) * kKeys + key] = tile * kKeys + key < visible ? value : -INFINITY;
      }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      if (r >= 2 && !paired)
        continue;
      const int row = (r / 2) * 8 + warp * 2 + r % 2;
      const float s = scores[row * kKeys + lane];
      const float next = fmaxf(maximum[row], WarpMax(s)),
                  a = isfinite(maximum[row]) ? expf(maximum[row] - next) : 0.f;
      const float prob = isfinite(s) ? expf(s - next) : 0.f, total = WarpSum(prob);
      p[row * kProbStride + lane] = __float2bfloat16_rn(prob);
      __syncwarp(); // Complete all old context_limit reads before lane0 updates shared state.
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
      if (paired) {
        accum[c][2] *= alpha[group + 8];
        accum[c][3] *= alpha[group + 8];
      }
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
          nonfinite |= NonfinitePair(b[j]);
        }
        Mma(a, b, accum[c]);
      }
    }
    if (nonfinite)
      atomicExch(bad, 1);
    __syncthreads();
  }
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    if (t && !paired)
      continue;
    const int outputToken = paired ? t : token;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int d = warp * 32 + c * 8 + tid * 2 + j;
        const float value = accum[c][j + t * 2] / denominator[group + t * 8];
        partial[outputToken * workspace + (split * kHeads + kv * 8 + group) * kDim + d] = value;
        Finite(value, bad);
      }
    }
  }
  if (threadIdx.x < (paired ? 16 : 8)) {
    const int row = threadIdx.x, outputToken = paired ? row / 8 : token;
    const float value = maximum[row] + logf(denominator[row]);
    lse[outputToken * workspace + split * kHeads + kv * 8 + row % 8] = value;
    Finite(value, bad);
  }
}
template <bool Local>
__global__ void MergeRows(const float *partial, const float *lse, __nv_bfloat16 *out,
                          const DecodeState *state, int cap, int splits, int *bad,
                          int maximum = kW4A8FContextCapacity) {
  if (!gqa8_attention::Admitted<Local>(state + blockIdx.y, cap, splits, maximum) || !partial ||
      !lse || !out || !bad || blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
      gridDim.x != 32 || gridDim.y != 2 || gridDim.z != 1) {
    if (!threadIdx.x && bad)
      atomicExch(bad, 1);
    return;
  }
  const int workspace = splits * 32 * 129;
  state += blockIdx.y;
  partial += blockIdx.y * workspace;
  lse += blockIdx.y * workspace;
  out += blockIdx.y * 4096;
  const int n = Local ? min(state->position + 1, 1024) : state->position + 1,
            active = min(splits, (n + 31) / 32), head = blockIdx.x, lane = threadIdx.x % 32;
  __shared__ float weights[32];
  if (threadIdx.x < 32) {
    const float v = lane < active ? lse[lane * 32 + head] : -INFINITY;
    const float mx = gqa8_attention::WarpMax(v), w = lane < active ? expf(v - mx) : 0.f;
    const float den = gqa8_attention::WarpSum(w);
    if (lane < active) {
      weights[lane] = w / den;
      Finite(weights[lane], bad);
    }
  }
  __syncthreads();
  float sum = 0.f;
  for (int s = 0; s < active; ++s)
    sum = fmaf(weights[s], partial[(s * 32 + head) * 128 + threadIdx.x], sum);
  const float v = B(sum);
  out[head * 128 + threadIdx.x] = __float2bfloat16_rn(v);
  Finite(v, bad);
}
__global__ void QuantizeRows(const __nv_bfloat16 *x, unsigned char *encoded, float *scales,
                             int *bad) {
  if (!x || !encoded || !scales || !bad || blockDim.x != 512 || blockDim.y != 1 ||
      blockDim.z != 1 || gridDim.x != 2 || gridDim.y != 1 || gridDim.z != 1) {
    if (!threadIdx.x && bad)
      atomicExch(bad, 1);
    return;
  }
  x += blockIdx.x * 4096;
  encoded += blockIdx.x * 4096;
  scales += blockIdx.x;
  __shared__ float maxima[16], scale, inverse;
  float row[8], mx = 0.f;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float v = __bfloat162float(x[threadIdx.x + j * 512]);
    row[j] = v;
    Finite(v, bad);
    mx = fmaxf(mx, fabsf(v));
  }
  mx = gqa8_attention::WarpMax(mx);
  if (lane == 0)
    maxima[warp] = mx;
  __syncthreads();
  if (warp == 0) {
    const float maximum = gqa8_attention::WarpMax(lane < 16 ? maxima[lane] : 0.f);
    if (lane == 0) {
      scale = maximum == 0 ? 1.f : __fdiv_rn(maximum, 448.f);
      scales[0] = scale;
      inverse = __fdiv_rn(1.f, scale);
    }
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; ++j)
    encoded[threadIdx.x + j * 512] = fp8_exact_reciprocal::Encode(row[j], scale, inverse);
}
inline void Launch(const float *queries, const __nv_bfloat16 *key, const __nv_bfloat16 *value,
                   __nv_bfloat16 *merged, unsigned char *encoded, float *scales,
                   const DecodeState *states, int cap, bool local, float *scratch,
                   const __nv_bfloat16 *oldSecondRow, int *bad, cudaStream_t stream,
                   int maximum = kW4A8FContextCapacity) {
  const int splits = gqa8_attention::MaxSplits(cap, local, maximum);
  auto *lse = scratch + splits * 32 * 128;
  if (local) {
    SplitRows<true><<<dim3(4 * splits, 1, 2), 128, 0, stream>>>(
        queries, key, value, scratch, lse, states, cap, splits, bad, oldSecondRow, maximum);
    MergeRows<true>
        <<<dim3(32, 2), 128, 0, stream>>>(scratch, lse, merged, states, cap, splits, bad, maximum);
  } else {
    SplitRows<false><<<dim3(4 * splits, 1, 2), 128, 0, stream>>>(
        queries, key, value, scratch, lse, states, cap, splits, bad, oldSecondRow, maximum);
    MergeRows<false>
        <<<dim3(32, 2), 128, 0, stream>>>(scratch, lse, merged, states, cap, splits, bad, maximum);
  }
  QuantizeRows<<<2, 512, 0, stream>>>(merged, encoded, scales, bad);
}
} // namespace mtp_attention
