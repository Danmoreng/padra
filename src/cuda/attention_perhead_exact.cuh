#pragma once
#include "fp8_exact_reciprocal.cuh"
// Decode only: per-head BF16 merge followed by exact row FP8 encoding.
// Attention split arithmetic, encoded values, scale and O projection are unchanged.
namespace attention_perhead_exact {
template <bool Local>
__global__ void Merge(const float *partial, const float *lse, __nv_bfloat16 *out,
                      const DecodeState *state, int cap, int splits, int *bad,
                      int maximum = kW4A8FContextCapacity) {
  if (!gqa8_attention::Admitted<Local>(state, cap, splits, maximum) || !partial || !lse || !out ||
      !bad || blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 || gridDim.x != 32 ||
      gridDim.y != 1 || gridDim.z != 1) {
    if (!threadIdx.x && bad)
      atomicExch(bad, 1);
    return;
  }
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
// Lossless alternative: retain the parent's FP8 activation bytes after the
// per-head BF16 merge. Max reduction is exact for admitted finite inputs.
__global__ void Quantize(const __nv_bfloat16 *x, unsigned char *encoded, float *scales, int *bad) {
  if (!x || !encoded || !scales || !bad || blockDim.x != 512 || blockDim.y != 1 ||
      blockDim.z != 1 || gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1) {
    if (!threadIdx.x && bad)
      atomicExch(bad, 1);
    return;
  }
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
inline void Launch(const float *q, const __nv_bfloat16 *kc, const __nv_bfloat16 *vc,
                   __nv_bfloat16 *out, const DecodeState *state, int cap, bool local,
                   float *scratch, int *bad, cudaStream_t stream,
                   int maximum = kW4A8FContextCapacity) {
  const int splits = gqa8_attention::MaxSplits(cap, local, maximum);
  auto *lse = scratch + splits * 32 * 128;
  if (local) {
    gqa8_fragment_validation::SplitBF16<true>
        <<<4 * splits, 128, 0, stream>>>(q, kc, vc, scratch, lse, state, cap, splits, bad, maximum);
    Merge<true><<<32, 128, 0, stream>>>(scratch, lse, out, state, cap, splits, bad, maximum);
  } else {
    gqa8_fragment_validation::SplitBF16<false>
        <<<4 * splits, 128, 0, stream>>>(q, kc, vc, scratch, lse, state, cap, splits, bad, maximum);
    Merge<false><<<32, 128, 0, stream>>>(scratch, lse, out, state, cap, splits, bad, maximum);
  }
}
} // namespace attention_perhead_exact
