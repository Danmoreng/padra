// Exact N4a merge/encoded bytes;only final E4 encoder uses proved boundary correction.
#pragma once
#include "fp8_exact_reciprocal.cuh"
// N4a row ownership/staging follows Gem16 c01789a2 fp8/reference.cu
// (Apache-2.0); Mellum N3a merge arithmetic and direct E4M3 division retained.
// A single CTA owns all32 heads: amax is over the complete4096-element row.
namespace attention_o_exact {
constexpr int kThreads = 256;
template <bool Local>
__launch_bounds__(kThreads, 1) __global__
    void MergeQuantize(const float *partial, const float *lse, float *out, unsigned char *encoded,
                       float *scales, const DecodeState *state, int cap, int max_splits, int *bad) {
  if (!gqa8_attention::Admitted<Local>(state, cap, max_splits) || !partial || !lse || !out ||
      !encoded || !scales || !bad || blockDim.x != kThreads || blockDim.y != 1 || blockDim.z != 1 ||
      gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  const int n = Local ? min(state->position + 1, 1024) : state->position + 1;
  const int active = min(max_splits, (n + 31) / 32), lane = threadIdx.x % 32,
            warp = threadIdx.x / 32;
  __shared__ float weights[32 * 32], maxima[8], scale, inverse;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int head = warp * 4 + r;
    const float v = lane < active ? lse[lane * 32 + head] : -INFINITY;
    const float mx = gqa8_attention::WarpMax(v), w = lane < active ? expf(v - mx) : 0.f;
    const float den = gqa8_attention::WarpSum(w);
    if (lane < active) {
      const float value = w / den;
      weights[head * 32 + lane] = value;
      Finite(value, bad);
    }
  }
  __syncthreads();
  float row[16], amax = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int index = threadIdx.x + i * kThreads, head = index / 128, d = index % 128;
    float sum = 0.f;
    for (int s = 0; s < active; ++s)
      sum = fmaf(weights[head * 32 + s], partial[(s * 32 + head) * 128 + d], sum);
    const float value = B(sum);
    row[i] = value;
    out[index] = value;
    Finite(value, bad);
    amax = fmaxf(amax, fabsf(value));
  }
  amax = gqa8_attention::WarpMax(amax);
  if (lane == 0)
    maxima[warp] = amax;
  __syncthreads();
  if (warp == 0) {
    const float mx = gqa8_attention::WarpMax(lane < 8 ? maxima[lane] : 0.f);
    if (lane == 0) {
      scale = mx == 0.f ? 1.f : __fdiv_rn(mx, 448.f);
      scales[0] = scale;
      inverse = __fdiv_rn(1.f, scale);
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 16; ++i)
    encoded[threadIdx.x + i * kThreads] = fp8_exact_reciprocal::Encode(row[i], scale, inverse);
}
inline void Launch(const float *q, const __nv_bfloat16 *kc, const __nv_bfloat16 *vc, float *out,
                   unsigned char *encoded, float *scales, const DecodeState *state, int cap,
                   bool local, float *scratch, int *bad, cudaStream_t stream,
                   bool fragments = false) {
  const int splits = gqa8_attention::MaxSplits(cap, local);
  auto *lse = scratch + splits * 32 * 128;
  if (local) {
    if (fragments)
      gqa8_fragment_validation::SplitBF16<true>
          <<<4 * splits, 128, 0, stream>>>(q, kc, vc, scratch, lse, state, cap, splits, bad);
    else
      gqa8_attention::SplitBF16<true>
          <<<4 * splits, 128, 0, stream>>>(q, kc, vc, scratch, lse, state, cap, splits, bad);
    MergeQuantize<true>
        <<<1, kThreads, 0, stream>>>(scratch, lse, out, encoded, scales, state, cap, splits, bad);
  } else {
    if (fragments)
      gqa8_fragment_validation::SplitBF16<false>
          <<<4 * splits, 128, 0, stream>>>(q, kc, vc, scratch, lse, state, cap, splits, bad);
    else
      gqa8_attention::SplitBF16<false>
          <<<4 * splits, 128, 0, stream>>>(q, kc, vc, scratch, lse, state, cap, splits, bad);
    MergeQuantize<false>
        <<<1, kThreads, 0, stream>>>(scratch, lse, out, encoded, scales, state, cap, splits, bad);
  }
}
} // namespace attention_o_exact
