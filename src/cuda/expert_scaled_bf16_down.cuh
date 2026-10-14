// E2 matching J64/I64/K64 down, single weight operand, existing E4M3 codec.
// Same E1/Gem16 async lifetime/fragment mapping; down gathers original slots.
#pragma once
#include "expert_scaled_bf16_layer.cuh"
namespace mellum16::expert_scaled_bf16_layer {
struct alignas(16) RawDown {
  unsigned char a[J][K], w[I][K / 2], sf[I][K / 16];
};
static_assert(sizeof(RawDown) == 6400);
__device__ __forceinline__ void StageDown(RawDown &dst, Matrix w, const unsigned char *a,
                                          const int *slots, int rowBase, int base) {
  const int t = threadIdx.x, row = t / 4, chunk = t % 4;
  Copy<16>(&dst.a[row][chunk * 16], a + slots[row] * w.cols + base + chunk * 16);
  if (t < 128) {
    const int r = t / 2, c = t % 2;
    Copy<16>(&dst.w[r][c * 16], w.packed + (rowBase + r) * (w.cols / 2) + base / 2 + c * 16);
  }
  if (t < 64)
    Copy<4>(dst.sf[t], w.sf + (rowBase + t) * (w.cols / 16) + base / 16);
  Commit();
}
__device__ __forceinline__ void DecodeDown(const RawDown &raw, __nv_bfloat16 *a, __nv_bfloat16 *w,
                                           int *invalid) {
  for (int i = threadIdx.x; i < 64 * 16; i += Threads) {
    const int row = i / 16, c = (i % 16) * 4;
    const unsigned word = *reinterpret_cast<const unsigned *>(&raw.a[row][c]);
    if ((word & 127) == 127 || ((word >> 8) & 127) == 127 || ((word >> 16) & 127) == 127 ||
        ((word >> 24) & 127) == 127)
      atomicExch(invalid, 1);
    Activation4(word, a + row * Stride + c);
    const unsigned char sf = raw.sf[row][c / 16];
    if (sf >= 127)
      atomicExch(invalid, 1);
    Weight4(*reinterpret_cast<const unsigned short *>(&raw.w[row][c / 2]), sf,
            w + row * Stride + c);
  }
}
__global__ void Down(const Matrix *matrices, const int *groups, const Tile *tiles, int capacity,
                     const int *liveCount, const unsigned char *a8, const float *rowScales,
                     float *output, int tokens, int *bad) {
  if (!bad)
    return;
  if (!matrices || !groups || !tiles || !liveCount || !a8 || !rowScales || !output ||
      tokens <= 128 || tokens > 2048 || capacity < 1 || capacity > 256 || blockDim.x != Threads ||
      blockDim.y != 1 || blockDim.z != 1 || gridDim.x != 36 || gridDim.y != unsigned(capacity) ||
      gridDim.z != 1 || (reinterpret_cast<std::uintptr_t>(a8) & 15)) {
    if (threadIdx.x == 0)
      Error(bad);
    return;
  }
  const int live = *liveCount;
  if (live < 0 || live > capacity) {
    if (threadIdx.x == 0)
      Error(bad);
    return;
  }
  if (blockIdx.y >= unsigned(live))
    return;
  const Tile tile = tiles[blockIdx.y];
  if (tile.expert < 0 || tile.expert >= 64 || tile.count != 64 || tile.begin < 0 ||
      tile.begin % 64 || tile.begin > tokens - 64) {
    if (threadIdx.x == 0)
      Error(bad);
    return;
  }
  const Matrix w = matrices[tile.expert];
  if (!w.packed || !w.sf || w.rows != 2304 || w.cols != 896 || !isfinite(w.divisor) ||
      w.divisor <= 0 || (reinterpret_cast<std::uintptr_t>(w.packed) & 15) ||
      (reinterpret_cast<std::uintptr_t>(w.sf) & 3)) {
    if (threadIdx.x == 0)
      Error(bad);
    return;
  }
  __shared__ RawDown raw[2];
  __shared__ __align__(16) __nv_bfloat16 ad[64 * Stride], wd[64 * Stride];
  __shared__ int slots[64], invalid;
  __shared__ float scales[64];
  if (threadIdx.x == 0)
    invalid = 0;
  __syncthreads();
  if (threadIdx.x < 64) {
    const int j = threadIdx.x, slot = groups[tile.expert * tokens + tile.begin + j];
    slots[j] = slot;
    const bool valid = slot >= 0 && slot < tokens * 8;
    if (!valid)
      atomicExch(&invalid, 1);
    scales[j] = valid ? rowScales[slot] : 1.f;
    if (!isfinite(scales[j]) || scales[j] <= 0)
      atomicExch(&invalid, 1);
  }
  __syncthreads();
  if (threadIdx.x < 64)
    for (int j = 0; j < threadIdx.x; ++j)
      if (slots[j] == slots[threadIdx.x])
        atomicExch(&invalid, 1);
  __syncthreads();
  if (invalid) {
    if (threadIdx.x == 0)
      Error(bad);
    return;
  }
  const int rb = blockIdx.x * 64, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int group = lane / 4, tid = lane % 4, mb = (warp / 2) * 16, nb = (warp % 2) * 32;
  float acc[4][4] = {};
  StageDown(raw[0], w, a8, slots, rb, 0);
  Wait();
  __syncthreads();
  for (int base = 0, stage = 0; base < 896; base += 64, stage ^= 1) {
    if (base + 64 < 896)
      StageDown(raw[stage ^ 1], w, a8, slots, rb, base + 64);
    DecodeDown(raw[stage], ad, wd, &invalid);
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 64; k += 16) {
      unsigned a[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        a[j] = Pack(ad + (mb + group + (j % 2) * 8) * Stride + k + tid * 2 + (j / 2) * 8);
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        unsigned b[2];
#pragma unroll
        for (int j = 0; j < 2; ++j)
          b[j] = Pack(wd + (nb + n * 8 + group) * Stride + k + tid * 2 + j * 8);
        Mma(a, b, acc[n]);
      }
    }
    Wait();
    __syncthreads();
  }
  if (invalid) {
    if (threadIdx.x == 0)
      Error(bad);
    return;
  }
#pragma unroll
  for (int n = 0; n < 4; ++n) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int t = mb + group + (j / 2) * 8, r = rb + nb + n * 8 + tid * 2 + j % 2;
      const float v = B(__fdiv_rn(__fmul_rn(acc[n][j], scales[t]), w.divisor));
      if (!isfinite(v)) {
        Error(bad);
        continue;
      }
      output[slots[t] * 2304 + r] = v;
    }
  }
}
} // namespace mellum16::expert_scaled_bf16_layer
