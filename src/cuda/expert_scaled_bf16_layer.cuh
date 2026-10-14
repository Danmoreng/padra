// E2 standalone hybrid layer; frozen E1 v3 mainloop, optional live descriptors.
// Packed pipeline/grouped indexing adapted from Gem16 c01789a (Apache-2.0).
// Vector conversions use locked CUTLASS db1c2889 (BSD-3-Clause).
// No inference/default wiring. See NOTICE and E0/E1 evidence.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cutlass/numeric_conversion.h>
#include <cstdint>

namespace mellum16::expert_scaled_bf16_layer {
constexpr int Threads = 256, J = 64, I = 64, K = 64, Stride = 72;
struct Matrix {
  const unsigned char *packed;
  const unsigned char *sf;
  float divisor;
  int rows, cols;
};
struct Tile {
  int expert, begin, count;
};
struct alignas(16) Raw {
  unsigned char a[J][K], g[I][K / 2], u[I][K / 2], gs[I][K / 16], us[I][K / 16];
};
static_assert(sizeof(Raw) == 8704);
__device__ __forceinline__ void Error(int *bad) {
  if (bad)
    atomicExch(bad, 1);
}
__device__ __forceinline__ float E4(unsigned char code) {
  __nv_fp8_e4m3 x;
  x.__x = code;
  return float(x);
}
__device__ __forceinline__ unsigned Pack(const __nv_bfloat16 *p) {
  return unsigned(__bfloat16_as_ushort(p[0])) | (unsigned(__bfloat16_as_ushort(p[1])) << 16);
}
__device__ __forceinline__ float B(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}
// Direct pinned converter, adjacent nibble order. The matrix divisor is excluded.
__device__ __forceinline__ void Weight4(unsigned word, unsigned char scale, __nv_bfloat16 *dst) {
  unsigned lo, hi;
  cutlass::detail::_e2m1_to_bf16_x4(word, lo, hi);
  const float s = E4(scale);
  const unsigned bits[4] = {lo & 65535, lo >> 16, hi & 65535, hi >> 16};
#pragma unroll
  for (int j = 0; j < 4; ++j)
    dst[j] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(__ushort_as_bfloat16(bits[j])), s));
}
__device__ __forceinline__ void Activation4(unsigned word, __nv_bfloat16 *dst) {
  cutlass::Array<cutlass::float_e4m3_t, 4> input;
  *reinterpret_cast<unsigned *>(&input) = word;
  const auto output =
      cutlass::NumericArrayConverter<cutlass::bfloat16_t, cutlass::float_e4m3_t, 4>::convert(input);
  const auto *p = reinterpret_cast<const unsigned short *>(&output);
#pragma unroll
  for (int j = 0; j < 4; ++j)
    dst[j] = __ushort_as_bfloat16(p[j]);
}
template <int Bytes> __device__ __forceinline__ void Copy(void *dst, const void *src) {
  static_assert(Bytes == 4 || Bytes == 16);
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], %2;" ::"r"(unsigned(__cvta_generic_to_shared(dst))),
      "l"(src), "n"(Bytes)
      : "memory");
}
__device__ __forceinline__ void Commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void Wait() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}
__device__ __forceinline__ void Mma(const unsigned *a, const unsigned *b, float *d) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// All input slots/metadata are checked collectively before any raw loads.
__device__ __forceinline__ void StagePackedPair(Raw &dst, Matrix g, Matrix u,
                                                const unsigned char *a, const int *slots, int count,
                                                int rowBase, int base) {
  const int t = threadIdx.x;
  // One vector copy per thread, gathered source reused for both projections.
  const int row = t / 4, chunk = t % 4;
  if (row < count)
    Copy<16>(&dst.a[row][chunk * 16], a + (slots[row] / 8) * g.cols + base + chunk * 16);
  else
    *reinterpret_cast<uint4 *>(&dst.a[row][chunk * 16]) = make_uint4(0, 0, 0, 0);
  if (t < 128) {
    const int r = t / 2, c = t % 2;
    Copy<16>(&dst.g[r][c * 16], g.packed + (rowBase + r) * (g.cols / 2) + base / 2 + c * 16);
    Copy<16>(&dst.u[r][c * 16], u.packed + (rowBase + r) * (u.cols / 2) + base / 2 + c * 16);
  }
  if (t < 64) {
    Copy<4>(dst.gs[t], g.sf + (rowBase + t) * (g.cols / 16) + base / 16);
    Copy<4>(dst.us[t], u.sf + (rowBase + t) * (u.cols / 16) + base / 16);
  }
  Commit();
}
__device__ __forceinline__ void Decode(const Raw &raw, __nv_bfloat16 *a, __nv_bfloat16 *g,
                                       __nv_bfloat16 *u, int *invalid) {
  for (int i = threadIdx.x; i < 64 * 16; i += Threads) {
    const int row = i / 16, c = (i % 16) * 4;
    const unsigned word = *reinterpret_cast<const unsigned *>(&raw.a[row][c]);
    if ((word & 127) == 127 || ((word >> 8) & 127) == 127 || ((word >> 16) & 127) == 127 ||
        ((word >> 24) & 127) == 127)
      atomicExch(invalid, 1);
    Activation4(word, a + row * Stride + c);
    const unsigned char gs = raw.gs[row][c / 16], us = raw.us[row][c / 16];
    if (gs >= 127 || us >= 127)
      atomicExch(invalid, 1);
    const unsigned gw = *reinterpret_cast<const unsigned short *>(&raw.g[row][c / 2]);
    const unsigned uw = *reinterpret_cast<const unsigned short *>(&raw.u[row][c / 2]);
    Weight4(gw, gs, g + row * Stride + c);
    Weight4(uw, us, u + row * Stride + c);
  }
}

// One fixed J64/I64/K64 geometry. No full expanded weights/global BF16 slab.
// Gate/up accumulators survive the whole K loop; all scale folding is per tile.
__global__ void GateUp(const Matrix *pairs, const int *groups, const Tile *tiles, int tileCount,
                       const unsigned char *a8, const float *rowScales, float *product, int tokens,
                       int rows, int cols, int *bad, float *gateCapture = nullptr,
                       float *upCapture = nullptr, const int *liveCount = nullptr) {
  if (!bad)
    return;
  if (!pairs || !groups || !tiles || !a8 || !rowScales || !product || tokens < 1 || tokens > 2048 ||
      rows < 64 || rows > 896 || rows % 64 || cols < 64 || cols > 2304 || cols % 64 ||
      tileCount < 1 || tileCount > 256 || blockDim.x != Threads || blockDim.y != 1 ||
      blockDim.z != 1 || gridDim.x != unsigned(rows / 64) || gridDim.y != unsigned(tileCount) ||
      gridDim.z != 1 || (reinterpret_cast<std::uintptr_t>(a8) & 15)) {
    if (threadIdx.x == 0)
      Error(bad);
    return;
  }
  const int live = liveCount ? *liveCount : tileCount;
  if (live < 0 || live > tileCount) {
    if (threadIdx.x == 0)
      Error(bad);
    return;
  }
  if (blockIdx.y >= unsigned(live))
    return;
  const Tile tile = tiles[blockIdx.y];
  if (tile.expert < 0 || tile.expert >= 64 || tile.begin < 0 || tile.count < 1 || tile.count > 64 ||
      tile.begin > tokens - tile.count) {
    if (threadIdx.x == 0)
      Error(bad);
    return;
  }
  const Matrix g = pairs[tile.expert * 2], u = pairs[tile.expert * 2 + 1];
  if (!g.packed || !g.sf || !u.packed || !u.sf || g.rows != rows || u.rows != rows ||
      g.cols != cols || u.cols != cols || !isfinite(g.divisor) || !isfinite(u.divisor) ||
      g.divisor <= 0 || u.divisor <= 0 || (reinterpret_cast<std::uintptr_t>(g.packed) & 15) ||
      (reinterpret_cast<std::uintptr_t>(u.packed) & 15) ||
      (reinterpret_cast<std::uintptr_t>(g.sf) & 3) ||
      (reinterpret_cast<std::uintptr_t>(u.sf) & 3)) {
    if (threadIdx.x == 0)
      Error(bad);
    return;
  }
  __shared__ Raw raw[2];
  __shared__ __align__(16) __nv_bfloat16 ad[64 * Stride], gd[64 * Stride], ud[64 * Stride];
  __shared__ int slots[64], invalid;
  __shared__ float scales[64];
  if (threadIdx.x == 0)
    invalid = 0;
  __syncthreads();
  if (threadIdx.x < 64) {
    const int j = threadIdx.x;
    const int slot = j < tile.count ? groups[tile.expert * tokens + tile.begin + j] : -1;
    slots[j] = slot;
    const bool valid = j >= tile.count || (slot >= 0 && slot < tokens * 8);
    if (!valid)
      atomicExch(&invalid, 1);
    const float sf = j < tile.count && valid ? rowScales[slot / 8] : 1.f;
    scales[j] = sf;
    if (!isfinite(sf) || sf <= 0)
      atomicExch(&invalid, 1);
  }
  __syncthreads();
  if (threadIdx.x < 64 && threadIdx.x < tile.count) {
    for (int j = 0; j < threadIdx.x; ++j)
      if (slots[j] == slots[threadIdx.x])
        atomicExch(&invalid, 1);
  }
  __syncthreads();
  if (invalid) {
    if (threadIdx.x == 0)
      Error(bad);
    return;
  }
  const int rowBase = blockIdx.x * 64, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int group = lane / 4, tid = lane % 4, mBase = (warp / 2) * 16, nBase = (warp % 2) * 32;
  float acc[4][4] = {}, up[4][4] = {};
  StagePackedPair(raw[0], g, u, a8, slots, tile.count, rowBase, 0);
  Wait();
  __syncthreads();
  for (int base = 0, stage = 0; base < cols; base += 64, stage ^= 1) {
    // Copy next packed stage before the current reconstruction/contraction.
    // Only packed operands are double buffered; decoded shared storage is reused.
    if (base + 64 < cols)
      StagePackedPair(raw[stage ^ 1], g, u, a8, slots, tile.count, rowBase, base + 64);
    Decode(raw[stage], ad, gd, ud, &invalid);
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 64; k += 16) {
      unsigned a[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        a[j] = Pack(ad + (mBase + group + (j % 2) * 8) * Stride + k + tid * 2 + (j / 2) * 8);
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        unsigned b[2], ub[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int at = (nBase + n * 8 + group) * Stride + k + tid * 2 + j * 8;
          b[j] = Pack(gd + at);
          ub[j] = Pack(ud + at);
        }
        Mma(a, b, acc[n]);
        Mma(a, ub, up[n]);
      }
    }
    // All decoded reads complete before reuse; all raw writes complete before
    // consume or recycling the ping-pong stage, including the final iteration.
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
      const int t = mBase + group + (j / 2) * 8;
      if (t >= tile.count)
        continue;
      const int slot = slots[t], r = rowBase + nBase + n * 8 + tid * 2 + j % 2;
      const float gate = B(__fdiv_rn(__fmul_rn(acc[n][j], scales[t]), g.divisor));
      const float other = B(__fdiv_rn(__fmul_rn(up[n][j], scales[t]), u.divisor));
      const float value = B(__fmul_rn(B(__fdiv_rn(gate, 1.f + expf(-gate))), other));
      if (!isfinite(gate) || !isfinite(other) || !isfinite(value)) {
        Error(bad);
        continue;
      }
      if (gateCapture)
        gateCapture[slot * rows + r] = gate;
      if (upCapture)
        upCapture[slot * rows + r] = other;
      product[slot * rows + r] = value;
    }
  }
}
} // namespace mellum16::expert_scaled_bf16_layer
