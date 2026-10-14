// X1 single M32/N128/K32 eight-warp exact-FP16 Trellis prefill candidate.
// Gem16 grouped staging/H128 mechanisms; Mellum shapes and arithmetic.
#pragma once
#include "expert_exl3_codec.cuh"
namespace exl3_fp16 {
constexpr int M = 32, N = 128, K = 32, S = 40;
struct Tile {
  int expert, begin;
};
__host__ __device__ constexpr int Capacity(int tokens) {
  return tokens * 8 / 32;
}
__global__ void Schedule(const int *counts, Tile *tiles, int *tileCount, int tokens, int *bad) {
  if (threadIdx.x)
    return;
  int total = 0, sum = 0;
  for (int e = 0; e < 64; ++e) {
    const int c = counts[e];
    if (c < 0 || c > tokens) {
      atomicExch(bad, 1);
      *tileCount = 0;
      return;
    }
    sum += c;
    for (int j = 0; j < c / 32; ++j) {
      if (total >= Capacity(tokens)) {
        atomicExch(bad, 1);
        *tileCount = 0;
        return;
      }
      tiles[total++] = {e, j * 32};
    }
  }
  if (sum != tokens * 8) {
    atomicExch(bad, 1);
    total = 0;
  }
  *tileCount = total;
}
// Baseline BF16 boundary followed by an explicit FP16 execution conversion.
// Range counters: max abs, count, subnormal, underflow, overflow, inexact.
__global__ void Input(const DeviceMatrix *matrices, int layer, int part, const int *ids,
                      const float *input, __half *transformed, int pairs, int *bad,
                      unsigned *stats = nullptr) {
  const int pair = blockIdx.y, i = blockIdx.x * 128 + threadIdx.x;
  const auto m = exl3_expert::Matrix(matrices, layer, part, ids, pair, bad);
  __shared__ float x[128];
  const int t = threadIdx.x;
  x[t] = __fmul_rn(input[(part == 2 ? pair : pair / 8) * m.cols + i],
                   __half2float(__ushort_as_half(m.suh[i])));
  __syncthreads();
  for (int d = 1; d < 128; d *= 2) {
    const float a = x[t], b = x[t ^ d];
    __syncthreads();
    x[t] = (t & d) ? __fsub_rn(b, a) : __fadd_rn(a, b);
    __syncthreads();
  }
  const float value = B(__fmul_rn(x[t], 0.08838834764831845f));
  const __half h = __float2half_rn(value);
  const unsigned bits = __half_as_ushort(h);
  if (!isfinite(value) || (bits & 0x7c00) == 0x7c00)
    atomicExch(bad, 1); // No clipping or fallback.
  if (stats) {
    atomicMax(stats, __float_as_uint(fabsf(value)));
    atomicAdd(stats + 1, 1U);
    if ((bits & 0x7c00) == 0 && (bits & 1023))
      atomicAdd(stats + 2, 1U);
    if ((bits & 0x7fff) == 0 && value != 0)
      atomicAdd(stats + 3, 1U);
    if ((bits & 0x7c00) == 0x7c00)
      atomicAdd(stats + 4, 1U);
    if (__half2float(h) != value)
      atomicAdd(stats + 5, 1U);
  }
  transformed[pair * m.cols + i] = h;
}
__device__ __forceinline__ unsigned Pack(const __half *p) {
  return unsigned(__half_as_ushort(p[0])) | (unsigned(__half_as_ushort(p[1])) << 16);
}
__device__ __forceinline__ void Mma(const unsigned *a, const unsigned *b, float *d) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void Stage(__half *a, __half *b, const __half *input, const int *slots,
                                      DeviceMatrix matrix, int nBase, int kBase) {
  const int t = threadIdx.x, lane = t & 31, warp = t / 32;
  if (t < 128) {
    const int row = t / 4, col = (t % 4) * 8;
    const void *src = input + slots[row] * matrix.cols + kBase + col;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(
                     unsigned(__cvta_generic_to_shared(a + row * S + col))),
                 "l"(src)
                 : "memory");
  }
// Eight output tiles × two K16 tiles; each packed tile loaded once per warp.
#pragma unroll
  for (int halfK = 0; halfK < 2; ++halfK) {
    const auto *packed = reinterpret_cast<const unsigned *>(
        matrix.data +
        std::uint64_t(((kBase / 16) + halfK) * (matrix.rows / 16) + (nBase / 16) + warp) * 128);
    const unsigned word = packed[lane];
    const int out = lane / 2, in = (lane % 2) * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      b[(warp * 16 + out) * S + halfK * 16 + in + j] = DecodeK4(word, in + j, out);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__global__ void Projection(const DeviceMatrix *matrices, int layer, int part, const int *groups,
                           const Tile *tiles, const int *tileCount, const __half *input,
                           float *output, int tokens, int *bad) {
  if (int(blockIdx.y) >= *tileCount)
    return;
  const Tile tile = tiles[blockIdx.y];
  const auto matrix = matrices[(layer * 64 + tile.expert) * 3 + part];
  if (matrix.format != 3) {
    if (!threadIdx.x)
      atomicExch(bad, 1);
    return;
  }
  __shared__ __align__(16) __half ad[2][M * S], bd[2][N * S];
  __shared__ float projected[M][N];
  __shared__ int slots[M];
  if (threadIdx.x < M)
    slots[threadIdx.x] = groups[tile.expert * tokens + tile.begin + threadIdx.x];
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32, group = lane / 4, tid = lane % 4;
  const int nBase = blockIdx.x * 128;
  float accum[2][2][4] = {};
  Stage(ad[0], bd[0], input, slots, matrix, nBase, 0);
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  for (int kBase = 0; kBase < matrix.cols; kBase += 32) {
    const int stage = (kBase / 32) & 1;
    if (kBase + 32 < matrix.cols)
      Stage(ad[stage ^ 1], bd[stage ^ 1], input, slots, matrix, nBase, kBase + 32);
#pragma unroll
    for (int k = 0; k < 32; k += 16) {
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        unsigned a[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          a[j] = Pack(ad[stage] + (m * 16 + group + (j % 2) * 8) * S + k + tid * 2 + (j / 2) * 8);
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          unsigned b[2];
#pragma unroll
          for (int j = 0; j < 2; ++j)
            b[j] = Pack(bd[stage] + (warp * 16 + n * 8 + group) * S + k + tid * 2 + j * 8);
          Mma(a, b, accum[m][n]);
        }
      }
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
  }
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
      for (int j = 0; j < 4; ++j)
        projected[m * 16 + group + (j / 2) * 8][warp * 16 + n * 8 + tid * 2 + (j % 2)] =
            accum[m][n][j];
    }
  }
  __syncthreads();
  for (int row = warp; row < 32; row += 8) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      v[j] = projected[row][lane * 4 + j];
    H128(v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = nBase + lane * 4 + j;
      const float value = B(__fmul_rn(v[j], __half2float(__ushort_as_half(matrix.svh[col]))));
      if (!isfinite(value))
        atomicExch(bad, 1);
      output[slots[row] * matrix.rows + col] = value;
    }
  }
}
// Declared SIMT narrow tails use the frozen BF16 input, not FP16 conversion.
__global__ void TailDot(const DeviceMatrix *matrices, int layer, int part, const int *groups,
                        const int *counts, const int *ids, const float *input, float *raw,
                        int tokens, int *bad) {
  const int e = blockIdx.z, j = (counts[e] / 32) * 32 + blockIdx.y;
  if (j >= counts[e])
    return;
  const int pair = groups[e * tokens + j], row = blockIdx.x * 4 + threadIdx.x / 32,
            lane = threadIdx.x & 31;
  const auto m = exl3_expert::Matrix(matrices, layer, part, ids, pair, bad);
  float sum = 0, error = 0;
  for (int k = lane; k < m.cols; k += 32)
    Accumulate(__fmul_rn(input[pair * m.cols + k], exl3_expert::Weight(m, k, row)), sum, error);
  for (int d = 16; d; d /= 2) {
    const float a = __shfl_down_sync(0xffffffff, sum, d),
                b = __shfl_down_sync(0xffffffff, error, d);
    if (lane + d < 32) {
      Accumulate(a, sum, error);
      Accumulate(b, sum, error);
    }
  }
  if (!lane)
    raw[pair * m.rows + row] = __fadd_rn(sum, error);
}
__global__ void TailOutput(const DeviceMatrix *matrices, int layer, int part, const int *groups,
                           const int *counts, const float *raw, float *output, int tokens,
                           int *bad) {
  const int e = blockIdx.z, j = (counts[e] / 32) * 32 + blockIdx.y;
  if (j >= counts[e])
    return;
  const int pair = groups[e * tokens + j], col = blockIdx.x * 128 + threadIdx.x;
  const auto m = matrices[(layer * 64 + e) * 3 + part];
  __shared__ float v[128];
  const int t = threadIdx.x;
  v[t] = raw[pair * m.rows + col];
  __syncthreads();
  for (int d = 1; d < 128; d *= 2) {
    const float a = v[t], b = v[t ^ d];
    __syncthreads();
    v[t] = (t & d) ? __fsub_rn(b, a) : __fadd_rn(a, b);
    __syncthreads();
  }
  const float value = B(
      __fmul_rn(__fmul_rn(v[t], 0.08838834764831845f), __half2float(__ushort_as_half(m.svh[col]))));
  if (!isfinite(value))
    atomicExch(bad, 1);
  output[pair * m.rows + col] = value;
}
// Workspace: FP16 input + FP32 raw output. Tail input temporarily leases raw
// output plus the unused upper FP16 input extent; explicit phase lifetimes.
inline void Launch(const DeviceMatrix *matrices, int layer, int part, const int *ids,
                   const int *groups, const int *counts, const Tile *tiles, const int *tileCount,
                   const float *input, float *output, __half *transformed, float *raw,
                   float *tailInput, int tokens, int *bad, cudaStream_t stream,
                   unsigned *stats = nullptr) {
  const int cols = part == 2 ? 896 : 2304, rows = part == 2 ? 2304 : 896, pairs = tokens * 8;
  if (tokens < 4) {
    exl3_expert::Launch(matrices, layer, part, ids, input, output, tailInput,
                        tailInput + pairs * cols, tokens, bad, stream);
    return;
  } // Exact singleton/narrow dispatch, no TC padding.
  Input<<<dim3(cols / 128, pairs), 128, 0, stream>>>(matrices, layer, part, ids, input, transformed,
                                                     pairs, bad, stats);
  Projection<<<dim3(rows / 128, Capacity(tokens)), 256, 0, stream>>>(
      matrices, layer, part, groups, tiles, tileCount, transformed, output, tokens, bad);
  // FP16 transformed inputs are dead before their region is reused for tails.
  exl3_expert::InputHad<<<dim3(cols / 128, pairs), 128, 0, stream>>>(matrices, layer, part, ids,
                                                                     input, tailInput, pairs, bad);
  raw = tailInput + pairs * cols; // Recarve after TC inputs die; max(cols+rows)=3200.
  TailDot<<<dim3(rows / 4, 31, 64), 128, 0, stream>>>(matrices, layer, part, groups, counts, ids,
                                                      tailInput, raw, tokens, bad);
  TailOutput<<<dim3(rows / 128, 31, 64), 128, 0, stream>>>(matrices, layer, part, groups, counts,
                                                           raw, output, tokens, bad);
}
} // namespace exl3_fp16
