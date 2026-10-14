// Owner10K: exact BF16 gate/up/down storage and SiLU+downH128 input consumer.
// Same nativeFP16 weights,MMA/H128/RN/BF16 points,no activation precision change.
#pragma once
#include "expert_exl3_prefill_warp.cuh"
#include "exl3_silu_lut.cuh"
#include "exl3_input_wide_vector.cuh"
#include "expert_exl3_dot_codec.cuh"
#include "runtime/exl3_masked_schedule.h"
namespace exl3_packed_activations {
using Tile = mellum16::exl3_masked_schedule::Tile;
using exl3_fp16::DecodeK4;
using exl3_fp16::H128;
using mellum16::exl3_masked_schedule::Capacity;
constexpr int M = 32, N = 128, K = 32, S = 40;
using exl3_masked::Schedule;
__device__ __forceinline__ float Act(unsigned short gate, unsigned short up,
                                     const unsigned short *table, int *bad) {
  const float g = __uint_as_float(unsigned(gate) << 16), u = __uint_as_float(unsigned(up) << 16);
  if (!table || !isfinite(g) || !isfinite(u)) {
    atomicExch(bad, 1);
    return 0.f;
  }
  const float a = B(__fmul_rn(__uint_as_float(unsigned(table[gate]) << 16), u));
  if (!isfinite(a))
    atomicExch(bad, 1);
  return a;
}
__global__ void InputActivated(const DeviceMatrix *matrices, int layer, const int *ids,
                               const unsigned short *gate, const unsigned short *up,
                               const unsigned short *table, __half *transformed, int pairs,
                               int *bad, unsigned *stats = nullptr) {
  const int lane = threadIdx.x & 31, globalWarp = blockIdx.x * 4 + threadIdx.x / 32,
            blocks = 896 / 128, pair = globalWarp / blocks, h = globalWarp % blocks;
  if (pair >= pairs)
    return;
  const auto m = exl3_expert::Matrix(matrices, layer, 2, ids, pair, bad);
  if (m.format != 3 || m.cols != 896 || blockDim.x != 128) {
    if (!lane)
      atomicExch(bad, 1);
    return;
  }
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = h * 128 + lane * 4 + j;
    const float a =
        Act(gate[std::uint64_t(pair) * 896 + i], up[std::uint64_t(pair) * 896 + i], table, bad);
    v[j] = __fmul_rn(a, __half2float(__ushort_as_half(m.suh[i])));
  }
  H128(v);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float value = B(v[j]);
    const auto a = __float2half_rn(value);
    const unsigned bits = __half_as_ushort(a);
    if (!isfinite(value) || (bits & 0x7c00) == 0x7c00)
      atomicExch(bad, 1);
    transformed[std::uint64_t(pair) * 896 + h * 128 + lane * 4 + j] = a;
    if (stats) {
      atomicMax(stats, __float_as_uint(fabsf(value)));
      atomicAdd(stats + 1, 1U);
      if ((bits & 0x7c00) == 0 && (bits & 1023))
        atomicAdd(stats + 2, 1U);
      if ((bits & 0x7fff) == 0 && value != 0)
        atomicAdd(stats + 3, 1U);
      if ((bits & 0x7c00) == 0x7c00)
        atomicAdd(stats + 4, 1U);
      if (__half2float(a) != value)
        atomicAdd(stats + 5, 1U);
    }
  }
}
// Diagnostic adapters only;not on the native/timed forward path.
__global__ void ActivateToFloat(const unsigned short *gate, const unsigned short *up,
                                const unsigned short *table, float *result, int count, int *bad) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i < count)
    result[i] = Act(gate[i], up[i], table, bad);
}
__global__ void ExpandToFloat(const unsigned short *input, float *result, int count) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i < count)
    result[i] = __uint_as_float(unsigned(input[i]) << 16);
}
__global__ void Combine(float *x, const unsigned short *down, const float *weights, int tokens,
                        int *bad, bool residual) {
  const int token = blockIdx.y, col = blockIdx.x * 128 + threadIdx.x;
  if (token >= tokens || col >= 2304)
    return;
  float sum = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float d =
        __uint_as_float(unsigned(down[(std::uint64_t(token) * 8 + j) * 2304 + col]) << 16);
    sum = B(__fadd_rn(sum, B(__fmul_rn(d, weights[token * 8 + j]))));
  }
  const int i = token * 2304 + col;
  const float value = residual ? B(__fadd_rn(x[i], sum)) : sum;
  if (!isfinite(value))
    atomicExch(bad, 1);
  x[i] = value;
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
__device__ __forceinline__ void Stage(__half *a, const __half *input, const int *slots,
                                      DeviceMatrix matrix, int nBase, int kBase) {
  const int t = threadIdx.x;
  if (t < 128) {
    const int row = t / 4, col = (t % 4) * 8;
    const bool valid = slots[row] >= 0;
    const void *src = valid ? input + std::uint64_t(slots[row]) * matrix.cols + kBase + col : input;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(
                     unsigned(__cvta_generic_to_shared(a + row * S + col))),
                 "l"(src), "r"(valid ? 16 : 0)
                 : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__global__ void Projection(const DeviceMatrix *matrices, int layer, int part, const int *groups,
                           const Tile *tiles, const int *tileCount, const __half *input,
                           unsigned short *output, int tokens, int *bad) {
  if (*bad || int(blockIdx.y) >= *tileCount)
    return;
  const Tile tile = tiles[blockIdx.y];
  if (tile.expert < 0 || tile.expert >= 64 || tile.valid_rows < 1 || tile.valid_rows > 32 ||
      tile.begin < 0 || tile.begin > tokens - tile.valid_rows) {
    if (!threadIdx.x)
      atomicExch(bad, 1);
    return;
  }
  const auto matrix = matrices[(layer * 64 + tile.expert) * 3 + part];
  if (matrix.format != 3) {
    if (!threadIdx.x)
      atomicExch(bad, 1);
    return;
  }
  __shared__ __align__(16) __half ad[2][M * S];
  __shared__ float projected[M][N];
  __shared__ int slots[M];
  __shared__ int invalid;
  if (!threadIdx.x)
    invalid = 0;
  __syncthreads();
  if (threadIdx.x < M) {
    const int row = threadIdx.x;
    const int slot = row < tile.valid_rows ? groups[tile.expert * tokens + tile.begin + row] : -1;
    slots[row] = slot;
    if (row < tile.valid_rows && (slot < 0 || slot >= tokens * 8))
      atomicExch(&invalid, 1);
  }
  __syncthreads();
  if (invalid) {
    if (!threadIdx.x)
      atomicExch(bad, 1);
    return;
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32, group = lane / 4, tid = lane % 4;
  const int nBase = blockIdx.x * 128;
  float accum[2][2][4] = {};
  Stage(ad[0], input, slots, matrix, nBase, 0);
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  for (int kBase = 0; kBase < matrix.cols; kBase += 32) {
    const int stage = (kBase / 32) & 1;
    if (kBase + 32 < matrix.cols)
      Stage(ad[stage ^ 1], input, slots, matrix, nBase, kBase + 32);
#pragma unroll
    for (int k = 0; k < 32; k += 16) {
      const std::uint64_t tileIndex =
          std::uint64_t((kBase + k) / 16) * std::uint64_t(matrix.rows / 16) + nBase / 16 + warp;
      const unsigned word = reinterpret_cast<const unsigned *>(matrix.data + tileIndex * 128)[lane];
      unsigned fragment[2][2];
#pragma unroll
      for (int n = 0; n < 2; ++n) {
#pragma unroll
        for (int j = 0; j < 2; ++j)
          fragment[n][j] =
              exl3_dot_codec::Bits(exl3_dot_codec::Pair(word, tid * 2 + j * 8, group + n * 8));
      }
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        unsigned a[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          a[j] = Pack(ad[stage] + (m * 16 + group + (j % 2) * 8) * S + k + tid * 2 + (j / 2) * 8);
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          Mma(a, fragment[n], accum[m][n]);
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
    unsigned short packed[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = nBase + lane * 4 + j;
      const float value = B(__fmul_rn(v[j], __half2float(__ushort_as_half(matrix.svh[col]))));
      if (row < tile.valid_rows && !isfinite(value))
        atomicExch(bad, 1);
      packed[j] = static_cast<unsigned short>(__float_as_uint(value) >> 16);
    }
    if (row < tile.valid_rows) {
      const uint2 store = {unsigned(packed[0]) | (unsigned(packed[1]) << 16),
                           unsigned(packed[2]) | (unsigned(packed[3]) << 16)};
      *reinterpret_cast<uint2 *>(output + std::uint64_t(slots[row]) * matrix.rows + nBase +
                                 lane * 4) = store;
    }
  }
}
inline void Launch(const DeviceMatrix *matrices, int layer, int part, const int *ids,
                   const int *groups, const int *counts, const Tile *tiles, const int *tileCount,
                   const float *input, unsigned short *output, const unsigned short *gate,
                   const unsigned short *up, const unsigned short *table, __half *transformed,
                   int tokens, int *bad, cudaStream_t stream, unsigned *stats = nullptr,
                   bool wideInput = false) {
  const int cols = part == 2 ? 896 : 2304, rows = part == 2 ? 2304 : 896, pairs = tokens * 8;
  if (tokens < 4 || part < 0 || part > 2)
    throw std::runtime_error("packed BF16 prefill part requires declared T>=4");
  if (part == 2)
    InputActivated<<<(pairs * 7 + 3) / 4, 128, 0, stream>>>(matrices, layer, ids, gate, up, table,
                                                            transformed, pairs, bad, stats);
  else if (wideInput)
    exl3_input_wide_vector::Input<<<(pairs * (cols / 128) + 7) / 8, 256, 0, stream>>>(
        matrices, layer, part, ids, input, transformed, pairs, bad, stats);
  else
    exl3_warp_input::Input<<<(pairs * (cols / 128) + 3) / 4, 128, 0, stream>>>(
        matrices, layer, part, ids, input, transformed, pairs, bad, stats);
  Projection<<<dim3(rows / 128, Capacity(tokens)), 256, 0, stream>>>(
      matrices, layer, part, groups, tiles, tileCount, transformed, output, tokens, bad);
}
} // namespace exl3_packed_activations
