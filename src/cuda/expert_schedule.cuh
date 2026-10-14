// Stable expert-major descriptors over existing Mellum slot ownership.
// Bounds: top8 unique per token; no per-layer host read/count/launch planning.
namespace fast_expert {
struct Tile {
  int expert, begin;
};
__host__ __device__ constexpr int DescriptorCapacity(int tokens) {
  return (tokens * 8 + 15) / 16 + 63 < 64 * ((tokens + 15) / 16) ? (tokens * 8 + 15) / 16 + 63
                                                                 : 64 * ((tokens + 15) / 16);
}
__global__ void Group(const int *ids, int *groups, int *counts, int tokens, int *bad) {
  if (!ids || !groups || !counts || !bad || tokens < 1 || tokens > kW4A8FMaxPrefillChunk ||
      blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 || gridDim.x != 64 ||
      gridDim.y != 1 || gridDim.z != 1) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  const int expert = blockIdx.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  __shared__ int total, warpCounts[4], offsets[4];
  if (threadIdx.x == 0)
    total = 0;
  __syncthreads();
  for (int base = 0; base < tokens * 8; base += 128) {
    const int slot = base + threadIdx.x;
    const int id = slot < tokens * 8 ? ids[slot] : -1;
    if (slot < tokens * 8) {
      if (id < 0 || id >= 64)
        atomicExch(bad, 1);
      if ((slot % 8) == 0) {
        unsigned long long seen = 0;
        for (int s = 0; s < 8; ++s) {
          const int e = ids[slot + s];
          if (e < 0 || e >= 64) {
            atomicExch(bad, 1);
            continue;
          }
          const auto bit = 1ull << e;
          if (seen & bit)
            atomicExch(bad, 1);
          seen |= bit;
        }
      }
    }
    const unsigned mask = __ballot_sync(0xffffffff, slot < tokens * 8 && id == expert);
    if (lane == 0)
      warpCounts[warp] = __popc(mask);
    __syncthreads();
    if (threadIdx.x == 0) {
      int n = total;
      for (int w = 0; w < 4; ++w) {
        offsets[w] = n;
        n += warpCounts[w];
      }
      total = n;
      if (n > tokens)
        atomicExch(bad, 1);
    }
    __syncthreads();
    if (slot < tokens * 8 && id == expert) {
      const int out = offsets[warp] + __popc(mask & ((1u << lane) - 1u));
      if (out < tokens)
        groups[expert * tokens + out] = slot;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    counts[expert] = min(total, tokens);
}
__global__ void Describe(const int *counts, Tile *tiles, int *tileCount, int tokens, int capacity,
                         int *bad) {
  if (!counts || !tiles || !tileCount || !bad || tokens < 1 || tokens > kW4A8FMaxPrefillChunk ||
      capacity != DescriptorCapacity(tokens) || blockDim.x != 64 || blockDim.y != 1 ||
      blockDim.z != 1 || gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  __shared__ int offsets[65], valid;
  if (threadIdx.x == 0) {
    valid = *bad == 0;
    int sum = 0, n = 0;
    for (int e = 0; e < 64; ++e) {
      const int c = counts[e];
      if (c < 0 || c > tokens)
        valid = 0;
      offsets[e] = n;
      if (c >= 0 && c <= tokens) {
        sum += c;
        n += (c + 15) / 16;
      }
    }
    offsets[64] = n;
    if (sum != tokens * 8 || n > capacity)
      valid = 0;
    *tileCount = valid ? n : 0;
    if (!valid)
      atomicExch(bad, 1);
  }
  __syncthreads();
  if (!valid)
    return;
  const int expert = threadIdx.x;
  for (int i = offsets[expert]; i < offsets[expert + 1]; ++i)
    tiles[i] = {expert, (i - offsets[expert]) * 16};
}
} // namespace fast_expert
