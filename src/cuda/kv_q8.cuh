// New Mellum-specific integer KV codec, no checkpoint quantization/calibration.
// Scale layout/commit mechanics follow existing kv_fp8.cuh only; no E4M3 math.
#pragma once
namespace kv_q8 {
constexpr int Heads = 4, Dim = 128, Features = 512, Threads = 128;
using Scales = mellum16::Q8KVScalePair;
__device__ __forceinline__ float WarpMax(float v) {
  for (int d = 16; d; d /= 2)
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, d));
  return v;
}
__device__ __forceinline__ signed char Code(float value, float scale) {
  return static_cast<signed char>(max(-127, min(127, __float2int_rn(__fdiv_rn(value, scale)))));
}
__device__ __forceinline__ __nv_bfloat16 Decode(signed char code, float scale, int *bad) {
  if (code == -128 || !isfinite(scale) || scale <= 0) {
    atomicExch(bad, 1);
    return __float2bfloat16_rn(NAN);
  }
  const float product = __fmul_rn(static_cast<float>(code), scale);
  const auto value = __float2bfloat16_rn(product);
  if (!isfinite(product) || (__bfloat16_as_ushort(value) & 0x7f80u) == 0x7f80u) {
    atomicExch(bad, 1);
    return __float2bfloat16_rn(NAN);
  }
  return value;
}
template <bool Controlled = false>
__global__ void Encode(const float *k, const float *v, signed char *codesK, signed char *codesV,
                       Scales *scales, int tokens, int *bad, const DecodeState *state = nullptr,
                       int cap = 0, int maximum = 8192) {
  if (!k || !v || !codesK || !codesV || !scales || !bad || tokens < 1 || tokens > 2048 ||
      blockDim.x != Threads || gridDim.x != static_cast<unsigned>(tokens) || blockDim.y != 1 ||
      blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1 ||
      (Controlled && (tokens != 1 || !DecodeReadyAtLimit(state, maximum) ||
                      (cap != state->capacity && cap != min(1024, state->capacity))))) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  const int row = blockIdx.x, target = Controlled ? state->position % cap : row,
            head = threadIdx.x / 32, lane = threadIdx.x % 32, feature = head * Dim + lane;
  float a[4], b[4], km = 0, vm = 0;
  bool invalid = false;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    a[i] = k[row * Features + feature + i * 32];
    b[i] = v[row * Features + feature + i * 32];
    invalid |= !isfinite(a[i]) || !isfinite(b[i]) || (__float_as_uint(a[i]) & 0xffffu) != 0 ||
               (__float_as_uint(b[i]) & 0xffffu) != 0;
    km = fmaxf(km, fabsf(a[i]));
    vm = fmaxf(vm, fabsf(b[i]));
  }
  if (__any_sync(0xffffffff, invalid)) {
    if (lane == 0)
      atomicExch(bad, 1);
    return;
  }
  km = WarpMax(km);
  vm = WarpMax(vm);
  const float ks = km == 0 ? 1 : __fdiv_rn(km, 127.f), vs = vm == 0 ? 1 : __fdiv_rn(vm, 127.f);
  if (!isfinite(ks) || !isfinite(vs) || ks <= 0 || vs <= 0) {
    if (lane == 0)
      atomicExch(bad, 1);
    return;
  }
  if (lane == 0)
    scales[target * Heads + head] = {ks, vs};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    codesK[target * Features + feature + i * 32] = Code(a[i], ks);
    codesV[target * Features + feature + i * 32] = Code(b[i], vs);
  }
}
__global__ void Commit(const signed char *k, const signed char *v, const Scales *scales,
                       signed char *kc, signed char *vc, Scales *cached, const PrefillState *state,
                       int tokens, int cap, int first, int *bad, int maximum = 8192) {
  if (!state || !k || !v || !scales || !kc || !vc || !cached || !bad ||
      !ValidPrefillState(*state, tokens, state->capacity, 1, maximum) || cap < 1 ||
      (cap != state->capacity && cap != min(1024, state->capacity)) ||
      first != (cap == state->capacity ? 0 : max(0, tokens - cap)) || first < 0 ||
      first >= tokens || tokens - first > cap || blockDim.x != Threads || blockDim.y != 1 ||
      blockDim.z != 1 || gridDim.x != static_cast<unsigned>(tokens - first) || gridDim.y != 1 ||
      gridDim.z != 1) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  if (*bad)
    return;
  const int row = first + blockIdx.x, target = (state->base + row) % cap, d = threadIdx.x * 4;
  *reinterpret_cast<unsigned *>(kc + target * Features + d) =
      *reinterpret_cast<const unsigned *>(k + row * Features + d);
  *reinterpret_cast<unsigned *>(vc + target * Features + d) =
      *reinterpret_cast<const unsigned *>(v + row * Features + d);
  if (threadIdx.x < Heads)
    cached[target * Heads + threadIdx.x] = scales[row * Heads + threadIdx.x];
}
} // namespace kv_q8
