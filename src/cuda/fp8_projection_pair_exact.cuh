// Same ordinary SIMT per-row FMA/reduction/scales;two independent rows share input decode.
#pragma once
namespace fp8_pair_exact {
__device__ __forceinline__ void Rows(DeviceMatrix m, const unsigned char *x, const float *scales,
                                     float *y, int r, int *bad) {
  if (r >= m.rows)
    return;
  const int lane = threadIdx.x % 32;
  float s0 = 0, s1 = 0;
  for (int c = lane * 4; c < m.cols; c += 128) {
    const auto a = DecodeFP8Four(Load32(x + c)),
               b0 = DecodeFP8Four(Load32(m.data + r * m.cols + c));
    const auto b1 = r + 1 < m.rows ? DecodeFP8Four(Load32(m.data + (r + 1) * m.cols + c))
                                   : make_float4(0, 0, 0, 0);
    s0 = fmaf(a.x, b0.x, s0);
    s1 = fmaf(a.x, b1.x, s1);
    s0 = fmaf(a.y, b0.y, s0);
    s1 = fmaf(a.y, b1.y, s1);
    s0 = fmaf(a.z, b0.z, s0);
    s1 = fmaf(a.z, b1.z, s1);
    s0 = fmaf(a.w, b0.w, s0);
    s1 = fmaf(a.w, b1.w, s1);
  }
  for (int d = 16; d; d /= 2) {
    s0 += __shfl_down_sync(0xffffffff, s0, d);
    s1 += __shfl_down_sync(0xffffffff, s1, d);
  }
  if (lane == 0) {
    const float v0 = B((s0 * scales[0]) * BF(m.scales, r));
    y[r] = v0;
    Finite(v0, bad);
    if (r + 1 < m.rows) {
      const float v1 = B((s1 * scales[0]) * BF(m.scales, r + 1));
      y[r + 1] = v1;
      Finite(v1, bad);
    }
  }
}
__global__ void Vector(DeviceMatrix m, const unsigned char *x, const float *scales, float *y,
                       int *bad) {
  if (!m.data || !m.scales || !x || !scales || !y || !bad || m.format != 1 || m.rows < 1 ||
      m.rows > 4096 || (m.cols != 2304 && m.cols != 4096) || blockDim.x != 128 || blockDim.y != 1 ||
      blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1 ||
      gridDim.x != unsigned((m.rows + 7) / 8) || (reinterpret_cast<std::uintptr_t>(m.data) & 3) ||
      (reinterpret_cast<std::uintptr_t>(x) & 3)) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  Rows(m, x, scales, y, blockIdx.x * 8 + (threadIdx.x / 32) * 2, bad);
}
__global__ void QKV(DeviceMatrix q, DeviceMatrix k, DeviceMatrix v, const unsigned char *x,
                    const float *scales, float *oq, float *ok, float *ov, int *bad) {
  if (!q.data || !k.data || !v.data || !q.scales || !k.scales || !v.scales || !x || !scales ||
      !oq || !ok || !ov || !bad || q.format != 1 || k.format != 1 || v.format != 1 ||
      q.rows != 4096 || k.rows != 512 || v.rows != 512 || q.cols != 2304 || k.cols != 2304 ||
      v.cols != 2304 || blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
      gridDim.x != 640 || gridDim.y != 1 || gridDim.z != 1) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  const int block = blockIdx.x;
  const auto m = block < 512 ? q : (block < 576 ? k : v);
  float *out = block < 512 ? oq : (block < 576 ? ok : ov);
  const int local = block < 512 ? block : (block < 576 ? block - 512 : block - 576);
  Rows(m, x, scales, out, local * 8 + (threadIdx.x / 32) * 2, bad);
}
} // namespace fp8_pair_exact
