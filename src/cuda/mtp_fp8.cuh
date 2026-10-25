// Packed verifier rows inspired by Gem16 f6de38e (Apache-2.0, see NOTICE).
// Keep Padra's ordinary SIMT FMA sequence and warp reduction for every row;
// switching to Gemma's FP8 MMA sequence would change the target arithmetic.
#pragma once
namespace mtp_fp8 {
template <int Rows>
__global__ void QKV(DeviceMatrix q, DeviceMatrix k, DeviceMatrix v, const unsigned char *inputs,
                    const float *scales, float *oq, float *ok, float *ov, int *bad) {
  const int block = blockIdx.x, lane = threadIdx.x & 31;
  const auto m = block < 1024 ? q : block < 1152 ? k : v;
  float *output = block < 1024 ? oq : block < 1152 ? ok : ov;
  const int local = block < 1024 ? block : block < 1152 ? block - 1024 : block - 1152;
  const int row = local * 4 + threadIdx.x / 32;
  float sum[Rows] = {};
  for (int c = lane * 4; c < 2304; c += 128) {
    const auto w = DecodeFP8Four(Load32(m.data + row * 2304 + c));
#pragma unroll
    for (int t = 0; t < Rows; ++t) {
      const auto a = DecodeFP8Four(Load32(inputs + t * 2304 + c));
      sum[t] = fmaf(a.x, w.x, sum[t]);
      sum[t] = fmaf(a.y, w.y, sum[t]);
      sum[t] = fmaf(a.z, w.z, sum[t]);
      sum[t] = fmaf(a.w, w.w, sum[t]);
    }
  }
#pragma unroll
  for (int t = 0; t < Rows; ++t) {
    for (int d = 16; d; d /= 2)
      sum[t] += __shfl_down_sync(0xffffffff, sum[t], d);
    if (lane == 0) {
      const float value = B((sum[t] * scales[t]) * BF(m.scales, row));
      output[t * m.rows + row] = value;
      Finite(value, bad);
    }
  }
}
inline void Launch(int rows, DeviceMatrix q, DeviceMatrix k, DeviceMatrix v,
                   const unsigned char *input, const float *scales, float *oq, float *ok, float *ov,
                   int *bad, cudaStream_t s) {
  switch (rows) {
  case 2:
    QKV<2><<<1280, 128, 0, s>>>(q, k, v, input, scales, oq, ok, ov, bad);
    break;
  case 3:
    QKV<3><<<1280, 128, 0, s>>>(q, k, v, input, scales, oq, ok, ov, bad);
    break;
  case 4:
    QKV<4><<<1280, 128, 0, s>>>(q, k, v, input, scales, oq, ok, ov, bad);
    break;
  case 5:
    QKV<5><<<1280, 128, 0, s>>>(q, k, v, input, scales, oq, ok, ov, bad);
    break;
  default:
    throw std::runtime_error("MTP QKV row count unsupported");
  }
}
template <int Rows>
__global__ void Output(DeviceMatrix m, const unsigned char *input, const float *scales,
                       float *output, int *bad) {
  const int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x & 31;
  float sum[Rows] = {};
  for (int c = lane * 4; c < 4096; c += 128) {
    const auto w = DecodeFP8Four(Load32(m.data + row * 4096 + c));
#pragma unroll
    for (int t = 0; t < Rows; ++t) {
      const auto a = DecodeFP8Four(Load32(input + t * 4096 + c));
      sum[t] = fmaf(a.x, w.x, sum[t]);
      sum[t] = fmaf(a.y, w.y, sum[t]);
      sum[t] = fmaf(a.z, w.z, sum[t]);
      sum[t] = fmaf(a.w, w.w, sum[t]);
    }
  }
#pragma unroll
  for (int t = 0; t < Rows; ++t) {
    for (int d = 16; d; d /= 2)
      sum[t] += __shfl_down_sync(0xffffffff, sum[t], d);
    if (lane == 0) {
      const float value = B((sum[t] * scales[t]) * BF(m.scales, row));
      output[t * 2304 + row] = value;
      Finite(value, bad);
    }
  }
}
inline void LaunchOutput(int rows, DeviceMatrix m, const unsigned char *input, const float *scales,
                         float *output, int *bad, cudaStream_t s) {
  switch (rows) {
  case 2:
    Output<2><<<576, 128, 0, s>>>(m, input, scales, output, bad);
    break;
  case 3:
    Output<3><<<576, 128, 0, s>>>(m, input, scales, output, bad);
    break;
  case 4:
    Output<4><<<576, 128, 0, s>>>(m, input, scales, output, bad);
    break;
  case 5:
    Output<5><<<576, 128, 0, s>>>(m, input, scales, output, bad);
    break;
  default:
    throw std::runtime_error("MTP output projection row count unsupported");
  }
}
} // namespace mtp_fp8
