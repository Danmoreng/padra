// Four independent router rows share each normalized input load.
// Each original lane c=lane+32*j/compensated tree/cast is unchanged.
#pragma once
__global__ void RouterBF16BatchQuad(DeviceMatrix m, const float *x, float *y, int tokens,
                                    int *bad) {
  if (m.format != 0 || m.rows != 64 || m.cols != 2304 || !m.data || !x || !y || !bad ||
      tokens < 1 || tokens > 2048 || blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
      gridDim.x != 4 || gridDim.y != unsigned(tokens) || gridDim.z != 1) {
    if (!threadIdx.x && bad)
      atomicExch(bad, 1);
    return;
  }
  const int lane = threadIdx.x & 31, r = blockIdx.x * 16 + (threadIdx.x / 32) * 4, t = blockIdx.y;
  float sum[4] = {}, error[4] = {};
  for (int c = lane; c < 2304; c += 32) {
    const float a = x[std::uint64_t(t) * 2304 + c];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      Accumulate(__fmul_rn(BF(m.data, (r + j) * 2304 + c), a), sum[j], error[j]);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    sum[j] = WarpAccurateReduce(sum[j], error[j]);
    if (!lane) {
      y[t * 64 + r + j] = B(sum[j]);
      Finite(y[t * 64 + r + j], bad);
    }
  }
}
