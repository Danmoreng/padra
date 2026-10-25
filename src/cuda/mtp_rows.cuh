// Each router output keeps GemvWarp's lane ownership, compensated accumulation
// and reduction. Rows are independent CTAs, with no cross-row arithmetic.
#pragma once
namespace mtp_rows {
__global__ void Router(DeviceMatrix m, const float *input, float *output, int rows, int *bad) {
  if (!m.data || !input || !output || !bad || m.format != 0 || m.rows != 64 || m.cols != 2304 ||
      rows < 2 || rows > 5 || blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
      gridDim.x != 16 || gridDim.y != unsigned(rows) || gridDim.z != 1) {
    if (!threadIdx.x && bad)
      atomicExch(bad, 1);
    return;
  }
  const int lane = threadIdx.x & 31, r = blockIdx.x * 4 + threadIdx.x / 32;
  input += blockIdx.y * 2304;
  output += blockIdx.y * 64;
  float sum = 0.f, error = 0.f;
  for (int c = lane; c < 2304; c += 32)
    Accumulate(__fmul_rn(WWarp(m, r, c), input[c]), sum, error);
  sum = WarpAccurateReduce(sum, error);
  if (!lane) {
    output[r] = B(sum);
    Finite(output[r], bad);
  }
}
} // namespace mtp_rows
