// Packed-load mechanism follows read-only gem16 FP8 sm120.cu, Apache-2.0.
// Mellum SIMT adaptation: exact pair decode, compensated four-value work partition.
__device__ __forceinline__ float4 DecodeFP8FourReference(unsigned packed){
  const auto lo=__half22float2(static_cast<__half2>(__nv_cvt_fp8x2_to_halfraw2(static_cast<unsigned short>(packed),__NV_E4M3)));
  const auto hi=__half22float2(static_cast<__half2>(__nv_cvt_fp8x2_to_halfraw2(static_cast<unsigned short>(packed>>16),__NV_E4M3)));
  return make_float4(lo.x,lo.y,hi.x,hi.y);
}
// Direct FP8 -> FP32 bit mapping on GP104; all 256 encodings are tested.
__device__ __forceinline__ float DecodeFP8Pascal(unsigned code){
  const unsigned absolute=code&127,exponent=absolute>>3,mantissa=absolute&7;
  if(absolute==127)return __half2float(__ushort_as_half(0x7fff));
  unsigned bits=(code&128)<<24;
  if(exponent)bits|=((exponent+120)<<23)|(mantissa<<20);
  else if(mantissa){const unsigned leading=31-__clz(mantissa);bits|=((118+leading)<<23)|((mantissa<<(23-leading))&0x7fffff);}
  return __uint_as_float(bits);
}
__device__ __forceinline__ float4 DecodeFP8Four(unsigned packed){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 610
  return make_float4(DecodeFP8Pascal(packed&255),DecodeFP8Pascal((packed>>8)&255),DecodeFP8Pascal((packed>>16)&255),DecodeFP8Pascal(packed>>24));
#else
  return DecodeFP8FourReference(packed);
#endif
}
__device__ __forceinline__ float4 DecodeFP8Lookup(unsigned packed,const float* table){return make_float4(table[packed&255],table[(packed>>8)&255],table[(packed>>16)&255],table[packed>>24]);}
__device__ __forceinline__ void InitFP8Lookup(float* table){for(unsigned code=threadIdx.x;code<256;code+=blockDim.x)table[code]=DecodeFP8Pascal(code);__syncthreads();}
__device__ __forceinline__ void ProjectFP8VectorRow(DeviceMatrix m, const unsigned char *x,
                                                    const float *scales, float *y, int *bad, int t,
                                                    int r) {
  if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 || t != 0 || m.format != 1 ||
      (m.cols != 2304 && m.cols != 4096) || m.rows <= 0 || m.rows > 4096 || !m.data || !m.scales ||
      !x || !scales || !y || (reinterpret_cast<std::uintptr_t>(m.data) & 3) ||
      (reinterpret_cast<std::uintptr_t>(x) & 3)) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  const unsigned lane = threadIdx.x & 31;
  if (r >= m.rows)
    return;
  float sum = 0, error = 0;
  for (int c = lane * 4; c < m.cols; c += 128) {
    const float4 w = DecodeFP8Four(Load32(m.data + r * m.cols + c)),
                 a = DecodeFP8Four(Load32(x + c));
    Accumulate(__fmul_rn(w.x, a.x), sum, error);
    Accumulate(__fmul_rn(w.y, a.y), sum, error);
    Accumulate(__fmul_rn(w.z, a.z), sum, error);
    Accumulate(__fmul_rn(w.w, a.w), sum, error);
  }
  sum = WarpAccurateReduce(sum, error);
  if (lane == 0) {
    const float value = B(__fmul_rn(__fmul_rn(sum, scales[0]), BF(m.scales, r)));
    y[r] = value;
    Finite(value, bad);
  }
}
__global__ void ProjectFP8Vector(DeviceMatrix m, const unsigned char *x, const float *scales,
                                 float *y, int *bad) {
  if (gridDim.y != 1 || gridDim.z != 1 || gridDim.x != static_cast<unsigned>((m.rows + 3) / 4)) {
    if (threadIdx.x == 0 && bad)
      atomicExch(bad, 1);
    return;
  }
  ProjectFP8VectorRow(m, x, scales, y, bad, 0, blockIdx.x * 4 + threadIdx.x / 32);
}
