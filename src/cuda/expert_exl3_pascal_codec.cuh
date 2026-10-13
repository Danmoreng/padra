#pragma once
namespace exl3_pascal {
// GP104's FP16 FMA is slow. The exact half constants and final RN half cast
// preserve the original codec for every 16-bit state; DP4A only sums hash bytes.
__device__ __forceinline__ float Decode(unsigned state){
  const unsigned hash=state*0x83DCD12DU;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  const unsigned sum=__dp4a(hash,0x01010101U,0U);
#else
  const unsigned sum=(hash&255)+((hash>>8)&255)+((hash>>16)&255)+(hash>>24);
#endif
  const float value=__fmaf_rn(float(1024+sum),__half2float(__ushort_as_half(0x1eee)),__half2float(__ushort_as_half(0xc931)));
  return __half2float(__float2half_rn(value));
}
__device__ __forceinline__ void InitLookup(float* lookup){for(unsigned sum=threadIdx.x;sum<1024;sum+=blockDim.x)lookup[sum]=__half2float(__float2half_rn(__fmaf_rn(float(1024+sum),__half2float(__ushort_as_half(0x1eee)),__half2float(__ushort_as_half(0xc931)))));__syncthreads();}
__device__ __forceinline__ float DecodeLookup(unsigned state,const float* lookup){const unsigned hash=state*0x83DCD12DU;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  return lookup[__dp4a(hash,0x01010101U,0U)];
#else
  return lookup[(hash&255)+((hash>>8)&255)+((hash>>16)&255)+(hash>>24)];
#endif
}
}
