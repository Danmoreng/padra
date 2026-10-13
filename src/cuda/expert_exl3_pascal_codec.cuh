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
// With col=lane/4 and ki=(lane%4)*2, all eight weights owned by this
// lane are in its own compressed word and the preceding word. Reuse that
// window instead of performing two arbitrary shuffles for every weight.
// The two states in each pair overlap by twelve bits.
__device__ __forceinline__ float2 PairLookup(unsigned window,const float* lookup){
  return make_float2(DecodeLookup((window>>4)&65535U,lookup),DecodeLookup(window&65535U,lookup));
}
__device__ __forceinline__ float2 Pair(unsigned window){
  return make_float2(Decode((window>>4)&65535U),Decode(window&65535U));
}
__device__ __forceinline__ unsigned Previous(unsigned word){
  return __shfl_sync(0xffffffffU,word,((threadIdx.x&31)+31)&31);
}
}
