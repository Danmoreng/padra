// Generic Gem16 c0a807a adjacent-state window, exact native K4 FP16 mul1.
// No E4M3/BF16 rounding; two warp shuffles for the shared20-bit window.
#pragma once
#include "expert_exl3_codec.cuh"
namespace exl3_register_codec {
__device__ __forceinline__ unsigned Hash(unsigned state){
  const unsigned h=state*0x83DCD12DU;
  return (h&255)+((h>>8)&255)+((h>>16)&255)+(h>>24);
}
__device__ __forceinline__ __half2 FromStates(unsigned a,unsigned b){
  return __hfma2(__halves2half2(__ushort_as_half(0x6400+Hash(a)),__ushort_as_half(0x6400+Hash(b))),
    __half2half2(__ushort_as_half(0x1eee)),__half2half2(__ushort_as_half(0xc931)));
}
__device__ __forceinline__ __half2 Pair(unsigned word,unsigned input,unsigned output){
  // input is even: adjacent input values are adjacent trellis positions.
  const int p=int(exl3_fp16::Permutation(input,output));
  const int begin=p*4+4-16+1024,end=p*4+8+1024;
  const unsigned first=__shfl_sync(0xffffffffU,word,(begin/32)&31);
  const unsigned last=__shfl_sync(0xffffffffU,word,((end-1)/32)&31);
  const int shift=(((end-1)/32)+1)*32-end;
  const unsigned window=__funnelshift_r(last,first,shift);
  return FromStates((window>>4)&65535U,window&65535U);
}
__device__ __forceinline__ unsigned Bits(__half2 h){
  return unsigned(__half_as_ushort(__low2half(h)))|(unsigned(__half_as_ushort(__high2half(h)))<<16);
}
}
