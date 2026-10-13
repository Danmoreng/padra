// X2 N128/eight-warp selected-vector FP32-FMA projection. Gem16 donor mapping
// in PROVENANCE.md. Standard K4 exact FP16 codec, no FP8 weight/activation math.
#pragma once
#include "expert_exl3_codec.cuh"
#include "expert_exl3_pascal_codec.cuh"
namespace exl3_selected {
__device__ __forceinline__ unsigned State(unsigned word,unsigned input,unsigned output){
  const int end=int(exl3_fp16::Permutation(input,output))*4+4+1024;
  const int last=(end-1)/32,first=(end-16)/32,shift=(last+1)*32-end;
  const unsigned a=__shfl_sync(0xffffffffU,word,last&31),b=__shfl_sync(0xffffffffU,word,first&31);
  return __funnelshift_r(a,b,shift)&65535U;
}
__device__ __forceinline__ unsigned HashSum(unsigned state){
  const unsigned h=state*0x83DCD12DU;
  return (h&255)+((h>>8)&255)+((h>>16)&255)+(h>>24);
}
__device__ __forceinline__ __half2 DecodePair(unsigned word,unsigned input,unsigned output){
  const unsigned a=HashSum(State(word,input,output)),b=HashSum(State(word,input+1,output));
  return __hfma2(__halves2half2(__ushort_as_half(0x6400+a),__ushort_as_half(0x6400+b)),
    __half2half2(__ushort_as_half(0x1eee)),__half2half2(__ushort_as_half(0xc931)));
}
template<bool Reference=false> __global__ void Projection(const DeviceMatrix* matrices,int layer,int part,const int* ids,
    const float* input,float* output,int pairs,int* bad,const int* active_pairs=nullptr){
  const int pair=blockIdx.y,lane=threadIdx.x&31,warp=threadIdx.x/32;
  if(active_pairs && pair>=*active_pairs*8)return;
  const auto m=exl3_expert::Matrix(matrices,layer,part,ids,pair,bad);
  const int base=blockIdx.x*128,col=lane/4,ki=(lane%4)*2;
  if(pair>=pairs||m.format!=3||base+128>m.rows||m.cols%16){if(!threadIdx.x)atomicExch(bad,1);return;}
  // Matrix/payload pointers are native64-bit; no uint32 pool-offset cast.
  __shared__ float transformed[2304],raw[128];
  for(int k=threadIdx.x;k<m.cols;k+=256)transformed[k]=input[pair*m.cols+k];
  __syncthreads();
  float low=0,high=0;
  for(int k=0;k<m.cols;k+=16){
    const std::uint64_t tile=std::uint64_t(k/16)*std::uint64_t(m.rows/16)+base/16+warp;
    const unsigned word=reinterpret_cast<const unsigned*>(m.data+tile*128)[lane];
    #pragma unroll
    for(int half=0;half<2;++half){
      const int index=ki+half*8;const float x=transformed[k+index],y=transformed[k+index+1];
      float2 a,b;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 610
      if constexpr(!Reference){
        a=make_float2(exl3_pascal::Decode(State(word,index,col)),exl3_pascal::Decode(State(word,index+1,col)));
        b=make_float2(exl3_pascal::Decode(State(word,index,col+8)),exl3_pascal::Decode(State(word,index+1,col+8)));
      }else
#endif
      {a=__half22float2(DecodePair(word,index,col));b=__half22float2(DecodePair(word,index,col+8));}
      low=__fmaf_rn(x,a.x,low);low=__fmaf_rn(y,a.y,low);
      high=__fmaf_rn(x,b.x,high);high=__fmaf_rn(y,b.y,high);
    }
  }
  #pragma unroll
  for(int d=1;d<=2;d*=2){low=__fadd_rn(low,__shfl_xor_sync(0xffffffffU,low,d));high=__fadd_rn(high,__shfl_xor_sync(0xffffffffU,high,d));}
  if((lane&3)==0){raw[warp*16+col]=low;raw[warp*16+col+8]=high;}
  __syncthreads();
  if(!warp){const int index=lane*4;float v[4]={raw[index],raw[index+1],raw[index+2],raw[index+3]};exl3_fp16::H128(v);
    #pragma unroll
    for(int j=0;j<4;++j){const int i=base+index+j;const float value=B(__fmul_rn(v[j],__half2float(__ushort_as_half(m.svh[i]))));if(!isfinite(value))atomicExch(bad,1);output[pair*m.rows+i]=value;}}
}
inline void Launch(const DeviceMatrix* matrices,int layer,int part,const int* ids,
    const float* input,float* output,float* scratch,int pairs,int* bad,cudaStream_t stream,bool reference=false){
  const int rows=part==2?2304:896,cols=part==2?896:2304;
  exl3_expert::InputHad<<<dim3(cols/128,pairs),128,0,stream>>>(matrices,layer,part,ids,input,scratch,pairs,bad);
  if(reference)Projection<true><<<dim3(rows/128,pairs),256,0,stream>>>(matrices,layer,part,ids,scratch,output,pairs,bad);
  else Projection<false><<<dim3(rows/128,pairs),256,0,stream>>>(matrices,layer,part,ids,scratch,output,pairs,bad);
}
}
