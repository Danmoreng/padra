// Overnight D2: one N64/four-warp M1-on-M16 FP16-MMA candidate.
// Exact K4 FP16 weights. Explicit BF16-H128 -> FP16 input rounding.
// Distinct expert B matrices, four-deep packed prefetch, full H128 after raw.
#pragma once
#include "expert_exl3_dot_codec.cuh"
#include "expert_exl3_prefill_masked.cuh"
namespace exl3_mma_dot {
__device__ __forceinline__ unsigned PackHalf(const __half* x){
  return unsigned(__half_as_ushort(x[0]))|(unsigned(__half_as_ushort(x[1]))<<16);
}
__global__ void Projection(const DeviceMatrix* matrices,int layer,int part,const int* ids,
    const float* input,float* raw,int pairs,int* bad){
  const int pair=blockIdx.y,lane=threadIdx.x&31,warp=threadIdx.x/32;
  if(pair>=pairs)return;
  const int expert=ids[pair];if(expert<0||expert>=64){if(!threadIdx.x)atomicExch(bad,1);return;}
  const auto m=exl3_expert::Matrix(matrices,layer,part,ids,pair,bad);
  const int base=blockIdx.x*64,group=lane/4,ki=(lane%4)*2;
  if(m.format!=3||base+64>m.rows||m.cols<64||m.cols%64||m.cols>2304||blockDim.x!=128){if(!threadIdx.x)atomicExch(bad,1);return;}
  __shared__ __align__(16) __half transformed[2304];
  for(int k=threadIdx.x;k<m.cols;k+=128){
    const float x=input[std::uint64_t(pair)*m.cols+k];const auto h=__float2half_rn(x);
    if(!isfinite(x)||(__half_as_ushort(h)&0x7c00)==0x7c00)atomicExch(bad,1);
    transformed[k]=h;
  }
  __syncthreads();
  // Four independent compressed words in registers hide dependent load latency.
  unsigned prefetched[4];const int iterations=m.cols/16;
  #pragma unroll
  for(int d=0;d<4;++d){const std::uint64_t t=std::uint64_t(d)*std::uint64_t(m.rows/16)+base/16+warp;prefetched[d]=reinterpret_cast<const unsigned*>(m.data+t*128)[lane];}
  float accum[2][4]={};
  for(int at=0;at<iterations;at+=4){
    #pragma unroll
    for(int d=0;d<4;++d){
      const int iteration=at+d,k=iteration*16;const unsigned word=prefetched[d];
      if(iteration+4<iterations){const std::uint64_t t=std::uint64_t(iteration+4)*std::uint64_t(m.rows/16)+base/16+warp;prefetched[d]=reinterpret_cast<const unsigned*>(m.data+t*128)[lane];}
      const unsigned lo=PackHalf(transformed+k+ki),hi=PackHalf(transformed+k+ki+8);
      const unsigned a[4]={lo,lo,hi,hi}; // sixteen identical M rows; same expert B.
      #pragma unroll
      for(int n=0;n<2;++n){const unsigned b[2]={
        exl3_dot_codec::Bits(exl3_dot_codec::Pair(word,ki,group+n*8)),
        exl3_dot_codec::Bits(exl3_dot_codec::Pair(word,ki+8,group+n*8))};
        exl3_masked::Mma(a,b,accum[n]);
      }
    }
  }
  // Only M row0 is real; group0 owns its sixteen unique output columns.
  if(group==0){
    #pragma unroll
    for(int n=0;n<2;++n){
      #pragma unroll
      for(int j=0;j<2;++j){const float x=accum[n][j];if(!isfinite(x))atomicExch(bad,1);raw[std::uint64_t(pair)*m.rows+base+warp*16+n*8+ki+j]=x;}
    }
  }
}
inline void Launch(const DeviceMatrix* matrices,int layer,int part,const int* ids,
    const float* input,float* output,float* scratch,int pairs,int* bad,cudaStream_t stream,unsigned* ranges=nullptr){
  const int rows=part==2?2304:896,cols=part==2?896:2304;
  float* raw=scratch+std::uint64_t(pairs)*cols;
  exl3_expert::InputHad<<<dim3(cols/128,pairs),128,0,stream>>>(matrices,layer,part,ids,input,scratch,pairs,bad);
  if(ranges)exl3_expert::Range<<<(cols*pairs+127)/128,128,0,stream>>>(scratch,cols*pairs,ranges);
  Projection<<<dim3(rows/64,pairs),128,0,stream>>>(matrices,layer,part,ids,scratch,raw,pairs,bad);
  exl3_expert::OutputHad<<<dim3(rows/128,pairs),128,0,stream>>>(matrices,layer,part,ids,raw,output,pairs,bad);
}
}
