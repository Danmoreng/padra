// Same per-element ordered BF16 expert combine,followed by unchanged staged
// compensated RMSNorm. One cached row spans both operations;no new precision.
#pragma once
#include "fp8_exact_reciprocal.cuh"
namespace combine_norm_exact {
template<bool EncodeFP8> __global__ void CombineNorm(float* x,const float* down,
 const float* routing,float* normalized,DeviceMatrix weight,unsigned char* packed,
 float* scales,int* bad,float* statistics=nullptr){
 if(blockDim.x!=128||blockDim.y!=1||blockDim.z!=1||gridDim.x!=1||gridDim.y!=1||gridDim.z!=1||!x||!down||!routing||!normalized||x==normalized||!weight.data||!bad||weight.format!=0||weight.rows!=2304||weight.cols!=1||(EncodeFP8&&(!packed||!scales))){if(threadIdx.x==0&&bad)atomicExch(bad,1);return;}
 __shared__ float sums[8],maxima[4],inv,scale,inverse;float row[18]={};
 // Each element sees experts0..7 in the original order. Interleaving
 // independent elements changes transport/register scheduling only.
 for(int e=0;e<8;++e){const float w=routing[e];
  #pragma unroll
  for(int j=0;j<18;++j){const int i=threadIdx.x+j*128;row[j]=B(__fadd_rn(row[j],B(__fmul_rn(down[e*2304+i],w))));}
 }
 float sum=0,error=0;
 #pragma unroll
 for(int j=0;j<18;++j){const int i=threadIdx.x+j*128;const float v=B(__fadd_rn(x[i],row[j]));x[i]=v;row[j]=v;Finite(v,bad);Accumulate(__fmul_rn(v,v),sum,error);}
 sum=AccurateReduce(sum,error,sums);if(!threadIdx.x){const float variance=__fdiv_rn(sum,2304.f);inv=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(variance,1e-6f)));if(statistics){statistics[0]=variance;statistics[1]=inv;}}
 __syncthreads();float maximum=0.f;
 #pragma unroll
 for(int j=0;j<18;++j){const int i=threadIdx.x+j*128;const float v=B(B(row[j]*inv)*BF(weight.data,i));normalized[i]=v;row[j]=v;Finite(v,bad);if constexpr(EncodeFP8)maximum=fmaxf(maximum,fabsf(v));}
 if constexpr(EncodeFP8){maximum=MaxReduce(maximum,maxima);if(!threadIdx.x){scale=maximum?__fdiv_rn(maximum,448.f):1.f;scales[0]=scale;inverse=__fdiv_rn(1.f,scale);}__syncthreads();
  #pragma unroll
  for(int j=0;j<18;++j)packed[threadIdx.x+j*128]=fp8_exact_reciprocal::Encode(row[j],scale,inverse);
 }
}
}
