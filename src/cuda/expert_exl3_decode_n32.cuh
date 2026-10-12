// R-D1 sole N32/four-warp/two K-partition ordinary decode candidate.
// Frozen X2 exact FP16 codec; BF16 inputs widened to FP32. Full H128 separate.
#pragma once
#include "expert_exl3_selected.cuh"
namespace exl3_n32 {
__global__ void Projection(const DeviceMatrix* matrices,int layer,int part,const int* ids,
    const float* input,float* raw,int pairs,int* bad){
  const int pair=blockIdx.y,lane=threadIdx.x&31,warp=threadIdx.x/32;
  if(pair>=pairs)return;
  const int expert=ids[pair];if(expert<0||expert>=64){if(!threadIdx.x)atomicExch(bad,1);return;}
  const auto m=exl3_expert::Matrix(matrices,layer,part,ids,pair,bad);
  const int base=blockIdx.x*32,col=lane/4,ki=(lane%4)*2;
  if(m.format!=3||base+32>m.rows||m.cols%32||m.cols>2304||blockDim.x!=128){if(!threadIdx.x)atomicExch(bad,1);return;}
  __shared__ float transformed[2304],partial[2][32];
  for(int k=threadIdx.x;k<m.cols;k+=128)transformed[k]=input[std::uint64_t(pair)*m.cols+k];
  __syncthreads();
  const int nTile=warp&1,kPart=warp/2;
  float low=0,high=0;
  for(int k=kPart*16;k<m.cols;k+=32){
    const std::uint64_t tile=std::uint64_t(k/16)*std::uint64_t(m.rows/16)+base/16+nTile;
    const unsigned word=reinterpret_cast<const unsigned*>(m.data+tile*128)[lane];
    #pragma unroll
    for(int half=0;half<2;++half){
      const int index=ki+half*8;const float x=transformed[k+index],y=transformed[k+index+1];
      const float2 a=__half22float2(exl3_selected::DecodePair(word,index,col));
      const float2 b=__half22float2(exl3_selected::DecodePair(word,index,col+8));
      low=__fmaf_rn(x,a.x,low);low=__fmaf_rn(y,a.y,low);
      high=__fmaf_rn(x,b.x,high);high=__fmaf_rn(y,b.y,high);
    }
  }
  #pragma unroll
  for(int d=1;d<=2;d*=2){low=__fadd_rn(low,__shfl_xor_sync(0xffffffffU,low,d));high=__fadd_rn(high,__shfl_xor_sync(0xffffffffU,high,d));}
  if((lane&3)==0){partial[kPart][nTile*16+col]=low;partial[kPart][nTile*16+col+8]=high;}
  __syncthreads();
  if(warp<2&&(lane&3)==0){
    #pragma unroll
    for(int half=0;half<2;++half){const int n=warp*16+col+half*8;
      const float value=__fadd_rn(partial[0][n],partial[1][n]);if(!isfinite(value))atomicExch(bad,1);
      raw[std::uint64_t(pair)*m.rows+base+n]=value;
    }
  }
}
inline void Launch(const DeviceMatrix* matrices,int layer,int part,const int* ids,
    const float* input,float* output,float* scratch,int pairs,int* bad,cudaStream_t stream){
  const int rows=part==2?2304:896,cols=part==2?896:2304;
  float* raw=scratch+std::uint64_t(pairs)*cols; // <=72KiB raw; existing arena only.
  exl3_expert::InputHad<<<dim3(cols/128,pairs),128,0,stream>>>(matrices,layer,part,ids,input,scratch,pairs,bad);
  Projection<<<dim3(rows/32,pairs),128,0,stream>>>(matrices,layer,part,ids,scratch,raw,pairs,bad);
  exl3_expert::OutputHad<<<dim3(rows/128,pairs),128,0,stream>>>(matrices,layer,part,ids,raw,output,pairs,bad);
}
}
