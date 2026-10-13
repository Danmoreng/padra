#pragma once
#include "simt_batch_state.cuh"
template<bool Local>
__global__ void AttendSimtBatch(const float* q,const __nv_bfloat16* kc,const __nv_bfloat16* vc,float* out,
    const SimtBatch* batch,int cap,float* workspace,int* bad,const __nv_bfloat16* current,int first){
  const int row=first+blockIdx.y;if(row>=batch->rows)return;
  const int pos=batch->base+row;q+=row*4096;out+=row*4096;
  if(pos<0 || (!Local && pos>=cap) || cap<1 || blockDim.x!=256){if(!threadIdx.x)atomicExch(bad,1);return;}
  const int h=blockIdx.x,kvh=h/8,start=Local?max(0,pos-1023):0,n=pos-start+1;
  float* scores=workspace+(blockIdx.y*32+h)*cap;
  __shared__ float scratch[16],maximum,total;
  const int lane=threadIdx.x&7,group=threadIdx.x/8;
  const unsigned mask=0xffu<<(threadIdx.x&24);
  float query[16];
  #pragma unroll
  for(int i=0;i<16;++i)query[i]=q[h*128+lane+i*8];
  float mx=-INFINITY;
  for(int j=group;j<n;j+=32){
    float dot=0,error=0;const int offset=((start+j)%cap)*512+kvh*128+lane;
    #pragma unroll
    for(int i=0;i<16;++i)Accumulate(__fmul_rn(query[i],__bfloat162float(start+j>=batch->base?current[(start+j-batch->base)*512+kvh*128+lane+i*8]:kc[offset+i*8])),dot,error);
    #pragma unroll
    for(int d=4;d;d/=2){const float other=__shfl_down_sync(mask,dot,d,8),low=__shfl_down_sync(mask,error,d,8);
      if(lane+d<8){Accumulate(other,dot,error);Accumulate(low,dot,error);}}
    if(!lane){const float score=B(B(__fadd_rn(dot,error))*.08838834764831845f);scores[j]=score;mx=fmaxf(mx,score);}
  }
  mx=AttentionMaxReduce8(mx,scratch);if(!threadIdx.x)maximum=mx;__syncthreads();
  float sum=0,error=0;
  if(threadIdx.x<128)for(int j=threadIdx.x;j<n;j+=128){scores[j]=ExpRN(__fsub_rn(scores[j],maximum));Accumulate(scores[j],sum,error);}
  sum=AttentionAccurateReduce8(sum,error,scratch);if(!threadIdx.x)total=sum;__syncthreads();
  if(threadIdx.x<128)for(int j=threadIdx.x;j<n;j+=128)scores[j]=B(__fdiv_rn(scores[j],total));
  __syncthreads();
  if(threadIdx.x<128){float value=0,low=0;
    for(int j=0;j<n;++j)Accumulate(__fmul_rn(scores[j],__bfloat162float(start+j>=batch->base?current[(kSimtBatch+start+j-batch->base)*512+kvh*128+threadIdx.x]:vc[((start+j)%cap)*512+kvh*128+threadIdx.x])),value,low);
    out[h*128+threadIdx.x]=B(__fadd_rn(value,low));Finite(value,bad);
  }
}
