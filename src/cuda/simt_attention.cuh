#pragma once
// The compensated eight-lane QK / BF16-softmax / compensated PV arithmetic
// from AttendDecodeControlled, with device-global scores instead of context-
// sized shared memory. Supports long contexts on Pascal's 48 KiB/CTA limit.
struct SimtCommand { int token,position; };
static_assert(sizeof(SimtCommand)==8);
__global__ void SetSimtCommand(SimtCommand* command,int token,int position){
  if(!threadIdx.x && !blockIdx.x)*command={token,position};
}
__global__ void EmbedSimt(DeviceMatrix m,const SimtCommand* command,float* x,int* bad,int capacity){
  if(command->token<0 || command->token>=98304 || command->position<0 || command->position>=capacity){if(!threadIdx.x)atomicExch(bad,1);return;}
  for(int i=threadIdx.x;i<2304;i+=128)x[i]=BF(m.data,command->token*2304+i);
}
__global__ void StoreSimt(const float* k,const float* v,__nv_bfloat16* kc,__nv_bfloat16* vc,const SimtCommand* command,int cap){
  const int i=blockIdx.x*128+threadIdx.x,offset=(command->position%cap)*512+i;
  kc[offset]=__float2bfloat16_rn(k[i]);vc[offset]=__float2bfloat16_rn(v[i]);
}
#include "attention_cta.cuh"
template<bool Local>
__global__ void AttendSimt(const float* q,const __nv_bfloat16* kc,const __nv_bfloat16* vc,float* out,
    const SimtCommand* command,int cap,float* workspace,int* bad){
  const int pos=command->position;
  if(pos<0 || (!Local && pos>=cap) || cap<1 || blockDim.x!=256){if(!threadIdx.x)atomicExch(bad,1);return;}
  const int h=blockIdx.x,kvh=h/8,start=Local?max(0,pos-1023):0,n=pos-start+1;
  float* scores=workspace+h*cap;
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
    for(int i=0;i<16;++i)Accumulate(__fmul_rn(query[i],__bfloat162float(kc[offset+i*8])),dot,error);
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
    for(int j=0;j<n;++j)Accumulate(__fmul_rn(scores[j],__bfloat162float(vc[((start+j)%cap)*512+kvh*128+threadIdx.x])),value,low);
    out[h*128+threadIdx.x]=B(__fadd_rn(value,low));Finite(value,bad);
  }
}
__global__ void QuantizeSimt(const float* x,unsigned char* packed,float* scales,int cols,int* bad){
  __shared__ float scratch[4],scale,inverse;float amax=0;
  for(int c=threadIdx.x;c<cols;c+=128){Finite(x[c],bad);amax=fmaxf(amax,fabsf(x[c]));}
  amax=MaxReduce(amax,scratch);
  if(!threadIdx.x){scale=amax==0?1.f:__fdiv_rn(amax,448.f);scales[0]=scale;inverse=__fdiv_rn(1.f,scale);}
  __syncthreads();
  for(int c=threadIdx.x;c<cols;c+=128)packed[c]=fp8_exact_reciprocal::Encode(x[c],scale,inverse);
}
