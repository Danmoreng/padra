#pragma once
#include "kv_q8_native.cuh"
namespace q8_simt {
__global__ void BatchCommand(const SimtBatch* batch,int row,SimtCommand* command){
  if(!threadIdx.x)*command={0,batch->base+row};
}
__global__ void State(const SimtCommand* command,DecodeState* state,int capacity){
  if(threadIdx.x)return;const int p=command->position;
  *state={command->token,p,capacity,capacity-p,p,min(p,1024),0,0};
}
// A single layer's dequantized prefix is reused by prefill. Cache storage and
// rollback remain encoded Q8_0; the current chunk stays separate until commit.
template<bool Current> __global__ void Dequantize(const q8_native::CacheRow* k,const q8_native::CacheRow* v,__nv_bfloat16* outk,__nv_bfloat16* outv,const SimtBatch* batch,int capacity,int* bad){
  const int rows=Current?batch->rows:min(batch->base,capacity);
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<rows*512;i+=blockDim.x*gridDim.x){
    const auto& a=k[i/512];const auto& b=v[i/512];
    const float ks=__half2float(a.d[(i%512)/32]),vs=__half2float(b.d[(i%512)/32]);
    if(!isfinite(ks)||!isfinite(vs)||ks<0||vs<0||a.qs[i%512]==-128||b.qs[i%512]==-128)atomicExch(bad,1);
    outk[i]=__float2bfloat16_rn(float(a.qs[i%512])*ks);
    outv[i]=__float2bfloat16_rn(float(b.qs[i%512])*vs);
  }
}
__global__ void Commit(const q8_native::CacheRow* k,const q8_native::CacheRow* v,q8_native::CacheRow* keys,q8_native::CacheRow* values,const SimtBatch* batch,int capacity){
  const int row=blockIdx.x;if(row>=batch->rows)return;
  const int target=(batch->base+row)%capacity;
  for(int i=threadIdx.x;i<136;i+=blockDim.x){
    reinterpret_cast<unsigned*>(keys+target)[i]=reinterpret_cast<const unsigned*>(k+row)[i];
    reinterpret_cast<unsigned*>(values+target)[i]=reinterpret_cast<const unsigned*>(v+row)[i];
  }
}
}
