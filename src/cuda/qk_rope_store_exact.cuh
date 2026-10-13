#pragma once
// Same Q/K normalization and RoPE arithmetic; each K-head CTA writes its BF16 KV row.
// The next attention kernel observes all rows through normal stream ordering.
__global__ void QKNormRopeStoreControlled(float* q,float* k,DeviceMatrix qn,DeviceMatrix kn,
    const float* table,const DecodeState* state,int* bad,const float* v,__nv_bfloat16* kc,__nv_bfloat16* vc,int cap,int maximum=kW4A8FContextCapacity){
  if(!DecodeReadyAtLimit(state,maximum)||!v||!kc||!vc||cap<1||cap>state->capacity||gridDim.y!=1){if(threadIdx.x==0 && bad)atomicExch(bad,1);return;}
  int position=state->position;
  if(blockDim.x!=128 || blockDim.y!=1 || blockDim.z!=1 || gridDim.x!=36 || gridDim.z!=1 ||
      !q || !k || !qn.data || !kn.data || !table || position<0 ||
      static_cast<unsigned>(position)>0x7fffffffU-gridDim.y || qn.format!=0 || kn.format!=0 ||
      qn.rows!=128 || kn.rows!=128 || qn.cols!=1 || kn.cols!=1){
    if(threadIdx.x==0)atomicExch(bad,1);return;
  }
  position+=blockIdx.y;
  const bool query=blockIdx.x<32;
  const int head=query?blockIdx.x:blockIdx.x-32,i=threadIdx.x;
  float* x=(query?q+blockIdx.y*4096:k+blockIdx.y*512)+head*128;
  const auto weights=query?qn:kn;
  __shared__ float sums[8],inv,old[128];
  float sum=0,error=0;Accumulate(__fmul_rn(x[i],x[i]),sum,error);
  sum=AccurateReduce(sum,error,sums);
  if(i==0){const float variance=__fdiv_rn(sum,128.f);inv=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(variance,1e-6f)));}
  __syncthreads();
  old[i]=B(B(x[i]*inv)*BF(weights.data,i));
  __syncthreads();
  const float c=table[position*128+(i%64)],s=table[position*128+64+(i%64)];
  const float rotated=i<64?-old[i+64]:old[i-64];
  const float value=B(B(old[i]*c)+B(rotated*s));x[i]=value;Finite(value,bad);
  if(!query){const int slot=(position%cap)*512+head*128+i;kc[slot]=__float2bfloat16_rn(value);vc[slot]=__float2bfloat16_rn(v[head*128+i]);}
}
