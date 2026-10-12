// Device-state transport for the existing selected-profile chunk arithmetic.
#pragma once
__global__ void EmbedPrefillControlled(DeviceMatrix matrix,const int* prompt,float* output,const PrefillState* state,int tokens,int* bad,int maximum=8192){
  if(!state || !prompt || !output || !bad || matrix.format!=0 || matrix.rows!=98304 || matrix.cols!=2304 || !matrix.data || blockDim.x!=128 || gridDim.x!=static_cast<unsigned>(tokens) || !ValidPrefillState(*state,tokens,state->capacity,1,maximum)){
    if(threadIdx.x==0 && bad)atomicExch(bad,1);return;
  }
  const int token=prompt[state->prompt_offset+blockIdx.x];
  if(token<0 || token>=98304){if(threadIdx.x==0)atomicExch(bad,1);return;}
  output+=blockIdx.x*2304;for(int i=threadIdx.x;i<2304;i+=128)output[i]=BF(matrix.data,token*2304+i);
}
__global__ void StorePrefillControlled(const float* k,const float* v,__nv_bfloat16* kc,__nv_bfloat16* vc,const PrefillState* state,int tokens,int cap,int first,int* bad,int maximum=8192){
  if(!state || !k || !v || !kc || !vc || !bad || !ValidPrefillState(*state,tokens,state->capacity,1,maximum) || cap<1 || cap>state->capacity || first<0 || first>=tokens || blockDim.x!=128 || gridDim.x!=4 || gridDim.y!=static_cast<unsigned>(tokens-first)){
    if(threadIdx.x==0 && bad)atomicExch(bad,1);return;
  }
  const int row=first+blockIdx.y,i=blockIdx.x*128+threadIdx.x;
  kc[((state->base+row)%cap)*512+i]=__float2bfloat16_rn(k[row*512+i]);vc[((state->base+row)%cap)*512+i]=__float2bfloat16_rn(v[row*512+i]);
}
__global__ void CompletePrefillChunk(PrefillState* state,int tokens,int* bad,int maximum=8192){
  if(threadIdx.x || blockIdx.x)return;
  if(!state || !bad || *bad || !ValidPrefillState(*state,tokens,state->capacity,1,maximum)){if(bad)atomicExch(bad,1);return;}
  state->committed+=tokens;
}
