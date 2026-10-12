// Split/merge schedule adapted from Gem16 (Apache-2.0), revision
// c01789a2b46520df8a699ee3338cb19aaccebae2, attention/decode_sm120.cu
// lines 275-504, 1924-2002. BF16 Mellum KV, D128/Q32/KV4, score scaling,
// device-state admission, final BF16 boundary and finite checks are adaptations.
namespace fast_attention {
// Explicit diagnostic fixture input only; actual model routes and both GPU
// output heads remain free. Never called by ordinary generation/benchmarks.
__global__ void ForceDiagnosticDecodeToken(DecodeState* state,int token,int* bad){
  if(threadIdx.x || blockIdx.x)return;
  if(!state || token<0 || token>=98304 || state->remaining!=1 || state->capacity<1 || state->capacity>kW4A8FContextCapacity ||
      state->position<0 || state->position>=state->capacity || state->global_valid!=state->position ||
      state->local_valid!=min(state->position,1024) || *bad){atomicExch(bad,1);return;}
  state->token=token;state->eos=0;state->stopped=0;
}
constexpr int kThreads=256,kWarps=8,kHeads=32,kDimension=128,kGroup=4,kChunk=128;
constexpr float kScoreScale=.08838834764831845f;
std::size_t WorkspaceElements(int capacity){
  if(capacity<1 || capacity>kW4A8FContextCapacity)throw std::runtime_error("fast attention requires admitted capacity 1..8192");
  return static_cast<std::size_t>((capacity+kChunk-1)/kChunk)*kHeads*(kDimension+1);
}
__device__ float BlockReduce(float value,float* shared,bool maximum){
  for(int offset=16;offset;offset>>=1){const float other=__shfl_down_sync(0xffffffffU,value,offset);value=maximum?fmaxf(value,other):value+other;}
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  if(lane==0)shared[warp]=value;
  __syncthreads();
  if(warp==0){value=lane<kWarps?shared[lane]:(maximum?-INFINITY:0.f);
    for(int offset=16;offset;offset>>=1){const float other=__shfl_down_sync(0xffffffffU,value,offset);value=maximum?fmaxf(value,other):value+other;}
    if(lane==0)shared[0]=value;
  }
  __syncthreads();const float result=shared[0];__syncthreads();return result;
}
template<bool Local>
__device__ bool Admitted(const DecodeState* state,int cap,int max_splits){
  return DecodeReady(state) && cap>0 && cap<=state->capacity &&
      max_splits==(cap+kChunk-1)/kChunk &&
      (Local?cap==min(1024,state->capacity):cap==state->capacity);
}
template<bool Local>
__launch_bounds__(kThreads,1) __global__ void SplitOnlineDecodeBF16(
    const float* __restrict__ q,const __nv_bfloat16* __restrict__ kc,
    const __nv_bfloat16* __restrict__ vc,float* __restrict__ partial,
    float* __restrict__ lse,const DecodeState* state,int cap,int max_splits,int* bad){
  if(!Admitted<Local>(state,cap,max_splits) || !q || !kc || !vc || !partial || !lse || !bad ||
      blockDim.x!=kThreads || blockDim.y!=1 || blockDim.z!=1 ||
      gridDim.x!=static_cast<unsigned>(max_splits*8) || gridDim.y!=1 || gridDim.z!=1){
    if(threadIdx.x==0 && bad)atomicExch(bad,1);return;
  }
  __shared__ float scores[kGroup*kChunk],reduction[kWarps];
  const int split=blockIdx.x/8,group=blockIdx.x%8;
  const int pos=state->position,start=Local?max(0,pos-1023):0,n=pos-start+1;
  const int split_start=split*kChunk;
  if(split_start>=n)return; // Merge never reads inactive splits, including stale scratch.
  const int count=min(kChunk,n-split_start),kvhead=group/2,headbase=group*kGroup;
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  for(int token=warp;token<count;token+=kWarps){
    const int slot=(start+split_start+token)%cap;
    const auto* key=kc+slot*512+kvhead*kDimension;
    float score[kGroup]={};
    for(int d=lane;d<kDimension;d+=32){const float key_value=__bfloat162float(key[d]);
      #pragma unroll
      for(int g=0;g<kGroup;++g)score[g]=fmaf(q[(headbase+g)*kDimension+d],key_value,score[g]);
    }
    #pragma unroll
    for(int g=0;g<kGroup;++g){
      for(int offset=16;offset;offset>>=1)score[g]+=__shfl_down_sync(0xffffffffU,score[g],offset);
      if(lane==0){const float scaled=score[g]*kScoreScale;Finite(scaled,bad);scores[g*kChunk+token]=scaled;}
    }
  }
  __syncthreads();
  float inverse[kGroup];
  #pragma unroll
  for(int g=0;g<kGroup;++g){
    float maximum=-INFINITY;
    for(int token=threadIdx.x;token<count;token+=kThreads)maximum=fmaxf(maximum,scores[g*kChunk+token]);
    maximum=BlockReduce(maximum,reduction,true);
    float sum=0;
    for(int token=threadIdx.x;token<count;token+=kThreads){const float p=expf(scores[g*kChunk+token]-maximum);scores[g*kChunk+token]=p;sum+=p;}
    const float denominator=BlockReduce(sum,reduction,false);
    inverse[g]=1.f/denominator;
    if(threadIdx.x==0){const float value=maximum+logf(denominator);lse[split*kHeads+headbase+g]=value;Finite(value,bad);}
  }
  for(int d=threadIdx.x;d<kDimension;d+=kThreads){
    float acc[kGroup]={};
    for(int token=0;token<count;++token){const int slot=(start+split_start+token)%cap;
      const float value=__bfloat162float(vc[slot*512+kvhead*kDimension+d]);
      #pragma unroll
      for(int g=0;g<kGroup;++g)acc[g]=fmaf(scores[g*kChunk+token],value,acc[g]);
    }
    #pragma unroll
    for(int g=0;g<kGroup;++g){const float value=acc[g]*inverse[g];partial[(split*kHeads+headbase+g)*kDimension+d]=value;Finite(value,bad);}
  }
}
template<bool Local>
__launch_bounds__(kThreads,1) __global__ void MergeOnlineDecodeBF16(
    const float* __restrict__ partial,float* __restrict__ lse,float* __restrict__ out,
    const DecodeState* state,int cap,int max_splits,int* bad){
  if(!Admitted<Local>(state,cap,max_splits) || !partial || !lse || !out || !bad ||
      blockDim.x!=kThreads || blockDim.y!=1 || blockDim.z!=1 || gridDim.x!=kHeads || gridDim.y!=1 || gridDim.z!=1){
    if(threadIdx.x==0 && bad)atomicExch(bad,1);return;
  }
  __shared__ float reduction[kWarps];
  const int head=blockIdx.x,n=Local?min(state->position+1,1024):state->position+1;
  const int splits=(n+kChunk-1)/kChunk;
  float maximum=-INFINITY;
  for(int s=threadIdx.x;s<splits;s+=kThreads)maximum=fmaxf(maximum,lse[s*kHeads+head]);
  maximum=BlockReduce(maximum,reduction,true);
  float sum=0;
  for(int s=threadIdx.x;s<splits;s+=kThreads){const float value=expf(lse[s*kHeads+head]-maximum);lse[s*kHeads+head]=value;sum+=value;}
  const float denominator=BlockReduce(sum,reduction,false);
  __syncthreads();
  for(int d=threadIdx.x;d<kDimension;d+=kThreads){float value=0;
    for(int s=0;s<splits;++s)value=fmaf(lse[s*kHeads+head],partial[(s*kHeads+head)*kDimension+d],value);
    value=B(value/denominator);out[head*kDimension+d]=value;Finite(value,bad);
  }
}
void Launch(const float* q,const __nv_bfloat16* kc,const __nv_bfloat16* vc,float* out,
    const DecodeState* state,int cap,bool local,float* scratch,int* bad,cudaStream_t stream){
  (void)WorkspaceElements(cap);
  const int splits=(cap+kChunk-1)/kChunk;
  auto* lse=scratch+splits*kHeads*kDimension;
  if(local){SplitOnlineDecodeBF16<true><<<splits*8,kThreads,0,stream>>>(q,kc,vc,scratch,lse,state,cap,splits,bad);
    MergeOnlineDecodeBF16<true><<<kHeads,kThreads,0,stream>>>(scratch,lse,out,state,cap,splits,bad);
  }else{SplitOnlineDecodeBF16<false><<<splits*8,kThreads,0,stream>>>(q,kc,vc,scratch,lse,state,cap,splits,bad);
    MergeOnlineDecodeBF16<false><<<kHeads,kThreads,0,stream>>>(scratch,lse,out,state,cap,splits,bad);
  }
}
} // namespace fast_attention
