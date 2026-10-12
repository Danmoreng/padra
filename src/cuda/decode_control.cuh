// Gem16 device token/position and graph lifecycle mechanics; Mellum math only.
#include "context_admission.cuh"
struct DecodeState {int token,position,capacity,remaining,global_valid,local_valid,eos,stopped;};
static_assert(sizeof(DecodeState)==32);
__device__ bool DecodeReady(const DecodeState* s){return s && !s->stopped && !s->eos && s->remaining>0 && s->token>=0 && s->token<98304 && s->capacity>0 && s->capacity<=kW4A8FContextCapacity && s->position>=0 && s->position<s->capacity && s->global_valid==s->position && s->local_valid==min(s->position,1024);}
__global__ void InitializeDecodeState(DecodeState* s,Control* chosen,int position,int capacity,int remaining,int* bad,int maximum=kW4A8FContextCapacity){
  if(threadIdx.x || blockIdx.x)return;
  if(!s || !chosen || !bad)return;
  *s={0,0,capacity,0,0,0,0,1};
  if(*bad || chosen->bad || chosen->token<0 || chosen->token>=98304 || capacity<1 || (maximum<1 || maximum>131072 || capacity>maximum) || position<0 || position>capacity || remaining<0 || remaining>capacity-position){atomicExch(bad,1);chosen->bad=1;return;}
  const bool eos=chosen->token==28;
  *s={chosen->token,position,capacity,remaining,position,min(position,1024),eos,eos || remaining==0};
}
__global__ void AdvanceDecodeState(DecodeState* s,Control* chosen,int* bad,int maximum=kW4A8FContextCapacity){
  if(threadIdx.x || blockIdx.x)return;
  if(!chosen || !bad)return;
  if(*bad || chosen->bad || !DecodeReadyAtLimit(s,maximum) || chosen->token<0 || chosen->token>=98304){atomicExch(bad,1);chosen->bad=1;if(s)s->stopped=1;return;}
  ++s->position;--s->remaining;s->global_valid=s->position;s->local_valid=min(s->position,1024);
  s->token=chosen->token;s->eos=s->token==28;s->stopped=s->eos || s->remaining==0;
}
// Owner/F4 lifecycle: cancellation is observed at a completed token boundary.
__global__ void CancelDecodeState(DecodeState* state,Control* chosen,int* bad,int maximum=kW4A8FContextCapacity){
 if(threadIdx.x || blockIdx.x)return;
 if(!bad || !chosen)return;
 if(*bad || chosen->bad || !DecodeReadyAtLimit(state,maximum)){atomicExch(bad,1);chosen->bad=1;return;}
 state->stopped=1; // Preserve already-consumed position, KV lengths and remaining budget.
}
__global__ void EmbedControlled(DeviceMatrix m,const DecodeState* state,float* x,int* bad,int maximum=kW4A8FContextCapacity){
  if(!DecodeReadyAtLimit(state,maximum) || blockDim.x!=128 || gridDim.x!=1 || !x || m.format!=0 || m.rows!=98304 || m.cols!=2304 || !m.data){if(threadIdx.x==0 && bad)atomicExch(bad,1);return;}
  const int token=state->token;for(int i=threadIdx.x;i<2304;i+=128)x[i]=BF(m.data,token*2304+i);
}
__global__ void QKNormRopeControlled(float* q,float* k,DeviceMatrix qn,DeviceMatrix kn,
    const float* table,const DecodeState* state,int* bad,float* capturedQ=nullptr,float* capturedK=nullptr,int maximum=kW4A8FContextCapacity){
  if(!DecodeReadyAtLimit(state,maximum)){if(threadIdx.x==0 && bad)atomicExch(bad,1);return;}
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
  float* captured=query?capturedQ:capturedK;
  // Model snapshot admission allows exactly one prompt token. Diagnostic
  // captures below keep the same first-token layout for a batched probe.
  if(captured && blockIdx.y==0)captured[head*128+i]=old[i];
  __syncthreads();
  const float c=table[position*128+(i%64)],s=table[position*128+64+(i%64)];
  const float rotated=i<64?-old[i+64]:old[i-64];
  const float value=B(B(old[i]*c)+B(rotated*s));x[i]=value;Finite(value,bad);
}

template<bool Local>
__global__ void AttendDecodeControlled(const float* q,const __nv_bfloat16* kc,const __nv_bfloat16* vc,float* out,const DecodeState* state,int cap,int* bad,float* captured_scores=nullptr,float* captured_probabilities=nullptr){
  if(!DecodeReady(state)){if(threadIdx.x==0 && bad)atomicExch(bad,1);return;}
  const int pos=state->position;
  if(blockDim.x!=256 || blockDim.y!=1 || blockDim.z!=1 || gridDim.x!=32 || gridDim.y!=1 || gridDim.z!=1 ||
      !q || !kc || !vc || !out || !bad || pos<0 || pos>=kW4A8FContextCapacity || cap<1 || cap>kW4A8FContextCapacity ||
      (Local?(cap<min(pos+1,1024)):(pos>=cap))){
    if(threadIdx.x==0 && bad)atomicExch(bad,1);return;
  }
  extern __shared__ float scores[];__shared__ float scratch[16];__shared__ float maxv,total;
  const int h=blockIdx.x,kvh=h/8,start=Local?max(0,pos-1023):0,n=pos-start+1;
  const int lane=threadIdx.x&7,group=threadIdx.x/8;
  const unsigned mask=0xffu<<(threadIdx.x&24);
  float query[16];
  #pragma unroll
  for(int i=0;i<16;++i)query[i]=q[h*128+lane+i*8];
  float mx=-INFINITY;
  for(int j=group;j<n;j+=32){float dot=0,error=0;const int offset=((start+j)%cap)*512+kvh*128+lane;
    #pragma unroll
    for(int i=0;i<16;++i)Accumulate(__fmul_rn(query[i],__bfloat162float(kc[offset+i*8])),dot,error);
    #pragma unroll
    for(int d=4;d;d/=2){const float other=__shfl_down_sync(mask,dot,d,8),low=__shfl_down_sync(mask,error,d,8);
      if(lane+d<8){Accumulate(other,dot,error);Accumulate(low,dot,error);}}
    dot=__fadd_rn(dot,error);
    if(lane==0){const float score=B(B(dot)*.08838834764831845f);scores[j]=score;if(captured_scores)captured_scores[h*n+j]=score;mx=fmaxf(mx,score);}
  }
  mx=AttentionMaxReduce8(mx,scratch);if(threadIdx.x==0)maxv=mx;__syncthreads();
  float sum=0,sum_error=0;if(threadIdx.x<128)for(int j=threadIdx.x;j<n;j+=128){scores[j]=ExpRN(__fsub_rn(scores[j],maxv));Accumulate(scores[j],sum,sum_error);}
  sum=AttentionAccurateReduce8(sum,sum_error,scratch);if(threadIdx.x==0)total=sum;__syncthreads();
  if(threadIdx.x<128)for(int j=threadIdx.x;j<n;j+=128){scores[j]=B(__fdiv_rn(scores[j],total));if(captured_probabilities)captured_probabilities[h*n+j]=scores[j];}__syncthreads();
  if(threadIdx.x<128){float value=0,error=0;for(int j=0;j<n;++j)Accumulate(__fmul_rn(scores[j],__bfloat162float(vc[((start+j)%cap)*512+kvh*128+threadIdx.x])),value,error);
  out[h*128+threadIdx.x]=B(__fadd_rn(value,error));Finite(value,bad);}
}

__global__ void StoreControlled(const float* k,const float* v,__nv_bfloat16* kc,__nv_bfloat16* vc,const DecodeState* state,int cap,int* bad,int maximum=kW4A8FContextCapacity){
  if(!DecodeReadyAtLimit(state,maximum) || blockDim.x!=128 || gridDim.x!=4 || gridDim.y!=1 || !k || !v || !kc || !vc || cap<1 || cap>state->capacity){if(threadIdx.x==0 && bad)atomicExch(bad,1);return;}
  const int position=state->position,i=blockIdx.x*128+threadIdx.x;kc[(position%cap)*512+i]=__float2bfloat16_rn(k[i]);vc[(position%cap)*512+i]=__float2bfloat16_rn(v[i]);
}
__global__ void TraceControlled(float* out,const float* x,const float* router,const int* ids,const float* weights,const DecodeState* state,int layer,int* bad){
  if(!DecodeReady(state) || layer<0 || layer>=28){if(threadIdx.x==0 && bad)atomicExch(bad,1);return;}
  out+=(state->position*28+layer)*2384;
  const int i=blockIdx.x*128+threadIdx.x;if(i<2304)out[i]=x[i];else if(i<2368)out[i]=router[i-2304];else if(i<2376)out[i]=static_cast<float>(ids[i-2368]);else if(i<2384)out[i]=weights[i-2376];
}
// Capture/instantiate/upload once; replay allocates nothing. No silent fallback.
struct DecodeGraph {
  cudaGraphExec_t executable=nullptr;std::size_t nodes=0;
  ~DecodeGraph(){if(executable)cudaGraphExecDestroy(executable);}
  void Reset(){if(executable){Check(cudaGraphExecDestroy(executable));executable=nullptr;}}
  DecodeGraph()=default;DecodeGraph(const DecodeGraph&)=delete;
  template<class Launcher> void Capture(cudaStream_t stream,Launcher&& launch){
    Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));cudaGraph_t graph=nullptr;
    try{launch();}catch(...){cudaStreamEndCapture(stream,&graph);if(graph)cudaGraphDestroy(graph);throw;}
    const auto ended=cudaStreamEndCapture(stream,&graph);
    if(ended!=cudaSuccess){if(graph)cudaGraphDestroy(graph);Check(ended);}
    Check(cudaGraphGetNodes(graph,nullptr,&nodes));
    const auto created=cudaGraphInstantiate(&executable,graph,0);cudaGraphDestroy(graph);Check(created);
    Check(cudaGraphUpload(executable,stream));Check(cudaStreamSynchronize(stream));
  }
  void Launch(cudaStream_t stream){if(!executable)throw std::runtime_error("decode graph not prepared");Check(cudaGraphLaunch(executable,stream));}
};
