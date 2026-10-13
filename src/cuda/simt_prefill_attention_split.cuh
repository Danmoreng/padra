#pragma once
namespace simt_prefill_attention_split {
__global__ void QK(const float* q,const __nv_bfloat16* keys,const __nv_bfloat16* current,float* workspace,const SimtBatch* batch,int capacity,bool local,int first){
  const int group=threadIdx.x/8,lane=threadIdx.x&7,row=first+group/4,head=blockIdx.y*4+group%4;
  const int common_start=local?max(0,batch->base+first-1023):0,key_first=common_start+blockIdx.x*64;
  const int last=batch->base+min(batch->rows-1,first+7);
  if(first>=batch->rows || key_first>last)return;
  // Pad the BF16 row so adjacent eight-lane key groups use different banks.
  __shared__ __nv_bfloat16 tile[64][136];
  for(int i=threadIdx.x;i<64*128;i+=256){const int absolute=key_first+i/128,d=i%128;
    tile[i/128][d]=absolute>last?__float2bfloat16_rn(0.f):absolute>=batch->base?current[(absolute-batch->base)*512+(head/8)*128+d]:keys[(local?(absolute&1023):absolute)*512+(head/8)*128+d];}
  __syncthreads();if(row>=batch->rows)return;
  const int pos=batch->base+row,start=local?max(0,pos-1023):0;
  float query[16];
  #pragma unroll
  for(int i=0;i<16;++i)query[i]=q[row*4096+head*128+lane+i*8];
  const unsigned mask=0xffu<<(threadIdx.x&24);float* scores=workspace+((row-first)*32+head)*capacity;
  for(int j=0;j<64;++j){const int absolute=key_first+j;if(absolute<start || absolute>pos)continue;
    float dot=0,error=0;
    #pragma unroll
    for(int i=0;i<16;++i)Accumulate(__fmul_rn(query[i],__bfloat162float(tile[j][lane+i*8])),dot,error);
    for(int d=4;d;d/=2){const float other=__shfl_down_sync(mask,dot,d,8),low=__shfl_down_sync(mask,error,d,8);if(lane+d<8){Accumulate(other,dot,error);Accumulate(low,dot,error);}}
    if(!lane)scores[absolute-start]=B(B(__fadd_rn(dot,error))*.08838834764831845f);
  }
}
__global__ void Softmax(float* workspace,const SimtBatch* batch,int capacity,bool local,int first){
  const int row=first+blockIdx.y;if(row>=batch->rows)return;
  const int pos=batch->base+row,n=local?min(pos+1,1024):pos+1,h=blockIdx.x;
  float* scores=workspace+(blockIdx.y*32+h)*capacity;const int lane=threadIdx.x&7,group=threadIdx.x/8;
  __shared__ float scratch[16],maximum,total;float mx=-INFINITY;
  if(!lane)for(int j=group;j<n;j+=32)mx=fmaxf(mx,scores[j]);
  mx=AttentionMaxReduce8(mx,scratch);if(!threadIdx.x)maximum=mx;__syncthreads();
  float sum=0,error=0;if(threadIdx.x<128)for(int j=threadIdx.x;j<n;j+=128){scores[j]=ExpRN(__fsub_rn(scores[j],maximum));Accumulate(scores[j],sum,error);}
  sum=AttentionAccurateReduce8(sum,error,scratch);if(!threadIdx.x)total=sum;__syncthreads();
  if(threadIdx.x<128)for(int j=threadIdx.x;j<n;j+=128)scores[j]=B(__fdiv_rn(scores[j],total));
}
__device__ __forceinline__ float V(const __nv_bfloat16* cache,const __nv_bfloat16* current,int absolute,int base,int capacity,int head,int d,bool local){return __bfloat162float(absolute>=base?current[(256+absolute-base)*512+(head/8)*128+d]:cache[(local?(absolute&1023):absolute)*512+(head/8)*128+d]);}
__global__ void PV(const __nv_bfloat16* cache,const __nv_bfloat16* current,const float* workspace,float* output,const SimtBatch* batch,int capacity,bool local,int first,int* bad){
  const int row=first+blockIdx.y;if(row>=batch->rows)return;
  const int pos=batch->base+row,start=local?max(0,pos-1023):0,n=pos-start+1,h=blockIdx.x,d=threadIdx.x;
  const float* scores=workspace+(blockIdx.y*32+h)*capacity;float probability[8],value[8];
  #pragma unroll
  for(int i=0;i<8;++i){probability[i]=i<n?scores[i]:0.f;value[i]=i<n?V(cache,current,start+i,batch->base,capacity,h,d,local):0.f;}
  float sum=0,error=0;
  for(int base=0;base<n;base+=8){
    #pragma unroll
    for(int i=0;i<8;++i){const float p=probability[i],v=value[i];const int next=base+i+8;if(next<n){probability[i]=scores[next];value[i]=V(cache,current,start+next,batch->base,capacity,h,d,local);}if(base+i<n)Accumulate(__fmul_rn(p,v),sum,error);}
  }
  output[row*4096+h*128+d]=B(__fadd_rn(sum,error));Finite(sum,bad);
}
inline void Launch(const float* q,const __nv_bfloat16* k,const __nv_bfloat16* v,float* output,const SimtBatch* batch,int capacity,bool local,float* scores,int* bad,const __nv_bfloat16* current,int first,cudaStream_t stream){
  QK<<<dim3((capacity+7+63)/64,8),256,0,stream>>>(q,k,current,scores,batch,capacity,local,first);
  Softmax<<<dim3(32,8),256,0,stream>>>(scores,batch,capacity,local,first);
  PV<<<dim3(32,8),128,0,stream>>>(v,current,scores,output,batch,capacity,local,first,bad);
}
}
