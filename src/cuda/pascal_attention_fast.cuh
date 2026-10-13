#pragma once
namespace pascal_attention {
__device__ __forceinline__ float Reduce(float value,float* scratch,bool maximum){
  const int lane=threadIdx.x&31,warp=threadIdx.x/32;
  for(int d=16;d;d/=2){const float other=__shfl_down_sync(0xffffffffU,value,d);value=maximum?fmaxf(value,other):value+other;}
  if(!lane)scratch[warp]=value;__syncthreads();
  value=threadIdx.x<8?scratch[lane]:(maximum?-INFINITY:0.f);
  if(!warp)for(int d=16;d;d/=2){const float other=__shfl_down_sync(0xffffffffU,value,d);value=maximum?fmaxf(value,other):value+other;}
  if(!threadIdx.x)scratch[0]=value;__syncthreads();
  // Every warp must consume the broadcast before a later reduction can reuse
  // scratch. A barrier before the load alone permits a fast warp to overwrite it.
  const float result=scratch[0];__syncthreads();return result;
}
template<bool Batch> __device__ __forceinline__ float KV(const __nv_bfloat16* cache,const __nv_bfloat16* current,int absolute,int base,int head,int d,bool local,bool value){
  if constexpr(Batch)if(absolute>=base)return __bfloat162float(current[((value?kSimtBatch:0)+absolute-base)*512+(head/8)*128+d]);
  return __bfloat162float(cache[(local?(absolute&1023):absolute)*512+(head/8)*128+d]);
}
// Each CTA shares 64 keys across four GQA heads (and eight Prefill queries).
template<bool Batch> __global__ void QK(const float* q,const __nv_bfloat16* keys,const __nv_bfloat16* current,float* workspace,const SimtBatch* batch,const SimtCommand* command,int capacity,bool local,int first){
  const int base=Batch?batch->base:0,rows=Batch?batch->rows:1;
  const int group=threadIdx.x/8,lane=threadIdx.x&7,row=Batch?first+group/4:0,head=blockIdx.y*4+group%4;
  const int pos=Batch?base+row:command->position;
  const int first_pos=Batch?base+first:command->position;
  const int common_start=local?max(0,first_pos-1023):0,key_first=common_start+blockIdx.x*64;
  const int last=Batch?base+min(rows-1,first+7):command->position;
  if(first>=rows || key_first>last)return;
  __shared__ __nv_bfloat16 tile[64][136];
  for(int i=threadIdx.x;i<64*128;i+=256){const int absolute=key_first+i/128,d=i%128;
    tile[i/128][d]=absolute>last?__float2bfloat16_rn(0.f):__float2bfloat16_rn(KV<Batch>(keys,current,absolute,base,head,d,local,false));}
  __syncthreads();if(row>=rows)return;
  // Decode has 8 independent eight-lane groups per head; each handles eight keys.
  const int key_step=Batch?1:8,key_begin=Batch?0:group/4;
  const int start=local?max(0,pos-1023):0;
  float query[16];
  #pragma unroll
  for(int i=0;i<16;++i)query[i]=q[row*4096+head*128+lane+i*8];
  const unsigned mask=0xffu<<(threadIdx.x&24);
  float* scores=workspace+((Batch?(row-first)*32:0)+head)*capacity;
  for(int j=key_begin;j<64;j+=key_step){const int absolute=key_first+j;if(absolute<start || absolute>pos)continue;
    float dot=0;
    #pragma unroll
    for(int i=0;i<16;++i)dot=__fmaf_rn(query[i],__bfloat162float(tile[j][lane+i*8]),dot);
    for(int d=4;d;d/=2)dot+=__shfl_down_sync(mask,dot,d,8);
    if(!lane)scores[absolute-start]=B(B(dot)*.08838834764831845f);
  }
}
// Parallelize PV over eight warps. Keep the model's BF16 score/probability
// rounding boundaries; only accumulation order and exp precision change.
template<bool Batch> __global__ void SoftmaxPV(const __nv_bfloat16* values,const __nv_bfloat16* current,float* workspace,float* output,const SimtBatch* batch,const SimtCommand* command,int capacity,bool local,int first,int* bad){
  const int row=Batch?first+blockIdx.y:0,base=Batch?batch->base:0;if constexpr(Batch)if(row>=batch->rows)return;
  const int pos=Batch?base+row:command->position,start=local?max(0,pos-1023):0,n=pos-start+1,h=blockIdx.x;
  float* scores=workspace+((Batch?blockIdx.y*32:0)+h)*capacity;
  __shared__ float scratch[16],partial[8][128],total;float maximum=-INFINITY;
  for(int j=threadIdx.x;j<n;j+=256)maximum=fmaxf(maximum,scores[j]);
  maximum=Reduce(maximum,scratch,true);float sum=0,error=0;
  for(int j=threadIdx.x;j<n;j+=256){const float p=expf(scores[j]-maximum);scores[j]=p;Accumulate(p,sum,error);}
  sum=AttentionAccurateReduce8(sum,error,scratch);if(!threadIdx.x)total=sum;__syncthreads();
  for(int j=threadIdx.x;j<n;j+=256)scores[j]=B(scores[j]/total);
  __syncthreads();const int warp=threadIdx.x/32,lane=threadIdx.x&31;float accum[4]={};
  for(int j=warp;j<n;j+=8){const float probability=scores[j];
    #pragma unroll
    for(int d=0;d<4;++d)accum[d]=__fmaf_rn(probability,KV<Batch>(values,current,start+j,base,h,lane+d*32,local,true),accum[d]);}
  #pragma unroll
  for(int d=0;d<4;++d)partial[warp][lane+d*32]=accum[d];
  __syncthreads();if(threadIdx.x<128){float value=0;
    #pragma unroll
    for(int w=0;w<8;++w)value+=partial[w][threadIdx.x];
    output[row*4096+h*128+threadIdx.x]=B(value);Finite(value,bad);}
}
__global__ void SoftmaxDecode(float* workspace,const SimtCommand* command,int capacity,bool local){
  const int n=local?min(command->position+1,1024):command->position+1;
  float* scores=workspace+blockIdx.x*capacity;
  __shared__ float scratch[16],total;float maximum=-INFINITY;
  for(int j=threadIdx.x;j<n;j+=256)maximum=fmaxf(maximum,scores[j]);
  maximum=Reduce(maximum,scratch,true);float sum=0,error=0;
  for(int j=threadIdx.x;j<n;j+=256){const float p=expf(scores[j]-maximum);scores[j]=p;Accumulate(p,sum,error);}
  sum=AttentionAccurateReduce8(sum,error,scratch);if(!threadIdx.x)total=sum;__syncthreads();
  for(int j=threadIdx.x;j<n;j+=256)scores[j]=B(scores[j]/total);
}
__global__ void PartialPV(const __nv_bfloat16* values,const float* workspace,float* output,const SimtCommand* command,int capacity,bool local){
  const int pos=command->position,start=local?max(0,pos-1023):0,n=pos-start+1,h=blockIdx.x;
  const int part=blockIdx.y,warp=threadIdx.x/32,lane=threadIdx.x&31;
  const float* scores=workspace+h*capacity;float accum[4]={};__shared__ float partial[8][128];
  for(int j=part*8+warp;j<n;j+=32){const float probability=scores[j];
    #pragma unroll
    for(int d=0;d<4;++d)accum[d]=__fmaf_rn(probability,KV<false>(values,nullptr,start+j,0,h,lane+d*32,local,true),accum[d]);}
  #pragma unroll
  for(int d=0;d<4;++d)partial[warp][lane+d*32]=accum[d];
  __syncthreads();if(threadIdx.x<128){float value=0;
    #pragma unroll
    for(int w=0;w<8;++w)value+=partial[w][threadIdx.x];
    output[part*4096+h*128+threadIdx.x]=value;}
}
__global__ void MergePV(const float* partial,float* output,int* bad){
  const int i=blockIdx.x*128+threadIdx.x;float sum=0;
  #pragma unroll
  for(int part=0;part<4;++part)sum+=partial[part*4096+i];
  output[i]=B(sum);Finite(sum,bad);
}
inline void Decode(const float* q,const __nv_bfloat16* k,const __nv_bfloat16* v,float* out,const SimtCommand* command,int capacity,bool local,float* scores,int* bad,cudaStream_t stream,int maximum_keys,float* partial=nullptr){
  const int keys=local?1024:min(capacity,maximum_keys);
  QK<false><<<dim3((keys+63)/64,8),256,0,stream>>>(q,k,nullptr,scores,nullptr,command,capacity,local,0);
  if(partial){
    SoftmaxDecode<<<32,256,0,stream>>>(scores,command,capacity,local);
    PartialPV<<<dim3(32,4),256,0,stream>>>(v,scores,partial,command,capacity,local);
    MergePV<<<32,128,0,stream>>>(partial,out,bad);
  }else SoftmaxPV<false><<<32,256,0,stream>>>(v,nullptr,scores,out,nullptr,command,capacity,local,0,bad);
}
inline void Prefill(const float* q,const __nv_bfloat16* k,const __nv_bfloat16* v,float* out,const SimtBatch* batch,int capacity,bool local,float* scores,int* bad,const __nv_bfloat16* current,int first,cudaStream_t stream){
  const int keys=local?1031:capacity;
  QK<true><<<dim3((keys+63)/64,8),256,0,stream>>>(q,k,current,scores,batch,nullptr,capacity,local,first);
  SoftmaxPV<true><<<dim3(32,8),256,0,stream>>>(v,current,scores,out,batch,nullptr,capacity,local,first,bad);
}
}
