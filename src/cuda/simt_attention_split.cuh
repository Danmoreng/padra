#pragma once
// Exact QK / softmax / PV split. No changed exponential or summation tree.
namespace simt_attention_split {
__global__ void QK(const float* q,const __nv_bfloat16* keys,float* scores,const SimtCommand* command,int capacity,bool local,int* bad){
  const int pos=command->position,start=local?max(0,pos-1023):0,n=pos-start+1,first=blockIdx.x*64;
  if(first>=n)return;if(pos<0 || (!local && pos>=capacity)){if(!threadIdx.x)atomicExch(bad,1);return;}
  const int head=blockIdx.y*4+threadIdx.x/64,kv_head=head/8,lane=threadIdx.x&7,group=(threadIdx.x&63)/8;
  // Pad the BF16 row so adjacent eight-lane key groups use different banks.
  __shared__ __nv_bfloat16 tile[64][136];
  for(int i=threadIdx.x;i<64*128;i+=256){const int j=i/128,d=i%128;tile[j][d]=first+j<n?keys[(local?((start+first+j)&1023):(start+first+j))*512+kv_head*128+d]:__float2bfloat16_rn(0.f);}
  __syncthreads();float query[16];
  #pragma unroll
  for(int i=0;i<16;++i)query[i]=q[head*128+lane+i*8];
  const unsigned mask=0xffu<<(threadIdx.x&24);
  for(int j=group;j<64 && first+j<n;j+=8){float dot=0,error=0;
    #pragma unroll
    for(int i=0;i<16;++i)Accumulate(__fmul_rn(query[i],__bfloat162float(tile[j][lane+i*8])),dot,error);
    for(int d=4;d;d/=2){const float other=__shfl_down_sync(mask,dot,d,8),low=__shfl_down_sync(mask,error,d,8);if(lane+d<8){Accumulate(other,dot,error);Accumulate(low,dot,error);}}
    if(!lane)scores[head*capacity+first+j]=B(B(__fadd_rn(dot,error))*.08838834764831845f);
  }
}
__global__ void Softmax(float* workspace,const SimtCommand* command,int capacity,bool local){
  const int pos=command->position,n=local?min(pos+1,1024):pos+1,h=blockIdx.x;
  float* scores=workspace+h*capacity;const int lane=threadIdx.x&7,group=threadIdx.x/8;
  __shared__ float scratch[16],maximum,total;float mx=-INFINITY;
  if(!lane)for(int j=group;j<n;j+=32)mx=fmaxf(mx,scores[j]);
  mx=AttentionMaxReduce8(mx,scratch);if(!threadIdx.x)maximum=mx;__syncthreads();
  float sum=0,error=0;if(threadIdx.x<128)for(int j=threadIdx.x;j<n;j+=128){scores[j]=ExpRN(__fsub_rn(scores[j],maximum));Accumulate(scores[j],sum,error);}
  sum=AttentionAccurateReduce8(sum,error,scratch);if(!threadIdx.x)total=sum;__syncthreads();
  if(threadIdx.x<128)for(int j=threadIdx.x;j<n;j+=128)scores[j]=B(__fdiv_rn(scores[j],total));
}
__global__ void PV(const __nv_bfloat16* values,const float* workspace,float* output,const SimtCommand* command,int capacity,bool local,int* bad){
  const int pos=command->position,start=local?max(0,pos-1023):0,n=pos-start+1,h=blockIdx.x,d=threadIdx.x;
  const float* scores=workspace+h*capacity;float probability[8],value[8];
  #pragma unroll
  for(int i=0;i<8;++i){probability[i]=i<n?scores[i]:0.f;value[i]=i<n?__bfloat162float(values[(local?((start+i)&1023):(start+i))*512+(h/8)*128+d]):0.f;}
  float sum=0,error=0;
  for(int base=0;base<n;base+=8){
    #pragma unroll
    for(int i=0;i<8;++i){const float p=probability[i],v=value[i];const int next=base+i+8;if(next<n){probability[i]=scores[next];value[i]=__bfloat162float(values[(local?((start+next)&1023):(start+next))*512+(h/8)*128+d]);}if(base+i<n)Accumulate(__fmul_rn(p,v),sum,error);}
  }
  output[h*128+d]=B(__fadd_rn(sum,error));Finite(sum,bad);
}
inline void Launch(const float* q,const __nv_bfloat16* k,const __nv_bfloat16* v,float* output,const SimtCommand* command,int capacity,bool local,float* scores,int* bad,cudaStream_t stream,int maximum_keys=0){
  const int keys=maximum_keys?min(capacity,maximum_keys):capacity;
  QK<<<dim3((keys+63)/64,8),256,0,stream>>>(q,k,scores,command,capacity,local,bad);
  Softmax<<<32,256,0,stream>>>(scores,command,capacity,local);
  PV<<<32,128,0,stream>>>(v,scores,output,command,capacity,local,bad);
}
}
