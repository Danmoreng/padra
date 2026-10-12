// BF16 fragment/MMA mechanism adapted from gem16 c01789a2 (Apache-2.0).
// Explicit Mellum experiment: K16 partial dots, compensated partial merge;
// original BF16 score/probability boundaries and RNExp softmax retained.
__device__ __forceinline__ unsigned AttentionPackBF16(float a,float b){
  return static_cast<unsigned>(__bfloat16_as_ushort(__float2bfloat16_rn(a))) |
         (static_cast<unsigned>(__bfloat16_as_ushort(__float2bfloat16_rn(b)))<<16);
}
__device__ __forceinline__ void AttentionMmaBF16(const unsigned* a,const unsigned* b,float* c){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
__device__ __forceinline__ bool AttentionTCShape(int base,int tokens,int cap,bool local,int pitch){
  if(base<0 || tokens<1 || tokens>512 || cap<1 || cap>4096 || base>4096-tokens)return false;
  if(local ? cap<min(1024,base+tokens) : cap<base+tokens)return false;
  const int first=local?max(0,base-1023):0;
  return pitch==((base+tokens-first+15)/16)*16;
}
__device__ __forceinline__ float AttentionQuery(const float* q,int row,int head,int d,int tokens){
  return row<tokens?q[(row*32+head)*128+d]:0.f;
}
__device__ __forceinline__ float AttentionKV(const __nv_bfloat16* cache,const float* chunk,int key,int base,int end,int cap,int feature){
  if(key>=end)return 0.f;
  return key>=base?chunk[(key-base)*512+feature]:__bfloat162float(cache[(key%cap)*512+feature]);
}
// Conservative FP32 roundoff envelope. Only ambiguous BF16 rounding bins
// are recomputed with the admitted scalar RN product/compensated order.
// Tiny values also use that order, avoiding hardware denormal assumptions.
__device__ __forceinline__ bool AttentionNeedsRepair(float dot,float absolute_sum){
  if(!isfinite(dot) || !isfinite(absolute_sum) || fabsf(dot)<0x1p-100f)return true;
  const float rounded=B(dot);const unsigned bits=__float_as_uint(rounded);
  const float low=__uint_as_float(bits+((bits>>31)?0x10000u:static_cast<unsigned>(-0x10000)));
  const float high=__uint_as_float(bits+((bits>>31)?static_cast<unsigned>(-0x10000):0x10000u));
  const float lower_mid=__fadd_rn(rounded,__fmul_rn(__fsub_rn(low,rounded),.5f));
  const float upper_mid=__fadd_rn(rounded,__fmul_rn(__fsub_rn(high,rounded),.5f));
  const float distance=fminf(fabsf(__fsub_rn(dot,lower_mid)),fabsf(__fsub_rn(dot,upper_mid)));
  return distance<=__fmul_rn(absolute_sum,0x1p-17f);
}
template<bool Local>
__global__ void AttentionTCQK(const float* q,const __nv_bfloat16* kc,const float* chunkK,float* scores,int base,int tokens,int cap,int pitch,int* bad){
  if(!q || !kc || !chunkK || !scores || !bad || !AttentionTCShape(base,tokens,cap,Local,pitch) ||
     blockDim.x!=32 || blockDim.y!=1 || blockDim.z!=1 || blockIdx.z>=32 || blockIdx.y>=static_cast<unsigned>((tokens+15)/16) || blockIdx.x>=static_cast<unsigned>(pitch/8)){
    if(threadIdx.x==0 && bad)atomicExch(bad,1);return;
  }
  const int group=threadIdx.x/4,tid=threadIdx.x%4,head=blockIdx.z,rowbase=blockIdx.y*16,colbase=blockIdx.x*8;
  const int first=Local?max(0,base-1023):0,end=base+tokens,kvhead=head/8;
  float sum[4]={},error[4]={},absoluteSum[4]={};
  for(int k=0;k<128;k+=16){
    unsigned a[4],b[2];
    #pragma unroll
    for(int j=0;j<4;++j){const int row=rowbase+group+(j%2)*8,d=k+tid*2+(j/2)*8;
      a[j]=AttentionPackBF16(AttentionQuery(q,row,head,d,tokens),AttentionQuery(q,row,head,d+1,tokens));}
    #pragma unroll
    for(int j=0;j<2;++j){const int key=first+colbase+group,d=k+tid*2+j*8;
      b[j]=AttentionPackBF16(AttentionKV(kc,chunkK,key,base,end,cap,kvhead*128+d),AttentionKV(kc,chunkK,key,base,end,cap,kvhead*128+d+1));}
    float partial[4]={},magnitudes[4]={};AttentionMmaBF16(a,b,partial);
    #pragma unroll
    for(int j=0;j<4;++j)a[j]&=0x7fff7fffu;
    #pragma unroll
    for(int j=0;j<2;++j)b[j]&=0x7fff7fffu;
    AttentionMmaBF16(a,b,magnitudes);
    #pragma unroll
    for(int j=0;j<4;++j){Accumulate(partial[j],sum[j],error[j]);absoluteSum[j]=__fadd_rn(absoluteSum[j],magnitudes[j]);}
  }
  #pragma unroll
  for(int j=0;j<4;++j){const int row=rowbase+group+(j/2)*8,col=colbase+tid*2+j%2,key=first+col;
    if(row<tokens){const int pos=base+row;const bool valid=key<=pos && (!Local || key>=max(0,pos-1023));
      float dot=__fadd_rn(sum[j],error[j]);
      if(valid && AttentionNeedsRepair(dot,absoluteSum[j])){float exact=0,low=0;
        for(int d=0;d<128;++d)Accumulate(__fmul_rn(q[(row*32+head)*128+d],AttentionKV(kc,chunkK,key,base,end,cap,kvhead*128+d)),exact,low);
        dot=__fadd_rn(exact,low);}
      const float score=valid?B(__fmul_rn(B(dot),.08838834764831845f)):-INFINITY;
      scores[(row*32+head)*pitch+col]=score;if(valid)Finite(score,bad);}}
}
template<bool Local>
__global__ void AttentionTCSoftmax(float* scores,__nv_bfloat16* probabilities,int base,int tokens,int cap,int pitch,int* bad){
  if(!scores || !probabilities || !bad || !AttentionTCShape(base,tokens,cap,Local,pitch) || blockDim.x!=128 || blockDim.y!=1 || blockDim.z!=1 || blockIdx.x>=32 || blockIdx.y>=static_cast<unsigned>(tokens) || blockIdx.z!=0){
    if(threadIdx.x==0 && bad)atomicExch(bad,1);return;
  }
  const int row=blockIdx.y,head=blockIdx.x,pos=base+row,first=Local?max(0,base-1023):0,start=Local?max(0,pos-1023):0,n=pos-start+1;
  float* s=scores+(row*32+head)*pitch;auto* p=probabilities+(row*32+head)*pitch;
  __shared__ float scratch[8];__shared__ float maximum,total;
  float mx=-INFINITY;
  for(int j=threadIdx.x;j<n;j+=128)mx=fmaxf(mx,s[start-first+j]);
  mx=MaxReduce(mx,scratch);if(threadIdx.x==0)maximum=mx;__syncthreads();
  float sum=0,error=0;
  for(int j=threadIdx.x;j<n;j+=128){const int key=start-first+j;s[key]=ExpRN(__fsub_rn(s[key],maximum));Accumulate(s[key],sum,error);}
  sum=AccurateReduce(sum,error,scratch);if(threadIdx.x==0){total=sum;Finite(sum,bad);}__syncthreads();
  for(int j=threadIdx.x;j<pitch;j+=128)p[j]=__float2bfloat16_rn(0.f);
  __syncthreads();
  for(int j=threadIdx.x;j<n;j+=128)p[start-first+j]=__float2bfloat16_rn(__fdiv_rn(s[start-first+j],total));
}
template<bool Local>
__global__ void AttentionTCPV(const __nv_bfloat16* probabilities,const __nv_bfloat16* vc,const float* chunkV,float* output,int base,int tokens,int cap,int pitch,int* bad){
  if(!probabilities || !vc || !chunkV || !output || !bad || !AttentionTCShape(base,tokens,cap,Local,pitch) || blockDim.x!=32 || blockDim.y!=1 || blockDim.z!=1 || blockIdx.x>=16 || blockIdx.y>=static_cast<unsigned>((tokens+15)/16) || blockIdx.z>=32){
    if(threadIdx.x==0 && bad)atomicExch(bad,1);return;
  }
  const int group=threadIdx.x/4,tid=threadIdx.x%4,head=blockIdx.z,rowbase=blockIdx.y*16,colbase=blockIdx.x*8;
  const int first=Local?max(0,base-1023):0,end=base+tokens,kvhead=head/8;
  float sum[4]={},error[4]={},absoluteSum[4]={};
  for(int k=0;k<pitch;k+=16){
    unsigned a[4],b[2];
    #pragma unroll
    for(int j=0;j<4;++j){const int row=rowbase+group+(j%2)*8,key=k+tid*2+(j/2)*8;
      a[j]=row<tokens?static_cast<unsigned>(__bfloat16_as_ushort(probabilities[(row*32+head)*pitch+key])) | (static_cast<unsigned>(__bfloat16_as_ushort(probabilities[(row*32+head)*pitch+key+1]))<<16):0;}
    #pragma unroll
    for(int j=0;j<2;++j){const int feature=kvhead*128+colbase+group,key=first+k+tid*2+j*8;
      b[j]=AttentionPackBF16(AttentionKV(vc,chunkV,key,base,end,cap,feature),AttentionKV(vc,chunkV,key+1,base,end,cap,feature));}
    float partial[4]={},magnitudes[4]={};AttentionMmaBF16(a,b,partial);
    #pragma unroll
    for(int j=0;j<4;++j)a[j]&=0x7fff7fffu;
    #pragma unroll
    for(int j=0;j<2;++j)b[j]&=0x7fff7fffu;
    AttentionMmaBF16(a,b,magnitudes);
    #pragma unroll
    for(int j=0;j<4;++j){Accumulate(partial[j],sum[j],error[j]);absoluteSum[j]=__fadd_rn(absoluteSum[j],magnitudes[j]);}
  }
  #pragma unroll
  for(int j=0;j<4;++j){const int row=rowbase+group+(j/2)*8,col=colbase+tid*2+j%2;
    if(row<tokens){float dot=__fadd_rn(sum[j],error[j]);
      if(AttentionNeedsRepair(dot,absoluteSum[j])){float exact=0,low=0;const int pos=base+row,start=Local?max(0,pos-1023):0;
        for(int key=start;key<=pos;++key)Accumulate(__fmul_rn(__bfloat162float(probabilities[(row*32+head)*pitch+key-first]),AttentionKV(vc,chunkV,key,base,end,cap,kvhead*128+col)),exact,low);
        dot=__fadd_rn(exact,low);}
      const float value=B(dot);output[(row*32+head)*128+col]=value;Finite(value,bad);}}
}
#include "attention_staged.cuh"
void LaunchAttentionTC(const float* q,const __nv_bfloat16* kc,const __nv_bfloat16* vc,const float* chunkK,const float* chunkV,float* output,float* scores,__nv_bfloat16* probabilities,int base,int tokens,int cap,bool local,int* bad,cudaStream_t stream){
  if(!q || !kc || !vc || !chunkK || !chunkV || !output || !scores || !probabilities || !bad || base<0 || tokens<1 || tokens>512 || base>4096-tokens || cap<1 || cap>4096 || (local?cap<std::min(1024,base+tokens):cap<base+tokens))throw std::runtime_error("BF16 TC prefill attention shape/binding unsupported");
  const int first=local?std::max(0,base-1023):0,pitch=((base+tokens-first+15)/16)*16;
  if(local){
    AttentionQKStaged<true><<<dim3(pitch/16,(tokens+63)/64,32),256,0,stream>>>(q,kc,chunkK,scores,base,tokens,cap,pitch,bad);
    AttentionTCSoftmax<true><<<dim3(32,tokens),128,0,stream>>>(scores,probabilities,base,tokens,cap,pitch,bad);
    AttentionPVStaged<true><<<dim3(2,(tokens+15)/16,32),256,0,stream>>>(probabilities,vc,chunkV,output,base,tokens,cap,pitch,bad);
  }else{
    AttentionQKStaged<false><<<dim3(pitch/16,(tokens+63)/64,32),256,0,stream>>>(q,kc,chunkK,scores,base,tokens,cap,pitch,bad);
    AttentionTCSoftmax<false><<<dim3(32,tokens),128,0,stream>>>(scores,probabilities,base,tokens,cap,pitch,bad);
    AttentionPVStaged<false><<<dim3(2,(tokens+15)/16,32),256,0,stream>>>(probabilities,vc,chunkV,output,base,tokens,cap,pitch,bad);
  }
}
