// Multiwarp shared staging adapted from gem16 c01789a2 attention kernels.
// Keep Mellum's per-output K16 partial order, absolute envelope and repair.
__device__ __forceinline__ float4 AttentionLoadKV4(const __nv_bfloat16* cache,const float* chunk,int key,int base,int end,int cap,int feature){
  if(key>=end)return make_float4(0,0,0,0);
  if(key>=base)return *reinterpret_cast<const float4*>(chunk+(key-base)*512+feature);
  const uint2 bits=*reinterpret_cast<const uint2*>(cache+(key%cap)*512+feature);
  return make_float4(__bfloat162float(__ushort_as_bfloat16(bits.x&65535)),__bfloat162float(__ushort_as_bfloat16(bits.x>>16)),__bfloat162float(__ushort_as_bfloat16(bits.y&65535)),__bfloat162float(__ushort_as_bfloat16(bits.y>>16)));
}
template<bool Local>
__global__ void AttentionQKStaged(const float* q,const __nv_bfloat16* kc,const float* chunkK,float* scores,int base,int tokens,int cap,int pitch,int* bad){
  if(!q || !kc || !chunkK || !scores || !bad || !AttentionTCShape(base,tokens,cap,Local,pitch) ||
     (reinterpret_cast<std::uintptr_t>(q)&15) || (reinterpret_cast<std::uintptr_t>(kc)&7) || (reinterpret_cast<std::uintptr_t>(chunkK)&15) ||
     blockDim.x!=256 || blockDim.y!=1 || blockDim.z!=1 || blockIdx.z>=32 || blockIdx.y>=static_cast<unsigned>((tokens+63)/64) || blockIdx.x>=static_cast<unsigned>(pitch/16)){
    if(threadIdx.x==0 && bad)atomicExch(bad,1);return;
  }
  __shared__ float qs[64*132],ks[16*132];
  const int warp=threadIdx.x/32,lane=threadIdx.x%32,group=lane/4,tid=lane%4,head=blockIdx.z;
  const int rowbase=blockIdx.y*64,colbase=blockIdx.x*16,first=Local?max(0,base-1023):0,end=base+tokens,kvhead=head/8;
  for(int i=threadIdx.x;i<64*32;i+=256){const int row=i/32,d=(i%32)*4;const float4 v=rowbase+row<tokens?*reinterpret_cast<const float4*>(q+((rowbase+row)*32+head)*128+d):make_float4(0,0,0,0);
    qs[row*132+d]=v.x;qs[row*132+d+1]=v.y;qs[row*132+d+2]=v.z;qs[row*132+d+3]=v.w;}
  for(int i=threadIdx.x;i<16*32;i+=256){const int key=i/32,d=(i%32)*4;const float4 v=AttentionLoadKV4(kc,chunkK,first+colbase+key,base,end,cap,kvhead*128+d);
    ks[key*132+d]=v.x;ks[key*132+d+1]=v.y;ks[key*132+d+2]=v.z;ks[key*132+d+3]=v.w;}
  __syncthreads();
  const int warpRow=(warp/2)*16,warpCol=(warp%2)*8;
  float sum[4]={},error[4]={},absoluteSum[4]={};
  for(int k=0;k<128;k+=16){unsigned a[4],b[2];
    #pragma unroll
    for(int j=0;j<4;++j){const int row=warpRow+group+(j%2)*8,d=k+tid*2+(j/2)*8;a[j]=AttentionPackBF16(qs[row*132+d],qs[row*132+d+1]);}
    #pragma unroll
    for(int j=0;j<2;++j){const int key=warpCol+group,d=k+tid*2+j*8;b[j]=AttentionPackBF16(ks[key*132+d],ks[key*132+d+1]);}
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
  for(int j=0;j<4;++j){const int relativeRow=warpRow+group+(j/2)*8,row=rowbase+relativeRow,relativeCol=warpCol+tid*2+j%2,col=colbase+relativeCol,key=first+col;
    if(row<tokens){const int pos=base+row;const bool valid=key<=pos && (!Local || key>=max(0,pos-1023));float dot=__fadd_rn(sum[j],error[j]);
      if(valid && AttentionNeedsRepair(dot,absoluteSum[j])){float exact=0,low=0;
        for(int d=0;d<128;++d)Accumulate(__fmul_rn(qs[relativeRow*132+d],ks[relativeCol*132+d]),exact,low);dot=__fadd_rn(exact,low);}
      const float score=valid?B(__fmul_rn(B(dot),.08838834764831845f)):-INFINITY;scores[(row*32+head)*pitch+col]=score;if(valid)Finite(score,bad);}}
}
__device__ __forceinline__ void AttentionStagePV(__nv_bfloat16* ps,float* vs,const __nv_bfloat16* probabilities,const __nv_bfloat16* vc,const float* chunkV,int rowbase,int colbase,int head,int k,int first,int base,int end,int cap,int pitch,int tokens){
  for(int i=threadIdx.x;i<16*8;i+=256){const int row=i/8,key=(i%8)*2;const unsigned packed=rowbase+row<tokens?*reinterpret_cast<const unsigned*>(probabilities+((rowbase+row)*32+head)*pitch+k+key):0;
    *reinterpret_cast<unsigned*>(ps+row*24+key)=packed;}
  for(int i=threadIdx.x;i<16*16;i+=256){const int key=i/16,d=(i%16)*4;const float4 v=AttentionLoadKV4(vc,chunkV,first+k+key,base,end,cap,(head/8)*128+colbase+d);
    vs[key*68+d]=v.x;vs[key*68+d+1]=v.y;vs[key*68+d+2]=v.z;vs[key*68+d+3]=v.w;}
}
template<bool Local>
__global__ void AttentionPVStaged(const __nv_bfloat16* probabilities,const __nv_bfloat16* vc,const float* chunkV,float* output,int base,int tokens,int cap,int pitch,int* bad){
  if(!probabilities || !vc || !chunkV || !output || !bad || !AttentionTCShape(base,tokens,cap,Local,pitch) ||
     (reinterpret_cast<std::uintptr_t>(probabilities)&3) || (reinterpret_cast<std::uintptr_t>(vc)&7) || (reinterpret_cast<std::uintptr_t>(chunkV)&15) ||
     blockDim.x!=256 || blockDim.y!=1 || blockDim.z!=1 || blockIdx.x>=2 || blockIdx.y>=static_cast<unsigned>((tokens+15)/16) || blockIdx.z>=32){
    if(threadIdx.x==0 && bad)atomicExch(bad,1);return;
  }
  __shared__ __nv_bfloat16 ps[16*24];__shared__ float vs[16*68];
  const int lane=threadIdx.x%32,warp=threadIdx.x/32,group=lane/4,tid=lane%4,head=blockIdx.z,rowbase=blockIdx.y*16,colbase=blockIdx.x*64;
  const int first=Local?max(0,base-1023):0,end=base+tokens;
  float sum[4]={},error[4]={},absoluteSum[4]={};
  for(int k=0;k<pitch;k+=16){
    AttentionStagePV(ps,vs,probabilities,vc,chunkV,rowbase,colbase,head,k,first,base,end,cap,pitch,tokens);__syncthreads();
    unsigned a[4],b[2];
    #pragma unroll
    for(int j=0;j<4;++j){const int row=group+(j%2)*8,key=tid*2+(j/2)*8;a[j]=*reinterpret_cast<const unsigned*>(ps+row*24+key);}
    #pragma unroll
    for(int j=0;j<2;++j){const int key=tid*2+j*8,col=warp*8+group;b[j]=AttentionPackBF16(vs[key*68+col],vs[(key+1)*68+col]);}
    float partial[4]={},magnitudes[4]={};AttentionMmaBF16(a,b,partial);
    #pragma unroll
    for(int j=0;j<4;++j)a[j]&=0x7fff7fffu;
    #pragma unroll
    for(int j=0;j<2;++j)b[j]&=0x7fff7fffu;
    AttentionMmaBF16(a,b,magnitudes);
    #pragma unroll
    for(int j=0;j<4;++j){Accumulate(partial[j],sum[j],error[j]);absoluteSum[j]=__fadd_rn(absoluteSum[j],magnitudes[j]);}
    __syncthreads();
  }
  bool repair[4];bool any=false;float exact[4]={},low[4]={};
  #pragma unroll
  for(int j=0;j<4;++j){const int row=rowbase+group+(j/2)*8;repair[j]=row<tokens && AttentionNeedsRepair(__fadd_rn(sum[j],error[j]),absoluteSum[j]);any|=repair[j];}
  // Every CTA thread participates in staging/barriers, including query tails.
  if(__syncthreads_or(any))for(int k=0;k<pitch;k+=16){
    AttentionStagePV(ps,vs,probabilities,vc,chunkV,rowbase,colbase,head,k,first,base,end,cap,pitch,tokens);__syncthreads();
    for(int i=0;i<16;++i){const int key=first+k+i;
      #pragma unroll
      for(int j=0;j<4;++j){const int relativeRow=group+(j/2)*8,pos=base+rowbase+relativeRow,start=Local?max(0,pos-1023):0,col=warp*8+tid*2+j%2;
        if(repair[j] && key>=start && key<=pos)Accumulate(__fmul_rn(__bfloat162float(ps[relativeRow*24+i]),vs[i*68+col]),exact[j],low[j]);}}
    __syncthreads();
  }
  #pragma unroll
  for(int j=0;j<4;++j){const int row=rowbase+group+(j/2)*8,col=colbase+warp*8+tid*2+j%2;
    if(row<tokens){const float value=B(repair[j]?__fadd_rn(exact[j],low[j]):__fadd_rn(sum[j],error[j]));output[(row*32+head)*128+col]=value;Finite(value,bad);}}
}
