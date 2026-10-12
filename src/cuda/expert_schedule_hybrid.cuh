// E2 device-only full64/M16-tail partition. Gem16 c01789a2 hybrid schedule
// pattern (Apache-2.0), adapted to Mellum's stable 64-expert slot groups.
// No Trellis codes/math; unlike the donor, a partial64 block is never broad.
#pragma once
namespace mellum16::expert_scaled_bf16_layer {
struct Tail {int expert,begin;};
__host__ __device__ constexpr int WideCapacity(int tokens){return tokens*8/64;}
__host__ __device__ constexpr int TailCapacity(int tokens){
  return (tokens*8+15)/16+63<64*((tokens+15)/16)?(tokens*8+15)/16+63:64*((tokens+15)/16);
}
// C<=128 bypasses this kernel entirely at graph construction. Explicit
// capacities allow bounded overflow tests; descriptor writes follow validation.
__global__ void DescribeFull64Tail16(const int* counts,Tile* wide,Tail* tail,
    int* wideCount,int* tailCount,int tokens,int wideCap,int tailCap,int* bad){
  if(!bad)return;
  if(!counts||!wide||!tail||!wideCount||!tailCount||tokens<=128||tokens>2048||
      wideCap<0||wideCap>WideCapacity(tokens)||tailCap<0||tailCap>TailCapacity(tokens)||
      blockDim.x!=64||blockDim.y!=1||blockDim.z!=1||
      gridDim.x!=1||gridDim.y!=1||gridDim.z!=1){
    if(threadIdx.x==0)Error(bad);return;
  }
  __shared__ int wo[65],to[65],valid;
  const int e=threadIdx.x;
  if(e==0){
    int nw=0,nt=0,sum=0;valid=*bad==0;
    for(int j=0;j<64;++j){
      wo[j]=nw;to[j]=nt;
      const int c=counts[j];
      if(c<0||c>tokens){valid=0;continue;}
      sum+=c;nw+=c/64;nt+=(c%64+15)/16;
    }
    wo[64]=nw;to[64]=nt;
    if(sum!=tokens*8||nw>wideCap||nt>tailCap)valid=0;
    *wideCount=valid?nw:0;*tailCount=valid?nt:0;
    if(!valid)Error(bad);
  }
  __syncthreads();
  if(!valid)return;
  const int full=counts[e]/64;
  for(int j=wo[e];j<wo[e+1];++j)wide[j]={e,(j-wo[e])*64,64};
  for(int j=to[e];j<to[e+1];++j)tail[j]={e,full*64+(j-to[e])*16};
}
} // namespace mellum16::expert_scaled_bf16_layer
