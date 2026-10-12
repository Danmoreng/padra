// Hierarchical candidate reduction, following gem16 output_head.cu scheduling.
// Mellum logits/head arithmetic is unchanged. See NOTICE.
#pragma once
#include <climits>
constexpr int kSelectionThreads=256;
constexpr int kSelectionTile=1024;
constexpr int kVocabulary=98304;
constexpr int kSelectionBlocks=(kVocabulary+kSelectionTile-1)/kSelectionTile;
struct SelectionCandidate {float value;int token;};
static_assert(sizeof(SelectionCandidate)==8);
__device__ SelectionCandidate Better(SelectionCandidate a,SelectionCandidate b){
  return b.value>a.value || (b.value==a.value && b.token<a.token)?b:a;
}
__device__ SelectionCandidate WarpBest(SelectionCandidate best){
  for(int distance=16;distance;distance/=2){
    SelectionCandidate other{__shfl_down_sync(0xffffffff,best.value,distance),
                             __shfl_down_sync(0xffffffff,best.token,distance)};
    if((threadIdx.x&31)+distance<32)best=Better(best,other);
  }
  return best;
}
__device__ SelectionCandidate BlockBest(SelectionCandidate best,SelectionCandidate* scratch){
  best=WarpBest(best);
  if((threadIdx.x&31)==0)scratch[threadIdx.x/32]=best;
  __syncthreads();
  if(threadIdx.x<32){
    best=threadIdx.x<kSelectionThreads/32?scratch[threadIdx.x]:SelectionCandidate{-INFINITY,INT_MAX};
    best=WarpBest(best);
  }
  return best;
}
__global__ void SelectTiles(const float* logits,SelectionCandidate* candidates,int count,int* bad){
  __shared__ SelectionCandidate scratch[kSelectionThreads/32];
  SelectionCandidate best{-INFINITY,INT_MAX};bool invalid=false;
  const int begin=blockIdx.x*kSelectionTile,end=min(begin+kSelectionTile,count);
  for(int token=begin+threadIdx.x;token<end;token+=kSelectionThreads){
    const float value=logits[token];
    if(isfinite(value))best=Better(best,{value,token});else invalid=true;
  }
  const bool warp_invalid=__any_sync(0xffffffff,invalid);
  if((threadIdx.x&31)==0 && warp_invalid)atomicExch(bad,1);
  best=BlockBest(best,scratch);
  if(threadIdx.x==0)candidates[blockIdx.x]=best;
}
__global__ void Select(const SelectionCandidate* candidates,Control* control,int count,const int* bad){
  __shared__ SelectionCandidate scratch[kSelectionThreads/32];
  SelectionCandidate best=threadIdx.x<count?candidates[threadIdx.x]:SelectionCandidate{-INFINITY,INT_MAX};
  best=BlockBest(best,scratch);
  if(threadIdx.x==0){
    // Return a safe ID on failure; the host rejects sticky bad before emission.
    control->token=best.token==INT_MAX?0:best.token;control->bad=*bad;
  }
}
