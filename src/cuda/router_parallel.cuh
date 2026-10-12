// Scheduling adapted from Gem16 RouterAssignmentsKernel, Apache-2.0.
// Fixed Mellum E64/top8: parallel RNexp and stable probability/ID selection.
// Original compensated ascending softmax/selected sums and BF16 route casts stay.
__global__ void RouteParallel(const float* logits,int* ids,float* weights,int* bad){
  if(blockDim.x!=64 || !logits || !ids || !weights || !bad){
    if(threadIdx.x==0 && bad){atomicExch(bad,1);if(ids && weights)for(int s=0;s<8;++s){ids[blockIdx.x*8+s]=0;weights[blockIdx.x*8+s]=0;}}return;
  }
  logits+=blockIdx.x*64;ids+=blockIdx.x*8;weights+=blockIdx.x*8;
  __shared__ float probabilities[64],maximum,total,warpProbability[2];
  __shared__ int abort,chosen,warpId[2];
  const int lane=threadIdx.x&31,warp=threadIdx.x/32;
  if(threadIdx.x==0){
    abort=*bad!=0;maximum=-INFINITY;
    for(int i=0;i<64;++i){if(!isfinite(logits[i]))abort=1;maximum=fmaxf(maximum,logits[i]);}
    if(abort){atomicExch(bad,1);for(int s=0;s<8;++s){ids[s]=0;weights[s]=0;}}
  }
  __syncthreads();if(abort)return;
  probabilities[threadIdx.x]=ExpRN(__fsub_rn(logits[threadIdx.x],maximum));
  __syncthreads();
  if(threadIdx.x==0){
    float sum=0,error=0;
    for(int i=0;i<64;++i)Accumulate(probabilities[i],sum,error);
    total=__fadd_rn(sum,error);
  }
  __syncthreads();
  float own=__fdiv_rn(probabilities[threadIdx.x],total);
  float selected=0,selected_error=0;
  for(int s=0;s<8;++s){
    float bestProbability=own;int bestId=threadIdx.x;
    for(int distance=16;distance;distance/=2){
      const float other=__shfl_down_sync(0xffffffff,bestProbability,distance);
      const int otherId=__shfl_down_sync(0xffffffff,bestId,distance);
      if(lane+distance<32 && (other>bestProbability || (other==bestProbability && otherId<bestId))){bestProbability=other;bestId=otherId;}
    }
    if(lane==0){warpProbability[warp]=bestProbability;warpId[warp]=bestId;}
    __syncthreads();
    if(threadIdx.x==0){
      const bool second=warpProbability[1]>warpProbability[0] || (warpProbability[1]==warpProbability[0] && warpId[1]<warpId[0]);
      chosen=warpId[second?1:0];const float p=warpProbability[second?1:0];
      ids[s]=chosen;weights[s]=p;Accumulate(p,selected,selected_error);
    }
    __syncthreads();if(static_cast<int>(threadIdx.x)==chosen)own=-1;
    // All lanes complete their own removal before the next warp/CTA selection.
    __syncthreads();
  }
  if(threadIdx.x==0){
    selected=__fadd_rn(selected,selected_error);
    for(int s=0;s<8;++s)weights[s]=B(__fdiv_rn(weights[s],selected));
    for(int a=0;a<8;++a)for(int b=a+1;b<8;++b)if(ids[b]<ids[a]){int id=ids[a];ids[a]=ids[b];ids[b]=id;float w=weights[a];weights[a]=weights[b];weights[b]=w;}
    Finite(selected,bad);
  }
}
