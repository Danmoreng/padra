// Exact probability rank;same RNexp/ascending compensated sums/BF16 weights.
#pragma once
__global__ void RouteRankExact(const float* logits,int* ids,float* weights,int* bad){
 if(blockDim.x!=64||blockDim.y!=1||blockDim.z!=1||!logits||!ids||!weights||!bad){
  if(threadIdx.x==0&&bad){atomicExch(bad,1);if(ids&&weights)for(int s=0;s<8;++s){ids[blockIdx.x*8+s]=0;weights[blockIdx.x*8+s]=0;}}return;
 }
 logits+=blockIdx.x*64;ids+=blockIdx.x*8;weights+=blockIdx.x*8;
 __shared__ float probabilities[64],maximum,total;
 __shared__ int abort;
 if(threadIdx.x==0){abort=*bad!=0;maximum=-INFINITY;for(int i=0;i<64;++i){if(!isfinite(logits[i]))abort=1;maximum=fmaxf(maximum,logits[i]);}if(abort){atomicExch(bad,1);for(int s=0;s<8;++s){ids[s]=0;weights[s]=0;}}}
 __syncthreads();if(abort)return;
 probabilities[threadIdx.x]=ExpRN(__fsub_rn(logits[threadIdx.x],maximum));
 __syncthreads();
 if(threadIdx.x==0){float sum=0,error=0;for(int i=0;i<64;++i)Accumulate(probabilities[i],sum,error);total=__fadd_rn(sum,error);}
 __syncthreads();
 const float own=__fdiv_rn(probabilities[threadIdx.x],total);
 probabilities[threadIdx.x]=own;
 __syncthreads();
 int rank=0;
 for(int i=0;i<64;++i){const float other=probabilities[i];rank+=other>own||(other==own&&i<int(threadIdx.x));}
 if(rank<8){ids[rank]=threadIdx.x;weights[rank]=own;}
 __syncthreads();
 if(threadIdx.x==0){float sum=0,error=0;for(int i=0;i<8;++i)Accumulate(weights[i],sum,error);const float selected=__fadd_rn(sum,error);for(int i=0;i<8;++i)weights[i]=B(__fdiv_rn(weights[i],selected));for(int a=0;a<8;++a)for(int b=a+1;b<8;++b)if(ids[b]<ids[a]){int id=ids[a];ids[a]=ids[b];ids[b]=id;float w=weights[a];weights[a]=weights[b];weights[b]=w;}Finite(selected,bad);}
}
