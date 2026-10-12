// Same compensated norm tree;skip zero-only finalwarp16/8/4 merges.
#pragma once
__device__ float RmsTailReduceExact(float sum,float error,float* scratch){
  for(int d=16;d;d/=2){const float other=__shfl_down_sync(0xffffffff,sum,d),low=__shfl_down_sync(0xffffffff,error,d);if((threadIdx.x&31)+d<32){Accumulate(other,sum,error);Accumulate(low,sum,error);}}
  if((threadIdx.x&31)==0){scratch[threadIdx.x/32]=sum;scratch[4+threadIdx.x/32]=error;}
  __syncthreads();sum=threadIdx.x<4?scratch[threadIdx.x]:0;error=threadIdx.x<4?scratch[4+threadIdx.x]:0;
  // RMS has four real warp sums;the final16/8/4 branches add only zero.
  if(threadIdx.x<4)for(int d=2;d;d/=2){const float other=__shfl_down_sync(0xf,sum,d,4),low=__shfl_down_sync(0xf,error,d,4);if(threadIdx.x+d<4){Accumulate(other,sum,error);Accumulate(low,sum,error);}}
  return __fadd_rn(sum,error);
}
