// Shared Mellum scalar arithmetic; extracted unchanged from inference.cu.
#pragma once
__device__ float WarpAccurateReduce(float sum,float error){
  for(int d=16;d;d/=2){const float other=__shfl_down_sync(0xffffffff,sum,d),low=__shfl_down_sync(0xffffffff,error,d);
    if((threadIdx.x&31)+d<32){Accumulate(other,sum,error);Accumulate(low,sum,error);}}
  return __fadd_rn(sum,error);
}
