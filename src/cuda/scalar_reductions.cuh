// Shared Mellum scalar arithmetic; extracted unchanged from inference.cu.
#pragma once
// BF16 products are exactly representable in FP32. Retain FP32 roundoff from
// both the per-thread sum and cross-warp merge before the BF16 output cast.
// Explicit intrinsics prevent compiler reassociation from erasing compensation.
__device__ void Accumulate(float value,float& sum,float& error){
  const float next=__fadd_rn(sum,value);
  const float residual=fabsf(sum)>=fabsf(value)?__fadd_rn(__fsub_rn(sum,next),value):__fadd_rn(__fsub_rn(value,next),sum);
  error=__fadd_rn(error,residual);sum=next;
}
__device__ float AccurateReduce(float sum,float error,float* scratch){
  for(int d=16;d;d/=2){const float other=__shfl_down_sync(0xffffffff,sum,d),low=__shfl_down_sync(0xffffffff,error,d);if((threadIdx.x&31)+d<32){Accumulate(other,sum,error);Accumulate(low,sum,error);}}
  if((threadIdx.x&31)==0){scratch[threadIdx.x/32]=sum;scratch[4+threadIdx.x/32]=error;}
  __syncthreads();sum=threadIdx.x<4?scratch[threadIdx.x]:0;error=threadIdx.x<4?scratch[4+threadIdx.x]:0;
  if(threadIdx.x<32)for(int d=16;d;d/=2){const float other=__shfl_down_sync(0xffffffff,sum,d),low=__shfl_down_sync(0xffffffff,error,d);if(threadIdx.x+d<32){Accumulate(other,sum,error);Accumulate(low,sum,error);}}
  return __fadd_rn(sum,error);
}
__device__ void Finite(float x,int* bad){if(!isfinite(x))atomicExch(bad,1);}
