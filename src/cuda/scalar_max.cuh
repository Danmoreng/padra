// Shared Mellum scalar arithmetic; extracted unchanged from inference.cu.
#pragma once
__device__ float MaxReduce(float x,float* scratch){
  for(int d=16;d;d/=2)x=fmaxf(x,__shfl_down_sync(0xffffffff,x,d));
  if((threadIdx.x&31)==0)scratch[threadIdx.x/32]=x;__syncthreads();
  x=threadIdx.x<4?scratch[threadIdx.x]:-INFINITY;
  if(threadIdx.x<32)for(int d=16;d;d/=2)x=fmaxf(x,__shfl_down_sync(0xffffffff,x,d));return x;
}
