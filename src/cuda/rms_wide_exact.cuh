// Preserve first128 product ownership/warp tree;second128 onlyparallelize independent outputs.
#pragma once
__device__ __forceinline__ float Rms128In256Exact(float sum,float error,float* scratch){
 if(threadIdx.x<128){for(int d=16;d;d/=2){const float other=__shfl_down_sync(0xffffffff,sum,d),low=__shfl_down_sync(0xffffffff,error,d);if((threadIdx.x&31)+d<32){Accumulate(other,sum,error);Accumulate(low,sum,error);}}if((threadIdx.x&31)==0){scratch[threadIdx.x/32]=sum;scratch[4+threadIdx.x/32]=error;}}
 __syncthreads();sum=threadIdx.x<4?scratch[threadIdx.x]:0;error=threadIdx.x<4?scratch[4+threadIdx.x]:0;
 if(threadIdx.x<4)for(int d=2;d;d/=2){const float other=__shfl_down_sync(0xf,sum,d,4),low=__shfl_down_sync(0xf,error,d,4);if(threadIdx.x+d<4){Accumulate(other,sum,error);Accumulate(low,sum,error);}}
 return __fadd_rn(sum,error);
}
__device__ __forceinline__ float Max256Exact(float value,float* scratch){for(int d=16;d;d/=2)value=fmaxf(value,__shfl_xor_sync(0xffffffff,value,d));if((threadIdx.x&31)==0)scratch[threadIdx.x/32]=value;__syncthreads();value=threadIdx.x<8?scratch[threadIdx.x]:0;if(threadIdx.x<32)for(int d=16;d;d/=2)value=fmaxf(value,__shfl_xor_sync(0xffffffff,value,d));return value;}
