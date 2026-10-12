// Same RN Neumaier step for nonnegative squares/prefix sum;cross-warp errors retain general accumulator.
#pragma once
__device__ __forceinline__ void AccumulatePositiveExact(float value,float& sum,float& error){const float next=__fadd_rn(sum,value);const float residual=__fadd_rn(__fsub_rn(fmaxf(sum,value),next),fminf(sum,value));error=__fadd_rn(error,residual);sum=next;}
