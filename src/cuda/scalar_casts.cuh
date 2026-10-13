// Shared Mellum scalar arithmetic; extracted unchanged from inference.cu.
#pragma once
__device__ float B(float x){return __bfloat162float(__float2bfloat16_rn(x));}
// Accuracy baseline: evaluate this scalar transcendental in FP64 and immediately
// round to FP32. No FP64 model tensors or accumulation. Explicit rnexp profile.
__device__ float ExpRN(float x){return __double2float_rn(exp(static_cast<double>(x)));}
__device__ float BF(const unsigned char* p,int i){return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p)[i]);}
