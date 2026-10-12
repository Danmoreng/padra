// Exact E4M3 output bytes using cached RN reciprocal plus RN division at
// representational boundaries. Inputs/scale contract unchanged;no dtype fallback.
#pragma once
namespace fp8_exact_reciprocal {
__device__ __forceinline__ bool NeedsDivision(unsigned bits){
 if(bits<0x3a800000U)return bits>=0x3a800000U-4U; // FP8 zero/min-subnormal midpoint.
 const unsigned e=bits>>23,shift=e<121U?140U-e:19U;
 const unsigned residue=bits&((1U<<(shift+1))-1U),midpoint=1U<<shift;
 return residue>=midpoint-4U&&residue<=midpoint+4U;
}
__device__ __forceinline__ unsigned char Encode(float value,float scale,float inverse){
 const unsigned ib=__float_as_uint(inverse)&0x7fffffffU;
 if((ib&0x7f800000U)==0||(ib&0x7f800000U)==0x7f800000U)return __nv_cvt_float_to_fp8(__fdiv_rn(value,scale),__NV_SATFINITE,__NV_E4M3);
 float q=__fmul_rn(value,inverse);const unsigned b=__float_as_uint(q)&0x7fffffffU;
 if(NeedsDivision(b))q=__fdiv_rn(value,scale);
 return __nv_cvt_float_to_fp8(q,__NV_SATFINITE,__NV_E4M3);
}
}
