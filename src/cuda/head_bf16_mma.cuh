// Explicit numerical BF16 head candidate: same BF16 weights/input/output,
// FP32 Tensor-Core accumulation replaces compensated per-lane summation.
#pragma once
namespace head_bf16_mma {
__device__ __forceinline__ unsigned Input(const float* x,int k,int* bad){
 const unsigned lo=__float_as_uint(x[k]),hi=__float_as_uint(x[k+1]);
 if((lo&65535U)||(hi&65535U)||!isfinite(x[k])||!isfinite(x[k+1]))atomicExch(bad,1);
 return (lo>>16)|(hi&0xffff0000U);
}
__global__ void Projection(DeviceMatrix m,const float* x,float* y,int* bad){
 if(m.format!=0||m.rows!=98304||m.cols!=2304||!m.data||!x||!y||x==y||!bad||blockDim.x!=128||blockDim.y!=1||blockDim.z!=1||gridDim.x!=3072||gridDim.y!=1||gridDim.z!=1||(reinterpret_cast<std::uintptr_t>(m.data)&3)){if(threadIdx.x==0&&bad)atomicExch(bad,1);return;}
 const int lane=threadIdx.x&31,row=blockIdx.x*32+(threadIdx.x/32)*8+lane/4,ki=(lane%4)*2;float d[4]={};
 const auto* weight=reinterpret_cast<const unsigned*>(m.data)+row*1152;
 for(int k=0;k<2304;k+=16){const unsigned a0=weight[(k+ki)/2],a2=weight[(k+ki+8)/2],b0=Input(x,k+ki,bad),b1=Input(x,k+ki+8,bad);
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%5}, {%7,%8}, {%0,%1,%2,%3};"
   :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(0U),"r"(a2),"r"(b0),"r"(b1));
 }
 if((lane%4)==0){y[row]=B(d[0]);Finite(y[row],bad);}
}
}
