// Mellum batched operators. Native MMA register ABI adapted from NVIDIA CUTLASS
// (BSD-3-Clause), pinned source and license in NOTICE / licenses.
// This header is included inside inference.cu's anonymous namespace.
__global__ void ProjectBatch(DeviceMatrix m,const float* x,float* y,int* bad){
  const int t=blockIdx.y,lane=threadIdx.x&31,r=blockIdx.x*4+threadIdx.x/32;
  if(r>=m.rows)return;float sum=0,error=0;
  for(int c=lane;c<m.cols;c+=32)Accumulate(__fmul_rn(WWarp(m,r,c),x[t*m.cols+c]),sum,error);
  sum=WarpAccurateReduce(sum,error);if(lane==0){y[t*m.rows+r]=B(sum);Finite(y[t*m.rows+r],bad);}
}
__global__ void ExpertBatch(const DeviceMatrix* experts,int layer,int part,const int* ids,const float* x,float* y,int* bad){
  const int slot=blockIdx.y,t=slot/8,lane=threadIdx.x&31;const auto m=experts[(layer*64+ids[slot])*3+part];
  const int r=blockIdx.x*4+threadIdx.x/32;if(r>=m.rows)return;float sum=0,error=0;
  const float* input=x+(part==2?slot*896:t*2304);
  for(int c=lane;c<m.cols;c+=32)Accumulate(__fmul_rn(WWarp(m,r,c),input[c]),sum,error);
  sum=WarpAccurateReduce(sum,error);if(lane==0){y[slot*m.rows+r]=B(sum);Finite(y[slot*m.rows+r],bad);}
}
// Deterministic device grouping, at most one slot per token for an expert.
// Group order is increasing token/slot order, independent of CUDA scheduling.
__global__ void GroupExperts(const int* ids,int* groups,int* counts,int tokens){
  const int e=blockIdx.x;if(threadIdx.x)return;int n=0;
  for(int slot=0;slot<tokens*8;++slot)if(ids[slot]==e)groups[e*tokens+n++]=slot;
  counts[e]=n;
}
__global__ void QuantizeFP8(const float* x,unsigned char* packed,float* scales,int cols,int* bad){
  __shared__ float scratch[4];__shared__ float scale;const int row=blockIdx.x;float amax=0;
  for(int c=threadIdx.x;c<cols;c+=128){const float value=x[row*cols+c];Finite(value,bad);amax=fmaxf(amax,fabsf(value));}
  amax=MaxReduce(amax,scratch);if(threadIdx.x==0){scale=amax==0?1.f:__fdiv_rn(amax,448.f);scales[row]=scale;}__syncthreads();
  for(int c=threadIdx.x;c<cols;c+=128)packed[row*cols+c]=__nv_cvt_float_to_fp8(__fdiv_rn(x[row*cols+c],scale),__NV_SATFINITE,__NV_E4M3);
}
// Dynamic row divisor following the audited NVFP4 weight recipe. This is an
// explicit activation format, not a precision fallback. Block scales remain
// independent groups of 16; zero rows have divisor 1.
__global__ void FP4Divisors(const float* x,float* divisors,int cols,int* bad){
  __shared__ float scratch[4];const int row=blockIdx.x;float mx=0;
  for(int c=threadIdx.x;c<cols;c+=128){const float value=x[row*cols+c];Finite(value,bad);mx=fmaxf(mx,fabsf(value));}
  mx=MaxReduce(mx,scratch);if(threadIdx.x==0){const float divisor=mx==0?1.f:B(__fdiv_rn(2688.f,mx));if(!isfinite(divisor) || divisor<=0){atomicExch(bad,1);divisors[row]=1.f;}else divisors[row]=divisor;}
}
__global__ void QuantizeFP4(const float* x,unsigned char* packed,unsigned char* scales,int elements,int* bad,const float* divisors=nullptr,int cols=16){
  const int group=blockIdx.x*128+threadIdx.x;if(group>=elements/16)return;float amax=0;
  for(int c=0;c<16;++c){const float value=x[group*16+c];Finite(value,bad);amax=fmaxf(amax,fabsf(value));}
  const float divisor=divisors?divisors[(group*16)/cols]:1.f;
  const auto code=__nv_cvt_float_to_fp8(fminf(448.f,__fmul_rn(__fdiv_rn(amax,6.f),divisor)),__NV_SATFINITE,__NV_E4M3);
  scales[group]=code;const float scale=DecodeFP8(code);
  for(int c=0;c<8;++c){const float a=scale==0?0:__fdiv_rn(__fmul_rn(x[group*16+c*2],divisor),scale),b=scale==0?0:__fdiv_rn(__fmul_rn(x[group*16+c*2+1],divisor),scale);
    packed[group*8+c]=__nv_cvt_float_to_fp4(a,__NV_E2M1,cudaRoundNearest) | (__nv_cvt_float_to_fp4(b,__NV_E2M1,cudaRoundNearest)<<4);}
}
__device__ void MmaFP8(const unsigned* a,const unsigned* b,float* d){
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
__device__ void MmaFP4(const unsigned* a,const unsigned* b,unsigned sa,unsigned sb,float* d){
  const unsigned short zero=0;
  asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%12,%12}, {%11}, {%12,%12};"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sa),"r"(sb),"h"(zero));
}
__device__ unsigned Load32(const unsigned char* p){return *reinterpret_cast<const unsigned*>(p);}
// One warp per 16-token x 8-output tile; current baseline intentionally simple.
__global__ void ProjectNativeFP8(DeviceMatrix m,const unsigned char* x,const float* scales,float* y,int tokens,int* bad){
  const int lane=threadIdx.x,group=lane/4,tid=lane%4,base=blockIdx.y*16,colbase=blockIdx.x*8;
  float d[4]={};
  for(int k=0;k<m.cols;k+=32){unsigned a[4],b[2];
    for(int j=0;j<4;++j){const int row=base+group+(j%2)*8,c=k+tid*4+(j/2)*16;a[j]=row<tokens?Load32(x+row*m.cols+c):0;}
    for(int j=0;j<2;++j)b[j]=Load32(m.data+(colbase+group)*m.cols+k+tid*4+j*16);
    MmaFP8(a,b,d);
  }
  for(int j=0;j<4;++j){const int row=base+group+(j/2)*8,col=colbase+tid*2+j%2;if(row<tokens){const float value=B(__fmul_rn(__fmul_rn(d[j],scales[row]),BF(m.scales,col)));y[row*m.rows+col]=value;Finite(value,bad);}}
}
__global__ void ExpertNativeFP4(const DeviceMatrix* experts,int layer,int part,const int* groups,const int* counts,
    const unsigned char* x,const unsigned char* scales,float* y,int tokens,int* bad){
  const int e=blockIdx.z,base=blockIdx.y*16,count=counts[e];if(base>=count)return;
  const auto m=experts[(layer*64+e)*3+part];const int lane=threadIdx.x,group=lane/4,tid=lane%4,colbase=blockIdx.x*8;
  float d[4]={};
  for(int k=0;k<m.cols;k+=64){unsigned a[4],b[2];
    for(int j=0;j<4;++j){const int row=base+group+(j%2)*8;const int slot=row<count?groups[e*tokens+row]:0;
      const int input=part==2?slot:slot/8,c=k+tid*8+(j/2)*32;a[j]=row<count?Load32(x+input*(m.cols/2)+c/2):0;}
    for(int j=0;j<2;++j)b[j]=Load32(m.data+(colbase+group)*(m.cols/2)+(k+tid*8+j*32)/2);
    const int scale_row=base+group+(lane%2)*8,slot=scale_row<count?groups[e*tokens+scale_row]:0,input=part==2?slot:slot/8;
    const unsigned sa=scale_row<count?Load32(scales+input*(m.cols/16)+k/16):0;
    const unsigned sb=Load32(m.scales+(colbase+group)*(m.cols/16)+k/16);
    MmaFP4(a,b,sa,sb,d);
  }
  for(int j=0;j<4;++j){const int row=base+group+(j/2)*8,col=colbase+tid*2+j%2;if(row<count){const int slot=groups[e*tokens+row];const float value=B(__fdiv_rn(d[j],*m.divisor));y[slot*m.rows+col]=value;Finite(value,bad);}}
}

// Same explicit W8A8 activation codec, compensated exact FP32 dot control.
// This dispatch uses SIMT for FP8, while the experts still use native NVFP4 MMA.
__device__ __forceinline__ void ProjectQuantizedAccurateRow(DeviceMatrix m,const unsigned char* x,const float* scales,float* y,int* bad,int t,int r){
  const int lane=threadIdx.x&31;if(r>=m.rows)return;
  float sum=0,error=0;for(int c=lane;c<m.cols;c+=32)Accumulate(__fmul_rn(DecodeFP8(m.data[r*m.cols+c]),DecodeFP8(x[t*m.cols+c])),sum,error);
  sum=WarpAccurateReduce(sum,error);if(lane==0){y[t*m.rows+r]=B(__fmul_rn(__fmul_rn(sum,scales[t]),BF(m.scales,r)));Finite(y[t*m.rows+r],bad);}
}

__global__ void ProjectQuantizedAccurate(DeviceMatrix m,const unsigned char* x,const float* scales,float* y,int* bad){
  ProjectQuantizedAccurateRow(m,x,scales,y,bad,blockIdx.y,blockIdx.x*4+threadIdx.x/32);
}

#include "fp8_vector.cuh"
