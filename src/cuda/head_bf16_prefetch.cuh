// Exact compensated BF16 output-head ownership/order, four-deep register loads.
#pragma once
namespace head_bf16_prefetch {
__global__ void Projection(DeviceMatrix m,const float* x,float* y,int* bad){
 if(m.format!=0||m.rows!=98304||m.cols!=2304||!m.data||!x||!y||!bad||blockDim.x!=128||blockDim.y!=1||blockDim.z!=1||gridDim.x!=24576||gridDim.y!=1||gridDim.z!=1){if(threadIdx.x==0&&bad)atomicExch(bad,1);return;}
 const int lane=threadIdx.x&31,r=blockIdx.x*4+threadIdx.x/32;
 float w[4],v[4];
 #pragma unroll
 for(int d=0;d<4;++d){w[d]=BF(m.data,r*2304+lane+d*32);v[d]=x[lane+d*32];}
 float sum=0,error=0;
 for(int at=0;at<72;at+=4){
  #pragma unroll
  for(int d=0;d<4;++d){const float a=w[d],b=v[d];const int next=at+d+4;if(next<72){w[d]=BF(m.data,r*2304+lane+next*32);v[d]=x[lane+next*32];}Accumulate(__fmul_rn(a,b),sum,error);}
 }
 sum=WarpAccurateReduce(sum,error);if(lane==0){y[r]=B(sum);Finite(y[r],bad);}
}
}
