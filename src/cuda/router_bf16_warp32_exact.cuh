// Same compensated row order as native GemvWarp;one64-row router warp per CTA.
#pragma once
__global__ void RouterBF16Warp32Exact(DeviceMatrix m,const float* x,float* y,int* bad){
 if(blockDim.x!=32||blockDim.y!=1||blockDim.z!=1||gridDim.x!=64||gridDim.y!=1||gridDim.z!=1||m.format!=0||m.rows!=64||m.cols!=2304||!m.data||!x||!y||!bad){if(threadIdx.x==0&&bad)atomicExch(bad,1);return;}
 const int r=blockIdx.x,lane=threadIdx.x;float sum=0,error=0;for(int c=lane;c<2304;c+=32)Accumulate(__fmul_rn(BF(m.data,r*2304+c),x[c]),sum,error);sum=WarpAccurateReduce(sum,error);if(lane==0){y[r]=B(sum);Finite(y[r],bad);}
}
