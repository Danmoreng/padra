#pragma once
namespace simt_batch {
// Eight rows reuse each decoded weight, preserving the baseline lane/FMA tree.
template<int M=8> __global__ void FP8Projection(DeviceMatrix matrix,const unsigned char* input,const float* scales,float* output,int* bad,const SimtBatch* batch){
  const int first=blockIdx.y*M,lane=threadIdx.x&31,row=blockIdx.x*4+threadIdx.x/32;
  if(first>=batch->rows || row>=matrix.rows)return;
  __shared__ float values[M][128];float sums[M]={};
  for(int base=0;base<matrix.cols;base+=128){
    for(int i=threadIdx.x;i<M*128;i+=128){const int r=i/128,c=i%128;values[r][c]=first+r<batch->rows?__half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(input[(first+r)*matrix.cols+base+c],__NV_E4M3))):0.f;}
    __syncthreads();
    const float4 weight=DecodeFP8Four(Load32(matrix.data+row*matrix.cols+base+lane*4));
    #pragma unroll
    for(int r=0;r<M;++r){sums[r]=__fmaf_rn(values[r][lane*4],weight.x,sums[r]);sums[r]=__fmaf_rn(values[r][lane*4+1],weight.y,sums[r]);sums[r]=__fmaf_rn(values[r][lane*4+2],weight.z,sums[r]);sums[r]=__fmaf_rn(values[r][lane*4+3],weight.w,sums[r]);}
    __syncthreads();
  }
  #pragma unroll
  for(int r=0;r<M;++r){
    for(int d=16;d;d/=2)sums[r]=__fadd_rn(sums[r],__shfl_down_sync(0xffffffffU,sums[r],d));
    if(!lane && first+r<batch->rows){const float value=B(__fmul_rn(__fmul_rn(sums[r],scales[first+r]),BF(matrix.scales,row)));output[(first+r)*matrix.rows+row]=value;Finite(value,bad);}
  }
}
inline void FP8(DeviceMatrix m,const unsigned char* x,const float* scales,float* out,int* bad,const SimtBatch* batch,cudaStream_t stream){FP8Projection<<<dim3((m.rows+3)/4,kSimtBatch/8),128,0,stream>>>(m,x,scales,out,bad,batch);}
}
