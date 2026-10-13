#pragma once
namespace exl3_pascal {
__global__ void InputHad(const DeviceMatrix* matrices,int layer,int part,const int* ids,const float* input,float* output,int pairs,int* bad){
  const int pair=blockIdx.y*4+threadIdx.x/32,lane=threadIdx.x&31;if(pair>=pairs)return;
  const auto m=exl3_expert::Matrix(matrices,layer,part,ids,pair,bad);const int first=blockIdx.x*128+lane*4;float values[4];
  #pragma unroll
  for(int j=0;j<4;++j)values[j]=__fmul_rn(input[(part==2?pair:pair/8)*m.cols+first+j],__half2float(__ushort_as_half(m.suh[first+j])));
  exl3_fp16::H128(values);
  #pragma unroll
  for(int j=0;j<4;++j){const float value=B(values[j]);output[pair*m.cols+first+j]=value;Finite(value,bad);}
}
}
