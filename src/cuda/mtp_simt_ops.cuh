#pragma once
// BF16 nonexpert draft projections use the same compensated warp reduction as
// the SM120 path. EXL3 experts are dispatched through Pascal SIMT separately.
__global__ void GemvWarp(DeviceMatrix m,const float* x,float* y,int* bad){
  const int lane=threadIdx.x&31,row=blockIdx.x*4+threadIdx.x/32;
  if(row>=m.rows)return;
  float sum=0,error=0;
  for(int c=lane;c<m.cols;c+=32)Accumulate(__fmul_rn(BF(m.data,row*m.cols+c),x[c]),sum,error);
  sum=WarpAccurateReduce(sum,error);
  if(!lane){y[row]=B(sum);Finite(y[row],bad);}
}
__global__ void Store(const float* k,const float* v,__nv_bfloat16* keys,__nv_bfloat16* values,int position,int capacity){
  const int i=blockIdx.x*128+threadIdx.x;
  keys[(position%capacity)*512+i]=__float2bfloat16_rn(k[i]);
  values[(position%capacity)*512+i]=__float2bfloat16_rn(v[i]);
}
#include "context_admission.cuh"
struct DecodeState {
  int token, position, capacity, remaining, global_valid, local_valid, eos, stopped;
};
static_assert(sizeof(DecodeState)==32);
