// Shared Mellum scalar arithmetic; extracted unchanged from inference.cu.
#pragma once
__global__ void Embed(DeviceMatrix m,int token,float* x,const int* tokens=nullptr){if(tokens)token=tokens[token+blockIdx.x];x+=blockIdx.x*2304;for(int i=threadIdx.x;i<2304;i+=128)x[i]=BF(m.data,token*2304+i);}
__global__ void Norm(const float* x,float* out,DeviceMatrix weights,int dim,int heads,float* statistics=nullptr){
  __shared__ float sums[8];__shared__ float inv;const int h=blockIdx.x; x+=blockIdx.y*dim*heads;out+=blockIdx.y*dim*heads;
  if(h>=heads)return;
  float sum=0,error=0;for(int i=threadIdx.x;i<dim;i+=128)Accumulate(__fmul_rn(x[h*dim+i],x[h*dim+i]),sum,error);
  // Upstream CPU rsqrt uses rounded sqrt followed by rounded division. CUDA's
  // approximate reciprocal-root can move the subsequent BF16 cast across a tie.
  sum=AccurateReduce(sum,error,sums);if(threadIdx.x==0){const float variance=__fdiv_rn(sum,static_cast<float>(dim));inv=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(variance,1e-6f)));if(statistics){statistics[2*h]=variance;statistics[2*h+1]=inv;}}__syncthreads();
  for(int i=threadIdx.x;i<dim;i+=128)out[h*dim+i]=B(B(x[h*dim+i]*inv)*BF(weights.data,i));
}
__global__ void Rope(float* q,float* k,const float* table,int position){
  position+=blockIdx.y;q+=blockIdx.y*4096;k+=blockIdx.y*512;
  const int h=blockIdx.x,i=threadIdx.x;float* x=h<32?q+h*128:k+(h-32)*128;
  __shared__ float old[128];old[i]=x[i];__syncthreads();
  const float c=table[position*128+(i%64)],s=table[position*128+64+(i%64)];
  const float rotated=i<64?-old[i+64]:old[i-64];x[i]=B(B(old[i]*c)+B(rotated*s));
}
