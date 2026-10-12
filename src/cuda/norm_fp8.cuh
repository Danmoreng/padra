// Norm/amax/FP8 fusion scheduling adapted from Danmoreng/gem16, Apache-2.0.
// RmsNormQuantizeTokenKernel. Mellum reductions and all BF16/RN stages retained.
__global__ void NormFP8(const float* input,float* normalized,DeviceMatrix weight,
    unsigned char* packed,float* scales,int* bad,float* statistics=nullptr){
  if(blockDim.x!=128 || blockDim.y!=1 || blockDim.z!=1 || gridDim.y!=1 || gridDim.z!=1 ||
      !input || !normalized || !weight.data || !packed || !scales ||
      weight.format!=0 || weight.rows!=2304 || weight.cols!=1){
    if(threadIdx.x==0)atomicExch(bad,1);return;
  }
  constexpr int dim=2304;const int row=blockIdx.x;
  input+=row*dim;normalized+=row*dim;packed+=row*dim;
  __shared__ float sums[8],maxima[4],inv,scale;
  float sum=0,error=0;
  #pragma unroll
  for(int i=threadIdx.x;i<dim;i+=128)Accumulate(__fmul_rn(input[i],input[i]),sum,error);
  sum=AccurateReduce(sum,error,sums);
  if(threadIdx.x==0){const float variance=__fdiv_rn(sum,static_cast<float>(dim));inv=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(variance,1e-6f)));
    if(statistics && row==0){statistics[0]=variance;statistics[1]=inv;}}
  __syncthreads();
  float maximum=0;
  #pragma unroll
  for(int i=threadIdx.x;i<dim;i+=128){const float value=B(B(input[i]*inv)*BF(weight.data,i));normalized[i]=value;Finite(value,bad);maximum=fmaxf(maximum,fabsf(value));}
  // Separate shared arrays preserve the norm scalar until all threads have
  // consumed it; MaxReduce synchronizes all normalized stores before coding.
  maximum=MaxReduce(maximum,maxima);
  if(threadIdx.x==0){scale=maximum==0?1.f:__fdiv_rn(maximum,448.f);scales[row]=scale;}
  __syncthreads();
  #pragma unroll
  for(int i=threadIdx.x;i<dim;i+=128)packed[i]=__nv_cvt_float_to_fp8(__fdiv_rn(normalized[i],scale),__NV_SATFINITE,__NV_E4M3);
}
