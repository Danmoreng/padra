// Two opposite features per thread, same original four-warp compensated tree.
// Both RoPE operands stay local, removing the normalized-row shared barrier.
#pragma once
__global__ void QKNormRopePair64(float* q,float* k,DeviceMatrix qn,DeviceMatrix kn,const float* table,int position,int* bad,float* capturedQ=nullptr,float* capturedK=nullptr,const PrefillState* prefill=nullptr,int maximum=8192){
  if(prefill){if(!ValidPrefillState(*prefill,static_cast<int>(gridDim.y),prefill->capacity,1,maximum)){if(threadIdx.x==0 && bad)atomicExch(bad,1);return;}position=prefill->base;}
 if(blockDim.x!=64||blockDim.y!=1||blockDim.z!=1||gridDim.x!=36||gridDim.z!=1||!q||!k||!qn.data||!kn.data||!table||!bad||position<0||unsigned(position)>0x7fffffffU-gridDim.y||qn.format!=0||kn.format!=0||qn.rows!=128||kn.rows!=128||qn.cols!=1||kn.cols!=1){if(!threadIdx.x&&bad)atomicExch(bad,1);return;}
 const int i=threadIdx.x,lane=i&31,warp=i/32;position+=blockIdx.y;const bool query=blockIdx.x<32;const int head=query?blockIdx.x:blockIdx.x-32;float* x=(query?q+blockIdx.y*4096:k+blockIdx.y*512)+head*128;const auto weights=query?qn:kn;
 const float input0=x[i],input1=x[i+64];float sum0=0,error0=0,sum1=0,error1=0;Accumulate(__fmul_rn(input0,input0),sum0,error0);Accumulate(__fmul_rn(input1,input1),sum1,error1);
 for(int d=16;d;d/=2){const float a=__shfl_down_sync(0xffffffff,sum0,d),b=__shfl_down_sync(0xffffffff,error0,d),c=__shfl_down_sync(0xffffffff,sum1,d),e=__shfl_down_sync(0xffffffff,error1,d);if(lane+d<32){Accumulate(a,sum0,error0);Accumulate(b,sum0,error0);Accumulate(c,sum1,error1);Accumulate(e,sum1,error1);}}
 __shared__ float sums[8],inv;if(!lane){sums[warp]=sum0;sums[warp+4]=error0;sums[warp+2]=sum1;sums[warp+6]=error1;}__syncthreads();float sum=i<4?sums[i]:0,error=i<4?sums[4+i]:0;
 if(i<32)for(int d=16;d;d/=2){const float a=__shfl_down_sync(0xffffffff,sum,d),b=__shfl_down_sync(0xffffffff,error,d);if(i+d<32){Accumulate(a,sum,error);Accumulate(b,sum,error);}}
 sum=__fadd_rn(sum,error);if(!i){const float variance=__fdiv_rn(sum,128.f);inv=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(variance,1e-6f)));}__syncthreads();
 const float a=B(B(input0*inv)*BF(weights.data,i)),b=B(B(input1*inv)*BF(weights.data,i+64));float* captured=query?capturedQ:capturedK;if(captured&&blockIdx.y==0){captured[head*128+i]=a;captured[head*128+i+64]=b;}
 const float c=table[position*128+i],s=table[position*128+64+i];const float lo=B(B(a*c)+B((-b)*s)),hi=B(B(b*c)+B(a*s));x[i]=lo;x[i+64]=hi;Finite(lo,bad);Finite(hi,bad);
}
