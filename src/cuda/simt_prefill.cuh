#pragma once
#include "simt_batch_state.cuh"
namespace simt_batch {
__global__ void InputNorm(const float* input,float* normalized,DeviceMatrix weight,
    unsigned char* packed,float* scales,int* bad,const SimtBatch* batch){
  if(blockDim.x!=256 || blockDim.y!=1 || blockDim.z!=1 || gridDim.y!=1 || gridDim.z!=1 ||
      !input || !normalized || !weight.data || !packed || !scales ||
      weight.format!=0 || weight.rows!=2304 || weight.cols!=1){
    if(threadIdx.x==0)atomicExch(bad,1);return;
  }
  if(blockIdx.x>=static_cast<unsigned>(batch->rows))return;
  constexpr int dim=2304;const int row=blockIdx.x;
  input+=row*dim;normalized+=row*dim;packed+=row*dim;
  __shared__ float sums[8],maxima[8],inv,scale,inverse;
  __shared__ float source[2304];float rowValues[9];
  float sum=0,error=0;
  if(threadIdx.x<128){
   #pragma unroll
   for(int j=0;j<18;++j){const int i=threadIdx.x+j*128;const float value=input[i];source[i]=value;Accumulate(__fmul_rn(value,value),sum,error);}
  }
  sum=Rms128In256Exact(sum,error,sums);
  if(threadIdx.x==0){const float variance=__fdiv_rn(sum,static_cast<float>(dim));inv=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(variance,1e-6f)));
}
  __syncthreads();
  float maximum=0;
  #pragma unroll
  for(int j=0;j<9;++j){const int i=threadIdx.x+j*256;const float value=B(B(source[i]*inv)*BF(weight.data,i));rowValues[j]=value;normalized[i]=value;Finite(value,bad);maximum=fmaxf(maximum,fabsf(value));}
  // Separate shared arrays preserve the norm scalar until all threads have
  // consumed it; MaxReduce synchronizes all normalized stores before coding.
  maximum=Max256Exact(maximum,maxima);
  if(threadIdx.x==0){scale=maximum==0?1.f:__fdiv_rn(maximum,448.f);scales[row]=scale;inverse=__fdiv_rn(1.f,scale);}
  __syncthreads();
  #pragma unroll
  for(int j=0;j<9;++j)packed[threadIdx.x+j*256]=fp8_exact_reciprocal::Encode(rowValues[j],scale,inverse);
}
__global__ void PostNorm(float* x,const float* residual,
 float* normalized,DeviceMatrix weight,
 int* bad,const SimtBatch* batch){
 if(blockDim.x!=128||blockDim.y!=1||blockDim.z!=1||gridDim.x!=1||gridDim.z!=1||!x||!residual||!normalized||x==normalized||!weight.data||!bad||weight.format!=0||weight.rows!=2304||weight.cols!=1){if(threadIdx.x==0&&bad)atomicExch(bad,1);return;}
 if(blockIdx.y>=static_cast<unsigned>(batch->rows))return;
 x+=blockIdx.y*2304;residual+=blockIdx.y*2304;normalized+=blockIdx.y*2304;
 __shared__ float sums[8],inv;
 float row[18],sum=0.f,error=0.f;
 #pragma unroll
 for(int j=0;j<18;++j){const int i=threadIdx.x+j*128;
  const float value=B(x[i]+residual[i]);x[i]=value;row[j]=value;Finite(value,bad);
  Accumulate(__fmul_rn(value,value),sum,error);
 }
 sum=AccurateReduce(sum,error,sums);
 if(threadIdx.x==0){const float variance=__fdiv_rn(sum,2304.f);inv=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(variance,1e-6f)));}
 __syncthreads();
 #pragma unroll
 for(int j=0;j<18;++j){const int i=threadIdx.x+j*128;
  const float value=B(B(row[j]*inv)*BF(weight.data,i));normalized[i]=value;
  Finite(value,bad);
 }
}
__global__ void Router(DeviceMatrix m,const float* x,float* y,int* bad,const SimtBatch* batch){
 if(blockDim.x!=32||blockDim.y!=1||blockDim.z!=1||gridDim.x!=64||gridDim.z!=1||m.format!=0||m.rows!=64||m.cols!=2304||!m.data||!x||!y||!bad){if(threadIdx.x==0&&bad)atomicExch(bad,1);return;}
 if(blockIdx.y>=static_cast<unsigned>(batch->rows))return;
 x+=blockIdx.y*2304;y+=blockIdx.y*64;
 const int r=blockIdx.x,lane=threadIdx.x;float sum=0,error=0;for(int c=lane;c<2304;c+=32)Accumulate(__fmul_rn(BF(m.data,r*2304+c),x[c]),sum,error);sum=WarpAccurateReduce(sum,error);if(lane==0){y[r]=B(sum);Finite(y[r],bad);}
}
__global__ void Rank(const float* logits,int* ids,float* weights,int* bad,const SimtBatch* batch){
 if(blockDim.x!=64||blockDim.y!=1||blockDim.z!=1||!logits||!ids||!weights||!bad){
  if(threadIdx.x==0&&bad){atomicExch(bad,1);if(ids&&weights)for(int s=0;s<8;++s){ids[blockIdx.x*8+s]=0;weights[blockIdx.x*8+s]=0;}}return;
 }
 if(blockIdx.x>=static_cast<unsigned>(batch->rows))return;
 logits+=blockIdx.x*64;ids+=blockIdx.x*8;weights+=blockIdx.x*8;
 __shared__ float probabilities[64],maximum,total;
 __shared__ int abort;
 if(threadIdx.x==0){abort=*bad!=0;maximum=-INFINITY;for(int i=0;i<64;++i){if(!isfinite(logits[i]))abort=1;maximum=fmaxf(maximum,logits[i]);}if(abort){atomicExch(bad,1);for(int s=0;s<8;++s){ids[s]=0;weights[s]=0;}}}
 __syncthreads();if(abort)return;
 probabilities[threadIdx.x]=ExpRN(__fsub_rn(logits[threadIdx.x],maximum));
 __syncthreads();
 if(threadIdx.x==0){float sum=0,error=0;for(int i=0;i<64;++i)Accumulate(probabilities[i],sum,error);total=__fadd_rn(sum,error);}
 __syncthreads();
 const float own=__fdiv_rn(probabilities[threadIdx.x],total);
 probabilities[threadIdx.x]=own;
 __syncthreads();
 int rank=0;
 for(int i=0;i<64;++i){const float other=probabilities[i];rank+=other>own||(other==own&&i<int(threadIdx.x));}
 if(rank<8){ids[rank]=threadIdx.x;weights[rank]=own;}
 __syncthreads();
 if(threadIdx.x==0){float sum=0,error=0;for(int i=0;i<8;++i)Accumulate(weights[i],sum,error);const float selected=__fadd_rn(sum,error);for(int i=0;i<8;++i)weights[i]=B(__fdiv_rn(weights[i],selected));for(int a=0;a<8;++a)for(int b=a+1;b<8;++b)if(ids[b]<ids[a]){int id=ids[a];ids[a]=ids[b];ids[b]=id;float w=weights[a];weights[a]=weights[b];weights[b]=w;}Finite(selected,bad);}
}
__global__ void QKRope(float* q,float* k,DeviceMatrix qn,DeviceMatrix kn,const float* table,int* bad,const SimtBatch* batch){
  if(blockIdx.y>=static_cast<unsigned>(batch->rows))return;
  int position=batch->base;
  position+=blockIdx.y;
  const bool query=blockIdx.x<32;
  const int head=query?blockIdx.x:blockIdx.x-32,i=threadIdx.x;
  float* x=(query?q+blockIdx.y*4096:k+blockIdx.y*512)+head*128;
  const auto weights=query?qn:kn;
  __shared__ float sums[8],inv,old[128];
  float sum=0,error=0;Accumulate(__fmul_rn(x[i],x[i]),sum,error);
  sum=AccurateReduce(sum,error,sums);
  if(i==0){const float variance=__fdiv_rn(sum,128.f);inv=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(variance,1e-6f)));}
  __syncthreads();
  old[i]=B(B(x[i]*inv)*BF(weights.data,i));

  // Model snapshot admission allows exactly one prompt token. Diagnostic
  // captures below keep the same first-token layout for a batched probe.

  __syncthreads();
  const float c=table[position*128+(i%64)],s=table[position*128+64+(i%64)];
  const float rotated=i<64?-old[i+64]:old[i-64];
  const float value=B(B(old[i]*c)+B(rotated*s));x[i]=value;Finite(value,bad);
}
__global__ void EmbedBatch(DeviceMatrix m,const int* prompt,float* x,int* bad,const SimtBatch* batch){
  const int row=blockIdx.x;if(row>=batch->rows)return;const int token=prompt[batch->base+row];
  if(token<0 || token>=98304){if(!threadIdx.x)atomicExch(bad,1);return;}
  for(int i=threadIdx.x;i<2304;i+=128)x[row*2304+i]=BF(m.data,token*2304+i);
}
__global__ void Quantize(const float* x,unsigned char* encoded,float* scales,int* bad,const SimtBatch* batch){
  const int row=blockIdx.x;if(row>=batch->rows)return;x+=row*4096;encoded+=row*4096;
  __shared__ float scratch[4],scale,inverse;float maximum=0;
  for(int c=threadIdx.x;c<4096;c+=128){Finite(x[c],bad);maximum=fmaxf(maximum,fabsf(x[c]));}
  maximum=MaxReduce(maximum,scratch);if(!threadIdx.x){scale=maximum==0?1.f:__fdiv_rn(maximum,448.f);scales[row]=scale;inverse=__fdiv_rn(1.f,scale);}__syncthreads();
  for(int c=threadIdx.x;c<4096;c+=128)encoded[c]=fp8_exact_reciprocal::Encode(x[c],scale,inverse);
}
__global__ void Stage(const float* k,const float* v,__nv_bfloat16* current,const SimtBatch* batch){
  const int row=blockIdx.y,i=blockIdx.x*128+threadIdx.x;if(row>=batch->rows)return;
  current[row*512+i]=__float2bfloat16_rn(k[row*512+i]);current[(kSimtBatch+row)*512+i]=__float2bfloat16_rn(v[row*512+i]);
}
__global__ void Commit(const __nv_bfloat16* current,__nv_bfloat16* k,__nv_bfloat16* v,int capacity,const SimtBatch* batch){
  const int row=blockIdx.y,i=blockIdx.x*128+threadIdx.x;if(row>=batch->rows)return;
  const int offset=((batch->base+row)%capacity)*512+i;k[offset]=current[row*512+i];v[offset]=current[(kSimtBatch+row)*512+i];
}
__global__ void CombineBatch(float* x,const float* down,const float* weights,int* bad,const SimtBatch* batch){
  const int row=blockIdx.y,i=blockIdx.x*128+threadIdx.x;if(row>=batch->rows || i>=2304)return;
  x+=row*2304;down+=row*8*2304;weights+=row*8;float sum=0;
  for(int s=0;s<8;++s)sum=B(sum+B(down[s*2304+i]*weights[s]));x[i]=B(x[i]+sum);Finite(x[i],bad);
}
__global__ void Silu(float* gate,const float* up,const unsigned short* table,int* bad,const SimtBatch* batch){
  const int i=blockIdx.x*128+threadIdx.x;if(i>=batch->rows*8*896)return;
  const float g=gate[i],u=up[i];Finite(g,bad);Finite(u,bad);const unsigned short bits=__bfloat16_as_ushort(__float2bfloat16_rn(g));
  gate[i]=B(__fmul_rn(__bfloat162float(__ushort_as_bfloat16(table[bits])),u));Finite(gate[i],bad);
}
// Baseline arithmetic with multiple rows; tiled projection dispatch is separate.
__global__ void FP8Rows(DeviceMatrix m,const unsigned char* x,const float* scales,float* out,int* bad,const SimtBatch* batch){
  const int row=blockIdx.y;if(row>=batch->rows)return;fast_projection::VectorRow(m,x+row*m.cols,scales+row,out+row*m.rows,blockIdx.x*4+threadIdx.x/32,bad);
}
__global__ void InputHadRows(const DeviceMatrix* matrices,int layer,int part,const int* ids,const float* input,float* transformed,int* bad,const SimtBatch* batch){
  const int pair=blockIdx.y;if(pair>=batch->rows*8)return;
  const auto m=exl3_expert::Matrix(matrices,layer,part,ids,pair,bad);const int i=blockIdx.x*128+threadIdx.x,t=threadIdx.x;
  __shared__ float x[128];x[t]=__fmul_rn(input[(part==2?pair:pair/8)*m.cols+i],__half2float(__ushort_as_half(m.suh[i])));__syncthreads();
  for(int d=1;d<128;d*=2){const float a=x[t],b=x[t^d];__syncthreads();x[t]=(t&d)?__fsub_rn(b,a):__fadd_rn(a,b);__syncthreads();}
  const float value=B(__fmul_rn(x[t],.08838834764831845f));transformed[pair*m.cols+i]=value;Finite(value,bad);
}
} // namespace simt_batch
