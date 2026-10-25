#pragma once
namespace simt_batch {
// Eight rows reuse each decoded weight, preserving the baseline lane/FMA tree.
template<int M=8> __global__ void FP8Projection(DeviceMatrix matrix,const unsigned char* input,const float* scales,float* output,int* bad,const SimtBatch* batch){
  const int first=blockIdx.y*M,lane=threadIdx.x&31,row=blockIdx.x*4+threadIdx.x/32;
  if(first>=batch->rows || row>=matrix.rows)return;
  __shared__ __align__(16) float values[M][128];float sums[M]={};
  __shared__ float table[256];InitFP8Lookup(table);
  for(int base=0;base<matrix.cols;base+=128){
    for(int i=threadIdx.x;i<M*32;i+=128){const int r=i/32,c=(i%32)*4;const float4 value=first+r<batch->rows?DecodeFP8Lookup(Load32(input+(first+r)*matrix.cols+base+c),table):make_float4(0,0,0,0);*reinterpret_cast<float4*>(&values[r][c])=value;}
    __syncthreads();
    const float4 weight=DecodeFP8Lookup(Load32(matrix.data+row*matrix.cols+base+lane*4),table);
    #pragma unroll
    for(int r=0;r<M;++r){const float4 value=*reinterpret_cast<const float4*>(&values[r][lane*4]);sums[r]=__fmaf_rn(value.x,weight.x,sums[r]);sums[r]=__fmaf_rn(value.y,weight.y,sums[r]);sums[r]=__fmaf_rn(value.z,weight.z,sums[r]);sums[r]=__fmaf_rn(value.w,weight.w,sums[r]);}
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
namespace simt_batch {
// One output per thread, 32 original lane accumulators. N32 input reuse.
__global__ void FP8ExactTile(DeviceMatrix matrix,const unsigned char* input,const float* scales,float* output,int* bad,const SimtBatch* batch){
  const int first=blockIdx.y*8,n=threadIdx.x%32,m=threadIdx.x/32,row=blockIdx.x*32+n;
  if(first>=batch->rows)return;
  __shared__ float weights[128][33],values[8][128],lookup[256];InitFP8Lookup(lookup);
  float sums[32]={};
  for(int base=0;base<matrix.cols;base+=128){
    for(int i=threadIdx.x;i<32*32;i+=256){const int r=i/32,c=(i%32)*4;const float4 w=DecodeFP8Lookup(Load32(matrix.data+(blockIdx.x*32+r)*matrix.cols+base+c),lookup);weights[c][r]=w.x;weights[c+1][r]=w.y;weights[c+2][r]=w.z;weights[c+3][r]=w.w;}
    if(threadIdx.x<8*32){const int r=threadIdx.x/32,c=(threadIdx.x%32)*4;const float4 a=first+r<batch->rows?DecodeFP8Lookup(Load32(input+(first+r)*matrix.cols+base+c),lookup):make_float4(0,0,0,0);values[r][c]=a.x;values[r][c+1]=a.y;values[r][c+2]=a.z;values[r][c+3]=a.w;}
    __syncthreads();
    #pragma unroll
    for(int p=0;p<32;++p){
      #pragma unroll
      for(int j=0;j<4;++j)sums[p]=__fmaf_rn(values[m][p*4+j],weights[p*4+j][n],sums[p]);
    }
    __syncthreads();
  }
  #pragma unroll
  for(int p=0;p<16;++p)sums[p]=__fadd_rn(sums[p],sums[p+16]);
  #pragma unroll
  for(int p=0;p<8;++p)sums[p]=__fadd_rn(sums[p],sums[p+8]);
  #pragma unroll
  for(int p=0;p<4;++p)sums[p]=__fadd_rn(sums[p],sums[p+4]);
  #pragma unroll
  for(int p=0;p<2;++p)sums[p]=__fadd_rn(sums[p],sums[p+2]);
  if(first+m<batch->rows){const float value=B(__fmul_rn(__fmul_rn(__fadd_rn(sums[0],sums[1]),scales[first+m]),BF(matrix.scales,row)));output[(first+m)*matrix.rows+row]=value;Finite(value,bad);}
}
inline void FP8Tiled(DeviceMatrix matrix,const unsigned char* input,const float* scales,float* output,int* bad,const SimtBatch* batch,cudaStream_t stream){FP8ExactTile<<<dim3(matrix.rows/32,32),256,0,stream>>>(matrix,input,scales,output,bad,batch);}
}
namespace simt_batch {
// Verifier rows fit in cache. Keep one accumulator per row and reuse each
// decoded weight without a block barrier for every 128 input channels.
template<int Rows> __global__ void FP8Small(DeviceMatrix matrix,const unsigned char* input,const float* scales,float* output,int* bad,const SimtBatch* batch){
  const int lane=threadIdx.x&31,row=blockIdx.x*4+threadIdx.x/32;
  __shared__ float table[256];InitFP8Lookup(table);
  float sums[Rows]={};
  for(int c=lane*4;c<matrix.cols;c+=128){
    const auto w=DecodeFP8Lookup(Load32(matrix.data+row*matrix.cols+c),table);
    #pragma unroll
    for(int r=0;r<Rows;++r){
      const auto a=DecodeFP8Lookup(Load32(input+r*matrix.cols+c),table);
      sums[r]=__fmaf_rn(a.x,w.x,sums[r]);sums[r]=__fmaf_rn(a.y,w.y,sums[r]);
      sums[r]=__fmaf_rn(a.z,w.z,sums[r]);sums[r]=__fmaf_rn(a.w,w.w,sums[r]);
    }
  }
  #pragma unroll
  for(int r=0;r<Rows;++r){
    for(int d=16;d;d/=2)sums[r]=__fadd_rn(sums[r],__shfl_down_sync(0xffffffffU,sums[r],d));
    if(!lane){const float v=B(__fmul_rn(__fmul_rn(sums[r],scales[r]),BF(matrix.scales,row)));output[r*matrix.rows+row]=v;Finite(v,bad);}
  }
}
}
