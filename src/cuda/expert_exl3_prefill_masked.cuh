// R-P1: frozen X1 M32/N128/K32 arithmetic; masked ceil groups.
// Gem16 c0a807a scheduling/cp.async zero-fill only; no FP8/Gemma arithmetic.
#pragma once
#include "expert_exl3_prefill_fp16.cuh"
#include "runtime/exl3_masked_schedule.h"
namespace exl3_masked {
using Tile=mellum16::exl3_masked_schedule::Tile;
using mellum16::exl3_masked_schedule::Capacity;
using exl3_fp16::DecodeK4;
using exl3_fp16::H128;
constexpr int M=32,N=128,K=32,S=40;
__global__ void Schedule(const int* counts,Tile* tiles,int* tileCount,int tokens,int* bad){
  if(threadIdx.x)return;
  if(!counts||!tiles||!tileCount||!bad||!Capacity(tokens)||blockDim.x!=1||gridDim.x!=1){if(bad)atomicExch(bad,1);if(tileCount)*tileCount=0;return;}
  *tileCount=0;if(*bad)return;
  int total=0,sum=0;
  for(int e=0;e<64;++e){const int c=counts[e];if(c<0||c>tokens){atomicExch(bad,1);return;}sum+=c;total+=(c+31)/32;}
  if(sum!=tokens*8||total>Capacity(tokens)){atomicExch(bad,1);return;}
  int at=0;for(int e=0;e<64;++e)for(int begin=0;begin<counts[e];begin+=32)tiles[at++]={e,begin,min(32,counts[e]-begin)};
  *tileCount=at;
}
__global__ void Input(const DeviceMatrix* matrices,int layer,int part,const int* ids,
    const float* input,__half* transformed,int pairs,int* bad,unsigned* stats=nullptr){
  const int pair=blockIdx.y,i=blockIdx.x*128+threadIdx.x;
  const auto m=exl3_expert::Matrix(matrices,layer,part,ids,pair,bad);
  __shared__ float x[128];const int t=threadIdx.x;
  x[t]=__fmul_rn(input[(part==2?pair:pair/8)*m.cols+i],__half2float(__ushort_as_half(m.suh[i])));__syncthreads();
  for(int d=1;d<128;d*=2){const float a=x[t],b=x[t^d];__syncthreads();x[t]=(t&d)?__fsub_rn(b,a):__fadd_rn(a,b);__syncthreads();}
  const float value=B(__fmul_rn(x[t],0.08838834764831845f));const __half h=__float2half_rn(value);
  const unsigned bits=__half_as_ushort(h);
  if(!isfinite(value)||(bits&0x7c00)==0x7c00)atomicExch(bad,1); // No clipping or fallback.
  if(stats){atomicMax(stats,__float_as_uint(fabsf(value)));atomicAdd(stats+1,1U);
    if((bits&0x7c00)==0&&(bits&1023))atomicAdd(stats+2,1U);
    if((bits&0x7fff)==0&&value!=0)atomicAdd(stats+3,1U);
    if((bits&0x7c00)==0x7c00)atomicAdd(stats+4,1U);
    if(__half2float(h)!=value)atomicAdd(stats+5,1U);}
  transformed[pair*m.cols+i]=h;
}
__device__ __forceinline__ unsigned Pack(const __half* p){return unsigned(__half_as_ushort(p[0]))|(unsigned(__half_as_ushort(p[1]))<<16);}
__device__ __forceinline__ void Mma(const unsigned* a,const unsigned* b,float* d){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
__device__ __forceinline__ void Stage(__half* a,__half* b,const __half* input,
    const int* slots,DeviceMatrix matrix,int nBase,int kBase){
  const int t=threadIdx.x,lane=t&31,warp=t/32;
  if(t<128){const int row=t/4,col=(t%4)*8;
    const bool valid=slots[row]>=0;
    const void* src=valid?input+std::uint64_t(slots[row])*matrix.cols+kBase+col:input;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"::"r"(unsigned(__cvta_generic_to_shared(a+row*S+col))),"l"(src),"r"(valid?16:0):"memory");}
  // Eight output tiles × two K16 tiles; each packed tile loaded once per warp.
  #pragma unroll
  for(int halfK=0;halfK<2;++halfK){
    const auto* packed=reinterpret_cast<const unsigned*>(matrix.data+std::uint64_t(((kBase/16)+halfK)*(matrix.rows/16)+(nBase/16)+warp)*128);
    const unsigned word=packed[lane];const int out=lane/2,in=(lane%2)*8;
    #pragma unroll
    for(int j=0;j<8;++j)b[(warp*16+out)*S+halfK*16+in+j]=DecodeK4(word,in+j,out);
  }
  asm volatile("cp.async.commit_group;":::"memory");
}
__global__ void Projection(const DeviceMatrix* matrices,int layer,int part,const int* groups,
    const Tile* tiles,const int* tileCount,const __half* input,float* output,int tokens,int* bad){
  if(*bad || int(blockIdx.y)>=*tileCount)return;
  const Tile tile=tiles[blockIdx.y];
  if(tile.expert<0||tile.expert>=64||tile.valid_rows<1||tile.valid_rows>32||tile.begin<0||tile.begin>tokens-tile.valid_rows){if(!threadIdx.x)atomicExch(bad,1);return;}
  const auto matrix=matrices[(layer*64+tile.expert)*3+part];
  if(matrix.format!=3){if(!threadIdx.x)atomicExch(bad,1);return;}
  __shared__ __align__(16) __half ad[2][M*S],bd[2][N*S];
  __shared__ float projected[M][N];__shared__ int slots[M];
  __shared__ int invalid; if(!threadIdx.x)invalid=0; __syncthreads();
  if(threadIdx.x<M){const int row=threadIdx.x;const int slot=row<tile.valid_rows?groups[tile.expert*tokens+tile.begin+row]:-1; slots[row]=slot;
    if(row<tile.valid_rows&&(slot<0||slot>=tokens*8))atomicExch(&invalid,1);}
  __syncthreads();if(invalid){if(!threadIdx.x)atomicExch(bad,1);return;}
  const int lane=threadIdx.x&31,warp=threadIdx.x/32,group=lane/4,tid=lane%4;
  const int nBase=blockIdx.x*128;
  float accum[2][2][4]={};
  Stage(ad[0],bd[0],input,slots,matrix,nBase,0);
  asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
  for(int kBase=0;kBase<matrix.cols;kBase+=32){const int stage=(kBase/32)&1;
    if(kBase+32<matrix.cols)Stage(ad[stage^1],bd[stage^1],input,slots,matrix,nBase,kBase+32);
    #pragma unroll
    for(int k=0;k<32;k+=16){
      #pragma unroll
      for(int m=0;m<2;++m){unsigned a[4];
        #pragma unroll
        for(int j=0;j<4;++j)a[j]=Pack(ad[stage]+(m*16+group+(j%2)*8)*S+k+tid*2+(j/2)*8);
        #pragma unroll
        for(int n=0;n<2;++n){unsigned b[2];
          #pragma unroll
          for(int j=0;j<2;++j)b[j]=Pack(bd[stage]+(warp*16+n*8+group)*S+k+tid*2+j*8);
          Mma(a,b,accum[m][n]);}}
    }
    asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
  }
  #pragma unroll
  for(int m=0;m<2;++m){
    #pragma unroll
    for(int n=0;n<2;++n){
      #pragma unroll
      for(int j=0;j<4;++j)projected[m*16+group+(j/2)*8][warp*16+n*8+tid*2+(j%2)]=accum[m][n][j];}}
  __syncthreads();
  for(int row=warp;row<32;row+=8){float v[4];
    #pragma unroll
    for(int j=0;j<4;++j)v[j]=projected[row][lane*4+j];
    H128(v);
    #pragma unroll
    for(int j=0;j<4;++j){const int col=nBase+lane*4+j;const float value=B(__fmul_rn(v[j],__half2float(__ushort_as_half(matrix.svh[col]))));
      if(row<tile.valid_rows){if(!isfinite(value))atomicExch(bad,1);output[std::uint64_t(slots[row])*matrix.rows+col]=value;}}
  }
}
inline void Launch(const DeviceMatrix* matrices,int layer,int part,const int* ids,const int* groups,const int* counts,
    const Tile* tiles,const int* tileCount,const float* input,float* output,__half* transformed,float* raw,
    float* tailInput,int tokens,int* bad,cudaStream_t stream,unsigned* stats=nullptr){
  const int cols=part==2?896:2304,rows=part==2?2304:896,pairs=tokens*8;
  if(tokens<4){exl3_expert::Launch(matrices,layer,part,ids,input,output,tailInput,tailInput+pairs*cols,tokens,bad,stream);return;}
  Input<<<dim3(cols/128,pairs),128,0,stream>>>(matrices,layer,part,ids,input,transformed,pairs,bad,stats);
  Projection<<<dim3(rows/128,Capacity(tokens)),256,0,stream>>>(matrices,layer,part,groups,tiles,tileCount,transformed,output,tokens,bad);
  // All valid pairs consumed; no duplicate InputHad or SIMT tail launches.
}
}
