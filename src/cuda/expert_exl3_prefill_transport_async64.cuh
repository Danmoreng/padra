// Double-buffered M32/N128/K64 transport for native EXL3 K4 prefill.
// Exact FP16 codebook values, sequential K16 MMA order and BF16 boundaries.
// The A/B staging arena is reused for output exchange after the final wait.
// Requests below 128 tokens retain the existing K32 projection geometry.
#pragma once
#include "expert_exl3_prefill_packed_activations.cuh"
#include "expert_exl3_lane_codec.cuh"
namespace exl3_transport_async64 {
using exl3_packed_activations::Tile;
using exl3_packed_activations::Capacity;
using exl3_packed_activations::H128;
using exl3_packed_activations::Pack;
using exl3_packed_activations::Mma;
using exl3_packed_activations::InputActivated;
constexpr int M=32,N=128,K=64,S=72;
__device__ __forceinline__ void Stage(__half* a,unsigned (*b)[8][32],const __half* input,const int* slots,DeviceMatrix matrix,int nBase,int kBase){
  const int t=threadIdx.x,row=t/8,col=(t%8)*8;
  const bool valid=slots[row]>=0;
  const void* src=valid?input+std::uint64_t(slots[row])*matrix.cols+kBase+col:input;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"::"r"(unsigned(__cvta_generic_to_shared(a+row*S+col))),"l"(src),"r"(valid?16:0):"memory");
  const int tile=t/8,segment=t%8,ks=tile/8,nw=tile%8;
  const std::uint64_t sourceTile=(std::uint64_t(kBase)/16+std::uint64_t(ks))*std::uint64_t(matrix.rows/16)+std::uint64_t(nBase)/16+std::uint64_t(nw);
  const void* packed=matrix.data+sourceTile*128+std::uint64_t(segment)*16;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"::"r"(unsigned(__cvta_generic_to_shared(&b[ks][nw][segment*4]))),"l"(packed):"memory");
  asm volatile("cp.async.commit_group;":::"memory");
}
__global__ void Projection(const DeviceMatrix* matrices,int layer,int part,const int* groups,
    const Tile* tiles,const int* tileCount,const __half* input,unsigned short* output,int tokens,int* bad){
  if(*bad || int(blockIdx.y)>=*tileCount)return;
  const Tile tile=tiles[blockIdx.y];
  if(tile.expert<0||tile.expert>=64||tile.valid_rows<1||tile.valid_rows>32||tile.begin<0||tile.begin>tokens-tile.valid_rows){if(!threadIdx.x)atomicExch(bad,1);return;}
  const auto matrix=matrices[(layer*64+tile.expert)*3+part];
  if(matrix.format!=3){if(!threadIdx.x)atomicExch(bad,1);return;}
  constexpr int BufferBytes=2*M*S*sizeof(__half)+8192;
  static_assert(BufferBytes>=M*N*sizeof(float));
  // A/B pipeline and output exchange occupy disjoint temporal lifetimes.
  __shared__ __align__(16) unsigned char storage[BufferBytes];
  auto ad=reinterpret_cast<__half (*)[M*S]>(storage);
  auto projected=reinterpret_cast<float (*)[N]>(storage);
  auto b_stage=reinterpret_cast<unsigned (*)[4][8][32]>(storage+2*M*S*sizeof(__half));
  __shared__ int slots[M];
  __shared__ int invalid; if(!threadIdx.x)invalid=0; __syncthreads();
  if(threadIdx.x<M){const int row=threadIdx.x;const int slot=row<tile.valid_rows?groups[tile.expert*tokens+tile.begin+row]:-1; slots[row]=slot;
    if(row<tile.valid_rows&&(slot<0||slot>=tokens*8))atomicExch(&invalid,1);}
  __syncthreads();if(invalid){if(!threadIdx.x)atomicExch(bad,1);return;}
  const int lane=threadIdx.x&31,warp=threadIdx.x/32,group=lane/4,tid=lane%4;
  const int nBase=blockIdx.x*128;
  // Weights own operand A: Four reconstructs all fragments with one shuffle.
  // Four 8-token B fragments preserve the sequential K16 accumulation order.
  float accum[4][4]={};
  Stage(ad[0],b_stage[0],input,slots,matrix,nBase,0);
  asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
  for(int kBase=0;kBase<matrix.cols;kBase+=64){const int stage=(kBase/64)&1;
    if(kBase+64<matrix.cols)Stage(ad[stage^1],b_stage[stage^1],input,slots,matrix,nBase,kBase+64);
    #pragma unroll
    for(int k=0;k<64;k+=16){
      const unsigned word=b_stage[stage][k/16][warp][lane];
      unsigned a[4];exl3_lane_codec::Four(word,a);
      #pragma unroll
      for(int n=0;n<4;++n){
        const unsigned b[2]={Pack(ad[stage]+(n*8+group)*S+k+tid*2),Pack(ad[stage]+(n*8+group)*S+k+tid*2+8)};
        Mma(a,b,accum[n]);
      }
    }
    asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
  }
  #pragma unroll
  for(int n=0;n<4;++n){
    #pragma unroll
    for(int j=0;j<4;++j)projected[n*8+tid*2+(j%2)][warp*16+group+(j/2)*8]=accum[n][j];
  }
  __syncthreads();
  for(int row=warp;row<32;row+=8){float v[4];
    #pragma unroll
    for(int j=0;j<4;++j)v[j]=projected[row][lane*4+j];
    H128(v);
    unsigned short packed[4];
    #pragma unroll
    for(int j=0;j<4;++j){const int col=nBase+lane*4+j;const float value=B(__fmul_rn(v[j],__half2float(__ushort_as_half(matrix.svh[col]))));if(row<tile.valid_rows&&!isfinite(value))atomicExch(bad,1);packed[j]=static_cast<unsigned short>(__float_as_uint(value)>>16);}
    if(row<tile.valid_rows){const uint2 store={unsigned(packed[0])|(unsigned(packed[1])<<16),unsigned(packed[2])|(unsigned(packed[3])<<16)};*reinterpret_cast<uint2*>(output+std::uint64_t(slots[row])*matrix.rows+nBase+lane*4)=store;}
  }
}
inline void Launch(const DeviceMatrix* matrices,int layer,int part,const int* ids,const int* groups,const int* counts,const Tile* tiles,const int* tileCount,const float* input,unsigned short* output,const unsigned short* gate,const unsigned short* up,const unsigned short* table,__half* transformed,int tokens,int* bad,cudaStream_t stream,unsigned* stats=nullptr,bool wideInput=false){
 const int cols=part==2?896:2304,rows=part==2?2304:896,pairs=tokens*8;
 if(tokens<4||part<0||part>2)throw std::runtime_error("packed BF16 prefill part requires declared T>=4");
 // K64 is also faster for short expert groups; T<4 keeps the caller's SIMT path.
 if(part==2)InputActivated<<<(pairs*7+3)/4,128,0,stream>>>(matrices,layer,ids,gate,up,table,transformed,pairs,bad,stats);
 else if(wideInput)exl3_input_wide_vector::Input<<<(pairs*(cols/128)+7)/8,256,0,stream>>>(matrices,layer,part,ids,input,transformed,pairs,bad,stats);
 else exl3_warp_input::Input<<<(pairs*(cols/128)+3)/4,128,0,stream>>>(matrices,layer,part,ids,input,transformed,pairs,bad,stats);
 Projection<<<dim3(rows/128,Capacity(tokens)),256,0,stream>>>(matrices,layer,part,groups,tiles,tileCount,transformed,output,tokens,bad);
}
}
