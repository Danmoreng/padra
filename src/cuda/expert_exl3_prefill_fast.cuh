#pragma once
namespace simt_batch {
// M32/N128/K16 SGEMM tile. Each thread keeps sixteen independent FP32 sums;
// A warp spans 32 columns and four rows, reusing each weight four times.
// The codec and activations are shared by all rows/columns of the tile.
// Sparse experts retain M8: padding them to M32 is slower on Pascal.
template<int N=128,int K=16,int R=(N==128?4:2)> __global__ void ProjectionFast(const DeviceMatrix* matrices,int layer,int part,const int* groups,const int* counts,const ExpertTile* tiles,const int* total,const float* input,float* output){
  static_assert((N==128 && K==16 && (R==2 || R==4)) || (N==64 && K==32 && R==2),"supported Pascal tile");
  constexpr int Columns=R==4?32:16;
  if(blockIdx.y>=static_cast<unsigned>(*total))return;
  const auto tile=tiles[blockIdx.y];const auto matrix=matrices[(layer*64+tile.expert)*3+part];
  if(counts[tile.expert]<32 || tile.begin%32)return;
  const int lane=threadIdx.x&31,warp=threadIdx.x/32,first=blockIdx.x*N;
  const int mr=(threadIdx.x/Columns)*R,nc=threadIdx.x%Columns;
  __shared__ float weights[K][N+1],values[32][K+1],codec[1024];__shared__ int pairs[32];
  exl3_pascal::InitLookup(codec);
  if(threadIdx.x<32)pairs[threadIdx.x]=tile.begin+threadIdx.x<counts[tile.expert]?groups[tile.expert*256+tile.begin+threadIdx.x]:-1;
  __syncthreads();float sums[R][N/Columns]={};
  for(int base=0;base<matrix.cols;base+=K){
    for(int i=threadIdx.x;i<32*K;i+=256){const int r=i/K,k=i%K;values[r][k]=pairs[r]>=0?input[pairs[r]*matrix.cols+base+k]:0.f;}
    // Eight native K16/N16 codec tiles, one per warp.
    const int kt=warp/(N/16),nt=warp%(N/16),ki=(lane&3)*2,col=lane/4;
    const std::uint64_t at=std::uint64_t(base/16+kt)*(matrix.rows/16)+first/16+nt;
    const unsigned word=reinterpret_cast<const unsigned*>(matrix.data+at*128)[lane];
    const unsigned previous=exl3_pascal::Previous(word);
    #pragma unroll
    for(int half=0;half<2;++half){const int k=ki+half*8;
      const float2 lo=exl3_pascal::PairLookup(__funnelshift_r(word,previous,half==0?24:16),codec);
      const float2 hi=exl3_pascal::PairLookup(half==0?__funnelshift_r(word,word,8):word,codec);
      weights[kt*16+k][nt*16+col]=lo.x;
      weights[kt*16+k+1][nt*16+col]=lo.y;
      weights[kt*16+k][nt*16+col+8]=hi.x;
      weights[kt*16+k+1][nt*16+col+8]=hi.y;}
    __syncthreads();
    #pragma unroll
    for(int k=0;k<K;++k){float activations[R];
      #pragma unroll
      for(int r=0;r<R;++r)activations[r]=values[mr+r][k];
      #pragma unroll
      for(int n=0;n<N/Columns;++n){const float w=weights[k][nc+n*Columns];
        #pragma unroll
        for(int r=0;r<R;++r)sums[r][n]=__fmaf_rn(activations[r],w,sums[r][n]);}}
    __syncthreads();
  }
  #pragma unroll
  for(int r=0;r<R;++r)if(pairs[mr+r]>=0){
    #pragma unroll
    for(int n=0;n<N/Columns;++n)output[pairs[mr+r]*matrix.rows+first+nc+n*Columns]=sums[r][n];}
}
__global__ void OutputHadFast(const DeviceMatrix* matrices,int layer,int part,const int* ids,const int* counts,float* output,int* bad,const SimtBatch* batch){
  const int pair=blockIdx.y*4+threadIdx.x/32,lane=threadIdx.x&31;if(pair>=batch->rows*8)return;
  const int expert=ids[pair];if(expert<0 || expert>=64){if(!lane)atomicExch(bad,1);return;}
  if(counts[expert]<32)return;
  const auto matrix=exl3_expert::Matrix(matrices,layer,part,ids,pair,bad);const int first=blockIdx.x*128+lane*4;float values[4];
  #pragma unroll
  for(int j=0;j<4;++j)values[j]=output[pair*matrix.rows+first+j];
  exl3_fp16::H128(values);
  #pragma unroll
  for(int j=0;j<4;++j){const float value=B(__fmul_rn(values[j],__half2float(__ushort_as_half(matrix.svh[first+j]))));output[pair*matrix.rows+first+j]=value;Finite(value,bad);}
}
template<int N=128,int K=16> inline void ExpertsFast(const DeviceMatrix* matrices,int layer,int part,const int* ids,const float* input,float* output,float* transformed,const int* groups,const int* counts,const ExpertTile* tiles,const int* total,int* bad,const SimtBatch* batch,cudaStream_t stream){
  const int cols=part==2?896:2304,rows=part==2?2304:896;
  InputHad<<<dim3(cols/128,512),128,0,stream>>>(matrices,layer,part,ids,input,transformed,bad,batch);
  Projection<true><<<dim3(rows/128,kExpertTiles),256,0,stream>>>(matrices,layer,part,groups,counts,tiles,total,transformed,output,bad);
  ProjectionFast<N,K><<<dim3(rows/N,kExpertTiles),256,0,stream>>>(matrices,layer,part,groups,counts,tiles,total,transformed,output);
  OutputHadFast<<<dim3(rows/128,512),128,0,stream>>>(matrices,layer,part,ids,counts,output,bad,batch);
}
}
