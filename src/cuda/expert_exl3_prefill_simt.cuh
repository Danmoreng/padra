#pragma once
namespace simt_batch {
struct ExpertTile {int expert,begin;};
inline constexpr int kExpertTiles=kSimtBatch+63;
__global__ void Group(const int* ids,int* groups,int* counts,int* bad,const SimtBatch* batch){
  const int expert=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x/32;
  __shared__ int total,warp_counts[4],offsets[4];if(!threadIdx.x)total=0;__syncthreads();
  for(int base=0;base<batch->rows*8;base+=128){const int pair=base+threadIdx.x;const int id=pair<batch->rows*8?ids[pair]:-1;
    if(pair<batch->rows*8 && (id<0 || id>=64))atomicExch(bad,1);
    const unsigned mask=__ballot_sync(0xffffffffU,pair<batch->rows*8 && id==expert);if(!lane)warp_counts[warp]=__popc(mask);__syncthreads();
    if(!threadIdx.x){int n=total;for(int w=0;w<4;++w){offsets[w]=n;n+=warp_counts[w];}total=n;if(n>batch->rows)atomicExch(bad,1);}__syncthreads();
    if(pair<batch->rows*8 && id==expert){const int index=offsets[warp]+__popc(mask&((1u<<lane)-1));if(index<batch->rows)groups[expert*kSimtBatch+index]=pair;}__syncthreads();
  }
  if(!threadIdx.x)counts[expert]=min(total,batch->rows);
}
__global__ void Describe(const int* counts,ExpertTile* tiles,int* total,int* bad,const SimtBatch* batch){
  if(threadIdx.x)return;int n=0,sum=0;
  for(int e=0;e<64;++e){const int count=counts[e];if(count<0 || count>batch->rows){atomicExch(bad,1);*total=0;return;}sum+=count;for(int i=0;i<count;i+=8){if(n>=kExpertTiles){atomicExch(bad,1);*total=0;return;}tiles[n++]={e,i};}}
  if(sum!=batch->rows*8){atomicExch(bad,1);n=0;}*total=n;
}
// Four independent warp H128s per CTA; all operations match InputHad's tree.
__global__ void InputHad(const DeviceMatrix* matrices,int layer,int part,const int* ids,const float* input,float* transformed,int* bad,const SimtBatch* batch){
  const int pair=blockIdx.y*4+threadIdx.x/32,lane=threadIdx.x&31;
  if(pair>=batch->rows*8)return;
  const auto m=exl3_expert::Matrix(matrices,layer,part,ids,pair,bad);const int first=blockIdx.x*128+lane*4;float values[4];
  #pragma unroll
  for(int j=0;j<4;++j)values[j]=__fmul_rn(input[(part==2?pair:pair/8)*m.cols+first+j],__half2float(__ushort_as_half(m.suh[first+j])));
  exl3_fp16::H128(values);
  #pragma unroll
  for(int j=0;j<4;++j){const float value=B(values[j]);transformed[pair*m.cols+first+j]=value;Finite(value,bad);}
}
// Weights decode once per eight expert rows. Original K16/eight-warp FMA order.
template<bool SparseOnly=false> __global__ void Projection(const DeviceMatrix* matrices,int layer,int part,const int* groups,const int* counts,const ExpertTile* tiles,const int* total,const float* input,float* output,int* bad){
  if(blockIdx.y>=static_cast<unsigned>(*total))return;
  const auto tile=tiles[blockIdx.y];const auto m=matrices[(layer*64+tile.expert)*3+part];
  if constexpr(SparseOnly)if(counts[tile.expert]>=32)return;
  const int lane=threadIdx.x&31,warp=threadIdx.x/32,base=blockIdx.x*128,col=lane/4,ki=(lane&3)*2;
  __shared__ int pairs[8];
  if(threadIdx.x<8)pairs[threadIdx.x]=tile.begin+threadIdx.x<counts[tile.expert]?groups[tile.expert*kSimtBatch+tile.begin+threadIdx.x]:-1;
  __syncthreads();
  __shared__ float transformed[8][128],raw[8][128];float low[8]={},high[8]={};
  for(int at=0;at<m.cols;at+=128){
    for(int i=threadIdx.x;i<8*128;i+=256){const int r=i/128,c=i%128;transformed[r][c]=pairs[r]>=0?input[pairs[r]*m.cols+at+c]:0.f;}__syncthreads();
    #pragma unroll
    for(int k=0;k<128;k+=16){
      const std::uint64_t offset=std::uint64_t((at+k)/16)*(m.rows/16)+base/16+warp;
      const unsigned word=reinterpret_cast<const unsigned*>(m.data+offset*128)[lane];
      #pragma unroll
      for(int half=0;half<2;++half){const int index=ki+half*8;
        const float2 a=make_float2(exl3_pascal::Decode(exl3_selected::State(word,index,col)),exl3_pascal::Decode(exl3_selected::State(word,index+1,col)));
        const float2 b=make_float2(exl3_pascal::Decode(exl3_selected::State(word,index,col+8)),exl3_pascal::Decode(exl3_selected::State(word,index+1,col+8)));
        #pragma unroll
        for(int r=0;r<8;++r)if(pairs[r]>=0){const float x=transformed[r][k+index],y=transformed[r][k+index+1];low[r]=__fmaf_rn(x,a.x,low[r]);low[r]=__fmaf_rn(y,a.y,low[r]);high[r]=__fmaf_rn(x,b.x,high[r]);high[r]=__fmaf_rn(y,b.y,high[r]);}
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int r=0;r<8;++r){for(int d=1;d<=2;d*=2){low[r]=__fadd_rn(low[r],__shfl_xor_sync(0xffffffffU,low[r],d));high[r]=__fadd_rn(high[r],__shfl_xor_sync(0xffffffffU,high[r],d));}if((lane&3)==0){raw[r][warp*16+col]=low[r];raw[r][warp*16+col+8]=high[r];}}
  __syncthreads();
  if(!warp){
    #pragma unroll
    for(int r=0;r<8;++r){float values[4];
      #pragma unroll
      for(int j=0;j<4;++j)values[j]=raw[r][lane*4+j];
      exl3_fp16::H128(values);
      if(pairs[r]>=0){
        #pragma unroll
        for(int j=0;j<4;++j){const int i=base+lane*4+j;const float value=B(__fmul_rn(values[j],__half2float(__ushort_as_half(m.svh[i]))));output[pairs[r]*m.rows+i]=value;Finite(value,bad);}}
    }
  }
}
inline void Experts(const DeviceMatrix* matrices,int layer,int part,const int* ids,const float* input,float* output,float* transformed,const int* groups,const int* counts,const ExpertTile* tiles,const int* total,int* bad,const SimtBatch* batch,cudaStream_t stream){
  const int cols=part==2?896:2304,rows=part==2?2304:896;
  InputHad<<<dim3(cols/128,kSimtBatch*2),128,0,stream>>>(matrices,layer,part,ids,input,transformed,bad,batch);
  Projection<<<dim3(rows/128,kExpertTiles),256,0,stream>>>(matrices,layer,part,groups,counts,tiles,total,transformed,output,bad);
}
}
