#pragma once
namespace simt_batch {
// M16/N64, four original K16 accumulators per output, 44 KiB shared scratch including the codec table.
__global__ void ProjectionTiled(const DeviceMatrix* matrices,int layer,int part,const int* groups,const int* counts,const ExpertTile* tiles,const int* total,const float* input,float* output,int* bad){
  if(blockIdx.y>=static_cast<unsigned>(*total))return;
  const auto descriptor=tiles[blockIdx.y];const auto matrix=matrices[(layer*64+descriptor.expert)*3+part];
  const int lane=threadIdx.x&31,warp=threadIdx.x/32,first=blockIdx.x*64,m=warp*2,n=lane*2;
  __shared__ float codec[1024];exl3_pascal::InitLookup(codec);
  __shared__ float weights[128][64],values[16][128];__shared__ int pairs[16];
  if(threadIdx.x<16)pairs[threadIdx.x]=descriptor.begin+threadIdx.x<counts[descriptor.expert]?groups[descriptor.expert*256+descriptor.begin+threadIdx.x]:-1;__syncthreads();
  float sums[2][2][4]={};
  for(int base=0;base<matrix.cols;base+=128){
    for(int i=threadIdx.x;i<16*128;i+=256){const int r=i/128,c=i%128;values[r][c]=pairs[r]>=0?input[pairs[r]*matrix.cols+base+c]:0.f;}
    for(int t=warp;t<32;t+=8){const int kt=t/4,nt=t%4,ki=(lane&3)*2,col=lane/4;const std::uint64_t tile=std::uint64_t(base/16+kt)*(matrix.rows/16)+first/16+nt;const unsigned word=reinterpret_cast<const unsigned*>(matrix.data+tile*128)[lane];
      #pragma unroll
      for(int half=0;half<2;++half){const int k=ki+half*8;
        weights[kt*16+k][nt*16+col]=exl3_pascal::DecodeLookup(exl3_selected::State(word,k,col),codec);weights[kt*16+k+1][nt*16+col]=exl3_pascal::DecodeLookup(exl3_selected::State(word,k+1,col),codec);
        weights[kt*16+k][nt*16+col+8]=exl3_pascal::DecodeLookup(exl3_selected::State(word,k,col+8),codec);weights[kt*16+k+1][nt*16+col+8]=exl3_pascal::DecodeLookup(exl3_selected::State(word,k+1,col+8),codec);}
    }
    __syncthreads();
    #pragma unroll
    for(int k=0;k<128;k+=16){
      #pragma unroll
      for(int half=0;half<2;++half){
        #pragma unroll
        for(int p=0;p<4;++p){const int at=k+p*2+half*8;
          #pragma unroll
          for(int j=0;j<2;++j){const float x0=values[m][at+j],x1=values[m+1][at+j],w0=weights[at+j][n],w1=weights[at+j][n+1];
            sums[0][0][p]=__fmaf_rn(x0,w0,sums[0][0][p]);sums[0][1][p]=__fmaf_rn(x0,w1,sums[0][1][p]);sums[1][0][p]=__fmaf_rn(x1,w0,sums[1][0][p]);sums[1][1][p]=__fmaf_rn(x1,w1,sums[1][1][p]);}
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int r=0;r<2;++r)if(pairs[m+r]>=0){
    #pragma unroll
    for(int c=0;c<2;++c)output[pairs[m+r]*matrix.rows+first+n+c]=__fadd_rn(__fadd_rn(sums[r][c][0],sums[r][c][1]),__fadd_rn(sums[r][c][2],sums[r][c][3]));
  }
  (void)bad;
}
__global__ void OutputHadTiled(const DeviceMatrix* matrices,int layer,int part,const int* ids,float* output,int* bad,const SimtBatch* batch){
  const int pair=blockIdx.y*4+threadIdx.x/32,lane=threadIdx.x&31;if(pair>=batch->rows*8)return;
  const auto matrix=exl3_expert::Matrix(matrices,layer,part,ids,pair,bad);const int first=blockIdx.x*128+lane*4;float values[4];
  #pragma unroll
  for(int j=0;j<4;++j)values[j]=output[pair*matrix.rows+first+j];exl3_fp16::H128(values);
  #pragma unroll
  for(int j=0;j<4;++j){const float value=B(__fmul_rn(values[j],__half2float(__ushort_as_half(matrix.svh[first+j]))));output[pair*matrix.rows+first+j]=value;Finite(value,bad);}
}
__global__ void DescribeTiled(const int* counts,ExpertTile* tiles,int* total,int* bad,const SimtBatch* batch){
  if(threadIdx.x)return;int n=0,sum=0;for(int expert=0;expert<64;++expert){const int count=counts[expert];if(count<0 || count>batch->rows){*total=0;atomicExch(bad,1);return;}sum+=count;for(int i=0;i<count;i+=16){if(n>=191){*total=0;atomicExch(bad,1);return;}tiles[n++]={expert,i};}}if(sum!=batch->rows*8){n=0;atomicExch(bad,1);}*total=n;
}
inline void ExpertsTiled(const DeviceMatrix* matrices,int layer,int part,const int* ids,const float* input,float* output,float* transformed,const int* groups,const int* counts,const ExpertTile* tiles,const int* total,int* bad,const SimtBatch* batch,cudaStream_t stream){
  const int cols=part==2?896:2304,rows=part==2?2304:896;
  InputHad<<<dim3(cols/128,512),128,0,stream>>>(matrices,layer,part,ids,input,transformed,bad,batch);
  ProjectionTiled<<<dim3(rows/64,191),256,0,stream>>>(matrices,layer,part,groups,counts,tiles,total,transformed,output,bad);
  OutputHadTiled<<<dim3(rows/128,512),128,0,stream>>>(matrices,layer,part,ids,output,bad,batch);
}
}
