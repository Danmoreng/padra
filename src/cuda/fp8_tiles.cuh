// Scheduling/weight reuse reference: gem16 fp8/sm120.cu, pinned c01789a2.
// Mellum's own per-output compensated exponent-bin math is preserved.
template<int Tiles>
__device__ __forceinline__ void ProjectFP8TilesBody(DeviceMatrix m,const unsigned char* x,const float* scales,float* y,int tokens,int* bad,int base,int colbase){
  static_assert(Tiles>=1 && Tiles<=4);
  const int lane=threadIdx.x&31,group=lane/4,tid=lane%4;
  if(colbase>=m.rows)return;
  float sum[Tiles][4]={},error[Tiles][4]={};
  for(int k=0;k<m.cols;k+=32){
    unsigned b[2],bb[3][2];bool bv[3];
    #pragma unroll
    for(int j=0;j<2;++j)b[j]=Load32(m.data+static_cast<std::size_t>(colbase+group)*m.cols+k+tid*4+j*16);
    #pragma unroll
    for(int bin=0;bin<3;++bin){unsigned bx=0;
      #pragma unroll
      for(int j=0;j<2;++j){bb[bin][j]=FP8Bin(b[j],bin);bx|=bb[bin][j]&0x7f7f7f7fu;}
      bv[bin]=__any_sync(0xffffffff,bx!=0);
    }
    #pragma unroll
    for(int tile=0;tile<Tiles;++tile){
      if(base+tile*16>=tokens)continue;
      unsigned a[4],ab[3][4];bool av[3];
      #pragma unroll
      for(int j=0;j<4;++j){const int row=base+tile*16+group+(j%2)*8,c=k+tid*4+(j/2)*16;
        a[j]=row<tokens?Load32(x+static_cast<std::size_t>(row)*m.cols+c):0;}
      #pragma unroll
      for(int bin=0;bin<3;++bin){unsigned ax=0;
        #pragma unroll
        for(int j=0;j<4;++j){ab[bin][j]=FP8Bin(a[j],bin);ax|=ab[bin][j]&0x7f7f7f7fu;}
        av[bin]=__any_sync(0xffffffff,ax!=0);
      }
      #pragma unroll
      for(int ai=0;ai<3;++ai){
        #pragma unroll
        for(int bi=0;bi<3;++bi){if(av[ai] && bv[bi]){float partial[4]={};MmaFP8(ab[ai],bb[bi],partial);
          #pragma unroll
          for(int j=0;j<4;++j)Accumulate(partial[j],sum[tile][j],error[tile][j]);}}
      }
    }
  }
  #pragma unroll
  for(int tile=0;tile<Tiles;++tile){
    #pragma unroll
    for(int j=0;j<4;++j){const int row=base+tile*16+group+(j/2)*8,col=colbase+tid*2+j%2;
      if(row<tokens){const float dot=__fadd_rn(sum[tile][j],error[tile][j]);const float value=B(__fmul_rn(__fmul_rn(dot,scales[row]),BF(m.scales,col)));
        y[static_cast<std::size_t>(row)*m.rows+col]=value;Finite(value,bad);}}
  }
}

template<int Tiles>
__global__ void ProjectFP8Tiles(DeviceMatrix m,const unsigned char* x,const float* scales,float* y,int tokens,int* bad){
  if(m.format!=1 || m.rows<=0 || m.rows%8 || m.cols<=0 || m.cols%32 || tokens<=0 || tokens>512 ||
     !m.data || !m.scales || !x || !scales || !y || !bad || blockDim.x!=128 || blockDim.y!=1 || blockDim.z!=1 ||
     blockIdx.x>=static_cast<unsigned>((m.rows+31)/32) || blockIdx.y>=static_cast<unsigned>((tokens+16*Tiles-1)/(16*Tiles)) || blockIdx.z!=0){
    if(threadIdx.x==0 && bad)atomicExch(bad,1);return;
  }
  ProjectFP8TilesBody<Tiles>(m,x,scales,y,tokens,bad,blockIdx.y*(16*Tiles),(blockIdx.x*4+threadIdx.x/32)*8);
}

template<int Tiles>
__global__ void ProjectQKVTiles(DeviceMatrix query,DeviceMatrix key,DeviceMatrix value,const unsigned char* x,const float* scales,float* q,float* k,float* v,int tokens,int* bad){
  constexpr unsigned qtiles=128,kvtiles=16;
  if(query.format!=1 || key.format!=1 || value.format!=1 || query.cols!=2304 || key.cols!=2304 || value.cols!=2304 ||
     query.rows!=4096 || key.rows!=512 || value.rows!=512 || !query.data || !query.scales || !key.data || !key.scales || !value.data || !value.scales ||
     !x || !scales || !q || !k || !v || !bad || tokens<=0 || tokens>512 || blockDim.x!=128 || blockDim.y!=1 || blockDim.z!=1 ||
     blockIdx.x>=qtiles+2*kvtiles || blockIdx.y>=static_cast<unsigned>((tokens+16*Tiles-1)/(16*Tiles)) || blockIdx.z!=0){
    if(threadIdx.x==0 && bad)atomicExch(bad,1);return;
  }
  const unsigned block=blockIdx.x;
  const bool is_query=block<qtiles,is_key=!is_query && block<qtiles+kvtiles;
  const DeviceMatrix matrix=is_query?query:(is_key?key:value);
  float* output=is_query?q:(is_key?k:v);
  const unsigned local_block=is_query?block:(is_key?block-qtiles:block-qtiles-kvtiles);
  ProjectFP8TilesBody<Tiles>(matrix,x,scales,output,tokens,bad,blockIdx.y*(16*Tiles),(local_block*4+threadIdx.x/32)*8);
}
