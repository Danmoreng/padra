// Diagnostic candidate: existing Mellum/CUTLASS m16n8k32 register ABI.
// Three disjoint E4M3 exponent bins bound each K32 partial's integer dot.
__device__ unsigned FP8Bin(unsigned word,int bin){
  unsigned out=0;
  #pragma unroll
  for(int byte=0;byte<4;++byte){const unsigned code=(word>>(byte*8))&255,exp=(code&127)>>3;
    const int group=exp<6?0:exp<11?1:2;if(group==bin)out|=code<<(byte*8);}
  return out;
}
template<bool Binned>
__device__ __forceinline__ void ProjectFP8PartialTile(DeviceMatrix m,const unsigned char* x,const float* scales,float* y,int tokens,int* bad,int base,int colbase){
  if(m.format!=1 || m.rows<=0 || m.rows%8 || m.cols<=0 || m.cols%32 || tokens<=0){if(threadIdx.x==0)atomicExch(bad,1);return;}
  const int lane=threadIdx.x&31,group=lane/4,tid=lane%4;
  if(colbase>=m.rows)return;
  float sum[4]={},error[4]={};
  for(int k=0;k<m.cols;k+=32){unsigned a[4],b[2];
    #pragma unroll
    for(int j=0;j<4;++j){const int row=base+group+(j%2)*8,c=k+tid*4+(j/2)*16;
      a[j]=row<tokens?Load32(x+static_cast<std::size_t>(row)*m.cols+c):0;}
    #pragma unroll
    for(int j=0;j<2;++j)b[j]=Load32(m.data+static_cast<std::size_t>(colbase+group)*m.cols+k+tid*4+j*16);
    if constexpr(Binned){
      unsigned ab[3][4],bb[3][2];bool av[3],bv[3];
      #pragma unroll
      for(int bin=0;bin<3;++bin){unsigned ax=0,bx=0;
        #pragma unroll
        for(int j=0;j<4;++j){ab[bin][j]=FP8Bin(a[j],bin);ax|=ab[bin][j]&0x7f7f7f7fu;}
        #pragma unroll
        for(int j=0;j<2;++j){bb[bin][j]=FP8Bin(b[j],bin);bx|=bb[bin][j]&0x7f7f7f7fu;}
        av[bin]=__any_sync(0xffffffff,ax!=0);bv[bin]=__any_sync(0xffffffff,bx!=0);
      }
      #pragma unroll
      for(int ai=0;ai<3;++ai){
        #pragma unroll
        for(int bi=0;bi<3;++bi){if(av[ai] && bv[bi]){float partial[4]={};MmaFP8(ab[ai],bb[bi],partial);
          #pragma unroll
          for(int j=0;j<4;++j)Accumulate(partial[j],sum[j],error[j]);}}
      }
    }else{
      #pragma unroll
      for(int half=0;half<2;++half){unsigned ah[4]={};ah[half*2]=a[half*2];ah[half*2+1]=a[half*2+1];
        float partial[4]={};MmaFP8(ah,b,partial);
        #pragma unroll
        for(int j=0;j<4;++j)Accumulate(partial[j],sum[j],error[j]);}
    }
  }
  #pragma unroll
  for(int j=0;j<4;++j){const int row=base+group+(j/2)*8,col=colbase+tid*2+j%2;
    if(row<tokens){const float dot=__fadd_rn(sum[j],error[j]);const float value=B(__fmul_rn(__fmul_rn(dot,scales[row]),BF(m.scales,col)));
      y[static_cast<std::size_t>(row)*m.rows+col]=value;Finite(value,bad);}}
}

template<bool Binned>
__global__ void ProjectFP8Partial(DeviceMatrix m,const unsigned char* x,const float* scales,float* y,int tokens,int* bad){
  ProjectFP8PartialTile<Binned>(m,x,scales,y,tokens,bad,blockIdx.y*16,(blockIdx.x*4+threadIdx.x/32)*8);
}
