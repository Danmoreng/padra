// Working-tile BF16 reconstruction only; NVFP4 serialized weights unchanged.
// E2M1*E4M3 SF products and finite A8 codes are exactly BF16-representable.
namespace fast_expert {
__device__ __forceinline__ unsigned PackBF(const __nv_bfloat16* p){return static_cast<unsigned>(__bfloat16_as_ushort(p[0]))|(static_cast<unsigned>(__bfloat16_as_ushort(p[1]))<<16);}
__device__ __forceinline__ float WeightBasis(DeviceMatrix m,int row,int c){const auto byte=m.data[row*(m.cols/2)+c/2];return __fmul_rn(DecodeFP4((byte>>((c&1)*4))&15),DecodeFP8(m.scales[row*(m.cols/16)+c/16]));}
template<bool Fused,bool Selected=false>
__global__ void StagedBF16(const DeviceMatrix* bindings,int layer,const int* groups,const int* counts,const Tile* tiles,const int* tileCount,
    const unsigned char* activation,const float* rowScales,float* output,int tokens,int* bad){
 if(!bindings||!groups||!activation||!rowScales||!output||!bad||layer<0||layer>=28||tokens<1||tokens>512||blockDim.x!=128||blockDim.y!=1||blockDim.z!=1 ||
     (Selected?(tokens!=1||gridDim.y!=8||gridDim.z!=1):(!counts||!tiles||!tileCount||gridDim.y!=static_cast<unsigned>(DescriptorCapacity(tokens))||gridDim.z!=1))){if(threadIdx.x==0&&bad)atomicExch(bad,1);return;}
 int expert,begin,end;if constexpr(Selected){expert=groups[blockIdx.y];begin=0;end=1;}else{if(*tileCount<0 || *tileCount>DescriptorCapacity(tokens)){if(threadIdx.x==0)atomicExch(bad,1);return;}if(blockIdx.y>=static_cast<unsigned>(*tileCount))return;Tile t=tiles[blockIdx.y];expert=t.expert;begin=t.begin;if(expert<0||expert>=64){if(threadIdx.x==0)atomicExch(bad,1);return;}end=counts[expert];}
 if(expert<0||expert>=64||begin<0||begin>=tokens||end<begin||end>tokens){if(threadIdx.x==0)atomicExch(bad,1);return;}
 const DeviceMatrix m=bindings[(layer*64+expert)*3+(Fused?0:2)],u=bindings[(layer*64+expert)*3+1];const int rows=Fused?896:2304,cols=Fused?2304:896;
 if(m.format!=2||m.rows!=rows||m.cols!=cols||!m.data||!m.scales||!m.divisor||gridDim.x!=static_cast<unsigned>(rows/64)||(Fused&&(u.format!=2||u.rows!=rows||u.cols!=cols||!u.data||!u.scales||!u.divisor))){if(threadIdx.x==0)atomicExch(bad,1);return;}
 __shared__ __align__(16) __nv_bfloat16 aTile[16*32],wTile[64*32],upTile[Fused?64*32:1];__shared__ int slots[16],invalid;
 if(threadIdx.x==0)invalid=0;__syncthreads();
 if(threadIdx.x<16){int grouped=begin+threadIdx.x;int slot=grouped<end?(Selected?blockIdx.y:groups[expert*tokens+grouped]):-1;slots[threadIdx.x]=slot;if(slot>=tokens*8||slot< -1)atomicExch(&invalid,1);}__syncthreads();if(invalid){if(threadIdx.x==0)atomicExch(bad,1);return;}
 const int lane=threadIdx.x%32,warp=threadIdx.x/32,group=lane/4,tid=lane%4;float acc[2][4]={},up[2][4]={};
 for(int base=0;base<cols;base+=32){
  for(int i=threadIdx.x;i<16*32;i+=128){int slot=slots[i/32],row=Fused?slot/8:slot;aTile[i]=__float2bfloat16_rn(slot>=0?DecodeFP8(activation[row*cols+base+i%32]):0.f);}
  for(int i=threadIdx.x;i<64*32;i+=128){const int r=blockIdx.x*64+i/32,c=base+i%32;wTile[i]=__float2bfloat16_rn(WeightBasis(m,r,c));if constexpr(Fused)upTile[i]=__float2bfloat16_rn(WeightBasis(u,r,c));}
  __syncthreads();
  #pragma unroll
  for(int k=0;k<32;k+=16){unsigned a[4];for(int j=0;j<4;++j)a[j]=PackBF(aTile+(group+(j%2)*8)*32+k+tid*2+(j/2)*8);
   #pragma unroll
   for(int n=0;n<2;++n){unsigned b[2],ub[2];for(int j=0;j<2;++j){const int at=(warp*16+n*8+group)*32+k+tid*2+j*8;b[j]=PackBF(wTile+at);if constexpr(Fused)ub[j]=PackBF(upTile+at);}AttentionMmaBF16(a,b,acc[n]);if constexpr(Fused)AttentionMmaBF16(a,ub,up[n]);}}
  __syncthreads();
 }
 for(int n=0;n<2;++n)for(int j=0;j<4;++j){int slot=slots[group+(j/2)*8],row=blockIdx.x*64+warp*16+n*8+tid*2+j%2;if(slot<0)continue;float sf=rowScales[Fused?slot/8:slot],value=B((acc[n][j]*sf)/(*m.divisor));if constexpr(Fused){const float other=B((up[n][j]*sf)/(*u.divisor));value=B(B(value/(1.f+expf(-value)))*other);}output[slot*rows+row]=value;Finite(value,bad);}
}
} // namespace fast_expert
