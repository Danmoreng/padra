#pragma once
// Optional SM61 arithmetic profile. EXL3 storage is unchanged; dense Prefill
// temporarily approximates codec values and H128 activations as INT8.
namespace exl3_pascal_dp4a {
constexpr int DenseThreshold=16;
constexpr float WeightScale=3.453125f/127.f;
__device__ __forceinline__ unsigned QuantCode(unsigned state,const int* codec){return unsigned(codec[__dp4a(state*0x83DCD12DU,0x01010101U,0U)])&255U;}
__device__ __forceinline__ unsigned Pair(unsigned window,const int* codec,int rate){return QuantCode((window>>rate)&65535U,codec)|(QuantCode(window&65535U,codec)<<8);}
// Dense rows only: retain compressed storage; quantize transformed activations
// in their existing scratch, in independent H128 blocks (32 packed words+scale).
__global__ void Quantize(const DeviceMatrix* matrices,int layer,int part,const int* ids,const int* counts,float* input,const SimtBatch* batch){
 const int pair=blockIdx.y*4+threadIdx.x/32,lane=threadIdx.x&31;
 if(pair>=batch->rows*8 || counts[ids[pair]]<DenseThreshold)return;
 const auto m=matrices[(layer*64+ids[pair])*3+part];float* p=input+pair*m.cols+blockIdx.x*128;
 float v[4],maximum=0.f;for(int j=0;j<4;++j){v[j]=p[lane*4+j];maximum=fmaxf(maximum,fabsf(v[j]));}
 for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffU,maximum,d));
 const float scale=maximum>0?maximum/127.f:1.f;unsigned word=0;
 for(int j=0;j<4;++j)word|=(unsigned(max(-127,min(127,__float2int_rn(v[j]/scale))))&255U)<<(j*8);
 __syncwarp();reinterpret_cast<unsigned*>(p)[lane]=word;if(!lane)p[32]=scale;
}
template<int Rate> __global__ __launch_bounds__(256,3) void Projection(const DeviceMatrix* matrices,int layer,int part,const int* groups,const int* counts,const simt_batch::ExpertTile* tiles,const int* total,const float* input,float* output,int* bad){
 static_assert(Rate==3 || Rate==4);
 if(blockIdx.y>=unsigned(*total))return;const auto tile=tiles[blockIdx.y];if(tile.begin%32 || counts[tile.expert]-tile.begin<DenseThreshold)return;
 const auto m=matrices[(layer*64+tile.expert)*3+part];const int lane=threadIdx.x&31,warp=threadIdx.x/32,first=blockIdx.x*128,mr=warp*4;
 __shared__ int weights[16][129],values[32][17],codec[1024],pairs[32];__shared__ float factors[32];
 for(int s=threadIdx.x;s<1024;s+=256){const float w=__half2float(__float2half_rn(__fmaf_rn(float(1024+s),__half2float(__ushort_as_half(0x1eee)),__half2float(__ushort_as_half(0xc931)))));codec[s]=max(-127,min(127,__float2int_rn(w/WeightScale)));}
 if(threadIdx.x<32)pairs[threadIdx.x]=tile.begin+threadIdx.x<counts[tile.expert]?groups[tile.expert*256+tile.begin+threadIdx.x]:-1;
 __syncthreads();float sums[4][4]={};
 for(int base=0;base<m.cols;base+=64){
  for(int i=threadIdx.x;i<32*16;i+=256){const int r=i/16,k=i%16;values[r][k]=pairs[r]<0?0:reinterpret_cast<const int*>(input)[pairs[r]*m.cols+(base/128)*128+(base%128)/4+k];}
  if(threadIdx.x<32)factors[threadIdx.x]=pairs[threadIdx.x]<0?0.f:input[pairs[threadIdx.x]*m.cols+(base/128)*128+32]*WeightScale;
  #pragma unroll
  for(int kt=0;kt<4;++kt){const exl3_pascal::Windows windows(m,std::uint64_t(base/16+kt)*(m.rows/16)+first/16+warp,lane,Rate);const int ki=(lane&3)*2,col=lane/4;
   #pragma unroll
   for(int half=0;half<2;++half){const unsigned lo=Pair(windows.Low(half),codec,Rate),hi=Pair(windows.High(half),codec,Rate);const unsigned lop=__shfl_xor_sync(0xffffffffU,lo,1),hip=__shfl_xor_sync(0xffffffffU,hi,1);if(!(lane&1)){weights[kt*4+(ki+half*8)/4][warp*16+col]=lo|(lop<<16);weights[kt*4+(ki+half*8)/4][warp*16+col+8]=hi|(hip<<16);}}
  }
  __syncthreads();int dots[4][4]={};
  #pragma unroll
  for(int k=0;k<16;++k){int a[4];for(int r=0;r<4;++r)a[r]=values[mr+r][k];
   #pragma unroll
   for(int n=0;n<4;++n){const int w=weights[k][lane+n*32];
    #pragma unroll
    for(int r=0;r<4;++r)dots[r][n]=__dp4a(w,a[r],dots[r][n]);}}
  #pragma unroll
  for(int r=0;r<4;++r){const float scale=factors[mr+r];for(int n=0;n<4;++n)sums[r][n]=__fmaf_rn(float(dots[r][n]),scale,sums[r][n]);}
  __syncthreads();
 }
 // Physical Hadamard bit order 0..6 matches H128 despite the strided layout.
 #pragma unroll
 for(int r=0;r<4;++r){
  #pragma unroll
  for(int mask=1;mask<32;mask*=2){
   #pragma unroll
   for(int n=0;n<4;++n){const float other=__shfl_xor_sync(0xffffffffU,sums[r][n],mask);sums[r][n]=(lane&mask)?__fsub_rn(other,sums[r][n]):__fadd_rn(sums[r][n],other);}}
  const float a=__fadd_rn(sums[r][0],sums[r][1]),b=__fsub_rn(sums[r][0],sums[r][1]),c=__fadd_rn(sums[r][2],sums[r][3]),d=__fsub_rn(sums[r][2],sums[r][3]);
  sums[r][0]=__fadd_rn(a,c);sums[r][1]=__fadd_rn(b,d);sums[r][2]=__fsub_rn(a,c);sums[r][3]=__fsub_rn(b,d);
  if(pairs[mr+r]>=0){
   #pragma unroll
   for(int n=0;n<4;++n){const int i=first+lane+n*32;const float value=B(__fmul_rn(__fmul_rn(sums[r][n],.08838834764831845f),__half2float(__ushort_as_half(m.svh[i]))));output[pairs[mr+r]*m.rows+i]=value;Finite(value,bad);}}
 }
}
inline void Launch(const DeviceMatrix* matrices,int layer,int part,const int* ids,const float* input,float* output,float* transformed,const int* groups,const int* counts,const simt_batch::ExpertTile* tiles,const int* total,int* bad,const SimtBatch* batch,cudaStream_t stream,int rate=4){
 if(rate!=3 && rate!=4)throw std::invalid_argument("DP4A prefill requires EXL3 K3 or K4");
 const int cols=part==2?896:2304,rows=part==2?2304:896;
 simt_batch::InputHad<<<dim3(cols/128,512),128,0,stream>>>(matrices,layer,part,ids,input,transformed,bad,batch);
 simt_batch::Projection<true,DenseThreshold><<<dim3(rows/128,simt_batch::kExpertTiles),256,0,stream>>>(matrices,layer,part,groups,counts,tiles,total,transformed,output,bad);
 Quantize<<<dim3(cols/128,512),128,0,stream>>>(matrices,layer,part,ids,counts,transformed,batch);
 if(rate==3)exl3_pascal_dp4a::Projection<3><<<dim3(rows/128,simt_batch::kExpertTiles),256,0,stream>>>(matrices,layer,part,groups,counts,tiles,total,transformed,output,bad);
 else exl3_pascal_dp4a::Projection<4><<<dim3(rows/128,simt_batch::kExpertTiles),256,0,stream>>>(matrices,layer,part,groups,counts,tiles,total,transformed,output,bad);
}
}
