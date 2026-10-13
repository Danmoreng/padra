// Standalone SM61 experiment: lossless permutations of compressed EXL3 words.
// Layout 0: native storage, four scalar loads per K64.
// Layout 1: [N16][K64][lane][four K16 words], one uint4 load per lane.
// Layout 2: [N128][K64][N16 within N128][lane][four K16 words].
// No weight expansion, requantization, or sidecar changes. Not runtime dispatch.
#include "runtime/inference.h"
#include "cuda/device_matrix.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace mellum16 {namespace {
#include "cuda/simt_kernels.cuh"
namespace layout_candidate {
template<int Layout=0> __global__ void Projection(const DeviceMatrix* matrices,int layer,int part,const int* ids,
    const float* input,float* output,int pairs,int* bad,const int* active_pairs=nullptr){
  const int pair=blockIdx.y,lane=threadIdx.x&31,warp=threadIdx.x/32;
  if(active_pairs && pair>=*active_pairs*8)return;
  const auto m=exl3_expert::Matrix(matrices,layer,part,ids,pair,bad);
  const int base=blockIdx.x*128,col=lane/4,ki=(lane%4)*2;
  if(pair>=pairs||m.format!=3||base+128>m.rows||m.cols%64){if(!threadIdx.x)atomicExch(bad,1);return;}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 610
  __shared__ float codec[1024];exl3_pascal::InitLookup(codec);
#endif
  // Matrix/payload pointers are native64-bit; no uint32 pool-offset cast.
  __shared__ float transformed[2304],raw[128];
  for(int k=threadIdx.x;k<m.cols;k+=256)transformed[k]=input[pair*m.cols+k];
  __syncthreads();
  float low=0,high=0;uint4 prefetched;
  for(int k=0;k<m.cols;k+=16){
    if((k&63)==0){
      const int nt=base/16+warp,kb=k/64;
      if constexpr(Layout==0){const std::uint64_t t=std::uint64_t(k/16)*std::uint64_t(m.rows/16)+nt;
        const unsigned* p=reinterpret_cast<const unsigned*>(m.data)+t*32+lane;const int stride=(m.rows/16)*32;
        prefetched=make_uint4(p[0],p[stride],p[2*stride],p[3*stride]);
      }else{const std::uint64_t group=Layout==1?std::uint64_t(nt)*(m.cols/64)+kb:(std::uint64_t(nt/8)*(m.cols/64)+kb)*8+nt%8;
        prefetched=reinterpret_cast<const uint4*>(m.data)[group*32+lane];}
    }
    const int slot=(k/16)&3;const unsigned word=slot==0?prefetched.x:slot==1?prefetched.y:slot==2?prefetched.z:prefetched.w;
    const unsigned previous=exl3_pascal::Previous(word);
    #pragma unroll
    for(int half=0;half<2;++half){
      const int index=ki+half*8;const float x=transformed[k+index],y=transformed[k+index+1];
      float2 a,b;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 610
      if constexpr(true){
        a=exl3_pascal::PairLookup(__funnelshift_r(word,previous,half==0?24:16),codec);
        b=exl3_pascal::PairLookup(half==0?__funnelshift_r(word,word,8):word,codec);
      }else
#endif
      {a=__half22float2(exl3_selected::DecodePair(word,index,col));b=__half22float2(exl3_selected::DecodePair(word,index,col+8));}
      low=__fmaf_rn(x,a.x,low);low=__fmaf_rn(y,a.y,low);
      high=__fmaf_rn(x,b.x,high);high=__fmaf_rn(y,b.y,high);
    }
  }
  #pragma unroll
  for(int d=1;d<=2;d*=2){low=__fadd_rn(low,__shfl_xor_sync(0xffffffffU,low,d));high=__fadd_rn(high,__shfl_xor_sync(0xffffffffU,high,d));}
  if((lane&3)==0){raw[warp*16+col]=low;raw[warp*16+col+8]=high;}
  __syncthreads();
  if(!warp){const int index=lane*4;float v[4]={raw[index],raw[index+1],raw[index+2],raw[index+3]};exl3_fp16::H128(v);
    #pragma unroll
    for(int j=0;j<4;++j){const int i=base+index+j;const float value=B(__fmul_rn(v[j],__half2float(__ushort_as_half(m.svh[i]))));if(!isfinite(value))atomicExch(bad,1);output[pair*m.rows+i]=value;}}
}
}

void Require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
__global__ void CompareFP8Codec(int* bad){const unsigned code=threadIdx.x;if(code>=256)return;const auto original=DecodeFP8FourReference(code*0x01010101U),optimized=DecodeFP8Four(code*0x01010101U);if(__float_as_uint(original.x)!=__float_as_uint(optimized.x) || __float_as_uint(original.y)!=__float_as_uint(optimized.y) || __float_as_uint(original.z)!=__float_as_uint(optimized.z) || __float_as_uint(original.w)!=__float_as_uint(optimized.w))atomicAdd(bad,1);}
__global__ void CompareLookupCodec(int* bad){__shared__ float lookup[1024];exl3_pascal::InitLookup(lookup);const unsigned state=blockIdx.x*256+threadIdx.x;if(__float_as_uint(exl3_pascal::DecodeLookup(state,lookup))!=__float_as_uint(__half2float(exl3_fp16::Mul1(state))))atomicAdd(bad,1);}
__global__ void ComparePascalCodec(int* bad){const unsigned state=blockIdx.x*256+threadIdx.x;if(state<65536 && __float_as_uint(exl3_pascal::Decode(state))!=__float_as_uint(__half2float(exl3_fp16::Mul1(state))))atomicAdd(bad,1);}
template<class F> float Time(F launch){
  for(int i=0;i<30;++i)launch();Check(cudaDeviceSynchronize());cudaEvent_t a,b;Check(cudaEventCreate(&a));Check(cudaEventCreate(&b));Check(cudaEventRecord(a));for(int i=0;i<100;++i)launch();Check(cudaEventRecord(b));Check(cudaEventSynchronize(b));float ms=0;Check(cudaEventElapsedTime(&ms,a,b));cudaEventDestroy(a);cudaEventDestroy(b);return ms/100;
}
void Projection(int part){
  const int rows=part==2?2304:896,cols=part==2?896:2304,pairs=8;
  const std::size_t bytes=std::size_t(rows)*cols/2;
  std::vector<unsigned> payload(bytes*pairs/4);unsigned random=19;for(auto& x:payload){random=random*1664525U+1013904223U;x=random;}
  std::vector<float> input(pairs*cols);for(std::size_t i=0;i<input.size();++i)input[i]=__bfloat162float(__float2bfloat16_rn(i%13==0?0.f:i%17==0?std::ldexp(1.f,-30):std::sin(float(i))*.25f));
  std::vector<unsigned short> side(pairs*(rows+cols));for(int e=0;e<pairs;++e)for(int i=0;i<rows+cols;++i)side[e*(rows+cols)+i]=__half_as_ushort(__float2half_rn(float((i+e*5)%31-15)/128));
  Buffer packed(payload.size()*4),sides(side.size()*2),source(input.size()*4),transform(input.size()*4),warp_transform(input.size()*4),reference(pairs*rows*4),output(pairs*rows*4),descriptors(64*3*sizeof(DeviceMatrix)),ids(pairs*4),bad(4);
  Check(cudaMemcpy(packed.p,payload.data(),packed.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(sides.p,side.data(),sides.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(source.p,input.data(),source.bytes,cudaMemcpyHostToDevice));
  std::vector<DeviceMatrix> table(64*3);std::vector<int> chosen(pairs);
  for(int e=0;e<pairs;++e){chosen[e]=(e*7)%64;table[chosen[e]*3+part]=DeviceMatrix{packed.As<unsigned char>()+e*bytes,nullptr,nullptr,rows,cols,3,sides.As<unsigned short>()+e*(rows+cols),sides.As<unsigned short>()+e*(rows+cols)+cols};}
  Check(cudaMemcpy(descriptors.p,table.data(),descriptors.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(ids.p,chosen.data(),ids.bytes,cudaMemcpyHostToDevice));Check(cudaMemset(bad.p,0,4));
  exl3_expert::InputHad<<<dim3(cols/128,pairs),128>>>(descriptors.As<DeviceMatrix>(),0,part,ids.As<int>(),source.As<float>(),transform.As<float>(),pairs,bad.As<int>());
  exl3_pascal::InputHad<<<dim3(cols/128,2),128>>>(descriptors.As<DeviceMatrix>(),0,part,ids.As<int>(),source.As<float>(),warp_transform.As<float>(),pairs,bad.As<int>());Check(cudaDeviceSynchronize());std::vector<unsigned char> original_had(transform.bytes),warp_had(transform.bytes);Check(cudaMemcpy(original_had.data(),transform.p,transform.bytes,cudaMemcpyDeviceToHost));Check(cudaMemcpy(warp_had.data(),warp_transform.p,transform.bytes,cudaMemcpyDeviceToHost));Require(original_had==warp_had,"warp H128 differs for real gate/up/down inputs and sidecars");
  auto old=[&]{exl3_selected::Projection<false><<<dim3(rows/128,pairs),256>>>(descriptors.As<DeviceMatrix>(),0,part,ids.As<int>(),transform.As<float>(),reference.As<float>(),pairs,bad.As<int>());};
  Buffer repacked(packed.bytes),alternate(descriptors.bytes);
  for(int layout=0;layout<3;++layout){
    std::vector<unsigned> native(payload.size());
    for(int e=0;e<pairs;++e)for(int nt=0;nt<rows/16;++nt)for(int kt=0;kt<cols/16;++kt)for(int lane=0;lane<32;++lane){const size_t oldidx=size_t(e)*bytes/4+(size_t(kt)*(rows/16)+nt)*32+lane;
      const size_t group=layout==1?size_t(nt)*(cols/64)+kt/4:(size_t(nt/8)*(cols/64)+kt/4)*8+nt%8;
      const size_t dst=size_t(e)*bytes/4+group*128+lane*4+kt%4;native[dst]=payload[oldidx];}
    Check(cudaMemcpy(repacked.p,native.data(),repacked.bytes,cudaMemcpyHostToDevice));
    auto tab=table;for(int e=0;e<pairs;++e)tab[chosen[e]*3+part].data=(layout?repacked:packed).As<unsigned char>()+e*bytes;
    Check(cudaMemcpy(alternate.p,tab.data(),alternate.bytes,cudaMemcpyHostToDevice));
    auto opt=[&]{
#define RUN(L) layout_candidate::Projection<L><<<dim3(rows/128,pairs),256>>>(alternate.As<DeviceMatrix>(),0,part,ids.As<int>(),transform.As<float>(),output.As<float>(),pairs,bad.As<int>())
      switch(layout){case 0:RUN(0);break;case 1:RUN(1);break;case 2:RUN(2);break;}
#undef RUN
    };
    old();opt();Check(cudaDeviceSynchronize());std::vector<float> a(pairs*rows),b(pairs*rows);Check(cudaMemcpy(a.data(),reference.p,reference.bytes,cudaMemcpyDeviceToHost));Check(cudaMemcpy(b.data(),output.p,output.bytes,cudaMemcpyDeviceToHost));Require(!memcmp(a.data(),b.data(),reference.bytes),"layout changes result");
    int flag=0;Check(cudaMemcpy(&flag,bad.p,4,cudaMemcpyDeviceToHost));Require(!flag,"invalid layout projection");
    for(int rep=0;rep<5;++rep){float before,after;if(rep%2){after=Time(opt);before=Time(old);}else{before=Time(old);after=Time(opt);}std::cout<<"LAYOUT part="<<part<<" layout="<<layout<<" rep="<<rep<<" old_ms="<<before<<" new_ms="<<after<<'\n';}
  }

}
}}
int main(){try{using namespace mellum16;Buffer bad(4);Check(cudaMemset(bad.p,0,4));ComparePascalCodec<<<256,256>>>(bad.As<int>());CompareLookupCodec<<<256,256>>>(bad.As<int>());CompareFP8Codec<<<1,256>>>(bad.As<int>());Check(cudaDeviceSynchronize());int count=0;Check(cudaMemcpy(&count,bad.p,4,cudaMemcpyDeviceToHost));Require(count==0,"Pascal codec differs in exhaustive test");for(int part=0;part<3;++part)Projection(part);std::cout<<"All 65536 codec states and distinct real-shape gate/up/down projections are bit-identical\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
