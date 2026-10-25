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
void Require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
__global__ void ReadK3Windows(DeviceMatrix matrix,unsigned* output){
  const unsigned lane=threadIdx.x,tile=blockIdx.x;
  const exl3_pascal::Windows window(matrix,tile,lane,3);
  auto* row=output+(tile*32+lane)*6;
  row[0]=window.word;row[1]=window.previous;row[2]=window.Low(0);
  row[3]=window.Low(1);row[4]=window.High(0);row[5]=window.High(1);
}
void K3Windows(){
  constexpr unsigned tiles=1024;
  std::vector<unsigned char> payload(tiles*96);unsigned seed=0x17a2b3;
  for(auto& byte:payload){seed=1664525U*seed+1013904223U;byte=seed>>24;}
  Buffer source(payload.size()),output(tiles*32*6*4);
  Check(cudaMemcpy(source.p,payload.data(),source.bytes,cudaMemcpyHostToDevice));
  DeviceMatrix matrix{};matrix.data=source.As<unsigned char>();matrix.format=6;
  ReadK3Windows<<<tiles,32>>>(matrix,output.As<unsigned>());
  std::vector<unsigned> actual(tiles*32*6);
  Check(cudaMemcpy(actual.data(),output.p,output.bytes,cudaMemcpyDeviceToHost));
  for(unsigned tile=0;tile<tiles;++tile)for(unsigned lane=0;lane<32;++lane){
    const auto load=[&](unsigned l){const auto* bytes=payload.data()+tile*96;const auto at=l*3;
      return (unsigned(bytes[at^3])<<16)|(unsigned(bytes[(at+1)^3])<<8)|unsigned(bytes[(at+2)^3]);};
    const unsigned word=load(lane),previous=load((lane+31)%32);
    const std::array<unsigned,6> expected{word,previous,(previous<<6)|(word>>18),
      (previous<<12)|(word>>12),(previous<<18)|(word>>6),word};
    Require(std::equal(expected.begin(),expected.end(),actual.begin()+(tile*32+lane)*6),
            "K3 lane windows differ from byte-address oracle");
  }
}
__global__ void CompareFP8Codec(int* bad){const unsigned code=threadIdx.x;if(code>=256)return;const auto original=DecodeFP8FourReference(code*0x01010101U),optimized=DecodeFP8Four(code*0x01010101U);if(__float_as_uint(original.x)!=__float_as_uint(optimized.x) || __float_as_uint(original.y)!=__float_as_uint(optimized.y) || __float_as_uint(original.z)!=__float_as_uint(optimized.z) || __float_as_uint(original.w)!=__float_as_uint(optimized.w))atomicAdd(bad,1);}
__global__ void CompareLookupCodec(int* bad){__shared__ float lookup[1024];exl3_pascal::InitLookup(lookup);const unsigned state=blockIdx.x*256+threadIdx.x;if(__float_as_uint(exl3_pascal::DecodeLookup(state,lookup))!=__float_as_uint(__half2float(exl3_fp16::Mul1(state))))atomicAdd(bad,1);}
__global__ void ComparePascalCodec(int* bad){const unsigned state=blockIdx.x*256+threadIdx.x;if(state<65536 && __float_as_uint(exl3_pascal::Decode(state))!=__float_as_uint(__half2float(exl3_fp16::Mul1(state))))atomicAdd(bad,1);}
template<class F> float Time(F launch){
  for(int i=0;i<30;++i)launch();Check(cudaDeviceSynchronize());cudaEvent_t a,b;Check(cudaEventCreate(&a));Check(cudaEventCreate(&b));Check(cudaEventRecord(a));for(int i=0;i<100;++i)launch();Check(cudaEventRecord(b));Check(cudaEventSynchronize(b));float ms=0;Check(cudaEventElapsedTime(&ms,a,b));cudaEventDestroy(a);cudaEventDestroy(b);return ms/100;
}
void Projection(int part,bool benchmark){
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
  auto old=[&]{exl3_selected::Projection<true><<<dim3(rows/128,pairs),256>>>(descriptors.As<DeviceMatrix>(),0,part,ids.As<int>(),transform.As<float>(),reference.As<float>(),pairs,bad.As<int>());};
  auto opt=[&]{exl3_selected::Projection<false><<<dim3(rows/128,pairs),256>>>(descriptors.As<DeviceMatrix>(),0,part,ids.As<int>(),transform.As<float>(),output.As<float>(),pairs,bad.As<int>());};
  old();opt();Check(cudaDeviceSynchronize());std::vector<float> a(pairs*rows),b(pairs*rows);Check(cudaMemcpy(a.data(),reference.p,reference.bytes,cudaMemcpyDeviceToHost));Check(cudaMemcpy(b.data(),output.p,output.bytes,cudaMemcpyDeviceToHost));int flag=0;Check(cudaMemcpy(&flag,bad.p,4,cudaMemcpyDeviceToHost));Require(!flag,"selected projection invalid");Require(!memcmp(a.data(),b.data(),reference.bytes),"real-shape projection codec mismatch");
  if(benchmark)for(int i=0;i<5;++i){float before,after;if(i%2){after=Time(opt);before=Time(old);}else{before=Time(old);after=Time(opt);}std::cout<<"part="<<part<<" rows="<<rows<<" cols="<<cols<<" rep="<<i<<" original_ms="<<before<<" pascal_ms="<<after<<" speedup="<<before/after<<'\n';}
}
}}
int main(int argc,char** argv){try{using namespace mellum16;K3Windows();if(argc>1 && std::string(argv[1])=="--k3-windows"){std::cout<<"K3 packed lane windows match byte oracle\n";return 0;}Buffer bad(4);Check(cudaMemset(bad.p,0,4));ComparePascalCodec<<<256,256>>>(bad.As<int>());CompareLookupCodec<<<256,256>>>(bad.As<int>());CompareFP8Codec<<<1,256>>>(bad.As<int>());Check(cudaDeviceSynchronize());int count=0;Check(cudaMemcpy(&count,bad.p,4,cudaMemcpyDeviceToHost));Require(count==0,"Pascal codec differs in exhaustive test");for(int part=0;part<3;++part)Projection(part,argc>1);std::cout<<"All 65536 codec states and distinct real-shape gate/up/down projections are bit-identical\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
