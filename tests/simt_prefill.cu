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
#include "cuda/simt_prefill.cuh"
#include "cuda/simt_prefill_attention.cuh"
void Require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
__nv_bfloat16 Value(int pos,int d,bool key){return __float2bfloat16_rn(float((pos*(key?3:5)+d)%17-8)/(key?32:8));}
void Attention(bool local,int base,int rows){
  const int capacity=local?1024:8192;
  std::vector<__nv_bfloat16> keys(capacity*512,__float2bfloat16_rn(NAN)),values(keys),current(256*1024,__float2bfloat16_rn(NAN));
  for(int pos=local?std::max(0,base-1024):0;pos<base;++pos)for(int d=0;d<512;++d){keys[(pos%capacity)*512+d]=Value(pos,d,true);values[(pos%capacity)*512+d]=Value(pos,d,false);}
  for(int row=0;row<rows;++row)for(int d=0;d<512;++d){current[row*512+d]=Value(base+row,d,true);current[(256+row)*512+d]=Value(base+row,d,false);}
  std::vector<float> query(256*4096,NAN);for(int row=0;row<rows;++row)for(int i=0;i<4096;++i)query[row*4096+i]=float((row*5+i)%11-5)/32;
  Buffer q(query.size()*4),k(keys.size()*2),v(values.size()*2),stage(current.size()*2),state(sizeof(SimtBatch)),command(sizeof(SimtCommand)),workspace(8ULL*32*8192*4),result(256*4096*4),reference(256*4096*4),bad(4);
  Check(cudaMemcpy(q.p,query.data(),q.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(k.p,keys.data(),k.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(v.p,values.data(),v.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(stage.p,current.data(),stage.bytes,cudaMemcpyHostToDevice));Check(cudaMemset(bad.p,0,4));
  SetSimtBatch<<<1,1>>>(state.As<SimtBatch>(),base,rows,8192,bad.As<int>());
  for(int first=0;first<256;first+=8){if(local)AttendSimtBatch<true><<<dim3(32,8),256>>>(q.As<float>(),k.As<__nv_bfloat16>(),v.As<__nv_bfloat16>(),result.As<float>(),state.As<SimtBatch>(),capacity,workspace.As<float>(),bad.As<int>(),stage.As<__nv_bfloat16>(),first);else AttendSimtBatch<false><<<dim3(32,8),256>>>(q.As<float>(),k.As<__nv_bfloat16>(),v.As<__nv_bfloat16>(),result.As<float>(),state.As<SimtBatch>(),capacity,workspace.As<float>(),bad.As<int>(),stage.As<__nv_bfloat16>(),first);}
  // Only after all batched reads: sequential baseline updates the same ring.
  for(int row=0;row<rows;++row){const int pos=base+row;Check(cudaMemcpy(k.As<__nv_bfloat16>()+(pos%capacity)*512,current.data()+row*512,1024,cudaMemcpyHostToDevice));Check(cudaMemcpy(v.As<__nv_bfloat16>()+(pos%capacity)*512,current.data()+(256+row)*512,1024,cudaMemcpyHostToDevice));SetSimtCommand<<<1,1>>>(command.As<SimtCommand>(),0,pos);
    if(local)AttendSimt<true><<<32,256>>>(q.As<float>()+row*4096,k.As<__nv_bfloat16>(),v.As<__nv_bfloat16>(),reference.As<float>()+row*4096,command.As<SimtCommand>(),capacity,workspace.As<float>(),bad.As<int>());else AttendSimt<false><<<32,256>>>(q.As<float>()+row*4096,k.As<__nv_bfloat16>(),v.As<__nv_bfloat16>(),reference.As<float>()+row*4096,command.As<SimtCommand>(),capacity,workspace.As<float>(),bad.As<int>());
  }
  Check(cudaDeviceSynchronize());std::vector<float> a(rows*4096),b(a);Check(cudaMemcpy(a.data(),result.p,a.size()*4,cudaMemcpyDeviceToHost));Check(cudaMemcpy(b.data(),reference.p,b.size()*4,cudaMemcpyDeviceToHost));Require(!memcmp(a.data(),b.data(),a.size()*4),"batch attention differs from sequential causal/ring reference");int flag=0;Check(cudaMemcpy(&flag,bad.p,4,cudaMemcpyDeviceToHost));Require(!flag,"batch attention read poisoned future/tail rows");
}
}}
int main(){try{using namespace mellum16;for(bool local:{false,true})for(auto shape:std::array<std::pair<int,int>,5>{{{0,3},{1017,31},{1024,256},{2000,16},{8000,128}}})Attention(local,shape.first,shape.second);std::cout<<"Staged causal batch attention, tails and local ring wrap are bit-identical\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
