// Standalone benchmark of the optional SM61 EXL3-to-INT8 compute profile.
// Checkpoint/GPU payload stays EXL3 K3/K4. Transient weights and H128 activation
// blocks are approximated as signed INT8 for DP4A; this changes model arithmetic.
// Synthetic operator error <= 5% here is NOT the model-quality acceptance gate.
#include "runtime/inference.h"
#include "cuda/device_matrix.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
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
#include "cuda/fp8_prefill_simt.cuh"
#include "cuda/expert_exl3_prefill_simt.cuh"
#include "cuda/expert_exl3_prefill_tiled.cuh"
#include "cuda/simt_attention_split.cuh"
#include "cuda/simt_prefill_attention_split.cuh"
#include "cuda/expert_exl3_prefill_fast.cuh"
#include "cuda/expert_exl3_prefill_dp4a.cuh"
#include "cuda/pascal_attention_fast.cuh"
#include "cuda/pascal_sgemm.cuh"
#include "cuda/pascal_attention_sgemm.cuh"
void Close(const std::vector<float>& actual,const std::vector<float>& expected,const char* label,double tolerance=.002,double absolute_rmse=0,double absolute_max=0){
  double error=0,norm=0;float maximum=0;
  for(std::size_t i=0;i<actual.size();++i){if(!std::isfinite(actual[i]))throw std::runtime_error(label);const double delta=double(actual[i])-expected[i];error+=delta*delta;norm+=double(expected[i])*expected[i];maximum=fmaxf(maximum,fabsf(delta));}
  const double relative=std::sqrt(error/std::max(norm,1e-30));std::cout<<label<<" nrmse="<<relative<<" max_abs="<<maximum<<'\n';
  if(relative>tolerance && (std::sqrt(error/actual.size())>absolute_rmse || maximum>absolute_max))throw std::runtime_error(label);
}
void Require(bool value,const char* message){if(!value)throw std::runtime_error(message);}

template<class F> float ExpertTime(F launch){for(int i=0;i<5;++i)launch();Check(cudaDeviceSynchronize());cudaEvent_t a,b;Check(cudaEventCreate(&a));Check(cudaEventCreate(&b));Check(cudaEventRecord(a));for(int i=0;i<20;++i)launch();Check(cudaEventRecord(b));Check(cudaEventSynchronize(b));float ms=0;Check(cudaEventElapsedTime(&ms,a,b));cudaEventDestroy(a);cudaEventDestroy(b);return ms/20;}
void Experts(int tokens,int part,bool skew,bool benchmark,int rate){
  const int rows=part==2?2304:896,cols=part==2?896:2304;const std::size_t bytes=std::size_t(rows)*cols*rate/8;
  std::vector<unsigned> weights(bytes*64/4);unsigned seed=23;for(auto& word:weights){seed=1664525U*seed+1013904223U;word=seed;}
  std::vector<unsigned short> side(64*(rows+cols));for(int e=0;e<64;++e)for(int i=0;i<rows+cols;++i)side[e*(rows+cols)+i]=__half_as_ushort(__float2half_rn(float((e*7+i)%17-8)/64));
  std::vector<float> input(256*8*cols);for(std::size_t i=0;i<input.size();++i)input[i]=__bfloat162float(__float2bfloat16_rn(std::sin(float(i))*.5f));
  std::fill(input.begin(),input.begin()+cols,0.f); // Exercise the all-zero INT8 scale path.
  std::vector<int> ids(tokens*8);for(int row=0;row<tokens;++row){for(int slot=0;slot<8;++slot)ids[row*8+slot]=skew?slot:(row*11+slot*7)%64;std::sort(ids.begin()+row*8,ids.begin()+row*8+8);}
  Buffer w(weights.size()*4),sides(side.size()*2),source(input.size()*4),chosen(256*8*4),desc(64*3*sizeof(DeviceMatrix)),transformed(input.size()*4),reference_input(input.size()*4),output(256*8*rows*4),reference(256*8*rows*4),state(sizeof(SimtBatch)),groups(64*256*4),counts(64*4),tiles(319*sizeof(simt_batch::ExpertTile)),total(4),bad(4);
  Check(cudaMemcpy(w.p,weights.data(),w.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(sides.p,side.data(),sides.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(source.p,input.data(),source.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(chosen.p,ids.data(),ids.size()*4,cudaMemcpyHostToDevice));Check(cudaMemset(bad.p,0,4));
  std::vector<DeviceMatrix> table(64*3);for(int e=0;e<64;++e)table[e*3+part]=DeviceMatrix{w.As<unsigned char>()+e*bytes,nullptr,nullptr,rows,cols,rate==3?6:3,sides.As<unsigned short>()+e*(rows+cols),sides.As<unsigned short>()+e*(rows+cols)+cols};Check(cudaMemcpy(desc.p,table.data(),desc.bytes,cudaMemcpyHostToDevice));SetSimtBatch<<<1,1>>>(state.As<SimtBatch>(),0,tokens,8192,bad.As<int>());
  simt_batch::Group<<<64,128>>>(chosen.As<int>(),groups.As<int>(),counts.As<int>(),bad.As<int>(),state.As<SimtBatch>());simt_batch::DescribeTiled<<<1,64>>>(counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>());
  if(rate==4){
  simt_batch::ExpertsTiled(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),output.As<float>(),transformed.As<float>(),groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>(),nullptr);
  }else{
    simt_batch::Describe<<<1,64>>>(counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>());
    simt_batch::Experts(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),output.As<float>(),transformed.As<float>(),groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>(),nullptr);
  }
  simt_batch::InputHadRows<<<dim3(cols/128,256*8),128>>>(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),reference_input.As<float>(),bad.As<int>(),state.As<SimtBatch>());
  // Small K3 cases use the independent generic native codec oracle.
  if(rate==3 && tokens<=3)exl3_selected::Projection<true><<<dim3(rows/128,tokens*8),256>>>(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),reference_input.As<float>(),reference.As<float>(),tokens*8,bad.As<int>());
  else exl3_selected::Projection<false><<<dim3(rows/128,tokens*8),256>>>(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),reference_input.As<float>(),reference.As<float>(),tokens*8,bad.As<int>());Check(cudaDeviceSynchronize());
  std::vector<float> x(tokens*8*rows),y(x),a(tokens*8*cols),b(a);Check(cudaMemcpy(x.data(),output.p,x.size()*4,cudaMemcpyDeviceToHost));Check(cudaMemcpy(y.data(),reference.p,y.size()*4,cudaMemcpyDeviceToHost));Check(cudaMemcpy(a.data(),transformed.p,a.size()*4,cudaMemcpyDeviceToHost));Check(cudaMemcpy(b.data(),reference_input.p,b.size()*4,cudaMemcpyDeviceToHost));Require(!memcmp(a.data(),b.data(),a.size()*4),"warp input H128 changed arithmetic");Require(!memcmp(x.data(),y.data(),x.size()*4),"grouped expert reuse/scatter changed arithmetic");int flag=0,n=0;Check(cudaMemcpy(&flag,bad.p,4,cudaMemcpyDeviceToHost));Check(cudaMemcpy(&n,total.p,4,cudaMemcpyDeviceToHost));Require(!flag && n<=319,"invalid expert grouping/descriptor count");
  // Production M8 and the M16 candidate both retain the selected-vector result.
  simt_batch::Describe<<<1,64>>>(counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>());
  simt_batch::Experts(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),output.As<float>(),transformed.As<float>(),groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>(),nullptr);
  Check(cudaDeviceSynchronize());Check(cudaMemcpy(x.data(),output.p,x.size()*4,cudaMemcpyDeviceToHost));Require(!memcmp(x.data(),y.data(),x.size()*4),"production M8 differs from selected-vector reference");
  simt_batch::Describe<<<1,64>>>(counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>());
  auto fast=[&]{simt_batch::ExpertsFast(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),output.As<float>(),transformed.As<float>(),groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>(),nullptr);};
  fast();Check(cudaDeviceSynchronize());Check(cudaMemcpy(x.data(),output.p,x.size()*4,cudaMemcpyDeviceToHost));Close(x,y,"EXL3 M32 FP32 tile",.0005);
  // Retiling must preserve every accumulator's K order, including sparse/tail rows.
  simt_batch::ProjectionFast<128,16,2><<<dim3(rows/128,simt_batch::kExpertTiles),256>>>(desc.As<DeviceMatrix>(),0,part,groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),transformed.As<float>(),output.As<float>());
  simt_batch::OutputHadFast<<<dim3(rows/128,512),128>>>(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),counts.As<int>(),output.As<float>(),bad.As<int>(),state.As<SimtBatch>());
  Check(cudaDeviceSynchronize());std::vector<float> previous(x.size());Check(cudaMemcpy(previous.data(),output.p,previous.size()*4,cudaMemcpyDeviceToHost));Require(!memcmp(x.data(),previous.data(),x.size()*4),"R4 tile differs from R2 accumulator order");
  auto int8=[&]{exl3_pascal_dp4a::Launch(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),output.As<float>(),transformed.As<float>(),groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>(),nullptr,rate);};
  Check(cudaMemset(output.p,0xff,output.bytes));int8();Check(cudaDeviceSynchronize());Check(cudaMemcpy(y.data(),output.p,y.size()*4,cudaMemcpyDeviceToHost));Close(y,x,"DP4A candidate",.05);
  Check(cudaMemcpy(&flag,bad.p,4,cudaMemcpyDeviceToHost));Require(!flag,"DP4A raised a numerical error");
  if(benchmark)for(int rep=0;rep<3;++rep){float old,newer;if(rep%2){newer=ExpertTime(int8);old=ExpertTime(fast);}else{old=ExpertTime(fast);newer=ExpertTime(int8);}std::cout<<"DP4A rate="<<rate<<" tokens="<<tokens<<" part="<<part<<" skew="<<skew<<" rep="<<rep<<" fp32_ms="<<old<<" int8_ms="<<newer<<'\n';}

}
}}
int main(int argc,char** argv){try{using namespace mellum16;std::cout<<std::unitbuf;bool benchmark=true,sanitize=false;int rate=4;
  for(int i=1;i<argc;++i){const std::string arg=argv[i];if(arg=="--k3")rate=3;else if(arg=="--check")benchmark=false;else if(arg=="--sanitize"){sanitize=true;benchmark=false;}else throw std::runtime_error("unknown benchmark option");}
  const std::vector<int> sizes=benchmark?std::vector<int>{64,128,256}:sanitize?std::vector<int>{17,33}:std::vector<int>{1,3,15,16,17,31,32,33,65,256};
  for(int tokens:sizes)for(int part:{0,2})for(bool skew:{false,true})Experts(tokens,part,skew,benchmark,rate);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
