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
#include "cuda/pascal_attention_fast.cuh"
#include "cuda/pascal_sgemm.cuh"
#include "cuda/pascal_attention_sgemm.cuh"
// BF16 probability rounding can amplify tiny expf changes in a nearly
// cancelling PV sum. Attention additionally bounds absolute RMSE and max error
// for the fixture values in [-1,1]; the real-model NLL gate is independent.
void Close(const std::vector<float>& actual,const std::vector<float>& expected,const char* label,double tolerance=.002,double absolute_rmse=0,double absolute_max=0){
  double error=0,norm=0;float maximum=0;
  for(std::size_t i=0;i<actual.size();++i){if(!std::isfinite(actual[i]))throw std::runtime_error(label);const double delta=double(actual[i])-expected[i];error+=delta*delta;norm+=double(expected[i])*expected[i];maximum=fmaxf(maximum,fabsf(delta));}
  const double relative=std::sqrt(error/std::max(norm,1e-30));std::cout<<label<<" nrmse="<<relative<<" max_abs="<<maximum<<'\n';
  if(relative>tolerance && (std::sqrt(error/actual.size())>absolute_rmse || maximum>absolute_max))throw std::runtime_error(label);
}
bool measure_attention=false;
void Require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
__nv_bfloat16 Value(int pos,int d,bool key){return __float2bfloat16_rn(float((pos*(key?3:5)+d)%17-8)/(key?32:8));}
template<class F> float ExpertTime(F launch){for(int i=0;i<5;++i)launch();Check(cudaDeviceSynchronize());cudaEvent_t a,b;Check(cudaEventCreate(&a));Check(cudaEventCreate(&b));Check(cudaEventRecord(a));for(int i=0;i<20;++i)launch();Check(cudaEventRecord(b));Check(cudaEventSynchronize(b));float ms=0;Check(cudaEventElapsedTime(&ms,a,b));cudaEventDestroy(a);cudaEventDestroy(b);return ms/20;}
void Experts(int tokens,int part,bool skew,bool benchmark=false,int rate=4){
  const int rows=part==2?2304:896,cols=part==2?896:2304;const std::size_t bytes=std::size_t(rows)*cols*rate/8;
  std::vector<unsigned> weights(bytes*64/4);unsigned seed=23;for(auto& word:weights){seed=1664525U*seed+1013904223U;word=seed;}
  std::vector<unsigned short> side(64*(rows+cols));for(int e=0;e<64;++e)for(int i=0;i<rows+cols;++i)side[e*(rows+cols)+i]=__half_as_ushort(__float2half_rn(float((e*7+i)%17-8)/64));
  std::vector<float> input(256*8*cols);for(std::size_t i=0;i<input.size();++i)input[i]=__bfloat162float(__float2bfloat16_rn(std::sin(float(i))*.5f));
  std::vector<int> ids(tokens*8);for(int row=0;row<tokens;++row){for(int slot=0;slot<8;++slot)ids[row*8+slot]=skew?slot:(row*11+slot*7)%64;std::sort(ids.begin()+row*8,ids.begin()+row*8+8);}
  Buffer w(weights.size()*4),sides(side.size()*2),source(input.size()*4),chosen(256*8*4),desc(64*3*sizeof(DeviceMatrix)),transformed(input.size()*4),reference_input(input.size()*4),output(256*8*rows*4),reference(256*8*rows*4),state(sizeof(SimtBatch)),groups(64*256*4),counts(64*4),tiles(319*sizeof(simt_batch::ExpertTile)),total(4),bad(4);
  Check(cudaMemcpy(w.p,weights.data(),w.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(sides.p,side.data(),sides.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(source.p,input.data(),source.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(chosen.p,ids.data(),ids.size()*4,cudaMemcpyHostToDevice));Check(cudaMemset(bad.p,0,4));
  std::vector<DeviceMatrix> table(64*3);for(int e=0;e<64;++e)table[e*3+part]=DeviceMatrix{w.As<unsigned char>()+e*bytes,nullptr,nullptr,rows,cols,rate==3?6:3,sides.As<unsigned short>()+e*(rows+cols),sides.As<unsigned short>()+e*(rows+cols)+cols};Check(cudaMemcpy(desc.p,table.data(),desc.bytes,cudaMemcpyHostToDevice));SetSimtBatch<<<1,1>>>(state.As<SimtBatch>(),0,tokens,8192,bad.As<int>());
  simt_batch::Group<<<64,128>>>(chosen.As<int>(),groups.As<int>(),counts.As<int>(),bad.As<int>(),state.As<SimtBatch>());
  simt_batch::Describe<<<1,64>>>(counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>());
  simt_batch::Experts(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),output.As<float>(),transformed.As<float>(),groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>(),nullptr);
  simt_batch::InputHadRows<<<dim3(cols/128,256*8),128>>>(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),reference_input.As<float>(),bad.As<int>(),state.As<SimtBatch>());
  exl3_selected::Projection<true><<<dim3(rows/128,tokens*8),256>>>(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),reference_input.As<float>(),reference.As<float>(),tokens*8,bad.As<int>());Check(cudaDeviceSynchronize());
  std::vector<float> x(tokens*8*rows),y(x),a(tokens*8*cols),b(a);Check(cudaMemcpy(x.data(),output.p,x.size()*4,cudaMemcpyDeviceToHost));Check(cudaMemcpy(y.data(),reference.p,y.size()*4,cudaMemcpyDeviceToHost));Check(cudaMemcpy(a.data(),transformed.p,a.size()*4,cudaMemcpyDeviceToHost));Check(cudaMemcpy(b.data(),reference_input.p,b.size()*4,cudaMemcpyDeviceToHost));Require(!memcmp(a.data(),b.data(),a.size()*4),"warp input H128 changed arithmetic");Require(!memcmp(x.data(),y.data(),x.size()*4),"grouped expert reuse/scatter changed arithmetic");int flag=0,n=0;Check(cudaMemcpy(&flag,bad.p,4,cudaMemcpyDeviceToHost));Check(cudaMemcpy(&n,total.p,4,cudaMemcpyDeviceToHost));Require(!flag && n<=319,"invalid expert grouping/descriptor count");
  exl3_selected::Projection<false><<<dim3(rows/128,tokens*8),256>>>(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),reference_input.As<float>(),output.As<float>(),tokens*8,bad.As<int>());
  Check(cudaDeviceSynchronize());Check(cudaMemcpy(x.data(),output.p,x.size()*4,cudaMemcpyDeviceToHost));
  Require(!memcmp(x.data(),y.data(),x.size()*4),"optimized K3/K4 codec differs from independent reference");
  if(rate==3)exl3_selected::Projection<false,3><<<dim3(rows/128,tokens*8),256>>>(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),reference_input.As<float>(),output.As<float>(),tokens*8,bad.As<int>());
  else exl3_selected::Projection<false,4><<<dim3(rows/128,tokens*8),256>>>(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),reference_input.As<float>(),output.As<float>(),tokens*8,bad.As<int>());
  Check(cudaDeviceSynchronize());Check(cudaMemcpy(x.data(),output.p,x.size()*4,cudaMemcpyDeviceToHost));Require(!memcmp(x.data(),y.data(),x.size()*4),"fixed-rate codec differs from generic reference");
  // Production M8 and the M16 candidate both retain the selected-vector result.
  simt_batch::Describe<<<1,64>>>(counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>());
  simt_batch::Experts(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),output.As<float>(),transformed.As<float>(),groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>(),nullptr);
  Check(cudaDeviceSynchronize());Check(cudaMemcpy(x.data(),output.p,x.size()*4,cudaMemcpyDeviceToHost));Require(!memcmp(x.data(),y.data(),x.size()*4),"production M8 differs from selected-vector reference");
  simt_batch::Describe<<<1,64>>>(counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>());
  auto fast=[&]{simt_batch::ExpertsFast(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),output.As<float>(),transformed.As<float>(),groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>(),nullptr);};
  fast();Check(cudaDeviceSynchronize());Check(cudaMemcpy(x.data(),output.p,x.size()*4,cudaMemcpyDeviceToHost));Close(x,y,"EXL3 M32 FP32 tile",.0005);
  if(tokens>=2 && tokens<=5){
    switch(tokens){
    case 2:if(rate==3)simt_batch::ExpertsSmall<2,3>(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),output.As<float>(),transformed.As<float>(),groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>(),nullptr);else simt_batch::ExpertsSmall<2,4>(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),output.As<float>(),transformed.As<float>(),groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>(),nullptr);break;
    case 3:if(rate==3)simt_batch::ExpertsSmall<3,3>(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),output.As<float>(),transformed.As<float>(),groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>(),nullptr);else simt_batch::ExpertsSmall<3,4>(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),output.As<float>(),transformed.As<float>(),groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>(),nullptr);break;
    case 4:if(rate==3)simt_batch::ExpertsSmall<4,3>(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),output.As<float>(),transformed.As<float>(),groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>(),nullptr);else simt_batch::ExpertsSmall<4,4>(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),output.As<float>(),transformed.As<float>(),groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>(),nullptr);break;
    case 5:if(rate==3)simt_batch::ExpertsSmall<5,3>(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),output.As<float>(),transformed.As<float>(),groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>(),nullptr);else simt_batch::ExpertsSmall<5,4>(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),output.As<float>(),transformed.As<float>(),groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>(),nullptr);break;
    }
    Check(cudaDeviceSynchronize());Check(cudaMemcpy(x.data(),output.p,x.size()*4,cudaMemcpyDeviceToHost));Require(!memcmp(x.data(),y.data(),x.size()*4),"small verifier experts differ from reference");
    fast();Check(cudaDeviceSynchronize());Check(cudaMemcpy(x.data(),output.p,x.size()*4,cudaMemcpyDeviceToHost));
  }
  // Retiling must preserve every accumulator's K order, including sparse/tail rows.
  simt_batch::ProjectionFast<128,16,2><<<dim3(rows/128,simt_batch::kExpertTiles),256>>>(desc.As<DeviceMatrix>(),0,part,groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),transformed.As<float>(),output.As<float>());
  simt_batch::OutputHadFast<<<dim3(rows/128,512),128>>>(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),counts.As<int>(),output.As<float>(),bad.As<int>(),state.As<SimtBatch>());
  Check(cudaDeviceSynchronize());std::vector<float> previous(x.size());Check(cudaMemcpy(previous.data(),output.p,previous.size()*4,cudaMemcpyDeviceToHost));Require(!memcmp(x.data(),previous.data(),x.size()*4),"R4 tile differs from R2 accumulator order");
  if(benchmark){const float candidate=ExpertTime(fast);
    simt_batch::Describe<<<1,64>>>(counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>());
    const float original=ExpertTime([&]{simt_batch::Experts(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),reference.As<float>(),transformed.As<float>(),groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>(),nullptr);});
    std::cout<<"FAST tokens="<<tokens<<" part="<<part<<" skew="<<skew<<" exact_ms="<<original<<" fast_ms="<<candidate<<'\n';
    auto m16=[&]{simt_batch::DescribeTiled<<<1,64>>>(counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>());simt_batch::ExpertsTiled(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),output.As<float>(),transformed.As<float>(),groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>(),nullptr);};
    auto m8=[&]{simt_batch::Describe<<<1,64>>>(counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>());simt_batch::Experts(desc.As<DeviceMatrix>(),0,part,chosen.As<int>(),source.As<float>(),reference.As<float>(),transformed.As<float>(),groups.As<int>(),counts.As<int>(),tiles.As<simt_batch::ExpertTile>(),total.As<int>(),bad.As<int>(),state.As<SimtBatch>(),nullptr);};
    m8();m16();Check(cudaDeviceSynchronize());Check(cudaMemcpy(x.data(),output.p,x.size()*4,cudaMemcpyDeviceToHost));Check(cudaMemcpy(y.data(),reference.p,y.size()*4,cudaMemcpyDeviceToHost));Require(!memcmp(x.data(),y.data(),x.size()*4),"M8 and M16 operator benchmark mismatch");
    for(int rep=0;rep<5;++rep){float old,newer;if(rep%2){newer=ExpertTime(m16);old=ExpertTime(m8);}else{old=ExpertTime(m8);newer=ExpertTime(m16);}std::cout<<"tokens="<<tokens<<" part="<<part<<" skew="<<skew<<" rep="<<rep<<" m8_ms="<<old<<" m16_ms="<<newer<<" speedup="<<old/newer<<'\n';}
  }
}
void FP8(int tokens,int cols){
  const int rows=512;std::vector<unsigned char> weights(rows*cols),activation(tokens*cols);unsigned seed=23;
  for(auto* data:{&weights,&activation})for(auto& value:*data){seed=1664525U*seed+1013904223U;value=static_cast<unsigned char>(seed>>24);if((value&127)==127)--value;}
  std::vector<unsigned short> ws(rows,0x3c00);std::vector<float> scales(tokens);for(int i=0;i<tokens;++i)scales[i]=float(i%13+1)/128;
  Buffer w(weights.size()),a(256ULL*cols),side(rows*2),scale(256*4),output(256ULL*rows*4),reference(256ULL*rows*4),state(sizeof(SimtBatch)),bad(4);
  Check(cudaMemset(a.p,0x7f,a.bytes));Check(cudaMemset(scale.p,0xff,scale.bytes));
  Check(cudaMemcpy(w.p,weights.data(),w.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(a.p,activation.data(),activation.size(),cudaMemcpyHostToDevice));Check(cudaMemcpy(side.p,ws.data(),side.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(scale.p,scales.data(),scales.size()*4,cudaMemcpyHostToDevice));Check(cudaMemset(bad.p,0,4));SetSimtBatch<<<1,1>>>(state.As<SimtBatch>(),0,tokens,8192,bad.As<int>());
  DeviceMatrix m{w.As<unsigned char>(),side.As<unsigned char>(),nullptr,rows,cols,1};simt_batch::FP8ExactTile<<<dim3(rows/32,32),256>>>(m,a.As<unsigned char>(),scale.As<float>(),output.As<float>(),bad.As<int>(),state.As<SimtBatch>());
  simt_batch::FP8Rows<<<dim3(rows/4,256),128>>>(m,a.As<unsigned char>(),scale.As<float>(),reference.As<float>(),bad.As<int>(),state.As<SimtBatch>());Check(cudaDeviceSynchronize());std::vector<float> x(tokens*rows),y(x);Check(cudaMemcpy(x.data(),output.p,x.size()*4,cudaMemcpyDeviceToHost));Check(cudaMemcpy(y.data(),reference.p,y.size()*4,cudaMemcpyDeviceToHost));Require(!memcmp(x.data(),y.data(),x.size()*4),"tiled FP8 batch changed lane/FMA/scale arithmetic");int flag=0;Check(cudaMemcpy(&flag,bad.p,4,cudaMemcpyDeviceToHost));Require(!flag,"tiled FP8 batch flagged bad");
  if(tokens>=2 && tokens<=5){
    switch(tokens){
    case 2:simt_batch::FP8Small<2><<<rows/4,128>>>(m,a.As<unsigned char>(),scale.As<float>(),output.As<float>(),bad.As<int>(),state.As<SimtBatch>());break;
    case 3:simt_batch::FP8Small<3><<<rows/4,128>>>(m,a.As<unsigned char>(),scale.As<float>(),output.As<float>(),bad.As<int>(),state.As<SimtBatch>());break;
    case 4:simt_batch::FP8Small<4><<<rows/4,128>>>(m,a.As<unsigned char>(),scale.As<float>(),output.As<float>(),bad.As<int>(),state.As<SimtBatch>());break;
    case 5:simt_batch::FP8Small<5><<<rows/4,128>>>(m,a.As<unsigned char>(),scale.As<float>(),output.As<float>(),bad.As<int>(),state.As<SimtBatch>());break;
    }
    Check(cudaDeviceSynchronize());Check(cudaMemcpy(x.data(),output.p,x.size()*4,cudaMemcpyDeviceToHost));Require(!memcmp(x.data(),y.data(),x.size()*4),"small verifier FP8 differs from reference");
  }
  Buffer decoded_weights(weights.size()*4),decoded_input(256ULL*cols*4),workspace(16ULL*1024*1024);Stream stream;pascal_sgemm::Handle blas(stream.s,workspace.p,workspace.bytes);
  auto fast=[&]{pascal_sgemm::Projection(blas,m,a.As<unsigned char>(),scale.As<float>(),output.As<float>(),decoded_weights.As<float>(),decoded_input.As<float>(),bad.As<int>(),state.As<SimtBatch>(),stream.s);};
  fast();Check(cudaDeviceSynchronize());DecodeGraph graph;graph.Capture(stream.s,fast);
  // Replay with smaller active tails and poisoned unused FP8 input/scale bytes.
  for(int tail:{tokens,std::max(1,tokens-1),1}){SetSimtBatch<<<1,1>>>(state.As<SimtBatch>(),0,tail,8192,bad.As<int>());graph.Launch(stream.s);Check(cudaDeviceSynchronize());x.resize(tail*rows);std::vector<float> expected(y.begin(),y.begin()+tail*rows);Check(cudaMemcpy(x.data(),output.p,x.size()*4,cudaMemcpyDeviceToHost));Close(x,expected,"FP8 captured SGEMM",.0005);}
  Check(cudaMemcpy(&flag,bad.p,4,cudaMemcpyDeviceToHost));Require(!flag,"SGEMM tail flagged bad");
}
void Attention(bool local,int base,int rows,int context_capacity=8192){
  const int capacity=local?1024:context_capacity;
  std::vector<__nv_bfloat16> keys(capacity*512,__float2bfloat16_rn(NAN)),values(keys),current(256*1024,__float2bfloat16_rn(NAN));
  for(int pos=local?std::max(0,base-1024):0;pos<base;++pos)for(int d=0;d<512;++d){keys[(pos%capacity)*512+d]=Value(pos,d,true);values[(pos%capacity)*512+d]=Value(pos,d,false);}
  for(int row=0;row<rows;++row)for(int d=0;d<512;++d){current[row*512+d]=Value(base+row,d,true);current[(256+row)*512+d]=Value(base+row,d,false);}
  std::vector<float> query(256*4096,NAN);for(int row=0;row<rows;++row)for(int i=0;i<4096;++i)query[row*4096+i]=float((row*5+i)%11-5)/32;
  Buffer q(query.size()*4),k(keys.size()*2),v(values.size()*2),stage(current.size()*2),state(sizeof(SimtBatch)),command(sizeof(SimtCommand)),workspace(8ULL*32*context_capacity*4),result(256*4096*4),reference(256*4096*4),split_batch(256*4096*4),bad(4);
  Check(cudaMemcpy(q.p,query.data(),q.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(k.p,keys.data(),k.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(v.p,values.data(),v.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(stage.p,current.data(),stage.bytes,cudaMemcpyHostToDevice));Check(cudaMemset(bad.p,0,4));
  SetSimtBatch<<<1,1>>>(state.As<SimtBatch>(),base,rows,context_capacity,bad.As<int>());
  for(int first=0;first<256;first+=8){if(local)AttendSimtBatch<true><<<dim3(32,8),256>>>(q.As<float>(),k.As<__nv_bfloat16>(),v.As<__nv_bfloat16>(),result.As<float>(),state.As<SimtBatch>(),capacity,workspace.As<float>(),bad.As<int>(),stage.As<__nv_bfloat16>(),first);else AttendSimtBatch<false><<<dim3(32,8),256>>>(q.As<float>(),k.As<__nv_bfloat16>(),v.As<__nv_bfloat16>(),result.As<float>(),state.As<SimtBatch>(),capacity,workspace.As<float>(),bad.As<int>(),stage.As<__nv_bfloat16>(),first);}
  for(int first=0;first<256;first+=8)simt_prefill_attention_split::Launch(q.As<float>(),k.As<__nv_bfloat16>(),v.As<__nv_bfloat16>(),split_batch.As<float>(),state.As<SimtBatch>(),capacity,local,workspace.As<float>(),bad.As<int>(),stage.As<__nv_bfloat16>(),first,nullptr);
  for(int first=0;first<256;first+=8)pascal_attention::Prefill(q.As<float>(),k.As<__nv_bfloat16>(),v.As<__nv_bfloat16>(),result.As<float>(),state.As<SimtBatch>(),capacity,local,workspace.As<float>(),bad.As<int>(),stage.As<__nv_bfloat16>(),first,nullptr);
  Check(cudaDeviceSynchronize());std::vector<float> fast_batch(rows*4096);Check(cudaMemcpy(fast_batch.data(),result.p,fast_batch.size()*4,cudaMemcpyDeviceToHost));
  Buffer gemm_weights(std::max(1280,capacity)*128ULL*8),gemm_queries(8ULL*256*128*8),gemm_scores(8ULL*(capacity>8192?64:256)*std::max(1280,capacity)*4),gemm_workspace(16ULL*1024*1024);Stream attention_stream;pascal_sgemm::Handle attention_blas(attention_stream.s,gemm_workspace.p,gemm_workspace.bytes);
  auto batched_attention=[&]{pascal_attention_sgemm::Prefill(attention_blas,q.As<float>(),k.As<__nv_bfloat16>(),v.As<__nv_bfloat16>(),stage.As<__nv_bfloat16>(),result.As<float>(),state.As<SimtBatch>(),capacity,local,gemm_weights.As<float>(),gemm_queries.As<float>(),gemm_scores.As<float>(),bad.As<int>(),attention_stream.s);};
  batched_attention();Check(cudaDeviceSynchronize());DecodeGraph attention_graph;attention_graph.Capture(attention_stream.s,batched_attention);attention_graph.Launch(attention_stream.s);Check(cudaDeviceSynchronize());std::vector<float> gemm_batch(rows*4096);Check(cudaMemcpy(gemm_batch.data(),result.p,gemm_batch.size()*4,cudaMemcpyDeviceToHost));
  if(!local && context_capacity==8192){
    const int bound=base+rows<=1024?1024:base+rows<=4096?4096:8192;
    pascal_attention_sgemm::Prefill(attention_blas,q.As<float>(),k.As<__nv_bfloat16>(),v.As<__nv_bfloat16>(),stage.As<__nv_bfloat16>(),result.As<float>(),state.As<SimtBatch>(),capacity,local,gemm_weights.As<float>(),gemm_queries.As<float>(),gemm_scores.As<float>(),bad.As<int>(),attention_stream.s,bound);
    Check(cudaDeviceSynchronize());std::vector<float> bounded(rows*4096);Check(cudaMemcpy(bounded.data(),result.p,bounded.size()*4,cudaMemcpyDeviceToHost));Close(bounded,gemm_batch,"bounded context excludes poisoned future KV",.002,.0001,.0005);
  }
  SetSimtBatch<<<1,1>>>(state.As<SimtBatch>(),base,1,context_capacity,bad.As<int>());attention_graph.Launch(attention_stream.s);Check(cudaDeviceSynchronize());std::vector<float> tail(4096);Check(cudaMemcpy(tail.data(),result.p,4096*4,cudaMemcpyDeviceToHost));Close(tail,std::vector<float>(gemm_batch.begin(),gemm_batch.begin()+4096),"SGEMM graph active-row tail",.002,.0001,.0005);SetSimtBatch<<<1,1>>>(state.As<SimtBatch>(),base,rows,context_capacity,bad.As<int>());
  // Only after all batched reads: sequential baseline updates the same ring.
  for(int row=0;row<rows;++row){const int pos=base+row;Check(cudaMemcpy(k.As<__nv_bfloat16>()+(pos%capacity)*512,current.data()+row*512,1024,cudaMemcpyHostToDevice));Check(cudaMemcpy(v.As<__nv_bfloat16>()+(pos%capacity)*512,current.data()+(256+row)*512,1024,cudaMemcpyHostToDevice));SetSimtCommand<<<1,1>>>(command.As<SimtCommand>(),0,pos);
    if(local)AttendSimt<true,1,false><<<32,256>>>(q.As<float>()+row*4096,k.As<__nv_bfloat16>(),v.As<__nv_bfloat16>(),reference.As<float>()+row*4096,command.As<SimtCommand>(),capacity,workspace.As<float>(),bad.As<int>());else AttendSimt<false,1,false><<<32,256>>>(q.As<float>()+row*4096,k.As<__nv_bfloat16>(),v.As<__nv_bfloat16>(),reference.As<float>()+row*4096,command.As<SimtCommand>(),capacity,workspace.As<float>(),bad.As<int>());
  }
  Check(cudaDeviceSynchronize());std::vector<float> a(rows*4096),b(a);Check(cudaMemcpy(a.data(),split_batch.p,a.size()*4,cudaMemcpyDeviceToHost));Check(cudaMemcpy(b.data(),reference.p,b.size()*4,cudaMemcpyDeviceToHost));Require(!memcmp(a.data(),b.data(),a.size()*4),"batch attention differs from sequential causal/ring reference");Check(cudaMemcpy(a.data(),split_batch.p,a.size()*4,cudaMemcpyDeviceToHost));Require(!memcmp(a.data(),b.data(),a.size()*4),"query/GQA tiled prefill attention differs from sequential reference");simt_attention_split::Launch(q.As<float>()+(rows-1)*4096,k.As<__nv_bfloat16>(),v.As<__nv_bfloat16>(),result.As<float>(),command.As<SimtCommand>(),capacity,local,workspace.As<float>(),bad.As<int>(),nullptr);Check(cudaDeviceSynchronize());std::vector<float> split(4096);Check(cudaMemcpy(split.data(),result.p,4096*4,cudaMemcpyDeviceToHost));Require(!memcmp(split.data(),b.data()+(rows-1)*4096,4096*4),"split/prefetched attention differs at long/ring context");int flag=0;Check(cudaMemcpy(&flag,bad.p,4,cudaMemcpyDeviceToHost));Require(!flag,"batch attention read poisoned future/tail rows");
  Close(fast_batch,b,"Fast causal staged attention",.002,.0001,.0005);
  Close(gemm_batch,b,"SGEMM causal staged attention",.002,.0001,.0005);
  pascal_attention::Decode(q.As<float>()+(rows-1)*4096,k.As<__nv_bfloat16>(),v.As<__nv_bfloat16>(),result.As<float>(),command.As<SimtCommand>(),capacity,local,workspace.As<float>(),bad.As<int>(),nullptr,context_capacity);
  Check(cudaDeviceSynchronize());Check(cudaMemcpy(split.data(),result.p,4096*4,cudaMemcpyDeviceToHost));Close(split,std::vector<float>(b.end()-4096,b.end()),"Fast parallel Decode attention",.002,.0001,.0005);
  Buffer partials(4*4096*4);
  auto fused=[&]{pascal_attention::Decode(q.As<float>()+(rows-1)*4096,k.As<__nv_bfloat16>(),v.As<__nv_bfloat16>(),result.As<float>(),command.As<SimtCommand>(),capacity,local,workspace.As<float>(),bad.As<int>(),nullptr,context_capacity);};
  auto split_pv=[&]{pascal_attention::Decode(q.As<float>()+(rows-1)*4096,k.As<__nv_bfloat16>(),v.As<__nv_bfloat16>(),result.As<float>(),command.As<SimtCommand>(),capacity,local,workspace.As<float>(),bad.As<int>(),nullptr,context_capacity,partials.As<float>());};
  split_pv();Check(cudaDeviceSynchronize());Check(cudaMemcpy(split.data(),result.p,4096*4,cudaMemcpyDeviceToHost));Close(split,std::vector<float>(b.end()-4096,b.end()),"Fast split Decode PV",.002,.0001,.0005);
  if(measure_attention)std::cout<<"PV base="<<base<<" local="<<local<<" fused_ms="<<ExpertTime(fused)<<" split_ms="<<ExpertTime(split_pv)<<'\n';
  Check(cudaMemcpy(&flag,bad.p,4,cudaMemcpyDeviceToHost));Require(!flag,"fast attention read poisoned future/tail rows");
}
}}
int main(int argc,char** argv){try{using namespace mellum16;std::cout<<std::unitbuf;if(argc>1 && std::string(argv[1])=="--attention-benchmark"){measure_attention=true;for(bool local:{false,true})for(int base:{0,1000,2000,4000,8000})Attention(local,base,3);return 0;}if(argc>1 && std::string(argv[1])=="--benchmark"){for(int tokens:{16,64,256})for(int part:{0,1,2})for(bool skew:{false,true})Experts(tokens,part,skew,true);return 0;}if(argc>1 && std::string(argv[1])=="--racecheck"){Experts(33,0,true);Experts(65,2,true);Experts(3,2,false);for(int cols:{2304,4096})FP8(17,cols);for(bool local:{false,true})for(auto shape:std::vector<std::pair<int,int>>{{1017,3},{8000,3}})Attention(local,shape.first,shape.second);std::cout<<"Focused shared-memory race checks passed\n";return 0;}const bool full=argc==1;if(argc>1 && std::string(argv[1])=="--verifier"){for(int tokens:{2,3,4,5}){for(int rate:{3,4})for(int part:{0,1,2})for(bool skew:{false,true})Experts(tokens,part,skew,false,rate);for(int cols:{2304,4096})FP8(tokens,cols);}std::cout<<"All small verifier row shapes passed\n";return 0;}if(argc>1 && std::string(argv[1])=="--k3"){for(int tokens:{1,3,33})for(int part:{0,1,2})for(bool skew:{false,true})Experts(tokens,part,skew,false,3);std::cout<<"K3 selected, grouped and dense projections passed\n";return 0;}for(int tokens:full?std::vector<int>{1,3,17,256}:std::vector<int>{3,17})for(int part:{0,1,2})for(bool skew:{false,true})Experts(tokens,part,skew);for(int tokens:full?std::vector<int>{1,3,16,31,64,128,256}:std::vector<int>{3,17})for(int cols:{2304,4096})FP8(tokens,cols);for(bool local:{false,true})for(auto shape:full?std::vector<std::pair<int,int>>{{0,3},{1017,31},{1024,256},{2000,16},{8000,128}}:std::vector<std::pair<int,int>>{{1017,9},{8000,9}})Attention(local,shape.first,shape.second);if(full)Attention(false,16000,127,16384);std::cout<<"Reference paths are bit-identical; fast FP32 paths pass numerical, tail and ring checks\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
