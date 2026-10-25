#include "runtime/gpu_build_config.h"
#include "runtime/model_runtime.h"
#include "runtime/simt_memory.h"
#include "runtime/simt_features.h"
#include <iostream>
#include <stdexcept>
void Require(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
int main(){try{
  auto profile=mellum16::SelectedServerOptions("unused","unused");
#if PADRA_BUILD_SIMT
  Require(profile.exl3_experts && profile.exl3_rate==4,"K4 profile");
  Require(!profile.exl3_decode_mma && !profile.exl3_prefill_fp16 && !profile.gqa8_decode_attention,"SIMT profile cannot select Tensor Cores");
  Require(mellum16::kDefaultServerContext==8192 && mellum16::kDefaultServerChunk==256,"SIMT server defaults");
#else
  Require(profile.exl3_decode_mma && profile.exl3_prefill_fp16 && profile.gqa8_decode_attention,"SM120 optimizations must remain enabled");
  Require(mellum16::kDefaultServerContext==131072 && mellum16::kDefaultServerChunk==2048,"SM120 server defaults");
#endif
  mellum16::InferenceOptions simt;
  simt.exl3_experts=true;simt.exl3_rate=4;
  Require(mellum16::ValidateSimtFeatures(simt).ok(),"ordinary K4 SIMT feature admission");
  simt.mtp.artifact="unused-draft";
  Require(!mellum16::ValidateSimtFeatures(simt).ok(),"SIMT must reject MTP instead of ignoring it");
  simt.mtp.artifact.clear();simt.q8_head=true;
  Require(mellum16::ValidateSimtFeatures(simt).ok(),"stored Q8 head admission");
  simt.q8_head=false;simt.capture_hidden=true;
  Require(mellum16::ValidateSimtFeatures(simt).ok(),"SIMT hidden capture admission");
  simt.capture_hidden=false;simt.kv_cache_format=mellum16::KVCacheFormat::Q8Token512Block32F16ScaleRotatedV2;
  Require(mellum16::ValidateSimtFeatures(simt).ok(),"SIMT native Q8_0 KV admission");
  simt.server_kv_checkpoint=true;
  Require(mellum16::ValidateSimtFeatures(simt).ok(),"SIMT encoded checkpoint admission");
  simt.server_kv_checkpoint=false;
  simt.kv_cache_format=mellum16::KVCacheFormat::Q8TokenHead128F32ScaleV1;
  Require(!mellum16::ValidateSimtFeatures(simt).ok(),"SIMT must reject the different Q8 head-scale layout");
  simt.kv_cache_format=mellum16::KVCacheFormat::BF16;simt.exl3_rate=3;
  Require(mellum16::ValidateSimtFeatures(simt).ok(),"K3 SIMT admission");
  simt.q8_head=true;simt.mtp.artifact="unused-draft";simt.mtp.exl3_experts=true;
  Require(mellum16::ValidateSimtFeatures(simt).ok(),"K3/Q8/EXL3 MTP admission");
  simt.mtp.depth=5;
  Require(!mellum16::ValidateSimtFeatures(simt).ok(),"MTP depth bounds");
  simt.mtp.depth=1;simt.mtp.fp8_experts=true;
  Require(!mellum16::ValidateSimtFeatures(simt).ok(),"MTP draft format cannot be silently ignored");
  simt.mtp.fp8_experts=false;simt.simt_reference_prefill=true;
  Require(!mellum16::ValidateSimtFeatures(simt).ok(),"MTP requires available prefill arenas");
  const auto small=mellum16::PlanSimtMemory(7092933120ULL,8192,2*1024*1024,true);
  const auto large=mellum16::PlanSimtMemory(7092933120ULL,16384,2*1024*1024,true);
  Require(small.Fits(8ULL*1024*1024*1024),"8K SIMT inventory should fit 8GiB");
  Require(large.buffers.at("kv")-small.buffers.at("kv")==7ULL*8192*2048,"global/local KV inventory");
  Require(large.buffers.at("attention_scores")==32ULL*16384*4,"long attention uses global scores");
  Require(small.buffers.at("local_checkpoint")==42ULL*1024*1024,"snapshot inventory");
  const auto q8=mellum16::PlanSimtMemory(5493330432ULL,8192,2*1024*1024,true,true,true,true,true);
  Require(q8.buffers.at("local_checkpoint")==21ULL*1024*1088,"encoded snapshot must include all Q8 codes and scales");
  for(int anchor:{1,255,256,257,511,512,1023,1024,1025,8191}){
    const auto saved=mellum16::CheckpointCaptureAnchor(anchor,256,true);
    Require(saved%256==0 && saved<=anchor && anchor-saved<256,"SIMT restores only complete original expert chunks");
  }
  const auto fast=mellum16::PlanSimtMemory(7092933120ULL,8192,2*1024*1024,true,true,true);
  Require(fast.resident-small.resident==4096ULL*2304*4+256ULL*4096*4+16ULL*1024*1024+8192ULL*8*(256-32)*4+4ULL*4096*4,"bounded FP32 SGEMM inventory");
  const auto fast16=mellum16::PlanSimtMemory(7092933120ULL,16384,2*1024*1024,true,true,true);
  Require(fast16.buffers.at("batch_scores")==32ULL*1024*1024,"16K attention query tiles bound memory");
  const auto fast1=mellum16::PlanSimtMemory(7092933120ULL,1024,2*1024*1024,true,true,true);
  Require(fast1.buffers.at("batch_scores")==1280ULL*256*8*4,"local staged attention spans 1280 keys");
  Require(fast.Fits(8ULL*1024*1024*1024),"Pascal FP32 profile must retain the reserve");
  Require(mellum16::ServerAvailableBytes(7ULL*1024,8ULL*1024,8)==7ULL*1024,"actual free-memory admission");
  std::cout<<"GPU profile and memory checks passed (SM"<<mellum16::kBuildArchitecture<<")\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
