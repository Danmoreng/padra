// Portable EXL3 K4 backend. Compile separately from Tensor Core / FP4 kernels.
#include "runtime/inference.h"
#include "runtime/simt_memory.h"
#include "runtime/gpu_build_config.h"
#include "cuda/model_runtime_worker.h"
#include "cuda/device_matrix.cuh"
#include "compiler/sha256.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <array>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mellum16 {
namespace {
#include "simt_kernels.cuh"
#include "simt_diagnostics.h"
#include "simt_prefill.cuh"
#include "simt_prefill_attention.cuh"

struct Layer {
  DeviceMatrix norm,post,qn,kn,q,k,v,out,router;
  __nv_bfloat16 *kc,*vc;
  int capacity;
};
static_assert(sizeof(DeviceMatrix)==56);

Status ValidateSimtOptions(const InferenceOptions& o,bool persistent){
  if(o.kv_cache_format!=KVCacheFormat::BF16 || o.fp8_kv_cache)
    return {StatusCode::kUnsupported,"SIMT backend currently requires BF16 KV"};
  if(!o.exl3_experts || o.exl3_rate!=4)
    return {StatusCode::kUnsupported,"SIMT backend requires the existing EXL3 K4 artifact"};
  if(o.capacity<1024 || o.capacity>131072 || !ValidServerChunk(o.prefill_chunk) || !ValidServerBudget(o.server_device_budget_gib))
    return {StatusCode::kInvalidArgument,"invalid SIMT capacity/chunk/device budget"};
  if(o.capture_trace || o.snapshot_layer>=0 || o.capture_quantization_rows || o.diagnostic_routes || o.diagnostic_h128_fp32 || o.diagnostic_exl3_layer>=0 || !o.diagnostic_attention.empty())
    return {StatusCode::kUnsupported,"SM120 research captures are not implemented in the SIMT backend"};
  if(o.diagnostic_decode_token>=0)
    return {StatusCode::kUnsupported,"teacher-forced decode is not implemented in the SIMT backend"};
  if(o.simt_diagnostic_layer<0 || o.simt_diagnostic_layer>=28)
    return {StatusCode::kInvalidArgument,"invalid SIMT diagnostic layer"};
  for(int id:o.simt_teacher_tokens)if(id<0 || id>=98304)return {StatusCode::kInvalidArgument,"invalid SIMT teacher token"};
  if(!o.simt_teacher_tokens.empty() && (o.simt_diagnostics.empty() || o.simt_teacher_tokens.size()<static_cast<std::size_t>(std::max(0,o.generate-1))))
    return {StatusCode::kInvalidArgument,"SIMT teacher replay requires offline diagnostics and enough teacher tokens"};
  if(!persistent && o.server_kv_checkpoint)
    return {StatusCode::kUnsupported,"KV checkpoint capture requires a persistent runtime"};
  if(auto sampling=ValidateSampling(o.sampling);!sampling.ok())return sampling;
  if(o.input_ids.empty() || o.input_ids.size()>static_cast<std::size_t>(o.capacity) || o.generate<0 || static_cast<std::size_t>(o.generate)>o.capacity-o.input_ids.size())
    return {StatusCode::kInvalidArgument,"prompt plus generation exceeds SIMT context"};
  for(int id:o.input_ids)if(id<0 || id>=98304)return {StatusCode::kInvalidArgument,"invalid SIMT token ID"};
  return {};
}

Result<InferenceResult> RunSimt(const InferenceOptions& o,CudaModelWorker* worker=nullptr){
  const auto load_begin=std::chrono::steady_clock::now();
  try {
    if(auto status=ValidateSimtOptions(o,worker!=nullptr);!status.ok())return status;
    Check(cudaSetDevice(0));cudaDeviceProp prop{};Check(cudaGetDeviceProperties(&prop,0));
    if(prop.major*10+prop.minor<PADRA_CUDA_SM)
      return Status(StatusCode::kUnsupported,"GPU is older than the compiled SM" MELLUM16_ARCHITECTURE " target; rebuild for this GPU");
    if(worker){
      const auto manifest=o.artifact/"manifest.json";
      if(std::filesystem::file_size(manifest)>16*1024*1024)return Status(StatusCode::kDataLoss,"persistent manifest size unsupported");
      std::ifstream input(manifest,std::ios::binary);std::string source{std::istreambuf_iterator<char>(input),{}};
      if(compiler::Sha256Hex(source.data(),source.size())!="0f30d4ead6ae3d0683ec1eb9ad7af1adfebe343d804e3313e8c01d6c31f76f0b")
        return Status(StatusCode::kDataLoss,"persistent runtime manifest identity mismatch");
    }
    auto loaded=internal::LoadCompiledModel(o.artifact);if(!loaded.ok())return loaded.status();
    auto model=std::move(loaded).value();
    if(!model.exl3_experts || model.exl3_rate!=4 || model.tokenizer_sha256!=o.tokenizer_sha256)
      return Status(StatusCode::kDataLoss,"SIMT artifact/EXL3 K4/tokenizer identity mismatch");
    if(worker && (model.weights_sha256!="e614375f2eb26c17321365d4bd8ee7aa4b716f6badee171bb2040b85b754804c" || model.weights.size()!=7092933120ULL))
      return Status(StatusCode::kDataLoss,"persistent runtime requires pinned K4 checkpoint");
    const auto weights_bytes=model.weights.size();
    const auto sampler_bytes=gpu_sampling::WorkspaceBytes();
    const auto plan=PlanSimtMemory(weights_bytes,o.capacity,sampler_bytes,o.server_kv_checkpoint,!o.simt_reference_prefill);
    std::size_t free=0,total=0;Check(cudaMemGetInfo(&free,&total));
    const auto free_before=free;
    const auto available=ServerAvailableBytes(free,total,o.server_device_budget_gib);
    if(worker)worker->RecordAdmission(total,free,available,plan.required());
    if(!plan.Fits(available))return Status(StatusCode::kResourceExhausted,"SIMT weights/KV/workspace/700MiB reserve do not fit GPU; required="+std::to_string(plan.required())+" available="+std::to_string(available));

    // Allocate each inventory entry exactly once, before graph capture/requests.
    Stream stream;std::map<std::string,std::unique_ptr<Buffer>> buffers;
    for(const auto& [name,bytes]:plan.buffers)buffers.emplace(name,std::make_unique<Buffer>(bytes));
    const auto buffer=[&](const char* name)->Buffer&{return *buffers.at(name);};
    auto* arena=buffer("weights").As<unsigned char>();
    Check(cudaMemcpyAsync(arena,model.weights.data(),weights_bytes,cudaMemcpyHostToDevice,stream.s));
    const auto bind=[&](const std::string& name){return BindDeviceMatrix(model,arena,name);};
    const auto embed=bind("model.embed_tokens.weight"),head=bind("lm_head.weight"),norm=bind("model.norm.weight");
    std::array<Layer,28> layers{};std::vector<DeviceMatrix> matrices;
    auto* cache=buffer("kv").As<unsigned char>();std::size_t cache_offset=0;
    for(int l=0;l<28;++l){
      const auto prefix="model.layers."+std::to_string(l)+".";
      auto& a=layers[l];a.norm=bind(prefix+"input_layernorm.weight");a.post=bind(prefix+"post_attention_layernorm.weight");
      a.qn=bind(prefix+"self_attn.q_norm.weight");a.kn=bind(prefix+"self_attn.k_norm.weight");
      a.q=bind(prefix+"self_attn.q_proj.weight");a.k=bind(prefix+"self_attn.k_proj.weight");
      a.v=bind(prefix+"self_attn.v_proj.weight");a.out=bind(prefix+"self_attn.o_proj.weight");a.router=bind(prefix+"mlp.gate.weight");
      a.capacity=l%4==3?o.capacity:1024;
      a.kc=reinterpret_cast<__nv_bfloat16*>(cache+cache_offset);a.vc=a.kc+a.capacity*512;cache_offset+=static_cast<std::size_t>(a.capacity)*2048;
      for(int e=0;e<64;++e)for(const char* part:{"gate","up","down"})matrices.push_back(bind(prefix+"mlp.experts."+std::to_string(e)+"."+part+"_proj.weight"));
    }
    if(cache_offset!=buffer("kv").bytes || matrices.size()*sizeof(DeviceMatrix)!=buffer("descriptors").bytes)
      throw std::runtime_error("SIMT device inventory mismatch");
    auto* descriptors=buffer("descriptors").As<DeviceMatrix>();
    Check(cudaMemcpyAsync(descriptors,matrices.data(),buffer("descriptors").bytes,cudaMemcpyHostToDevice,stream.s));
    auto local_rope=RopeTable(o.capacity,true),global_rope=RopeTable(o.capacity,false);
    auto* rope_local=buffer("rope").As<float>();auto* rope_global=rope_local+o.capacity*128;
    Check(cudaMemcpyAsync(rope_local,local_rope.data(),local_rope.size()*4,cudaMemcpyHostToDevice,stream.s));
    Check(cudaMemcpyAsync(rope_global,global_rope.data(),global_rope.size()*4,cudaMemcpyHostToDevice,stream.s));
    auto* silu=buffer("silu").As<unsigned short>();exl3_silu_lut::Build<<<256,256,0,stream.s>>>(silu);
    Check(cudaStreamSynchronize(stream.s));

    auto* x=buffer("scratch").As<float>();auto* normalized=x+2304;auto* residual=normalized+2304;
    auto* q=residual+2304;auto* attention=q+4096;auto* k=attention+4096;auto* v=k+512;
    auto* router=v+512;auto* gate=router+64;auto* up=gate+8*896;auto* down=up+8*896;auto* logits=down+8*2304;
    auto* command=buffer("command").As<SimtCommand>();auto* bad=buffer("bad").As<int>();
    auto* control=buffer("control").As<Control>();auto* ids=buffer("ids").As<int>();auto* routing=buffer("routing").As<float>();
    auto* encoded=buffer("encoded").As<unsigned char>();auto* scale=buffer("scale").As<float>();
    auto* transformed=buffer("transform").As<float>();auto* scores=buffer("attention_scores").As<float>();
    auto* samples=buffer("sampling").As<float>();auto* adjusted=samples;auto* sorted=adjusted+98304;
    auto* sample_ids=reinterpret_cast<int*>(sorted+98304);auto* sorted_ids=sample_ids+98304;auto* counts=sorted_ids+98304;
    auto* cumulative=reinterpret_cast<double*>(counts+98304);auto* rng=reinterpret_cast<gpu_sampling::State*>(cumulative+98304);
    auto* candidates=buffer("selection").As<SelectionCandidate>();
    SimtDiagnostics diagnostics(o);
    DecodeGraph forward_graph,head_graph;
    const auto forward=[&]{
      EmbedSimt<<<1,128,0,stream.s>>>(embed,command,x,bad,o.capacity);
      for(int l=0;l<28;++l){const auto& a=layers[l];
        const auto dump=[&](const char* name,const void* pointer,std::size_t bytes){if(l==o.simt_diagnostic_layer)diagnostics.Dump(name,pointer,bytes,stream.s);};
        dump("input",x,2304*4);
        diagnostics.Measure("norm",l,stream.s,[&]{NormFP8WideExact<<<1,256,0,stream.s>>>(x,normalized,a.norm,encoded,scale,bad);});
        dump("input_norm",normalized,2304*4);
        diagnostics.Measure("qkv",l,stream.s,[&]{fast_projection::QKVVector<<<1280,128,0,stream.s>>>(a.q,a.k,a.v,encoded,scale,q,k,v,bad);});
        dump("q_proj",q,4096*4);dump("k_proj",k,512*4);dump("v_proj",v,512*4);
        diagnostics.Measure("qknorm_rope_store",l,stream.s,[&]{QKNormRope<<<36,128,0,stream.s>>>(q,k,a.qn,a.kn,l%4==3?rope_global:rope_local,0,bad,nullptr,nullptr,nullptr,o.capacity,buffer("command").As<int>()+1);
        StoreSimt<<<4,128,0,stream.s>>>(k,v,a.kc,a.vc,command,a.capacity);});
        dump("q_rope",q,4096*4);dump("k_rope",k,512*4);
        diagnostics.Measure("attention",l,stream.s,[&]{if(l%4==3)AttendSimt<false><<<32,256,0,stream.s>>>(q,a.kc,a.vc,attention,command,a.capacity,scores,bad);
        else AttendSimt<true><<<32,256,0,stream.s>>>(q,a.kc,a.vc,attention,command,a.capacity,scores,bad);});
        dump("attention",attention,4096*4);
        diagnostics.Measure("o",l,stream.s,[&]{QuantizeSimt<<<1,128,0,stream.s>>>(attention,encoded,scale,4096,bad);
        fast_projection::Vector<<<576,128,0,stream.s>>>(a.out,encoded,scale,residual,bad);});
        dump("o_proj",residual,2304*4);
        diagnostics.Measure("postnorm_router",l,stream.s,[&]{ResidualPostNormBF16<<<1,128,0,stream.s>>>(x,residual,normalized,a.post,bad);
        RouterBF16Warp32Exact<<<64,32,0,stream.s>>>(a.router,normalized,router,bad);
        RouteRankExact<<<1,64,0,stream.s>>>(router,ids,routing,bad);});
        dump("post_norm",normalized,2304*4);dump("router_ids_i32",ids,8*4);dump("routing_weights",routing,8*4);
        diagnostics.Measure("gate",l,stream.s,[&]{exl3_selected::Launch(descriptors,l,0,ids,normalized,gate,transformed,8,bad,stream.s,o.simt_reference_codec);});
        dump("expert_gate",gate,8*896*4);
        diagnostics.Measure("up",l,stream.s,[&]{exl3_selected::Launch(descriptors,l,1,ids,normalized,up,transformed,8,bad,stream.s,o.simt_reference_codec);});
        dump("expert_up",up,8*896*4);
        diagnostics.Measure("silu",l,stream.s,[&]{exl3_silu_lut::Activate<<<56,128,0,stream.s>>>(gate,up,silu,8*896,bad);});
        dump("expert_activation",gate,8*896*4);
        diagnostics.Measure("down",l,stream.s,[&]{exl3_selected::Launch(descriptors,l,2,ids,gate,down,transformed,8,bad,stream.s,o.simt_reference_codec);});
        dump("expert_down",down,8*2304*4);
        diagnostics.Measure("combine",l,stream.s,[&]{Combine<<<18,128,0,stream.s>>>(x,down,routing,bad);});
        dump("output",x,2304*4);
      }
    };
    forward_graph.Capture(stream.s,forward);
    head_graph.Capture(stream.s,[&]{Norm<<<1,128,0,stream.s>>>(x,normalized,norm,2304,1);head_bf16_prefetch::Projection<<<24576,128,0,stream.s>>>(head,normalized,logits,bad);});
    DecodeGraph prefill_graph;
    SimtBatch* batch=nullptr;float *bx=nullptr,*bn=nullptr,*br=nullptr,*bq=nullptr,*ba=nullptr,*bk=nullptr,*bv=nullptr,*brouter=nullptr,*bg=nullptr,*bu=nullptr,*bd=nullptr,*bt=nullptr;
    unsigned char* be=nullptr;float* bs=nullptr;float* bscore=nullptr;__nv_bfloat16* bkv=nullptr;
    if(!o.simt_reference_prefill){
      batch=buffer("batch_state").As<SimtBatch>();bx=buffer("batch_activations").As<float>();bn=bx+256*2304;br=bn+256*2304;
      bq=br+256*2304;ba=bq+256*4096;bk=ba+256*4096;bv=bk+256*512;brouter=bv+256*512;
      bg=buffer("batch_experts").As<float>();bu=bg+256*8*896;bd=bu+256*8*896;bt=buffer("batch_transform").As<float>();
      be=buffer("batch_encoded").As<unsigned char>();bs=buffer("batch_scales").As<float>();bscore=buffer("batch_scores").As<float>();bkv=buffer("batch_kv").As<__nv_bfloat16>();
    }
    int diagnostic_batch_base=0,diagnostic_batch_rows=0;
    const auto prefill=[&]{
      simt_batch::EmbedBatch<<<256,128,0,stream.s>>>(embed,buffer("prompt").As<int>(),bx,bad,batch);
      for(int l=0;l<28;++l){const auto& a=layers[l];
        const auto dump=[&](const char* name,const void* pointer,int stride){
          if(!diagnostics.active || l!=o.simt_diagnostic_layer)return;
          const int last=diagnostics.position,base=diagnostic_batch_base,rows=diagnostic_batch_rows;
          for(int pos:o.simt_diagnostic_positions)if(pos>=base && pos<base+rows){diagnostics.position=pos;diagnostics.Dump(name,static_cast<const unsigned char*>(pointer)+(pos-base)*stride*4,stride*4,stream.s);}diagnostics.position=last;
        };
        dump("input",bx,2304);
        simt_batch::InputNorm<<<256,256,0,stream.s>>>(bx,bn,a.norm,be,bs,bad,batch);dump("input_norm",bn,2304);
        simt_batch::FP8Rows<<<dim3(1024,256),128,0,stream.s>>>(a.q,be,bs,bq,bad,batch);
        simt_batch::FP8Rows<<<dim3(128,256),128,0,stream.s>>>(a.k,be,bs,bk,bad,batch);
        simt_batch::FP8Rows<<<dim3(128,256),128,0,stream.s>>>(a.v,be,bs,bv,bad,batch);
        dump("q_proj",bq,4096);dump("k_proj",bk,512);dump("v_proj",bv,512);
        simt_batch::QKRope<<<dim3(36,256),128,0,stream.s>>>(bq,bk,a.qn,a.kn,l%4==3?rope_global:rope_local,bad,batch);
        dump("q_rope",bq,4096);dump("k_rope",bk,512);
        simt_batch::Stage<<<dim3(4,256),128,0,stream.s>>>(bk,bv,bkv,batch);
        for(int first=0;first<256;first+=8){
          if(l%4==3)AttendSimtBatch<false><<<dim3(32,8),256,0,stream.s>>>(bq,a.kc,a.vc,ba,batch,a.capacity,bscore,bad,bkv,first);
          else AttendSimtBatch<true><<<dim3(32,8),256,0,stream.s>>>(bq,a.kc,a.vc,ba,batch,a.capacity,bscore,bad,bkv,first);
        }
        // Current chunk cannot overwrite old local-ring rows before attention.
        simt_batch::Commit<<<dim3(4,256),128,0,stream.s>>>(bkv,a.kc,a.vc,a.capacity,batch);dump("attention",ba,4096);
        simt_batch::Quantize<<<256,128,0,stream.s>>>(ba,be,bs,bad,batch);
        simt_batch::FP8Rows<<<dim3(576,256),128,0,stream.s>>>(a.out,be,bs,br,bad,batch);dump("o_proj",br,2304);
        simt_batch::PostNorm<<<dim3(1,256),128,0,stream.s>>>(bx,br,bn,a.post,bad,batch);
        simt_batch::Router<<<dim3(64,256),32,0,stream.s>>>(a.router,bn,brouter,bad,batch);
        simt_batch::Rank<<<256,64,0,stream.s>>>(brouter,buffer("batch_ids").As<int>(),buffer("batch_routing").As<float>(),bad,batch);
        auto* bids=buffer("batch_ids").As<int>();auto* weights=buffer("batch_routing").As<float>();
        dump("post_norm",bn,2304);dump("router_ids_i32",bids,8);dump("routing_weights",weights,8);
        const auto experts=[&](int part,const float* in,float* out){const int cols=part==2?896:2304,rows=part==2?2304:896;
          simt_batch::InputHadRows<<<dim3(cols/128,256*8),128,0,stream.s>>>(descriptors,l,part,bids,in,bt,bad,batch);
          exl3_selected::Projection<false><<<dim3(rows/128,256*8),256,0,stream.s>>>(descriptors,l,part,bids,bt,out,256*8,bad,&batch->rows);
        };
        experts(0,bn,bg);dump("expert_gate",bg,8*896);experts(1,bn,bu);dump("expert_up",bu,8*896);
        simt_batch::Silu<<<256*56,128,0,stream.s>>>(bg,bu,silu,bad,batch);dump("expert_activation",bg,8*896);
        experts(2,bg,bd);dump("expert_down",bd,8*2304);
        simt_batch::CombineBatch<<<dim3(18,256),128,0,stream.s>>>(bx,bd,weights,bad,batch);dump("output",bx,2304);
      }
    };
    if(batch)prefill_graph.Capture(stream.s,prefill);
    Check(cudaMemGetInfo(&free,&total));const auto after_graph=ServerAvailableBytes(free,total,o.server_device_budget_gib);
    if(after_graph<plan.reserve)return Status(StatusCode::kResourceExhausted,"SIMT post-graph admission violates 700MiB reserve");
    InferenceResult identity;identity.weights_sha256=model.weights_sha256;identity.weights_bytes=weights_bytes;identity.kv_bytes=buffer("kv").bytes;
    identity.resident_buffer_bytes=plan.resident;identity.device_free_before=free_before;identity.device_free_after_initialization=free;
    identity.kernel_profile="exl3-k4-simt-sm" MELLUM16_ARCHITECTURE "-token-graph-pascal-codec-v2";
    if(batch)identity.kernel_profile+=";layer-batch256-staged-kv-v1";
    identity.math_profile="mellum16-simt-exl3-k4-v1;FP8-E4M3-activations/FP32-FMA-projections/BF16-output;compensated-BF16-QK/softmax/PV;EXL3-K4-H128-BF16-input/FP16-codec/FP32-FMA/BF16-output;exact-RN-SiLU;compensated-router/head";
    identity.execution_profile=batch?"simt-layer-batch256-staged-kv-v1":"simt-token-graph-chunk-commits";identity.prefill_execution=batch?"layer-batch256-graph":"token-graph";identity.decode_execution="token-graph";identity.prefill_chunk=o.prefill_chunk;
    identity.actual_attention_projection="FP8 storage and activations, FP32 SIMT FMA";identity.actual_expert_dispatch="EXL3 K4 selected N128/eight-warps/FP32 FMA, all prompt rows";
    const double load_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-load_begin).count();
    if(worker){worker->RecordLoad(plan.resident);worker->RecordCapture();worker->RecordCapture();if(batch)worker->RecordCapture();worker->RecordGraphAdmission(after_graph);worker->RecordIdentity(identity.math_profile,identity.kernel_profile);}
    const auto check_numerics=[&]{int value=0;Check(cudaMemcpyAsync(&value,bad,sizeof(value),cudaMemcpyDeviceToHost,stream.s));Check(cudaStreamSynchronize(stream.s));if(value)throw std::runtime_error("nonfinite/invalid SIMT forward; session poisoned");};
    const auto checkpoint_copy=[&](bool restore){std::size_t offset=0;for(int l=0;l<28;++l)if(l%4!=3){
      auto* saved=buffer("local_checkpoint").As<unsigned char>()+offset;
      Check(cudaMemcpyAsync(restore?static_cast<void*>(layers[l].kc):saved,restore?saved:static_cast<void*>(layers[l].kc),1024*1024,cudaMemcpyDeviceToDevice,stream.s));
      Check(cudaMemcpyAsync(restore?static_cast<void*>(layers[l].vc):saved+1024*1024,restore?saved+1024*1024:static_cast<void*>(layers[l].vc),1024*1024,cudaMemcpyDeviceToDevice,stream.s));offset+=1024*2048;
    }Check(cudaStreamSynchronize(stream.s));};
    int diagnostic_request=0;
    do {
      ++diagnostic_request;
      if(!o.simt_diagnostics.empty()){diagnostics.directory=o.simt_diagnostics/("request-"+std::to_string(diagnostic_request));std::filesystem::create_directories(diagnostics.directory);}
      if(worker && !worker->AwaitPrepare())break;
      if(worker && worker->checkpoint_restore_pending){checkpoint_copy(true);worker->RecordCheckpointRestore();}
      Check(cudaMemsetAsync(bad,0,sizeof(int),stream.s));
      Check(cudaMemsetAsync(counts,0,98304*sizeof(int),stream.s));
      gpu_sampling::State initial{o.sampling.seed,0};Check(cudaMemcpyAsync(rng,&initial,sizeof(initial),cudaMemcpyHostToDevice,stream.s));
      if(batch || o.sampling.penalties())Check(cudaMemcpyAsync(buffer("prompt").p,o.input_ids.data(),o.input_ids.size()*4,cudaMemcpyHostToDevice,stream.s));
      if(o.sampling.penalties()){
        gpu_sampling::CountPrompt<<<(o.input_ids.size()+255)/256,256,0,stream.s>>>(buffer("prompt").As<int>(),static_cast<int>(o.input_ids.size()),counts,98304,bad);
      }
      Check(cudaStreamSynchronize(stream.s));
      if(worker && !worker->AwaitGenerate())break;
      if(worker && worker->dropped)continue;
      InferenceResult result=identity;result.initialization_seconds=load_seconds;
      const auto started=std::chrono::steady_clock::now();auto prefill_finished=started;
      int executed=worker?worker->append_base:0,token=0;
      const auto run_forward=[&](int next_token,int pos){
        diagnostics.position=pos;
        diagnostics.active=!o.simt_diagnostics.empty() && (o.simt_diagnostic_positions.empty()?pos>=static_cast<int>(o.input_ids.size())-1:std::find(o.simt_diagnostic_positions.begin(),o.simt_diagnostic_positions.end(),pos)!=o.simt_diagnostic_positions.end());
        SetSimtCommand<<<1,1,0,stream.s>>>(command,next_token,pos);
        if(diagnostics.active || o.simt_eager)forward();else forward_graph.Launch(stream.s);
        diagnostics.Flush(stream.s);
      };
      const auto cancelled=[&]{return (o.cancellation_requested && o.cancellation_requested->load()) || (o.diagnostic_cancel_after>=0 && static_cast<int>(result.generated_ids.size())>=o.diagnostic_cancel_after);};
      if(batch){
        while(executed<static_cast<int>(o.input_ids.size()) && !result.cancelled){
          if(cancelled()){result.cancelled=true;break;}
          int rows=std::min({256,o.prefill_chunk,static_cast<int>(o.input_ids.size())-executed});
          if(worker && worker->checkpoint_anchor>executed && worker->checkpoint_anchor<executed+rows)rows=worker->checkpoint_anchor-executed;
          diagnostic_batch_base=executed;diagnostic_batch_rows=rows;diagnostics.position=executed+rows-1;
          diagnostics.active=!o.simt_diagnostics.empty() && std::any_of(o.simt_diagnostic_positions.begin(),o.simt_diagnostic_positions.end(),[&](int pos){return pos>=executed && pos<executed+rows;});
          SetSimtBatch<<<1,1,0,stream.s>>>(batch,executed,rows,o.capacity,bad);
          if(diagnostics.active || o.simt_eager)prefill();else prefill_graph.Launch(stream.s);
          diagnostics.Flush(stream.s);check_numerics();
          Check(cudaMemcpyAsync(x,bx+(rows-1)*2304,2304*4,cudaMemcpyDeviceToDevice,stream.s));
          executed+=rows;
          if(worker && worker->checkpoint_anchor==executed){worker->BeginCheckpointCapture();checkpoint_copy(false);worker->CompleteCheckpointCapture();}
          if(worker)worker->CommitChunk(executed,rows!=256);
          if(cancelled())result.cancelled=true;
        }
      }else for(int base=executed;base<static_cast<int>(o.input_ids.size()) && !result.cancelled;base+=o.prefill_chunk){
        if(cancelled()){result.cancelled=true;break;}
        const int end=std::min(base+o.prefill_chunk,static_cast<int>(o.input_ids.size()));
        for(int pos=base;pos<end;++pos){
          run_forward(o.input_ids[pos],pos);
          if(worker && worker->checkpoint_anchor==pos+1){worker->BeginCheckpointCapture();check_numerics();checkpoint_copy(false);worker->CompleteCheckpointCapture();}
        }
        check_numerics();executed=end;
        if(worker)worker->CommitChunk(executed,end-base!=o.prefill_chunk);
        if(cancelled())result.cancelled=true;
      }
      result.prefill_tokens=executed;
      if(!result.cancelled){
        // Computing the head for a zero-output request preserves capture_logits.
        head_graph.Launch(stream.s);check_numerics();
        for(int step=0;step<o.generate && !result.cancelled;++step){
          if(cancelled()){result.cancelled=true;break;}
          if(step){run_forward(o.simt_teacher_tokens.empty()?token:o.simt_teacher_tokens[step-1],static_cast<int>(o.input_ids.size())+step-1);head_graph.Launch(stream.s);++result.decode_steps;}
          if(o.sampling.stochastic())gpu_sampling::Launch(logits,adjusted,sorted,sample_ids,sorted_ids,cumulative,rng,control,bad,buffer("sampling_workspace").p,sampler_bytes,stream.s,o.sampling,counts);
          else {
            const float* selection_logits=logits;
            if(o.sampling.penalties()){gpu_sampling::PrepareLaunch(logits,adjusted,sample_ids,counts,98304,bad,o.sampling,stream.s);selection_logits=adjusted;}
            SelectTiles<<<96,256,0,stream.s>>>(selection_logits,candidates,98304,bad);Select<<<1,256,0,stream.s>>>(candidates,control,96,bad);
          }
          if(o.sampling.penalties())gpu_sampling::RecordChoice<<<1,1,0,stream.s>>>(control,counts,98304,bad);
          Control chosen{};Check(cudaMemcpyAsync(&chosen,control,sizeof(chosen),cudaMemcpyDeviceToHost,stream.s));Check(cudaStreamSynchronize(stream.s));
          if(chosen.bad)throw std::runtime_error("invalid SIMT sampling/forward; session poisoned");
          if(!o.simt_diagnostics.empty()){
            diagnostics.position=static_cast<int>(o.input_ids.size())+step-1;diagnostics.active=true;
            diagnostics.Dump("logits",logits,98304*4,stream.s);diagnostics.active=false;
          }
          token=chosen.token;result.generated_ids.push_back(token);
          if(step==0)prefill_finished=std::chrono::steady_clock::now();
          if(worker && !worker->Emit(token,step==0))result.cancelled=true;
          if(token==28 && o.simt_teacher_tokens.empty())break;
        }
        if(o.generate==0)prefill_finished=std::chrono::steady_clock::now();
        if(o.capture_logits){result.logits.resize(98304);Check(cudaMemcpyAsync(result.logits.data(),logits,98304*4,cudaMemcpyDeviceToHost,stream.s));}
      }
      Check(cudaGetLastError());Check(cudaStreamSynchronize(stream.s));
      const auto finished=std::chrono::steady_clock::now();
      if(result.cancelled && result.generated_ids.empty())prefill_finished=finished;
      result.prefill_seconds=std::chrono::duration<double>(prefill_finished-started).count();
      result.decode_seconds=std::chrono::duration<double>(finished-prefill_finished).count();result.inference_seconds=std::chrono::duration<double>(finished-started).count();
      Check(cudaMemGetInfo(&free,&total));result.device_free_after_execution=free;
      if(worker)worker->Publish(result);else return result;
    }while(worker);
    return identity;
  }catch(const std::overflow_error& e){return Status(StatusCode::kResourceExhausted,e.what());}
   catch(const std::exception& e){return Status(StatusCode::kInternal,e.what());}
}
} // namespace

Result<InferenceResult> RunCudaInference(const InferenceOptions& o){return RunSimt(o);}
Result<std::unique_ptr<RuntimeBackend>> LoadCudaModelRuntime(InferenceOptions o){
  auto owner=std::make_unique<CudaModelWorker>(std::move(o));auto* worker=owner.get();
  worker->thread=std::thread([worker]{try{auto result=RunSimt(worker->options,worker);if(!result.ok())worker->Fail(result.status());}catch(const std::exception& e){worker->Fail({StatusCode::kInternal,e.what()});}});
  std::unique_lock lock(worker->mutex);worker->cv.wait(lock,[&]{return worker->loaded || worker->failed;});
  if(worker->failed){auto failure=worker->fault;lock.unlock();return failure;}
  lock.unlock();return std::unique_ptr<RuntimeBackend>(std::move(owner));
}
} // namespace mellum16
