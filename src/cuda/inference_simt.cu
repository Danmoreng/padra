// Portable EXL3 K3/K4 backend. Compile separately from Tensor Core / FP4 kernels.
#include "runtime/inference.h"
#include "runtime/simt_memory.h"
#include "runtime/simt_features.h"
#include "runtime/native_artifacts.h"
#include "runtime/mtp_memory.h"
#include "runtime/mtp_schedule.h"
#include "model/mtp.h"
#include <optional>
#include "runtime/gpu_build_config.h"
#include "cuda/model_runtime_worker.h"
#include "cuda/device_matrix.cuh"
#include "compiler/sha256.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#if PADRA_CUDA_SM == 61
#include <cublas_v2.h>
#endif
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
#include "head_q8.cuh"
#include "mtp_simt_ops.cuh"
#define PADRA_MTP_SIMT 1
#include "mtp.cuh"
#undef PADRA_MTP_SIMT
#include "simt_diagnostics.h"
#include "simt_prefill.cuh"
#include "kv_q8_simt.cuh"
#include "simt_prefill_attention.cuh"
#include "fp8_prefill_simt.cuh"
#include "expert_exl3_prefill_simt.cuh"
#include "simt_attention_split.cuh"
#include "simt_prefill_attention_split.cuh"
#include "expert_exl3_prefill_fast.cuh"
#if PADRA_PASCAL_DP4A_PREFILL
#include "expert_exl3_prefill_dp4a.cuh"
#endif
#include "pascal_attention_fast.cuh"
#if PADRA_CUDA_SM == 61
#include "pascal_sgemm.cuh"
#include "pascal_attention_sgemm.cuh"
#endif

struct Layer {
  DeviceMatrix norm,post,qn,kn,q,k,v,out,router;
  __nv_bfloat16 *kc,*vc;
  int capacity;
};
static_assert(sizeof(DeviceMatrix)==56);

Status ValidateSimtOptions(const InferenceOptions& o,bool persistent){
  if(auto status=ValidateSimtFeatures(o);!status.ok())return status;
  if(o.capacity<1024 || o.capacity>131072 || !ValidServerChunk(o.prefill_chunk) || !ValidServerBudget(o.server_device_budget_gib))
    return {StatusCode::kInvalidArgument,"invalid SIMT capacity/chunk/device budget"};
  if(o.capture_trace || o.snapshot_layer>=0 || o.capture_quantization_rows || o.diagnostic_routes || o.diagnostic_h128_fp32 || o.diagnostic_exl3_layer>=0 || !o.diagnostic_attention.empty())
    return {StatusCode::kUnsupported,"SM120 research captures are not implemented in the SIMT backend"};
  if(o.diagnostic_decode_token>=0)
    return {StatusCode::kUnsupported,"teacher-forced decode is not implemented in the SIMT backend"};
  if(o.simt_diagnostic_layer<0 || o.simt_diagnostic_layer>=28)
    return {StatusCode::kInvalidArgument,"invalid SIMT diagnostic layer"};
  for(int id:o.simt_teacher_tokens)if(id<0 || id>=98304)return {StatusCode::kInvalidArgument,"invalid SIMT teacher token"};
  for(const auto& sequence:o.simt_teacher_sequences)for(int id:sequence)if(id<0 || id>=98304)return {StatusCode::kInvalidArgument,"invalid SIMT teacher sequence token"};
  if(!o.simt_teacher_sequences.empty() && (o.simt_diagnostics.empty() || !o.simt_teacher_tokens.empty()))return {StatusCode::kInvalidArgument,"teacher sequences require offline diagnostics and no global teacher tokens"};
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
    auto loaded=internal::LoadCompiledModel(o.artifact);if(!loaded.ok())return loaded.status();
    auto model=std::move(loaded).value();
    if(!model.exl3_experts || model.exl3_rate!=o.exl3_rate || model.tokenizer_sha256!=o.tokenizer_sha256)
      return Status(StatusCode::kDataLoss,"SIMT artifact/expert rate/tokenizer identity mismatch");
    const auto* pinned=FindNativeArtifact(model.manifest_sha256);
    if(worker && (!pinned || model.weights_sha256!=pinned->weights_sha256 || model.weights.size()!=pinned->weights_bytes || model.exl3_rate!=pinned->expert_rate))
      return Status(StatusCode::kDataLoss,"persistent runtime requires a pinned native checkpoint");
    const auto head_format=model.tensors.at("lm_head.weight").format;
    if(head_format!=internal::WeightFormat::Bf16 && head_format!=internal::WeightFormat::Q8HeadBlock32)
      return Status(StatusCode::kUnsupported,"SIMT requires a BF16 or stored Q8 output head");
    const bool stored_q8=head_format==internal::WeightFormat::Q8HeadBlock32;
    if(stored_q8!=o.q8_head)
      return Status(StatusCode::kUnsupported,"SIMT Q8 head option must match the stored checkpoint");
#if PADRA_PASCAL_DP4A_PREFILL
    if(o.exl3_rate!=4)return Status(StatusCode::kUnsupported,"approximate DP4A prefill currently requires K4; rebuild with PADRA_PASCAL_DP4A_PREFILL=OFF for K3");
#endif
#if PADRA_CUDA_SM != 61
    if(!o.mtp.artifact.empty())return Status(StatusCode::kUnsupported,"SIMT MTP is currently implemented for SM61");
#endif
    std::optional<internal::MtpModel> mtp_model;
    if(!o.mtp.artifact.empty()){
      auto loaded_draft=internal::LoadMtpModel(o.mtp.artifact,false,true);
      if(!loaded_draft.ok())return loaded_draft.status();
      mtp_model=std::move(loaded_draft).value();
    }
    const auto weights_bytes=model.weights.size();
    const auto sampler_bytes=gpu_sampling::WorkspaceBytes();
#if PADRA_CUDA_SM == 61
    const bool fast_math=!o.simt_reference_math;
#else
    const bool fast_math=false;
#endif
    const bool q8_kv=o.kv_cache_format==KVCacheFormat::Q8Token512Block32F16ScaleRotatedV2;
    auto plan=PlanSimtMemory(weights_bytes,o.capacity,sampler_bytes,o.server_kv_checkpoint,!o.simt_reference_prefill,fast_math,o.q8_head,q8_kv);
    if(mtp_model){
      // Draft/Verification own their allocations; the remaining arenas belong
      // to the common inventory and are allocated exactly once below.
      plan.resident=admission::Sum({plan.resident,MtpDraftResidentBytes(false,true),mtp::kVerifyResidentBytes});
      const auto add=[&](const char* name,std::size_t bytes){plan.buffers.emplace(name,bytes);plan.resident=admission::Sum({plan.resident,bytes});};
      add("mtp_hidden",257ULL*2304*4);add("mtp_state",sizeof(DecodeState));
      if(o.server_kv_checkpoint)add("mtp_checkpoint",kMtpCheckpointBytes);
    }
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
      a.kc=reinterpret_cast<__nv_bfloat16*>(cache+cache_offset);a.vc=reinterpret_cast<__nv_bfloat16*>(cache+cache_offset+static_cast<std::size_t>(a.capacity)*(q8_kv?544:1024));cache_offset+=static_cast<std::size_t>(a.capacity)*(q8_kv?1088:2048);
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
    std::unique_ptr<mtp::Draft> draft;
    std::unique_ptr<mtp::Verification> verifier;
    float* mtp_hidden=nullptr;DecodeState* mtp_state=nullptr;
    if(mtp_model){
      draft=std::make_unique<mtp::Draft>(*mtp_model,embed,head,rope_local,o.capacity,stream.s,buffer("head_input").As<unsigned char>(),transformed);
      verifier=std::make_unique<mtp::Verification>();
      if(draft->ResidentBytes()!=MtpDraftResidentBytes(false,true) || verifier->ResidentBytes()!=mtp::kVerifyResidentBytes)throw std::runtime_error("SIMT MTP inventory mismatch");
      mtp_hidden=buffer("mtp_hidden").As<float>();mtp_state=buffer("mtp_state").As<DecodeState>();
      std::array<mtp::CacheBinding,29> bindings{};
      for(int l=0;l<28;++l)bindings[l]={layers[l].kc,layers[l].vc,layers[l].capacity,nullptr,q8_kv?544:1024,0};
      bindings[28]={draft->keys,draft->values,1024};
      Check(cudaMemcpyAsync(verifier->bindings.p,bindings.data(),sizeof(bindings),cudaMemcpyHostToDevice,stream.s));
      Check(cudaStreamSynchronize(stream.s));
    }
    SimtDiagnostics diagnostics(o);
    const auto q8_attention=[&](const Layer& a,bool local,const float* q,float* attention){
          auto* state=buffer("q8_state").As<DecodeState>();auto* query=buffer("q8_query").As<q8_native::QueryBlock>();
          auto* partial=buffer("q8_partial").As<float>();auto* lse=buffer("q8_lse").As<float>();
          const auto* keys=reinterpret_cast<const q8_native::CacheRow*>(a.kc);const auto* values=reinterpret_cast<const q8_native::CacheRow*>(a.vc);
          const int splits=local?4:32;
          q8_native::EncodeQuery<<<8,128,0,stream.s>>>(q,query,bad);
          if(!local){q8_native::Split<false><<<dim3(8,splits),128,0,stream.s>>>(query,keys,values,partial,lse,state,a.capacity,splits,bad,o.capacity);q8_native::Merge<false,float><<<8,128,0,stream.s>>>(partial,lse,attention,state,splits,bad,o.capacity);}
          else {q8_native::Split<true><<<dim3(8,splits),128,0,stream.s>>>(query,keys,values,partial,lse,state,a.capacity,splits,bad,o.capacity);q8_native::Merge<true,float><<<8,128,0,stream.s>>>(partial,lse,attention,state,splits,bad,o.capacity);}
    };
    DecodeGraph forward_graph,head_graph;
    const auto forward=[&](int attention_keys){
      EmbedSimt<<<1,128,0,stream.s>>>(embed,command,x,bad,o.capacity);
      if(q8_kv)q8_simt::State<<<1,1,0,stream.s>>>(command,buffer("q8_state").As<DecodeState>(),o.capacity);
      for(int l=0;l<28;++l){const auto& a=layers[l];
        const auto dump=[&](const char* name,const void* pointer,std::size_t bytes){if(l==o.simt_diagnostic_layer)diagnostics.Dump(name,pointer,bytes,stream.s);};
        dump("input",x,2304*4);
        diagnostics.Measure("norm",l,stream.s,[&]{NormFP8WideExact<<<1,256,0,stream.s>>>(x,normalized,a.norm,encoded,scale,bad);});
        dump("input_norm",normalized,2304*4);
        diagnostics.Measure("qkv",l,stream.s,[&]{fast_projection::QKVVector<<<1280,128,0,stream.s>>>(a.q,a.k,a.v,encoded,scale,q,k,v,bad);});
        dump("q_proj",q,4096*4);dump("k_proj",k,512*4);dump("v_proj",v,512*4);
        diagnostics.Measure("qknorm_rope_store",l,stream.s,[&]{QKNormRope<<<36,128,0,stream.s>>>(q,k,a.qn,a.kn,l%4==3?rope_global:rope_local,0,bad,nullptr,nullptr,nullptr,o.capacity,buffer("command").As<int>()+1);
        if(q8_kv)q8_native::EncodeKV<true><<<1,128,0,stream.s>>>(k,v,reinterpret_cast<q8_native::CacheRow*>(a.kc),reinterpret_cast<q8_native::CacheRow*>(a.vc),1,bad,buffer("q8_state").As<DecodeState>(),a.capacity,o.capacity);
        else StoreSimt<<<4,128,0,stream.s>>>(k,v,a.kc,a.vc,command,a.capacity);});
        dump("q_rope",q,4096*4);dump("k_rope",k,512*4);
        diagnostics.Measure("attention",l,stream.s,[&]{if(q8_kv){q8_attention(a,l%4!=3,q,attention);
        }else if(fast_math)pascal_attention::Decode(q,a.kc,a.vc,attention,command,a.capacity,l%4!=3,scores,bad,stream.s,attention_keys?attention_keys:1024,buffer("attention_partials").As<float>());else if(attention_keys)simt_attention_split::Launch(q,a.kc,a.vc,attention,command,a.capacity,l%4!=3,scores,bad,stream.s,attention_keys);else if(l%4==3)AttendSimt<false,8><<<32,256,0,stream.s>>>(q,a.kc,a.vc,attention,command,a.capacity,scores,bad);
        else AttendSimt<true,8><<<32,256,0,stream.s>>>(q,a.kc,a.vc,attention,command,a.capacity,scores,bad);});
        dump("attention",attention,4096*4);
        diagnostics.Measure("o",l,stream.s,[&]{QuantizeSimt<<<1,128,0,stream.s>>>(attention,encoded,scale,4096,bad);
        fast_projection::Vector<<<576,128,0,stream.s>>>(a.out,encoded,scale,residual,bad);});
        dump("o_proj",residual,2304*4);
        diagnostics.Measure("postnorm_router",l,stream.s,[&]{ResidualPostNormBF16<<<1,128,0,stream.s>>>(x,residual,normalized,a.post,bad);
        RouterBF16Warp32Exact<<<64,32,0,stream.s>>>(a.router,normalized,router,bad);
        RouteRankExact<<<1,64,0,stream.s>>>(router,ids,routing,bad);});
        dump("post_norm",normalized,2304*4);dump("router_ids_i32",ids,8*4);dump("routing_weights",routing,8*4);
        diagnostics.Measure("gate",l,stream.s,[&]{exl3_selected::Launch(descriptors,l,0,ids,normalized,gate,transformed,8,bad,stream.s,o.simt_reference_codec,o.exl3_rate);});
        dump("expert_gate",gate,8*896*4);
        diagnostics.Measure("up",l,stream.s,[&]{exl3_selected::Launch(descriptors,l,1,ids,normalized,up,transformed,8,bad,stream.s,o.simt_reference_codec,o.exl3_rate);});
        dump("expert_up",up,8*896*4);
        diagnostics.Measure("silu",l,stream.s,[&]{exl3_silu_lut::Activate<<<56,128,0,stream.s>>>(gate,up,silu,8*896,bad);});
        dump("expert_activation",gate,8*896*4);
        diagnostics.Measure("down",l,stream.s,[&]{exl3_selected::Launch(descriptors,l,2,ids,gate,down,transformed,8,bad,stream.s,o.simt_reference_codec,o.exl3_rate);});
        dump("expert_down",down,8*2304*4);
        diagnostics.Measure("combine",l,stream.s,[&]{Combine<<<18,128,0,stream.s>>>(x,down,routing,bad);});
        dump("output",x,2304*4);
      }
    };
    DecodeGraph middle_graph,long_graph;
    forward_graph.Capture(stream.s,[&]{forward(0);});
    if(fast_math || !o.simt_reference_attention){middle_graph.Capture(stream.s,[&]{forward(4096);});long_graph.Capture(stream.s,[&]{forward(o.capacity);});}
    head_graph.Capture(stream.s,[&]{Norm<<<1,128,0,stream.s>>>(x,normalized,norm,2304,1);if(o.q8_head)head_q8::Launch(1,head,normalized,buffer("head_input").As<unsigned char>(),logits,bad,stream.s);else head_bf16_prefetch::Projection<<<24576,128,0,stream.s>>>(head,normalized,logits,bad);});
    DecodeGraph prefill_graph,prefill_1024_graph,prefill_4096_graph;
    const bool bounded_prefill=PADRA_PASCAL_DP4A_PREFILL && fast_math && !o.simt_reference_prefill && o.capacity==8192;
    SimtBatch* batch=nullptr;float *bx=nullptr,*bn=nullptr,*br=nullptr,*bq=nullptr,*ba=nullptr,*bk=nullptr,*bv=nullptr,*brouter=nullptr,*bg=nullptr,*bu=nullptr,*bd=nullptr,*bt=nullptr;
    unsigned char* be=nullptr;float* bs=nullptr;float* bscore=nullptr;__nv_bfloat16* bkv=nullptr;
    if(!o.simt_reference_prefill){
      batch=buffer("batch_state").As<SimtBatch>();bx=buffer("batch_activations").As<float>();bn=bx+256*2304;br=bn+256*2304;
      bq=br+256*2304;ba=bq+256*4096;bk=ba+256*4096;bv=bk+256*512;brouter=bv+256*512;
      bg=buffer("batch_experts").As<float>();bu=bg+256*8*896;bd=bu+256*8*896;bt=buffer("batch_transform").As<float>();
      be=buffer("batch_encoded").As<unsigned char>();bs=buffer("batch_scales").As<float>();bscore=buffer("batch_scores").As<float>();bkv=buffer("batch_kv").As<__nv_bfloat16>();
    }
#if PADRA_CUDA_SM == 61
    std::unique_ptr<pascal_sgemm::Handle> blas;
    if(fast_math && batch){
      blas=std::make_unique<pascal_sgemm::Handle>(stream.s,buffer("sgemm_workspace").p,buffer("sgemm_workspace").bytes);
      // Initialize/warm cuBLAS outside capture. No request allocations/uploads.
      Check(cudaMemsetAsync(be,0,buffer("batch_encoded").bytes,stream.s));Check(cudaMemsetAsync(bs,0,buffer("batch_scales").bytes,stream.s));Check(cudaMemsetAsync(bad,0,4,stream.s));
      SetSimtBatch<<<1,1,0,stream.s>>>(batch,0,1,o.capacity,bad);
      for(auto matrix:{layers[0].q,layers[0].k,layers[0].v,layers[0].out})pascal_sgemm::Projection(*blas,matrix,be,bs,bq,buffer("sgemm_weights").As<float>(),buffer("sgemm_inputs").As<float>(),bad,batch,stream.s);
      Check(cudaStreamSynchronize(stream.s));
    }
#endif
    int active_batch_limit=256;
    const auto batch_projection=[&](DeviceMatrix matrix,float* output){
      if(active_batch_limit<=5){
        switch(active_batch_limit){
        case 2:simt_batch::FP8Small<2><<<dim3(matrix.rows/4,1),128,0,stream.s>>>(matrix,be,bs,output,bad,batch);break;
        case 3:simt_batch::FP8Small<3><<<dim3(matrix.rows/4,1),128,0,stream.s>>>(matrix,be,bs,output,bad,batch);break;
        case 4:simt_batch::FP8Small<4><<<dim3(matrix.rows/4,1),128,0,stream.s>>>(matrix,be,bs,output,bad,batch);break;
        case 5:simt_batch::FP8Small<5><<<dim3(matrix.rows/4,1),128,0,stream.s>>>(matrix,be,bs,output,bad,batch);break;
        }return;
      }
#if PADRA_CUDA_SM == 61
      if(blas){pascal_sgemm::Projection(*blas,matrix,be,bs,output,buffer("sgemm_weights").As<float>(),buffer("sgemm_inputs").As<float>(),bad,batch,stream.s);return;}
#endif
      simt_batch::FP8Tiled(matrix,be,bs,output,bad,batch,stream.s);
    };
    int diagnostic_batch_base=0,diagnostic_batch_rows=0;
    const auto prefill=[&](int maximum_keys,int row_limit=256){
      active_batch_limit=row_limit;
      simt_batch::EmbedBatch<<<row_limit,128,0,stream.s>>>(embed,buffer("prompt").As<int>(),bx,bad,batch);
      for(int l=0;l<28;++l){const auto& a=layers[l];
        const auto dump=[&](const char* name,const void* pointer,int stride){
          if(!diagnostics.active || l!=o.simt_diagnostic_layer)return;
          const int last=diagnostics.position,base=diagnostic_batch_base,rows=diagnostic_batch_rows;
          for(int pos:o.simt_diagnostic_positions)if(pos>=base && pos<base+rows){diagnostics.position=pos;diagnostics.Dump(name,static_cast<const unsigned char*>(pointer)+(pos-base)*stride*4,stride*4,stream.s);}diagnostics.position=last;
        };
        dump("input",bx,2304);
        diagnostics.Measure("norm",l,stream.s,[&]{simt_batch::InputNorm<<<row_limit,256,0,stream.s>>>(bx,bn,a.norm,be,bs,bad,batch);});dump("input_norm",bn,2304);
        diagnostics.Measure("qkv",l,stream.s,[&]{batch_projection(a.q,bq);batch_projection(a.k,bk);batch_projection(a.v,bv);});
        dump("q_proj",bq,4096);dump("k_proj",bk,512);dump("v_proj",bv,512);
        diagnostics.Measure("qknorm_rope",l,stream.s,[&]{simt_batch::QKRope<<<dim3(36,row_limit),128,0,stream.s>>>(bq,bk,a.qn,a.kn,l%4==3?rope_global:rope_local,bad,batch);});
        dump("q_rope",bq,4096);dump("k_rope",bk,512);
        diagnostics.Measure("attention",l,stream.s,[&]{
        if(row_limit<=5){
          // Same per-row attention as ordinary decoding. Store only the current
          // row before reading a wrapping ring; future verifier rows remain tentative.
          for(int row=0;row<row_limit;++row){
            q8_simt::BatchCommand<<<1,1,0,stream.s>>>(batch,row,command);
            if(q8_kv){
              q8_simt::State<<<1,1,0,stream.s>>>(command,buffer("q8_state").As<DecodeState>(),o.capacity);
              q8_native::EncodeKV<true><<<1,128,0,stream.s>>>(bk+row*512,bv+row*512,reinterpret_cast<q8_native::CacheRow*>(a.kc),reinterpret_cast<q8_native::CacheRow*>(a.vc),1,bad,buffer("q8_state").As<DecodeState>(),a.capacity,o.capacity);
              q8_attention(a,l%4!=3,bq+row*4096,ba+row*4096);
            }else{
              StoreSimt<<<4,128,0,stream.s>>>(bk+row*512,bv+row*512,a.kc,a.vc,command,a.capacity);
              pascal_attention::Decode(bq+row*4096,a.kc,a.vc,ba+row*4096,command,a.capacity,l%4!=3,scores,bad,stream.s,o.capacity,buffer("attention_partials").As<float>());
            }
          }
        }else{
        const __nv_bfloat16* attention_k=a.kc;const __nv_bfloat16* attention_v=a.vc;
        if(q8_kv){
          auto* chunk=buffer("q8_chunk").As<q8_native::CacheRow>();
          auto* prefix=buffer("q8_prefix").As<__nv_bfloat16>();
          q8_native::EncodeKV<<<row_limit,128,0,stream.s>>>(bk,bv,chunk,chunk+256,row_limit,bad,nullptr,0,o.capacity,&batch->rows);
          q8_simt::Dequantize<true><<<512,256,0,stream.s>>>(chunk,chunk+256,bkv,bkv+256*512,batch,a.capacity,bad);
          q8_simt::Dequantize<false><<<512,256,0,stream.s>>>(reinterpret_cast<const q8_native::CacheRow*>(a.kc),reinterpret_cast<const q8_native::CacheRow*>(a.vc),prefix,prefix+o.capacity*512,batch,a.capacity,bad);
          attention_k=prefix;attention_v=prefix+o.capacity*512;
          q8_native::RotateRows<false><<<row_limit*8,128,0,stream.s>>>(bq,row_limit,bad,&batch->rows);
        }else simt_batch::Stage<<<dim3(4,row_limit),128,0,stream.s>>>(bk,bv,bkv,batch);
#if PADRA_CUDA_SM == 61
        if(blas && row_limit>5 && o.capacity<=16384)pascal_attention_sgemm::Prefill(*blas,bq,attention_k,attention_v,bkv,ba,batch,a.capacity,l%4!=3,buffer("sgemm_weights").As<float>(),buffer("sgemm_inputs").As<float>(),bscore,bad,stream.s,maximum_keys);
        else
#endif
        for(int first=0;first<row_limit;first+=8){
          if(fast_math)pascal_attention::Prefill(bq,attention_k,attention_v,ba,batch,a.capacity,l%4!=3,bscore,bad,bkv,first,stream.s);
          else if(!o.simt_reference_attention)simt_prefill_attention_split::Launch(bq,attention_k,attention_v,ba,batch,a.capacity,l%4!=3,bscore,bad,bkv,first,stream.s);
          else if(l%4==3)AttendSimtBatch<false><<<dim3(32,8),256,0,stream.s>>>(bq,attention_k,attention_v,ba,batch,a.capacity,bscore,bad,bkv,first);
          else AttendSimtBatch<true><<<dim3(32,8),256,0,stream.s>>>(bq,attention_k,attention_v,ba,batch,a.capacity,bscore,bad,bkv,first);
        }
        // Current chunk cannot overwrite old local-ring rows before attention.
        if(q8_kv){
          auto* chunk=buffer("q8_chunk").As<q8_native::CacheRow>();
          q8_simt::Commit<<<row_limit,128,0,stream.s>>>(chunk,chunk+256,reinterpret_cast<q8_native::CacheRow*>(a.kc),reinterpret_cast<q8_native::CacheRow*>(a.vc),batch,a.capacity);
          q8_native::RotateRows<true><<<row_limit*8,128,0,stream.s>>>(ba,row_limit,bad,&batch->rows);
        }else simt_batch::Commit<<<dim3(4,row_limit),128,0,stream.s>>>(bkv,a.kc,a.vc,a.capacity,batch);        }});dump("attention",ba,4096);
        diagnostics.Measure("o",l,stream.s,[&]{simt_batch::Quantize<<<row_limit,128,0,stream.s>>>(ba,be,bs,bad,batch);
        batch_projection(a.out,br);});dump("o_proj",br,2304);
        diagnostics.Measure("postnorm_router",l,stream.s,[&]{simt_batch::PostNorm<<<dim3(1,row_limit),128,0,stream.s>>>(bx,br,bn,a.post,bad,batch);
        simt_batch::Router<<<dim3(64,row_limit),32,0,stream.s>>>(a.router,bn,brouter,bad,batch);
        simt_batch::Rank<<<row_limit,64,0,stream.s>>>(brouter,buffer("batch_ids").As<int>(),buffer("batch_routing").As<float>(),bad,batch);});
        auto* bids=buffer("batch_ids").As<int>();auto* weights=buffer("batch_routing").As<float>();
        dump("post_norm",bn,2304);dump("router_ids_i32",bids,8);dump("routing_weights",weights,8);
        auto* groups=buffer("batch_groups").As<int>();auto* counts_per_expert=buffer("batch_counts").As<int>();
        auto* tiles=buffer("batch_tiles").As<simt_batch::ExpertTile>();auto* tile_count=buffer("batch_tile_count").As<int>();
        diagnostics.Measure("group",l,stream.s,[&]{simt_batch::Group<<<64,128,0,stream.s>>>(bids,groups,counts_per_expert,bad,batch);
        simt_batch::Describe<<<1,64,0,stream.s>>>(counts_per_expert,tiles,tile_count,bad,batch);});
        const auto experts=[&](int part,const float* in,float* out){
          diagnostics.Measure(part==0?"gate":part==1?"up":"down",l,stream.s,[&]{
            if(row_limit<=5){
              switch(row_limit){
              case 2:if(o.exl3_rate==3)simt_batch::ExpertsSmall<2,3>(descriptors,l,part,bids,in,out,bt,groups,counts_per_expert,tiles,tile_count,bad,batch,stream.s);else simt_batch::ExpertsSmall<2,4>(descriptors,l,part,bids,in,out,bt,groups,counts_per_expert,tiles,tile_count,bad,batch,stream.s);break;
              case 3:if(o.exl3_rate==3)simt_batch::ExpertsSmall<3,3>(descriptors,l,part,bids,in,out,bt,groups,counts_per_expert,tiles,tile_count,bad,batch,stream.s);else simt_batch::ExpertsSmall<3,4>(descriptors,l,part,bids,in,out,bt,groups,counts_per_expert,tiles,tile_count,bad,batch,stream.s);break;
              case 4:if(o.exl3_rate==3)simt_batch::ExpertsSmall<4,3>(descriptors,l,part,bids,in,out,bt,groups,counts_per_expert,tiles,tile_count,bad,batch,stream.s);else simt_batch::ExpertsSmall<4,4>(descriptors,l,part,bids,in,out,bt,groups,counts_per_expert,tiles,tile_count,bad,batch,stream.s);break;
              case 5:if(o.exl3_rate==3)simt_batch::ExpertsSmall<5,3>(descriptors,l,part,bids,in,out,bt,groups,counts_per_expert,tiles,tile_count,bad,batch,stream.s);else simt_batch::ExpertsSmall<5,4>(descriptors,l,part,bids,in,out,bt,groups,counts_per_expert,tiles,tile_count,bad,batch,stream.s);break;
              }
            }else if(fast_math){
#if PADRA_PASCAL_DP4A_PREFILL
            exl3_pascal_dp4a::Launch(descriptors,l,part,bids,in,out,bt,groups,counts_per_expert,tiles,tile_count,bad,batch,stream.s);
#else
            simt_batch::ExpertsFast(descriptors,l,part,bids,in,out,bt,groups,counts_per_expert,tiles,tile_count,bad,batch,stream.s);
#endif
          }else simt_batch::Experts(descriptors,l,part,bids,in,out,bt,groups,counts_per_expert,tiles,tile_count,bad,batch,stream.s);});
        };
        experts(0,bn,bg);dump("expert_gate",bg,8*896);experts(1,bn,bu);dump("expert_up",bu,8*896);
        diagnostics.Measure("silu",l,stream.s,[&]{simt_batch::Silu<<<row_limit*56,128,0,stream.s>>>(bg,bu,silu,bad,batch);});dump("expert_activation",bg,8*896);
        experts(2,bg,bd);dump("expert_down",bd,8*2304);
        diagnostics.Measure("combine",l,stream.s,[&]{simt_batch::CombineBatch<<<dim3(18,row_limit),128,0,stream.s>>>(bx,bd,weights,bad,batch);});dump("output",bx,2304);
      }
    };
    if(batch)prefill_graph.Capture(stream.s,[&]{prefill(o.capacity);});
    if(bounded_prefill){prefill_1024_graph.Capture(stream.s,[&]{prefill(1024);});prefill_4096_graph.Capture(stream.s,[&]{prefill(4096);});}
    std::array<DecodeGraph,4> draft_graphs,verify_graphs;
    if(draft){
      for(int i=0;i<o.mtp.depth;++i){
        draft_graphs[i].Capture(stream.s,[&,i]{
          draft->Forward(0,0,i?draft->hidden:mtp_hidden,bad,stream.s,{},mtp_state,verifier->inputs.As<int>(),i);
          SelectTiles<<<96,256,0,stream.s>>>(draft->logits,candidates,98304,bad);
          Select<<<1,256,0,stream.s>>>(candidates,control,96,bad);
          mtp::SaveProposal<<<1,1,0,stream.s>>>(control,mtp_state,verifier->inputs.As<int>(),i,bad);
        });
        verify_graphs[i].Capture(stream.s,[&,i]{
          prefill(o.capacity,i+2);
          Norm<<<dim3(1,i+2),128,0,stream.s>>>(bx,verifier->hidden.As<float>(),norm,2304,1);
          head_q8::Launch(i+2,head,verifier->hidden.As<float>(),buffer("head_input").As<unsigned char>(),verifier->logits.As<float>(),bad,stream.s);
          Check(cudaMemcpyAsync(verifier->x.p,bx,(i+2)*2304*4,cudaMemcpyDeviceToDevice,stream.s));
        });
      }
    }
    Check(cudaMemGetInfo(&free,&total));const auto after_graph=ServerAvailableBytes(free,total,o.server_device_budget_gib);
    if(after_graph<plan.reserve)return Status(StatusCode::kResourceExhausted,"SIMT post-graph admission violates 700MiB reserve");
    InferenceResult identity;identity.weights_sha256=model.weights_sha256;identity.weights_bytes=weights_bytes;identity.kv_bytes=buffer("kv").bytes;
    identity.resident_buffer_bytes=plan.resident;identity.device_free_before=free_before;identity.device_free_after_initialization=free;
    identity.kernel_profile="exl3-k4-simt-sm" MELLUM16_ARCHITECTURE "-token-graph";
#if PADRA_CUDA_SM == 61
    identity.kernel_profile+=o.simt_reference_codec?"-decode-reference-codec-v1":"-pascal-codec-lut-warp-h128-v3";
#endif
    if(batch)identity.kernel_profile+=";layer-batch256-staged-kv-fp8-m8-n32-grouped-exl3-m8-n128-v4";
    identity.kernel_profile+=o.simt_reference_attention?";attention-direct-kv-prefetch8-v2":";attention-split-padded-direct-kv-candidate-v2";
    identity.math_profile="mellum16-simt-exl3-k4-v1;FP8-E4M3-activations/FP32-FMA-projections/BF16-output;compensated-BF16-QK/softmax/PV;EXL3-K4-H128-BF16-input/FP16-codec/FP32-FMA/BF16-output;exact-RN-SiLU;compensated-router/head";
    if(fast_math){
      identity.kernel_profile="exl3-k4-simt-sm61-token-graph-pascal-fp32-v3-lane-window;attention-gqa4-parallel-pv-v2";
      if(batch)identity.kernel_profile+=o.capacity<=16384?";prefill-gqa8-fp32-sgemm-v1":";prefill-gqa4-fp32-split-v1";
      if(batch)identity.kernel_profile+=";layer-batch256-staged-kv-sgemm-exl3-m32-n128-r4-sparse-m8";
      identity.math_profile="mellum16-pascal-fp32-v3;FP8-E4M3-activations/FP32-projections(SGEMM-batch/FMA-vector)/BF16-output;FP32-QK-PV/expf-compensated-softmax/BF16-score-probability-output;EXL3-K4-H128-BF16-input/exact-codec/FP32-FMA/BF16-output;exact-RN-SiLU;compensated-router/head";
    }
    if(bounded_prefill)identity.kernel_profile+=";prefill-attention-context-1024-4096-8192";
    identity.execution_profile=batch?"simt-layer-batch256-staged-kv-v1":"simt-token-graph-chunk-commits";identity.prefill_execution=batch?"layer-batch256-graph":"token-graph";identity.decode_execution="token-graph";identity.prefill_chunk=o.prefill_chunk;
    identity.actual_attention_projection="FP8 storage and activations, FP32 SIMT FMA";identity.actual_expert_dispatch=batch?"EXL3 K4 prefill grouped M8/N128; decode selected N128/eight-warps; FP32 FMA":"EXL3 K4 selected N128/eight-warps/FP32 FMA, all prompt rows";
    if(fast_math && batch){identity.actual_attention_projection="FP8 storage and activations; Prefill cuBLAS FP32 SGEMM; Decode FP32 SIMT FMA";identity.actual_expert_dispatch="EXL3 K4 Prefill grouped M32/N128 FP32 tiles, sparse experts M8/N128; Decode selected N128/eight-warps FP32 FMA";}
    if(draft){identity.mtp_depth=o.mtp.depth;identity.mtp_resident_bytes=MtpDraftResidentBytes(false,true)+mtp::kVerifyResidentBytes+buffer("mtp_hidden").bytes+sizeof(DecodeState)+(o.server_kv_checkpoint?kMtpCheckpointBytes:0);identity.kernel_profile+=";mtp-simt-verify-small";identity.decode_execution="mtp-simt-graph";}
    const double load_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-load_begin).count();
#if PADRA_PASCAL_DP4A_PREFILL
    if(fast_math && batch){
      identity.kernel_profile+=";dense-exl3-prefill-dp4a-m32-n128-k64-fused-tail16-v2";
      identity.math_profile="mellum16-pascal-dp4a-prefill-v2;FP8-E4M3-activations/FP32-projections(SGEMM-batch/FMA-vector)/BF16-output;FP32-QK-PV/expf-compensated-softmax/BF16-score-probability-output;EXL3-K4-storage/H128-input/dense-tail16-prefill-INT8-W-scale-3.453125-over-127-and-H128-A-scales/DP4A-INT32/FP32-block-combine/sparse-and-decode-exact-codec-FP32-FMA/BF16-output;exact-RN-SiLU;compensated-router/head";
      identity.actual_expert_dispatch="EXL3 K4 storage; Prefill dense M32/N128/K64 INT8 DP4A with H128 activation scales; sparse Prefill M8/N128 and Decode selected N128 exact FP32";
    }
#endif
    if(o.exl3_rate==3){
      for(auto* label:{&identity.kernel_profile,&identity.math_profile,&identity.actual_expert_dispatch}){
        for(const auto& pair:std::array<std::pair<std::string,std::string>,3>{{{"K4","K3"},{"k4","k3"},{"-r4-","-r3-"}}}){
          std::size_t pos=0;while((pos=label->find(pair.first,pos))!=std::string::npos){label->replace(pos,pair.first.size(),pair.second);pos+=pair.second.size();}
        }
      }
    }
    if(q8_kv){
      identity.kernel_profile+=";q8_0-kv-pascal-sync-dp4a";
      identity.math_profile="mellum16-pascal-q8_0-v1;FP8-E4M3-activations/FP32-projections/BF16-output;rotated-Q8_0-KV/FP16-scales/DP4A-QK/FP32-online-softmax-PV/BF16-output;prefill-BF16-dequantized-rotated-KV/FP32-attention;EXL3-K"+std::to_string(o.exl3_rate)+"-H128-BF16-input/exact-codec/FP32-FMA/BF16-output;exact-RN-SiLU;compensated-router";
    }
    if(draft)identity.kernel_profile+=";draft-exl3-k4";
    if(o.server_kv_checkpoint)identity.kernel_profile+=";prefix-chunk256-bitcopy";
    if(o.q8_head){
      const auto head_label=identity.math_profile.find("compensated-router/head");
      if(head_label!=std::string::npos)identity.math_profile.replace(head_label,23,"compensated-router");
      identity.kernel_profile+=";stored-q8-shared-head-dp4a";identity.math_profile+=";head-Q8-block32/BF16-scales/DP4A-FP32";}
    if(worker){worker->RecordLoad(plan.resident,identity.mtp_resident_bytes);worker->RecordCapture();worker->RecordCapture();if(batch)worker->RecordCapture();if(fast_math || !o.simt_reference_attention){worker->RecordCapture();worker->RecordCapture();}if(bounded_prefill){worker->RecordCapture();worker->RecordCapture();}if(draft)for(int i=0;i<2*o.mtp.depth;++i)worker->RecordCapture();worker->RecordGraphAdmission(after_graph);worker->RecordIdentity(identity.math_profile,identity.kernel_profile,model.weights_sha256,model.manifest_sha256);}
    const auto check_numerics=[&]{int value=0;Check(cudaMemcpyAsync(&value,bad,sizeof(value),cudaMemcpyDeviceToHost,stream.s));Check(cudaStreamSynchronize(stream.s));if(value)throw std::runtime_error("nonfinite/invalid SIMT forward; session poisoned");};
    const auto checkpoint_copy=[&](bool restore){
      const std::size_t plane_bytes=1024ULL*(q8_kv?sizeof(q8_native::CacheRow):512*sizeof(__nv_bfloat16));
      std::size_t offset=0;for(int l=0;l<28;++l)if(l%4!=3){
      auto* saved=buffer("local_checkpoint").As<unsigned char>()+offset;
      Check(cudaMemcpyAsync(restore?static_cast<void*>(layers[l].kc):saved,restore?saved:static_cast<void*>(layers[l].kc),plane_bytes,cudaMemcpyDeviceToDevice,stream.s));
      Check(cudaMemcpyAsync(restore?static_cast<void*>(layers[l].vc):saved+plane_bytes,restore?saved+plane_bytes:static_cast<void*>(layers[l].vc),plane_bytes,cudaMemcpyDeviceToDevice,stream.s));offset+=2*plane_bytes;
    }
      if(draft){
        auto* saved=buffer("mtp_checkpoint").As<unsigned char>();
        Check(cudaMemcpyAsync(restore?draft->kv.p:saved,restore?saved:draft->kv.p,draft->kv.bytes,cudaMemcpyDeviceToDevice,stream.s));
        Check(cudaMemcpyAsync(restore?static_cast<void*>(mtp_hidden):saved+draft->kv.bytes,restore?saved+draft->kv.bytes:static_cast<void*>(mtp_hidden),2304*4,cudaMemcpyDeviceToDevice,stream.s));
      }
      Check(cudaStreamSynchronize(stream.s));};
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
      if(!o.simt_teacher_sequences.empty() && diagnostic_request>static_cast<int>(o.simt_teacher_sequences.size()))throw std::runtime_error("missing SIMT request teacher sequence");
      const auto& teacher_tokens=o.simt_teacher_sequences.empty()?o.simt_teacher_tokens:o.simt_teacher_sequences[diagnostic_request-1];
      if(!teacher_tokens.empty() && teacher_tokens.size()<static_cast<std::size_t>(std::max(0,o.generate-1)))throw std::runtime_error("too few SIMT request teacher tokens");
      InferenceResult result=identity;result.initialization_seconds=load_seconds;
      const auto started=std::chrono::steady_clock::now();auto prefill_finished=started;
      int executed=worker?worker->append_base:0,token=0;
      if(draft && executed==0){Check(cudaMemsetAsync(mtp_hidden,0,2304*4,stream.s));Check(cudaMemsetAsync(draft->kv.p,0,draft->kv.bytes,stream.s));}
      const auto run_forward=[&](int next_token,int pos){
        diagnostics.position=pos;diagnostics.phase="decode";diagnostics.rows=1;
        diagnostics.active=!o.simt_diagnostics.empty() && (o.simt_diagnostic_positions.empty()?pos>=static_cast<int>(o.input_ids.size())-1:std::find(o.simt_diagnostic_positions.begin(),o.simt_diagnostic_positions.end(),pos)!=o.simt_diagnostic_positions.end());
        SetSimtCommand<<<1,1,0,stream.s>>>(command,next_token,pos);
        if(diagnostics.active || o.simt_eager)forward((!fast_math && o.simt_reference_attention) || pos<1024?0:pos<4096?4096:o.capacity);else if((!fast_math && o.simt_reference_attention) || pos<1024)forward_graph.Launch(stream.s);else if(pos<4096)middle_graph.Launch(stream.s);else long_graph.Launch(stream.s);
        diagnostics.Flush(stream.s);
      };
      const auto run_head=[&](int pos){
        diagnostics.position=pos;diagnostics.rows=1;diagnostics.phase="head";
        diagnostics.active=!o.simt_diagnostics.empty() && (o.simt_diagnostic_positions.empty() || std::find(o.simt_diagnostic_positions.begin(),o.simt_diagnostic_positions.end(),pos)!=o.simt_diagnostic_positions.end());
        diagnostics.Measure("head",-1,stream.s,[&]{head_graph.Launch(stream.s);});diagnostics.Flush(stream.s);
      };
      const auto cancelled=[&]{return (o.cancellation_requested && o.cancellation_requested->load()) || (o.diagnostic_cancel_after>=0 && static_cast<int>(result.generated_ids.size())>=o.diagnostic_cancel_after);};
      if(batch){
        while(executed<static_cast<int>(o.input_ids.size()) && !result.cancelled){
          if(cancelled()){result.cancelled=true;break;}
          const int chunk=std::min(256,o.prefill_chunk);
          int rows=std::min(chunk-executed%chunk,static_cast<int>(o.input_ids.size())-executed);
          if(worker && worker->checkpoint_anchor>executed && worker->checkpoint_anchor<executed+rows)rows=worker->checkpoint_anchor-executed;
          diagnostic_batch_base=executed;diagnostic_batch_rows=rows;diagnostics.position=executed+rows-1;diagnostics.phase="prefill";diagnostics.rows=rows;
          diagnostics.active=!o.simt_diagnostics.empty() && std::any_of(o.simt_diagnostic_positions.begin(),o.simt_diagnostic_positions.end(),[&](int pos){return pos>=executed && pos<executed+rows;});
          SetSimtBatch<<<1,1,0,stream.s>>>(batch,executed,rows,o.capacity,bad);
          const int maximum_keys=bounded_prefill?(executed+rows<=1024?1024:executed+rows<=4096?4096:o.capacity):o.capacity;
          if(diagnostics.active || o.simt_eager)prefill(maximum_keys);
          else if(bounded_prefill && maximum_keys==1024)prefill_1024_graph.Launch(stream.s);
          else if(bounded_prefill && maximum_keys==4096)prefill_4096_graph.Launch(stream.s);
          else prefill_graph.Launch(stream.s);
          diagnostics.Flush(stream.s);check_numerics();
          Check(cudaMemcpyAsync(x,bx+(rows-1)*2304,2304*4,cudaMemcpyDeviceToDevice,stream.s));
          if(draft){
            Norm<<<dim3(1,rows),128,0,stream.s>>>(bx,mtp_hidden+2304,norm,2304,1);
            // Prefill projection arenas are dead after target x was saved.
            draft->EncodeRows(buffer("prompt").As<int>(),executed,rows,mtp_hidden,bg,br,bn,bk,bv,bq,bad,stream.s);
            Check(cudaMemcpyAsync(mtp_hidden,mtp_hidden+rows*2304,2304*4,cudaMemcpyDeviceToDevice,stream.s));
            check_numerics();
          }
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
        run_head(static_cast<int>(o.input_ids.size())-1);check_numerics();
        const auto choose=[&](const float* row_logits){
          if(o.sampling.stochastic())gpu_sampling::Launch(row_logits,adjusted,sorted,sample_ids,sorted_ids,cumulative,rng,control,bad,buffer("sampling_workspace").p,sampler_bytes,stream.s,o.sampling,counts);
          else {
            const float* selection_logits=row_logits;
            if(o.sampling.penalties()){gpu_sampling::PrepareLaunch(row_logits,adjusted,sample_ids,counts,98304,bad,o.sampling,stream.s);selection_logits=adjusted;}
            SelectTiles<<<96,256,0,stream.s>>>(selection_logits,candidates,98304,bad);Select<<<1,256,0,stream.s>>>(candidates,control,96,bad);
          }
          if(o.sampling.penalties())gpu_sampling::RecordChoice<<<1,1,0,stream.s>>>(control,counts,98304,bad);
          Control chosen{};Check(cudaMemcpyAsync(&chosen,control,sizeof(chosen),cudaMemcpyDeviceToHost,stream.s));Check(cudaStreamSynchronize(stream.s));
          if(chosen.bad)throw std::runtime_error("invalid SIMT sampling/forward; session poisoned");
          return chosen.token;
        };
        const auto emit=[&](int chosen){
          token=chosen;result.generated_ids.push_back(token);
          if(result.generated_ids.size()==1)prefill_finished=std::chrono::steady_clock::now();
          if(worker && !worker->Emit(token,result.generated_ids.size()==1))result.cancelled=true;
          if(cancelled())result.cancelled=true;
        };
        if(draft){
          if(o.generate>0 && !cancelled())emit(choose(logits));
          int position=static_cast<int>(o.input_ids.size());
          while(static_cast<int>(result.generated_ids.size())<o.generate && token!=28 && !result.cancelled){
            if(cancelled()){result.cancelled=true;break;}
            const int remaining=o.generate-static_cast<int>(result.generated_ids.size());
            const int budget=MtpProposalBudget(o.mtp.depth,remaining,position,o.capacity);
            DecodeState state{token,position,o.capacity,remaining,position,std::min(position,1024),0,0};
            Check(cudaMemcpyAsync(mtp_state,&state,sizeof(state),cudaMemcpyHostToDevice,stream.s));
            const int saved_rows=budget+1;
            mtp::CopyCacheRows<false><<<dim3(saved_rows,29),128,0,stream.s>>>(verifier->bindings.As<mtp::CacheBinding>(),mtp_state,verifier->backup.As<__nv_bfloat16>(),saved_rows,0,bad);
            std::array<int,5> inputs{};inputs[0]=token;
            int proposed=0;
            if(budget){
              const bool profile=!o.simt_diagnostics.empty() && std::find(o.simt_diagnostic_positions.begin(),o.simt_diagnostic_positions.end(),position)!=o.simt_diagnostic_positions.end();
              diagnostics.active=profile;diagnostics.position=position;diagnostics.phase="draft";diagnostics.rows=budget;
              diagnostics.Measure("draft",-1,stream.s,[&]{for(int i=0;i<budget;++i)draft_graphs[i].Launch(stream.s);});diagnostics.Flush(stream.s);
              result.mtp_draft_calls+=budget;
              Check(cudaMemcpyAsync(inputs.data(),verifier->inputs.p,(budget+1)*4,cudaMemcpyDeviceToHost,stream.s));
              check_numerics();
              while(proposed<budget && inputs[proposed+1]!=28)++proposed;
            }
            if(!proposed){
              mtp::CopyCacheRows<true><<<dim3(saved_rows,29),128,0,stream.s>>>(verifier->bindings.As<mtp::CacheBinding>(),mtp_state,verifier->backup.As<__nv_bfloat16>(),saved_rows,1,bad);
              draft->Encode(token,position,mtp_hidden,bad,stream.s);
              run_forward(token,position);run_head(position);
              Check(cudaMemcpyAsync(mtp_hidden,normalized,2304*4,cudaMemcpyDeviceToDevice,stream.s));
              emit(choose(logits));++position;++result.decode_steps;++result.mtp_ordinary_tails;
              continue;
            }
            ++result.mtp_groups;result.mtp_proposed+=proposed;
            Check(cudaMemcpyAsync(buffer("prompt").As<int>()+position,inputs.data(),(proposed+1)*4,cudaMemcpyHostToDevice,stream.s));
            SetSimtBatch<<<1,1,0,stream.s>>>(batch,position,proposed+1,o.capacity,bad);
            const bool profile=!o.simt_diagnostics.empty() && std::find(o.simt_diagnostic_positions.begin(),o.simt_diagnostic_positions.end(),position)!=o.simt_diagnostic_positions.end();
            if(profile){
              diagnostics.active=true;diagnostics.position=position;diagnostics.phase="verify";diagnostics.rows=proposed+1;
              prefill(o.capacity,proposed+1);
              diagnostics.Measure("head",-1,stream.s,[&]{Norm<<<dim3(1,proposed+1),128,0,stream.s>>>(bx,verifier->hidden.As<float>(),norm,2304,1);head_q8::Launch(proposed+1,head,verifier->hidden.As<float>(),buffer("head_input").As<unsigned char>(),verifier->logits.As<float>(),bad,stream.s);});
              Check(cudaMemcpyAsync(verifier->x.p,bx,(proposed+1)*2304*4,cudaMemcpyDeviceToDevice,stream.s));diagnostics.Flush(stream.s);
            }else verify_graphs[proposed-1].Launch(stream.s);
            int retained=0;
            for(int row=0;row<=proposed;++row){
              if(cancelled()){result.cancelled=true;break;}
              emit(choose(verifier->logits.As<float>()+row*98304));++retained;
              const bool accepted=row<proposed && token==inputs[row+1];
              result.mtp_accepted+=accepted;
              if(!accepted || token==28 || result.cancelled)break;
            }
            mtp::CopyCacheRows<true><<<dim3(saved_rows,29),128,0,stream.s>>>(verifier->bindings.As<mtp::CacheBinding>(),mtp_state,verifier->backup.As<__nv_bfloat16>(),saved_rows,retained,bad);
            for(int row=1;row<retained;++row)draft->Encode(inputs[row],position+row,verifier->hidden.As<float>()+(row-1)*2304,bad,stream.s);
            if(retained){
              Check(cudaMemcpyAsync(mtp_hidden,verifier->hidden.As<float>()+(retained-1)*2304,2304*4,cudaMemcpyDeviceToDevice,stream.s));
              Check(cudaMemcpyAsync(normalized,mtp_hidden,2304*4,cudaMemcpyDeviceToDevice,stream.s));
              Check(cudaMemcpyAsync(x,verifier->x.As<float>()+(retained-1)*2304,2304*4,cudaMemcpyDeviceToDevice,stream.s));
              Check(cudaMemcpyAsync(logits,verifier->logits.As<float>()+(retained-1)*98304,98304*4,cudaMemcpyDeviceToDevice,stream.s));
            }
            position+=retained;result.decode_steps+=retained;
          }
          if(cancelled())result.cancelled=true;
          check_numerics();
        }else for(int step=0;step<o.generate && !result.cancelled;++step){
          if(cancelled()){result.cancelled=true;break;}
          if(step){run_forward(teacher_tokens.empty()?token:teacher_tokens[step-1],static_cast<int>(o.input_ids.size())+step-1);run_head(static_cast<int>(o.input_ids.size())+step-1);++result.decode_steps;}
          const int chosen=choose(logits);
          if(!o.simt_diagnostics.empty()){
            diagnostics.position=static_cast<int>(o.input_ids.size())+step-1;diagnostics.active=true;
            diagnostics.Dump("logits",logits,98304*4,stream.s);diagnostics.active=false;
          }
          emit(chosen);
          if(token==28 && teacher_tokens.empty())break;
        }
        if(o.generate==0)prefill_finished=std::chrono::steady_clock::now();
        if(o.capture_hidden){result.hidden.resize(2304);Check(cudaMemcpyAsync(result.hidden.data(),normalized,2304*4,cudaMemcpyDeviceToHost,stream.s));}
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
