#include "runtime/model_runtime.h"
namespace mellum16 {
InferenceOptions SelectedServerOptions(const std::filesystem::path& path,const std::string& tokenizer){
  InferenceOptions o;o.artifact=path;o.tokenizer_sha256=tokenizer;o.capacity=8192;o.prefill_chunk=2048;
  o.native_prefill=o.fp8_expert_activations=o.native_decode=true;
  o.online_prefill_attention=o.gqa8_prefill_attention=o.packed_bf16_prefill_attention=o.padded_bf16_prefill_attention=true;
  o.ordinary_fp8_projections=o.ordinary_fp8_simt=o.compact_experts=o.staged_experts=true;
  o.block_expert_prefill=o.ordinary_prefill_silu=o.gqa8_decode_attention=o.fused_attention_o=o.fused_residual_norm=true;
  o.exl3_experts=true;o.exl3_rate=4;o.exl3_prefill_fp16=o.exl3_prefill_masked=o.exl3_prefill_register=o.exl3_prefill_dot=o.exl3_prefill_warp=true;
  o.exl3_decode_mma=o.exl3_decode_dot=o.exl3_decode_fused=o.exl3_decode_lane=o.exl3_decode_dual=true;
  o.exl3_silu_table=o.exl3_bf16_postnorm=o.exact_fp8_encoding=o.attention_fragment_validation=o.exact_router_rank=true;
  o.wide_attention_merge=o.paired_attention_o=o.wide_rms_dataflow=o.exl3_packed_activations=true;
  o.quad_router_prefill=o.pair64_qknorm_prefill=o.exact_norm_prefill=o.prefill_graphs=o.exl3_wide_input=true;
  o.device_decode=o.decode_graphs=o.full_decode_graph=o.fast_w4a8=true;return o;
}
#if MELLUM16_HAS_CUDA
Result<std::unique_ptr<RuntimeBackend>> LoadCudaModelRuntime(InferenceOptions);
#endif
Result<std::unique_ptr<ModelRuntime>> ModelRuntime::Load(const std::filesystem::path& path,const std::string& tokenizer,bool controlled_chunks,NativeContextCapacity capacity,KVCacheFormat kv_format,int diagnostic_decode_token,int prefill_chunk,int device_budget_gib,bool enable_checkpoint){
#if MELLUM16_HAS_CUDA
  if(!ValidServerChunk(prefill_chunk) || !ValidServerBudget(device_budget_gib))return Status(StatusCode::kInvalidArgument,"server requires explicit chunk2048/512/256 and budget0/8/12GiB");
  if((prefill_chunk!=2048 || device_budget_gib) && (!controlled_chunks || kv_format!=KVCacheFormat::BF16))return Status(StatusCode::kUnsupported,"low-memory profiles require selected controlled BF16; Q8 quality stop retained");
  if(diagnostic_decode_token< -1 || diagnostic_decode_token>=98304)return Status(StatusCode::kInvalidArgument,"invalid fixed diagnostic decode token");
  if(enable_checkpoint && (!controlled_chunks || kv_format!=KVCacheFormat::BF16))return Status(StatusCode::kUnsupported,"checkpoint capture/restore requires explicit controlled BF16 profile");
  auto admitted=AdmitNativeContext(static_cast<int>(capacity));if(!admitted.ok())return admitted.status();
  if(capacity!=NativeContextCapacity::C8192 && !controlled_chunks)return Status(StatusCode::kUnsupported,"long context requires the explicit selected K4 controlled-chunk BF16 profile");
  if(kv_format!=KVCacheFormat::BF16 && kv_format!=KVCacheFormat::Q8TokenHead128F32ScaleV1)return Status(StatusCode::kUnsupported,"unsupported KV format");
  if(kv_format==KVCacheFormat::Q8TokenHead128F32ScaleV1 && (!controlled_chunks || capacity!=NativeContextCapacity::C8192))return Status(StatusCode::kUnsupported,"Q8 integrated research currently requires controlled8192; long qualification pending");
  auto profile=SelectedServerOptions(path,tokenizer);profile.capacity=static_cast<int>(capacity);profile.controlled_chunk_graphs=controlled_chunks;profile.kv_cache_format=kv_format;profile.diagnostic_decode_token=diagnostic_decode_token;profile.prefill_chunk=prefill_chunk;profile.server_device_budget_gib=device_budget_gib;profile.server_kv_checkpoint=enable_checkpoint;
  auto backend=LoadCudaModelRuntime(std::move(profile));if(!backend.ok())return backend.status();
  return std::make_unique<ModelRuntime>(std::move(backend).value());
#else
  (void)path;(void)tokenizer;(void)controlled_chunks;(void)capacity;(void)kv_format;(void)diagnostic_decode_token;(void)prefill_chunk;(void)device_budget_gib;(void)enable_checkpoint;return Status(StatusCode::kUnsupported,"persistent model runtime requires compiled CUDA/SM120");
#endif
}
}
