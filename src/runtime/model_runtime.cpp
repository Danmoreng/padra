#include "runtime/model_runtime.h"
#include "runtime/native_artifacts.h"
#include "compiler/sha256.h"
#include <fstream>
#include <stdexcept>
namespace mellum16 {
InferenceOptions SelectedServerOptions(const std::filesystem::path &path,
                                       const std::string &tokenizer, int expert_rate) {
  if (expert_rate != 3 && expert_rate != 4 && expert_rate != 6)
    throw std::invalid_argument("selected server supports EXL3 K3, K4 or K6");
  InferenceOptions o;
  o.artifact = path;
  o.tokenizer_sha256 = tokenizer;
  o.capacity = 8192;
  o.prefill_chunk = 2048;
  o.native_prefill = o.fp8_expert_activations = o.native_decode = true;
  o.online_prefill_attention = o.gqa8_prefill_attention = o.packed_bf16_prefill_attention =
      o.padded_bf16_prefill_attention = true;
  o.ordinary_fp8_projections = o.ordinary_fp8_simt = o.compact_experts = o.staged_experts = true;
  o.block_expert_prefill = o.ordinary_prefill_silu = o.gqa8_decode_attention = o.fused_attention_o =
      o.fused_residual_norm = true;
  o.exl3_experts = true;
  o.exl3_rate = expert_rate;
  o.exl3_prefill_fp16 = o.exl3_prefill_masked = o.exl3_prefill_register = o.exl3_prefill_dot =
      o.exl3_prefill_warp = true;
  o.exl3_decode_mma = o.exl3_decode_dot = o.exl3_decode_fused = o.exl3_decode_lane =
      o.exl3_decode_dual = true;
  o.exl3_silu_table = o.exl3_bf16_postnorm = o.exact_fp8_encoding =
      o.attention_fragment_validation = o.exact_router_rank = true;
  o.wide_attention_merge = o.paired_attention_o = o.wide_rms_dataflow = o.exl3_packed_activations =
      true;
  o.quad_router_prefill = o.pair64_qknorm_prefill = o.exact_norm_prefill = o.prefill_graphs =
      o.exl3_wide_input = true;
  o.device_decode = o.decode_graphs = o.full_decode_graph = o.fast_w4a8 = true;
  return o;
}
#if MELLUM16_HAS_CUDA
Result<std::unique_ptr<RuntimeBackend>> LoadCudaModelRuntime(InferenceOptions);
#endif
Result<std::unique_ptr<ModelRuntime>>
ModelRuntime::Load(const std::filesystem::path &path, const std::string &tokenizer,
                   bool controlled_chunks, NativeContextCapacity capacity, KVCacheFormat kv_format,
                   int diagnostic_decode_token, int prefill_chunk, int device_budget_gib,
                   bool enable_checkpoint, MtpOptions mtp, std::optional<bool> q8_head) {
#if MELLUM16_HAS_CUDA
  if (!mtp.artifact.empty() && (mtp.depth < 1 || mtp.depth > 4))
    return Status(StatusCode::kInvalidArgument, "MTP depth must be D1, D2, D3 or D4");
  if (!mtp.artifact.empty() && !controlled_chunks)
    return Status(StatusCode::kUnsupported, "MTP requires controlled execution");
  if (!ValidServerChunk(prefill_chunk) || !ValidServerBudget(device_budget_gib))
    return Status(StatusCode::kInvalidArgument,
                  "server requires explicit chunk2048/512/256 and budget0/8/12GiB");
  if ((prefill_chunk != 2048 || device_budget_gib) && !controlled_chunks)
    return Status(StatusCode::kUnsupported,
                  "low-memory profiles require the selected controlled-chunk runtime");
  if (diagnostic_decode_token < -1 || diagnostic_decode_token >= 98304)
    return Status(StatusCode::kInvalidArgument, "invalid fixed diagnostic decode token");
  if (enable_checkpoint && !controlled_chunks)
    return Status(StatusCode::kUnsupported,
                  "checkpoint capture/restore requires explicit controlled profile");
  auto admitted = AdmitNativeContext(static_cast<int>(capacity));
  if (!admitted.ok())
    return admitted.status();
  if (capacity != NativeContextCapacity::C8192 && !controlled_chunks)
    return Status(StatusCode::kUnsupported,
                  "long context requires the explicit selected K3/K4/K6 controlled-chunk profile");
  if (kv_format != KVCacheFormat::BF16 && kv_format != KVCacheFormat::Q8TokenHead128F32ScaleV1 &&
      kv_format != KVCacheFormat::Q8Token512Block32F16ScaleRotatedV2)
    return Status(StatusCode::kUnsupported, "unsupported KV format");
  if (kv_format != KVCacheFormat::BF16 && !controlled_chunks)
    return Status(StatusCode::kUnsupported,
                  "Q8 requires the explicit selected controlled-chunk runtime");
  const auto manifestPath = path / "manifest.json";
  std::error_code error;
  const auto fileType = std::filesystem::symlink_status(manifestPath, error);
  if (error || !std::filesystem::is_regular_file(fileType) || std::filesystem::is_symlink(fileType))
    return Status(StatusCode::kDataLoss, "selected manifest must be a bounded regular file");
  const auto size = std::filesystem::file_size(manifestPath, error);
  if (error || size > 4 * 1024 * 1024)
    return Status(StatusCode::kDataLoss, "selected manifest must be a bounded regular file");
  std::ifstream source(manifestPath, std::ios::binary);
  if (!source)
    return Status(StatusCode::kIoError, "cannot open selected manifest");
  std::string manifest(static_cast<std::size_t>(size), '\0');
  source.read(manifest.data(), static_cast<std::streamsize>(manifest.size()));
  if (source.gcount() != static_cast<std::streamsize>(size) || source.bad() ||
      source.peek() != std::char_traits<char>::eof())
    return Status(StatusCode::kDataLoss, "selected manifest read failed or changed");
  const auto *identity = FindNativeArtifact(compiler::Sha256Hex(manifest.data(), manifest.size()));
  if (!identity)
    return Status(StatusCode::kDataLoss, "persistent runtime manifest identity mismatch");
  auto profile = SelectedServerOptions(path, tokenizer, identity->expert_rate);
  profile.capacity = static_cast<int>(capacity);
  profile.controlled_chunk_graphs = controlled_chunks;
  profile.kv_cache_format = kv_format;
  profile.diagnostic_decode_token = diagnostic_decode_token;
  profile.prefill_chunk = prefill_chunk;
  profile.server_device_budget_gib = device_budget_gib;
  profile.server_kv_checkpoint = enable_checkpoint;
  profile.mtp = std::move(mtp);
  if (identity->q8_head && q8_head == false)
    return Status(StatusCode::kInvalidArgument, "stored Q8 head cannot be read as BF16");
  profile.q8_head = q8_head.value_or(identity->q8_head);
  auto backend = LoadCudaModelRuntime(std::move(profile));
  if (!backend.ok())
    return backend.status();
  return std::make_unique<ModelRuntime>(std::move(backend).value());
#else
  (void)path;
  (void)tokenizer;
  (void)q8_head;
  (void)controlled_chunks;
  (void)capacity;
  (void)kv_format;
  (void)diagnostic_decode_token;
  (void)prefill_chunk;
  (void)device_budget_gib;
  (void)enable_checkpoint;
  (void)mtp;
  return Status(StatusCode::kUnsupported, "persistent model runtime requires compiled CUDA/SM120");
#endif
}
} // namespace mellum16
