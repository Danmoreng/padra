#pragma once
#include <filesystem>
#include <map>
#include <string>
#include <vector>
#include "mellum16/status.h"
#include "platform/mapped_file.h"
namespace mellum16::internal {
inline constexpr auto kHybridProfile="mellum16-nvfp4-fp8-bf16-w4a16-v1";
inline constexpr auto kModelRevision="92ddae9fc7665e9f801d141d2e5a6b2caf2460c4";
inline constexpr auto kExl3Profile="mellum16-exl3-k4-fp8-bf16-v1";
inline constexpr auto kExl3K8Profile="mellum16-exl3-k8-fp8-bf16-v1";
enum class WeightFormat { Bf16, Fp8, Nvfp4, Exl3K4, Exl3K8 };
struct Region { std::uint64_t offset=0, bytes=0; };
struct Matrix {
  WeightFormat format=WeightFormat::Bf16;
  std::vector<std::uint64_t> shape;
  std::map<std::string,Region,std::less<>> regions;
};
struct UploadSlice { std::uint64_t source_offset=0,device_offset=0,bytes=0; bool attention=false; };
struct CompiledModel {
  MappedFile diagnostic_attention;
  std::vector<UploadSlice> uploads;
  std::uint64_t resident_weights_bytes=0;
  std::string diagnostic_attention_sha256;
  bool exl3_experts=false;
  int exl3_rate=0;
  MappedFile weights;
  std::map<std::string,Matrix,std::less<>> tensors;
  std::string tokenizer_sha256;
  std::string weights_sha256;
  std::string manifest_sha256;
};
Status ApplyDiagnosticAttention(CompiledModel& model,const std::filesystem::path& directory);
Result<CompiledModel> LoadCompiledModel(const std::filesystem::path& directory);
}
