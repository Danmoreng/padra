#pragma once
#include "platform/mapped_file.h"
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace mellum16::internal {
inline constexpr char kMtpProfile[] = "mellum21-mtp-bf16-f32-v1";
inline constexpr char kMtpSourceSha256[] =
    "9c7b2c37dcdcb28140118debc81175cfc22bb00a7aa51bcaf0be1768bcfd139b";
inline constexpr char kMtpWeightsSha256[] =
    "b03d5abcf0adf5f5089694668f723fb262f78dc104aeb14dcd00be7db40b8389";
inline constexpr std::uint64_t kMtpWeightsBytes = 857061376;
struct MtpTensor {
  std::vector<std::uint64_t> shape;
  std::uint64_t offset = 0, bytes = 0;
  bool f32 = false, fp8 = false, exl3 = false;
  std::uint64_t scale_offset = 0, suh_offset = 0, svh_offset = 0;
};
struct MtpModel {
  MappedFile weights;
  std::map<std::string, MtpTensor, std::less<>> tensors;
  std::string manifest_sha256;
};
using MtpTensorTable = std::map<std::string, MtpTensor, std::less<>>;
// Validate metadata without opening a payload (also used by host contract tests).
Result<MtpTensorTable> ParseMtpManifest(std::string_view manifest, bool fp8_experts = false,
                                        bool exl3_experts = false);
// Default admits only the pinned full BF16/F32 draft. The separate pinned FP8
// expert profile requires explicit opt-in. Embedding, LM output and
// target output norm are aliases into the separately validated target arena.
Result<MtpModel> LoadMtpModel(const std::filesystem::path &directory, bool fp8_experts = false,
                              bool exl3_experts = false);
} // namespace mellum16::internal
