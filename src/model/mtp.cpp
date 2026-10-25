#include "model/mtp.h"
#include "model/mtp_fp8_pins.h"
#include "model/mtp_exl3_pins.h"
#include "compiler/sha256.h"
#include "model/compiled.h"
#include "util/json.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#if MELLUM16_HAS_OPENSSL
#include <openssl/evp.h>
#endif

namespace mellum16::internal {
namespace {
void Require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
const json::Value &Field(const json::Value &value, std::string_view name) {
  const auto *field = value.find(name);
  Require(field != nullptr, "missing MTP manifest field");
  return *field;
}
std::string Text(const json::Value &value) {
  Require(value.is_string(), "MTP string required");
  return value.as_string();
}
std::uint64_t Number(const json::Value &value) {
  Require(value.is_integer() && value.as_integer() >= 0, "MTP nonnegative integer required");
  return static_cast<std::uint64_t>(value.as_integer());
}
MappedFile OpenRegular(const std::filesystem::path &path) {
  const auto status = std::filesystem::symlink_status(path);
  Require(std::filesystem::is_regular_file(status) && !std::filesystem::is_symlink(status),
          "MTP artifact must be a regular file, not a symlink");
  auto opened = MappedFile::Open(path);
  if (!opened.ok())
    throw std::runtime_error(opened.status().message());
  return std::move(opened).value();
}
std::string Hash(const MappedFile &file) {
#if MELLUM16_HAS_OPENSSL
  unsigned char bytes[32];
  unsigned length = 0;
  Require(EVP_Digest(file.data(), file.size(), bytes, &length, EVP_sha256(), nullptr) == 1 &&
              length == 32,
          "MTP payload hash failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (auto byte : bytes) {
    result += hex[byte >> 4];
    result += hex[byte & 15];
  }
  return result;
#else
  return compiler::Sha256Hex(file.data(), file.size());
#endif
}
std::map<std::string, MtpTensor, std::less<>> Expected(bool fp8_experts, bool exl3_experts) {
  std::map<std::string, MtpTensor, std::less<>> result;
  const auto add = [&](const char *name, std::vector<std::uint64_t> shape, bool f32) {
    MtpTensor tensor;
    tensor.shape = std::move(shape);
    tensor.f32 = f32;
    tensor.bytes = f32 ? 4 : 2;
    for (const auto extent : tensor.shape)
      tensor.bytes *= extent;
    result.emplace(std::string("blk.28.") + name + ".weight", std::move(tensor));
  };
  for (const auto *name :
       {"attn_norm", "ffn_norm", "nextn.enorm", "nextn.hnorm", "nextn.shared_head_norm"})
    add(name, {2304}, true);
  for (const auto *name : {"attn_q_norm", "attn_k_norm"})
    add(name, {128}, true);
  add("attn_q", {4096, 2304}, false);
  add("attn_k", {512, 2304}, false);
  add("attn_v", {512, 2304}, false);
  add("attn_output", {2304, 4096}, false);
  add("ffn_gate_inp", {64, 2304}, true);
  add("ffn_gate_exps", {64, 896, 2304}, false);
  add("ffn_up_exps", {64, 896, 2304}, false);
  add("ffn_down_exps", {64, 2304, 896}, false);
  add("nextn.eh_proj", {2304, 4608}, false);
  if (fp8_experts)
    for (const auto &pin : kMtpFp8TensorPins) {
      auto &tensor = result.at(pin.name);
      tensor.fp8 = true;
      tensor.bytes = tensor.shape[0] * tensor.shape[1] * (tensor.shape[2] + 2);
    }
  if (exl3_experts)
    for (const auto &pin : kMtpExl3TensorPins) {
      auto &tensor = result.at(pin.name);
      tensor.exl3 = true;
      const auto rows = tensor.shape[1], cols = tensor.shape[2];
      tensor.bytes = 64 * (rows * cols / 2 + (rows + cols) * 2);
    }
  return result;
}
} // namespace

Result<MtpTensorTable> ParseMtpManifest(std::string_view manifest, bool fp8_experts,
                                        bool exl3_experts) {
  try {
    Require(!(fp8_experts && exl3_experts), "draft expert formats are mutually exclusive");
    Require(!manifest.empty() && manifest.size() <= 64 * 1024, "MTP manifest size unsupported");
    auto parsed =
        json::Parse(manifest, {.max_depth = 8, .max_values = 1024, .max_string_bytes = 64 * 1024});
    if (!parsed.ok())
      return parsed.status();
    const auto &value = parsed.value();
    Require(Number(Field(value, "schema_version")) == 1 &&
                Text(Field(value, "profile")) == (exl3_experts  ? kMtpExl3Profile
                                                  : fp8_experts ? kMtpFp8Profile
                                                                : kMtpProfile) &&
                Text(Field(value, "model_revision")) == kModelRevision &&
                Text(Field(value, "source_revision")) ==
                    "4159d684ffdb09ba89dc2a58feb3e75c54eef036" &&
                Text(Field(value, "source_sha256")) == kMtpSourceSha256 &&
                Number(Field(value, "source_bytes")) == 1766609280 &&
                Text(Field(value, "tokenizer_sha256")) ==
                    "58548a346eb073e5132bf7d8ad17dc6971bca36ade378ca4d2bfbc49bf60da2a",
            "MTP model/source/tokenizer identity mismatch");
    const auto weightsBytes = exl3_experts  ? kMtpExl3WeightsBytes
                              : fp8_experts ? kMtpFp8WeightsBytes
                                            : kMtpWeightsBytes;
    const auto *weightsHash = exl3_experts  ? kMtpExl3WeightsSha256
                              : fp8_experts ? kMtpFp8WeightsSha256
                                            : kMtpWeightsSha256;
    if (fp8_experts)
      Require(Text(Field(value, "conversion")) == kMtpFp8Conversion &&
                  Text(Field(value, "parent_weights_sha256")) == kMtpWeightsSha256,
              "MTP FP8 conversion/parent identity mismatch");
    if (exl3_experts)
      Require(Text(Field(value, "conversion")) == kMtpExl3Conversion &&
                  Text(Field(value, "calibration_sha256")) == kMtpExl3CalibrationSha256 &&
                  Text(Field(value, "parent_weights_sha256")) == kMtpWeightsSha256,
              "MTP EXL3 conversion/calibration/parent identity mismatch");
    Require(Text(Field(value, "weights_file")) == "weights.bin" &&
                Number(Field(value, "weights_bytes")) == weightsBytes &&
                Text(Field(value, "weights_sha256")) == weightsHash,
            "MTP payload identity mismatch");
    const auto &aliases = Field(value, "shared_target");
    Require(aliases.is_object() && aliases.as_object().size() == 3 &&
                Text(Field(aliases, "output.weight")) == "lm_head.weight" &&
                Text(Field(aliases, "output_norm.weight")) == "model.norm.weight" &&
                Text(Field(aliases, "token_embd.weight")) == "model.embed_tokens.weight",
            "MTP shared-target bindings mismatch");
    auto expected = Expected(fp8_experts, exl3_experts);
    const auto &table = Field(value, "tensors");
    Require(table.is_object() && table.as_object().size() == expected.size(),
            "MTP tensor inventory mismatch");
    std::uint64_t cursor = 0;
    for (auto &[name, tensor] : expected) {
      const auto &entry = Field(table, name);
      const auto &shape = Field(entry, "shape");
      Require(shape.is_array() && shape.as_array().size() == tensor.shape.size(),
              "MTP tensor rank mismatch");
      for (std::size_t i = 0; i < tensor.shape.size(); ++i)
        Require(Number(shape.as_array()[i]) == tensor.shape[i], "MTP tensor shape mismatch");
      tensor.offset = Number(Field(entry, "offset"));
      Require(Text(Field(entry, "dtype")) == (tensor.exl3  ? "EXL3_K4"
                                              : tensor.fp8 ? "FP8_E4M3FN"
                                              : tensor.f32 ? "F32"
                                                           : "BF16") &&
                  Number(Field(entry, "bytes")) == tensor.bytes &&
                  tensor.offset == (cursor + 255) / 256 * 256 && tensor.offset <= weightsBytes &&
                  tensor.bytes <= weightsBytes - tensor.offset,
              "MTP tensor dtype/range/order mismatch");
      if (tensor.fp8) {
        const auto scaleBytes = tensor.shape[0] * tensor.shape[1] * 2;
        tensor.scale_offset = tensor.offset + tensor.bytes - scaleBytes;
        Require(Text(Field(entry, "scale_dtype")) == "BF16" &&
                    Number(Field(entry, "scale_offset")) == tensor.scale_offset &&
                    Number(Field(entry, "scale_bytes")) == scaleBytes &&
                    tensor.scale_offset % 256 == 0,
                "MTP FP8 scale descriptor mismatch");
        for (const auto &pin : kMtpFp8TensorPins)
          if (name == pin.name)
            Require(Text(Field(entry, "source_sha256")) == pin.source &&
                        Text(Field(entry, "codes_sha256")) == pin.codes &&
                        Text(Field(entry, "scales_sha256")) == pin.scales,
                    "MTP FP8 tensor identities mismatch");
      }
      if (tensor.exl3) {
        const auto packedBytes = 64 * tensor.shape[1] * tensor.shape[2] / 2;
        const auto suhBytes = 64 * tensor.shape[2] * 2, svhBytes = 64 * tensor.shape[1] * 2;
        tensor.suh_offset = tensor.offset + packedBytes;
        tensor.svh_offset = tensor.suh_offset + suhBytes;
        Require(Number(Field(entry, "packed_offset")) == tensor.offset &&
                    Number(Field(entry, "packed_bytes")) == packedBytes &&
                    Number(Field(entry, "suh_offset")) == tensor.suh_offset &&
                    Number(Field(entry, "suh_bytes")) == suhBytes &&
                    Number(Field(entry, "svh_offset")) == tensor.svh_offset &&
                    Number(Field(entry, "svh_bytes")) == svhBytes && tensor.suh_offset % 256 == 0 &&
                    tensor.svh_offset % 256 == 0,
                "MTP EXL3 packed/F16-sidecar descriptor mismatch");
        for (const auto &pin : kMtpExl3TensorPins)
          if (name == pin.name)
            Require(Text(Field(entry, "source_sha256")) == pin.source &&
                        Text(Field(entry, "packed_sha256")) == pin.packed &&
                        Text(Field(entry, "suh_sha256")) == pin.suh &&
                        Text(Field(entry, "svh_sha256")) == pin.svh,
                    "MTP EXL3 tensor identities mismatch");
      }
      cursor = tensor.offset + tensor.bytes;
    }
    Require(cursor == weightsBytes, "MTP payload has unaccounted bytes");
    return expected;
  } catch (const std::exception &error) {
    return Status(StatusCode::kDataLoss, error.what());
  }
}

Result<MtpModel> LoadMtpModel(const std::filesystem::path &directory, bool fp8_experts,
                              bool exl3_experts) {
  try {
    auto manifest = OpenRegular(directory / "manifest.json");
    auto parsed = ParseMtpManifest({reinterpret_cast<const char *>(manifest.data()),
                                    static_cast<std::size_t>(manifest.size())},
                                   fp8_experts, exl3_experts);
    if (!parsed.ok())
      return parsed.status();
    MtpModel result;
    result.weights = OpenRegular(directory / "weights.bin");
    Require(result.weights.size() == (exl3_experts  ? kMtpExl3WeightsBytes
                                      : fp8_experts ? kMtpFp8WeightsBytes
                                                    : kMtpWeightsBytes) &&
                Hash(result.weights) == (exl3_experts  ? kMtpExl3WeightsSha256
                                         : fp8_experts ? kMtpFp8WeightsSha256
                                                       : kMtpWeightsSha256),
            "MTP payload size/SHA-256 mismatch");
    result.tensors = std::move(parsed).value();
    result.manifest_sha256 = compiler::Sha256Hex(manifest.data(), manifest.size());
    return result;
  } catch (const std::exception &error) {
    return Status(StatusCode::kDataLoss, error.what());
  }
}
} // namespace mellum16::internal
