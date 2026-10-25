#include "model/mtp.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

using namespace mellum16;
namespace {
void Require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void RejectMutation(const std::string &manifest, const std::string &from, const std::string &to,
                    bool fp8 = false, bool exl3 = false) {
  auto mutated = manifest;
  const auto at = mutated.find(from);
  Require(at != std::string::npos, "test mutation has no source");
  mutated.replace(at, from.size(), to);
  Require(!internal::ParseMtpManifest(mutated, fp8, exl3).ok(), "accepted invalid MTP metadata");
}
struct Scratch {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("padra-mtp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  Scratch() {
    std::filesystem::create_directory(path);
  }
  ~Scratch() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};
} // namespace
int main(int argc, char **argv) {
  try {
    Require(argc >= 2 && argc <= 4, "usage: mtp-model-tests MANIFEST [REAL_ARTIFACT]");
    std::ifstream file(argv[1]);
    Require(file.good(), "cannot read pinned manifest fixture");
    const std::string manifest{std::istreambuf_iterator<char>(file), {}};
    auto parsed = internal::ParseMtpManifest(manifest);
    Require(parsed.ok() && parsed.value().size() == 16, "rejected pinned MTP metadata");
    RejectMutation(manifest, "mellum21-mtp-bf16-f32-v1", "unknown");
    RejectMutation(manifest, "\"schema_version\": 1", "\"schema_version\": 2");
    RejectMutation(manifest, "92ddae9fc7665e9f801d141d2e5a6b2caf2460c4", "wrong-target");
    RejectMutation(manifest, internal::kMtpSourceSha256, std::string(64, '0'));
    RejectMutation(manifest, internal::kMtpWeightsSha256, std::string(64, '0'));
    RejectMutation(manifest, "\"weights.bin\"", "\"../weights.bin\"");
    RejectMutation(manifest, "\"offset\": 0", "\"offset\": -1");
    RejectMutation(manifest, "\"offset\": 0", "\"offset\": 256");
    RejectMutation(manifest, "\"offset\": 2359296", "\"offset\": 0");
    RejectMutation(manifest, "\"bytes\": 2359296", "\"bytes\": 2359295");
    RejectMutation(manifest, "\"BF16\"", "\"F32\"");
    RejectMutation(manifest, "512,", "513,");
    RejectMutation(manifest, "\"model.embed_tokens.weight\"", "\"lm_head.weight\"");
    RejectMutation(manifest, "\"tensors\": {", "\"tensors\": {\"unknown\": {},");
    RejectMutation(manifest, "\"schema_version\": 1", "\"schema_version\": 1.0");
    Require(!internal::ParseMtpManifest(std::string(65537, ' ')).ok(),
            "accepted oversized metadata");
    Require(!internal::ParseMtpManifest("{}").ok(), "accepted missing metadata");
    std::ifstream fp8file(std::filesystem::path(argv[1]).parent_path() / "mtp-fp8.json");
    Require(fp8file.good(), "missing FP8 fixture");
    const std::string fp8{std::istreambuf_iterator<char>(fp8file), {}};
    Require(internal::ParseMtpManifest(fp8, true).ok(), "rejected pinned FP8 metadata");
    Require(!internal::ParseMtpManifest(fp8).ok() &&
                !internal::ParseMtpManifest(manifest, true).ok(),
            "profile crossed opt-in boundary");
    RejectMutation(fp8, "FP8_E4M3FN", "BF16", true);
    RejectMutation(fp8, "ceilbf16", "rnebf16", true);
    RejectMutation(fp8, "\"scale_dtype\": \"BF16\"", "\"scale_dtype\": \"F32\"", true);
    RejectMutation(fp8, "\"scale_bytes\": 294912", "\"scale_bytes\": 294910", true);
    RejectMutation(fp8, "\"scale_offset\": 174598144", "\"scale_offset\": 174598400", true);
    RejectMutation(fp8, "\"codes_sha256\": \"", "\"codes_sha256\": \"0", true);
    RejectMutation(fp8, "\"source_sha256\": \"", "\"source_sha256\": \"0", true);
    RejectMutation(fp8, "\"parent_weights_sha256\": \"", "\"parent_weights_sha256\": \"0", true);
    std::ifstream exl3file(std::filesystem::path(argv[1]).parent_path() / "mtp-exl3.json");
    Require(exl3file.good(), "missing EXL3 fixture");
    const std::string exl3{std::istreambuf_iterator<char>(exl3file), {}};
    Require(internal::ParseMtpManifest(exl3, false, true).ok(), "rejected pinned EXL3 metadata");
    Require(!internal::ParseMtpManifest(exl3).ok() &&
                !internal::ParseMtpManifest(exl3, true).ok() &&
                !internal::ParseMtpManifest(exl3, true, true).ok(),
            "EXL3 crossed opt-in boundary");
    RejectMutation(exl3, "EXL3_K4", "BF16", false, true);
    RejectMutation(exl3, "hessian025", "hessian000", false, true);
    for (const auto field :
         {"packed_offset", "suh_offset", "svh_offset", "packed_bytes", "suh_bytes", "svh_bytes"}) {
      const std::string key = std::string("\"") + field + "\": ";
      auto modified = exl3;
      const auto at = modified.find(key);
      Require(at != std::string::npos, "missing EXL3 descriptor");
      modified.insert(at + key.size(), "9");
      Require(!internal::ParseMtpManifest(modified, false, true).ok(),
              "accepted EXL3 descriptor mutation");
    }
    for (const auto field : {"packed_sha256", "suh_sha256", "svh_sha256", "calibration_sha256"})
      RejectMutation(exl3, std::string("\"") + field + "\": \"",
                     std::string("\"") + field + "\": \"0", false, true);
    Scratch scratch;
    Require(!internal::LoadMtpModel(scratch.path).ok(), "accepted missing artifact");
    {
      std::ofstream out(scratch.path / "manifest.json");
      out << manifest;
    }
    {
      std::ofstream out(scratch.path / "weights.bin");
      out << 'x';
    }
    Require(!internal::LoadMtpModel(scratch.path).ok(), "accepted short payload");
#ifndef _WIN32
    std::filesystem::rename(scratch.path / "manifest.json", scratch.path / "real.json");
    std::filesystem::create_symlink("real.json", scratch.path / "manifest.json");
    Require(!internal::LoadMtpModel(scratch.path).ok(), "accepted symlink manifest");
#endif
    if (argc >= 3) {
      auto model = internal::LoadMtpModel(argv[2]);
      Require(model.ok(), model.status().message().c_str());
      Require(model.value().weights.size() == internal::kMtpWeightsBytes,
              "actual MTP payload size mismatch");
    }
    if (argc == 4) {
      Require(!internal::LoadMtpModel(argv[3]).ok(), "FP8 artifact admitted without opt-in");
      auto quantized = internal::LoadMtpModel(argv[3], true);
      Require(quantized.ok() && quantized.value().weights.size() == 461223936,
              "actual pinned FP8 payload rejected");
    }
    std::cout << "MTP metadata and payload contracts passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
