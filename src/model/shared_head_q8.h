#pragma once
#include "compiled.h"
#include "runtime/head_memory.h"
#include "compiler/sha256.h"
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <stdexcept>
namespace mellum16::internal {
// Convert the validated host head before upload. There is never a BF16 GPU head.
inline void ApplySharedHeadQ8(CompiledModel &model) {
  constexpr std::size_t rows = 98304, cols = 2304, codes = rows * cols, scales = codes / 32 * 2;
  static_assert(codes + scales == kHeadQ8WeightBytes);
  if (!model.uploads.empty() || !model.shared_head.empty() || std::fegetround() != FE_TONEAREST)
    throw std::runtime_error(
        "shared head requires original arena and round-to-nearest environment");
  const auto &head = model.tensors.at("lm_head.weight");
  // A validated v2 checkpoint already owns the only head payload. No temporary
  // BF16 head, repack, or quantization is needed at model load.
  if (head.format == WeightFormat::Q8HeadBlock32) {
    if (model.shared_head_sha256.empty())
      throw std::runtime_error("missing stored head identity");
    return;
  }
  if (head.format != WeightFormat::Bf16 || head.shape != std::vector<std::uint64_t>{rows, cols} ||
      head.regions.size() != 1 || !head.regions.contains("data"))
    throw std::runtime_error("shared head shape/format mismatch");
  const auto source = head.regions.at("data");
  if (source.bytes != codes * 2 || source.offset > model.weights.size() ||
      source.bytes > model.weights.size() - source.offset)
    throw std::runtime_error("shared head source bounds mismatch");
  const auto *raw = model.weights.data() + source.offset;
  std::vector<unsigned char> quantized(codes + scales);
  for (std::size_t at = 0; at < codes; at += 32) {
    float values[32], maximum = 0;
    for (int c = 0; c < 32; ++c) {
      std::uint16_t bits;
      std::memcpy(&bits, raw + (at + c) * 2, 2);
      values[c] = std::bit_cast<float>(std::uint32_t(bits) << 16);
      if (!std::isfinite(values[c]))
        throw std::runtime_error("nonfinite shared head weight");
      maximum = std::max(maximum, std::abs(values[c]));
    }
    const float unrounded = maximum == 0 ? 1.f : maximum / 127.f;
    const auto bits = std::bit_cast<std::uint32_t>(unrounded);
    const auto sbits = std::uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
    const float scale = std::bit_cast<float>(std::uint32_t(sbits) << 16);
    if (!std::isfinite(scale) || scale <= 0)
      throw std::runtime_error("invalid shared head scale");
    std::memcpy(quantized.data() + codes + at / 32 * 2, &sbits, 2);
    for (int c = 0; c < 32; ++c) {
      const auto code =
          static_cast<signed char>(std::clamp(std::nearbyint(values[c] / scale), -127.f, 127.f));
      std::memcpy(quantized.data() + at + c, &code, 1);
    }
  }
  auto table = model.tensors;
  auto &replacement = table.at("lm_head.weight");
  replacement.format = WeightFormat::Q8HeadBlock32;
  replacement.regions = {{"packed", {0, codes}}, {"scales", {codes, scales}}};
  std::vector<UploadSlice> uploads;
  std::uint64_t cursor = 0;
  for (auto &[name, matrix] : table)
    for (auto &[part, region] : matrix.regions) {
      const bool shared = name == "lm_head.weight";
      const auto source_size = shared ? quantized.size() : model.weights.size();
      if (region.offset > source_size || region.bytes > source_size - region.offset)
        throw std::runtime_error("shared head repack source bounds");
      cursor = (cursor + 255) / 256 * 256;
      uploads.push_back({region.offset, cursor, region.bytes, false, shared});
      region.offset = cursor;
      cursor += region.bytes;
    }
  if (cursor >= model.weights.size() || model.weights.size() - cursor < codes * 2 - codes - scales)
    throw std::runtime_error("shared head repack did not remove BF16 GPU storage");
  model.shared_head_sha256 = compiler::Sha256Hex(quantized.data(), quantized.size());
  model.shared_head = std::move(quantized);
  model.tensors = std::move(table);
  model.uploads = std::move(uploads);
  model.resident_weights_bytes = cursor;
}
} // namespace mellum16::internal
