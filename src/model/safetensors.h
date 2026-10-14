// Derived from Danmoreng/gem16, Apache-2.0; see NOTICE.
// Modified for Mellum16: namespace/includes; selected loader surface.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "mellum16/status.h"

namespace mellum16::internal {

struct StoredTensor {
  std::string name;
  std::vector<std::uint64_t> shape;
  std::string dtype;
  std::uint64_t absolute_offset = 0;
  std::uint64_t length = 0;
  std::uint64_t alignment = 1;
  std::string shard;
};

// Parse one explicitly selected Safetensors-compatible file.  The caller owns
// profile/metadata validation; this function owns bounded header, dtype, shape,
// range and overlap validation.
[[nodiscard]] Result<std::vector<StoredTensor>>
LoadSafetensorsFile(const std::filesystem::path &path);

} // namespace mellum16::internal
