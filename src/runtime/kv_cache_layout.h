#pragma once
#include "runtime/size_checked.h"
#include <cstdint>
#include <stdexcept>
namespace mellum16 {
enum class KVCacheFormat { BF16, Q8TokenHead128F32ScaleV1, Q8Token512Block32F16ScaleRotatedV2 };
struct KVCacheLayout {
  KVCacheFormat format = KVCacheFormat::BF16;
  int rows = 0;
  std::size_t k_offset = 0, v_offset = 0, scales_offset = 0, bytes = 0;
  static KVCacheLayout Make(KVCacheFormat format, int rows) {
    if (rows < 1 || rows > 131072) {
      throw std::invalid_argument("unsupported KV row capacity");
    }
    KVCacheLayout result;
    result.format = format;
    result.rows = rows;
    if (format == KVCacheFormat::BF16) {
      result.v_offset = admission::Bytes(rows, 1024);
      result.scales_offset = 0;
      result.bytes = admission::Bytes(rows, 2048);
    } else if (format == KVCacheFormat::Q8TokenHead128F32ScaleV1) {
      result.v_offset = admission::Bytes(rows, 512);
      result.scales_offset = admission::Bytes(rows, 1024);
      result.bytes = admission::Bytes(rows, 1056);
    } else if (format == KVCacheFormat::Q8Token512Block32F16ScaleRotatedV2) {
      result.v_offset = admission::Bytes(rows, 544);
      result.bytes = admission::Bytes(rows, 1088);
    } else {
      throw std::invalid_argument("unknown KV format");
    }
    return result;
  }
  const char *name() const {
    return format == KVCacheFormat::BF16 ? "bf16"
           : format == KVCacheFormat::Q8TokenHead128F32ScaleV1
               ? "q8_token_head128_f32scale_v1"
               : "q8_0_token512_block32_f16scale_rot_k128_v64_v2";
  }
};
struct KVCacheView {
  void *k = nullptr;
  void *v = nullptr;
  void *scales = nullptr;
  KVCacheLayout layout;
};
struct alignas(8) Q8KVScalePair {
  float k, v;
};
static_assert(sizeof(Q8KVScalePair) == 8);
} // namespace mellum16
