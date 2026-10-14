#pragma once
#include "mellum16/status.h"
namespace mellum16 {
inline constexpr int kSourceContextMaximum = 131072;
enum class NativeContextCapacity : int {
  C8192 = 8192,
  C16384 = 16384,
  C32768 = 32768,
  C65536 = 65536,
  C131072 = 131072
};
inline Result<NativeContextCapacity> AdmitNativeContext(int value) {
  switch (value) {
  case 8192:
    return NativeContextCapacity::C8192;
  case 16384:
    return NativeContextCapacity::C16384;
  case 32768:
    return NativeContextCapacity::C32768;
  case 65536:
    return NativeContextCapacity::C65536;
  case 131072:
    return NativeContextCapacity::C131072;
  default:
    return Status(
        StatusCode::kUnsupported,
        "native context must be an explicit8192/16384/32768/65536/131072 profile; no fallback");
  }
}
} // namespace mellum16
