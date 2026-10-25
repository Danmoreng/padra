#pragma once
#include "size_checked.h"
namespace mellum16 {
inline constexpr std::size_t kMtpFp8ExpertWeightSavingBytes = 395837440;
inline constexpr std::size_t kMtpDraftWorkspaceBytes =
    (4608 + 2304 * 4 + 4096 * 2 + 512 * 3 + 64 + 8 + 8 * 896 * 2 + 8 * 2304 + 98304 + 32 * 1024) *
    4;
inline constexpr std::size_t kMtpCheckpointBytes = 2 * 1024 * 512 * 2 + 2304 * 4;
inline constexpr std::size_t kMtpDraftResidentBytes =
    857061376 + 192 * 56 + 2 * 1024 * 512 * 2 + kMtpDraftWorkspaceBytes + 8 * 4;
inline constexpr std::size_t kMtpExl3ExpertWeightSavingBytes = 593313792;
inline constexpr std::size_t MtpDraftResidentBytes(bool fp8_experts, bool exl3_experts = false) {
  if (fp8_experts && exl3_experts)
    throw std::invalid_argument("draft expert formats are mutually exclusive");
  return kMtpDraftResidentBytes - (exl3_experts  ? kMtpExl3ExpertWeightSavingBytes
                                   : fp8_experts ? kMtpFp8ExpertWeightSavingBytes
                                                 : 0);
}
inline constexpr std::size_t kMtpVerifyResidentBytes =
    5 * (2304 * 2 + 98304) * 4 + 5 * (32 + 4) + 29 * 40 + 29 * 5 * 1024 * 2 +
    5 * (2304 + 8 * 896 * 2 + 8 * 2304) * 4 + 5 * 8 * 8 + 64 * (5 * 4 + 4) +
    5 * (4096 + 512 * 2) * 4 + 5 * (4096 + 4) + 65 * 4;
inline std::size_t MtpHiddenBytes(int chunk) {
  if (chunk < 2 || chunk > 2048)
    throw std::invalid_argument("invalid MTP prefill shape");
  return admission::Bytes(chunk + 1, 2304 * 4);
}
} // namespace mellum16
