#pragma once
#include <cstddef>
namespace mellum16 {
inline constexpr std::size_t kHeadBf16WeightBytes = 98304ULL * 2304 * 2;
inline constexpr std::size_t kHeadQ8WeightBytes = 98304ULL * (2304 + 72 * 2);
inline constexpr std::size_t kHeadQ8InputBytes = 5 * (2304 + 72 * 2);
inline constexpr std::size_t kHeadQ8WeightSavingBytes = kHeadBf16WeightBytes - kHeadQ8WeightBytes;
} // namespace mellum16
