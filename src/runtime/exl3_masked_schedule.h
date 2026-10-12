#pragma once
#include <cstddef>
#include <stdexcept>
#include "size_checked.h"
namespace mellum16::exl3_masked_schedule {
struct Tile { int expert, begin, valid_rows; };
#if defined(__CUDACC__)
#define MELLUM_EXL3_HD __host__ __device__
#else
#define MELLUM_EXL3_HD
#endif
// Bound first; multiplication cannot overflow even for hostile int inputs.
MELLUM_EXL3_HD constexpr int Capacity(int tokens) {
  return tokens >= 1 && tokens <= 2048 ? (tokens * 8 + 31) / 32 + 64 : 0;
}
MELLUM_EXL3_HD constexpr int Tiles(int count) { return (count + 31) / 32; }
#undef MELLUM_EXL3_HD
inline std::size_t Bytes(int tokens) {
  const int capacity = Capacity(tokens);
  if (!capacity) throw std::invalid_argument("EXL3 masked schedule tokens must be 1..2048");
  return admission::Sum({admission::Bytes(static_cast<std::size_t>(capacity), sizeof(Tile)), sizeof(int)});
}
}
