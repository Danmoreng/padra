#pragma once
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
namespace mellum16::admission {
inline std::size_t Bytes(std::size_t count, std::size_t width) {
  if (width && count > std::numeric_limits<std::size_t>::max() / width)
    throw std::overflow_error("native allocation size multiplication overflow");
  return count * width;
}
inline std::size_t Sum(std::initializer_list<std::size_t> parts) {
  std::size_t total = 0;
  for (auto part : parts) {
    if (part > std::numeric_limits<std::size_t>::max() - total) {
      throw std::overflow_error("native allocation size addition overflow");
    }
    total += part;
  }
  return total;
}
} // namespace mellum16::admission
