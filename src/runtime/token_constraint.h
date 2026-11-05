#pragma once
#include <cstdint>
#include <span>
namespace mellum16 {
// Request-owned state. Only the inference worker may advance it, once per
// committed target token. Draft proposals must never mutate this state.
class TokenConstraint {
public:
  virtual ~TokenConstraint() = default;
  virtual std::span<const std::uint8_t> Mask() = 0;
  virtual void Accept(int token) = 0;
};
} // namespace mellum16
