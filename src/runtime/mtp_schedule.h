#pragma once
#include "mellum16/status.h"
#include <algorithm>
#include <array>
#include <span>

namespace mellum16 {
inline constexpr int kMaximumMtpDepth = 4;
inline constexpr int kMaximumMtpRows = kMaximumMtpDepth + 1;
struct MtpGroup {
  std::array<int, kMaximumMtpRows> output{};
  int accepted = 0, count = 0;
  bool eos = false;
};
// The pending token is already an emitted output. Each verifier input consumes
// it (or one accepted proposal) and selects the next output. Thus count is also
// the number of target KV rows retained, even when the final output is EOS.
inline Result<MtpGroup> ResolveMtpGroup(std::span<const int> proposed,
                                        std::span<const int> selected, int outputBudget) {
  if (proposed.size() > kMaximumMtpDepth || selected.size() != proposed.size() + 1 ||
      outputBudget < 1)
    return Status(StatusCode::kInvalidArgument, "invalid MTP verifier shape/budget");
  for (const auto row : {proposed, selected})
    for (int token : row)
      if (token < 0 || token >= 98304)
        return Status(StatusCode::kInvalidArgument, "MTP token outside vocabulary");
  MtpGroup result;
  for (std::size_t i = 0; i < selected.size() && result.count < outputBudget; ++i) {
    result.output[result.count++] = selected[i];
    const bool match = i < proposed.size() && selected[i] == proposed[i];
    result.accepted += match;
    result.eos = selected[i] == 28;
    if (!match || result.eos)
      break;
  }
  return result;
}
// Reserve one target correction/bonus row and avoid unused draft work at tails.
inline int MtpProposalBudget(int depth, int remaining, int position, int capacity) {
  if (depth < 1 || depth > kMaximumMtpDepth || remaining < 1 || position < 0 ||
      position >= capacity || remaining > capacity - position)
    return -1;
  return std::min({depth, remaining - 1, capacity - position - 1});
}
} // namespace mellum16
