#include "runtime/mtp_schedule.h"
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace mellum16;
namespace {
void Require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
// Independent ordinary execution: advance the proposed input only while it is
// the same token ordinary selection emitted. Stop at EOS or an output boundary.
void Check(const std::vector<int> &draft, const std::vector<int> &chosen, int budget) {
  auto r = ResolveMtpGroup(draft, chosen, budget);
  Require(r.ok(), "rejected valid group");
  std::vector<int> ordinary;
  int accepted = 0;
  for (std::size_t row = 0; row < chosen.size(); ++row) {
    if (int(ordinary.size()) == budget)
      break;
    ordinary.push_back(chosen[row]);
    if (row < draft.size() && chosen[row] == draft[row])
      ++accepted;
    if (chosen[row] == 28 || row == draft.size() || chosen[row] != draft[row])
      break;
  }
  Require(r.value().count == int(ordinary.size()) && r.value().accepted == accepted &&
              std::equal(ordinary.begin(), ordinary.end(), r.value().output.begin()) &&
              r.value().eos == (ordinary.back() == 28),
          "ordinary/group commit differs");
}
} // namespace
int main() {
  try {
    std::mt19937 random(0x4d5450);
    for (int depth = 0; depth <= 4; ++depth)
      for (int mismatch = 0; mismatch <= depth; ++mismatch)
        for (int eos = -1; eos <= depth; ++eos)
          for (int budget = 1; budget <= 5; ++budget) {
            std::vector<int> proposed(depth), selected(depth + 1);
            for (int i = 0; i <= depth; ++i)
              selected[i] = 100 + i;
            std::copy_n(selected.begin(), depth, proposed.begin());
            if (mismatch < depth)
              ++proposed[mismatch];
            if (eos >= 0) {
              selected[eos] = 28;
              if (eos < depth && eos != mismatch)
                proposed[eos] = 28;
            }
            Check(proposed, selected, budget);
          }
    for (int i = 0; i < 10000; ++i) {
      const int depth = random() % 5;
      std::vector<int> proposed(depth), selected(depth + 1);
      for (auto &v : proposed)
        v = random() % 30;
      for (auto &v : selected)
        v = random() % 30;
      Check(proposed, selected, 1 + random() % 5);
    }
    for (int depth = 1; depth <= 4; ++depth)
      for (int position : {0, 1, 1022, 1023, 1024, 8190, 8191})
        for (int remaining = 1; remaining <= std::min(5, 8192 - position); ++remaining) {
          const int proposed = MtpProposalBudget(depth, remaining, position, 8192);
          Require(proposed >= 0 && proposed <= depth && proposed + 1 <= remaining &&
                      position + proposed < 8192,
                  "tail/context admission overflow");
        }
    Require(MtpProposalBudget(5, 2, 0, 8192) == -1 && MtpProposalBudget(1, 2, 8191, 8192) == -1 &&
                !ResolveMtpGroup({}, {}, 1).ok() &&
                !ResolveMtpGroup({}, std::array{98304}, 1).ok() &&
                !ResolveMtpGroup({}, std::array{1}, 0).ok(),
            "invalid group admitted");
    std::cout << "MTP D1-D4 acceptance, EOS and tail contracts passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
