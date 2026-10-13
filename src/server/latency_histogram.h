#pragma once
#include <array>
#include <cstdint>
#include <ostream>
#include <string_view>
namespace mellum16::server {
// Fixed storage and fixed labels: prompts/request IDs never enter metrics.
struct LatencyHistogram {
  static constexpr std::array<double, 16> bounds{.001, .005, .01, .025, .05, .1, .25, .5,
                                                 1,    2.5,  5,   10,   30,  60, 300, 600};
  std::array<std::uint64_t, bounds.size()> buckets{};
  std::uint64_t count = 0;
  double sum = 0;
  void Observe(double seconds) {
    ++count;
    sum += seconds;
    for (std::size_t i = 0; i < bounds.size(); ++i)
      if (seconds <= bounds[i]) {
        ++buckets[i];
        break;
      }
  }
  void Write(std::ostream &out, std::string_view name) const {
    out << "# TYPE " << name << " histogram\n";
    std::uint64_t cumulative = 0;
    for (std::size_t i = 0; i < bounds.size(); ++i) {
      cumulative += buckets[i];
      out << name << "_bucket{le=\"" << bounds[i] << "\"} " << cumulative << '\n';
    }
    out << name << "_bucket{le=\"+Inf\"} " << count << '\n'
        << name << "_count " << count << '\n'
        << name << "_sum " << sum << '\n';
  }
};
} // namespace mellum16::server
