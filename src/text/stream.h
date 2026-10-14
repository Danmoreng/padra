#pragma once
#include <string>
#include <string_view>
#include <vector>
namespace mellum16::text {
class StopMatcher {
public:
  StopMatcher(std::vector<std::string> stops, std::size_t max_chunk_bytes);
  std::string_view Push(std::string_view);
  std::string_view Finish();
  std::size_t BufferedBytes() const {
    return pending_.size();
  }
  bool stopped() const {
    return stopped_;
  }

private:
  std::vector<std::string> stops_;
  std::size_t max_chunk_;
  std::string pending_, output_;
  bool stopped_ = false;
};
struct ThinkingOutput {
  std::string_view content, reasoning;
};
// Mellum opens thinking only with the initial <think> marker. Later literal
// markup in a visible answer is retained. Closing-marker separator LFs are
// hidden. Reasoning payload is preserved (including its own newlines).
class ThinkingParser {
public:
  explicit ThinkingParser(std::size_t max_chunk_bytes);
  ThinkingOutput Push(std::string_view);
  ThinkingOutput Finish();
  std::size_t BufferedBytes() const {
    return pending_.size();
  }
  bool in_reasoning() const {
    return phase_ == Phase::Reasoning;
  }

private:
  enum class Phase { Detect, Reasoning, Answer };
  void Consume();
  void Answer(std::string_view);
  Phase phase_ = Phase::Detect;
  std::size_t max_chunk_;
  std::string pending_, content_, reasoning_;
  bool trim_answer_lf_ = false;
};
} // namespace mellum16::text
