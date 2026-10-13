// Test-only vocabulary/renderer. Production UTF-8 decoding, request validation,
// coordinator, stores and HTTP code are compiled without replacements.
#include "text/tokenizer.h"
#include "util/json.h"
#include <stdexcept>
namespace mellum16::text {
Tokenizer::~Tokenizer() = default;
Result<std::unique_ptr<Tokenizer>> Tokenizer::Load(const std::filesystem::path &,
                                                   const std::string &) {
  auto t = std::unique_ptr<Tokenizer>(new Tokenizer);
  t->raw_.resize(384);
  t->special_.resize(384);
  t->sha256_ = std::string(64, '0');
  t->raw_[0] = "x";
  t->raw_[1] = "STOP";
  t->special_[28] = true;
  t->max_raw_ = 4;
  for (int c = 0; c < 256; ++c)
    t->raw_[128 + c] = std::string(1, static_cast<char>(c));
  return t;
}
Result<std::vector<int>> Tokenizer::Encode(std::string_view text) const {
  std::vector<int> ids;
  for (unsigned char c : text)
    ids.push_back(128 + c);
  return ids;
}
Result<std::string> Tokenizer::Decode(std::span<const int> ids, bool skip) const {
  std::string out;
  for (int id : ids)
    out += RawToken(id, skip);
  return out;
}
std::string_view Tokenizer::RawToken(int id, bool skip) const {
  if (id < 0 || id >= static_cast<int>(raw_.size()))
    throw std::invalid_argument("synthetic token out of range");
  return skip && special_[id] ? std::string_view{} : raw_[id];
}
Result<std::string> RenderChat(std::string_view body) {
  auto parsed = json::Parse(body);
  if (!parsed.ok())
    return parsed.status();
  auto *messages = parsed.value().find("messages");
  if (!messages || !messages->is_array())
    return Status{StatusCode::kInvalidArgument, "synthetic renderer requires messages"};
  auto rendered = json::Stringify(*messages);
  auto *generation = parsed.value().find("add_generation_prompt");
  if (!generation || generation->as_bool())
    rendered += '>';
  return rendered;
}
} // namespace mellum16::text
