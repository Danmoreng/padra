#include "parser_properties.h"
#include "server/tool_protocol.h"
#include "text/stream.h"
#include "text/tokenizer.h"
#include "util/json.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace mellum16::server::test {
namespace {
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
bool SameJson(const json::Value &a, const json::Value &b) {
  if (a.is_number() && b.is_number())
    return a.as_number() == b.as_number();
  if (a.is_array() && b.is_array()) {
    if (a.as_array().size() != b.as_array().size())
      return false;
    for (std::size_t i = 0; i < a.as_array().size(); ++i)
      if (!SameJson(a.as_array()[i], b.as_array()[i]))
        return false;
    return true;
  }
  if (a.is_object() && b.is_object()) {
    if (a.as_object().size() != b.as_object().size())
      return false;
    for (const auto &[key, value] : a.as_object()) {
      auto *other = b.find(key);
      if (!other || !SameJson(value, *other))
        return false;
    }
    return true;
  }
  return json::Stringify(a) == json::Stringify(b);
}
struct Outcome {
  std::string stopped, content, reasoning, visible;
  std::array<std::string, 8> names, arguments, raw;
  bool stop = false, tool_error = false, incomplete = false;
  std::size_t calls = 0;
  bool operator==(const Outcome &) const = default;
};
Outcome Run(std::string_view input, std::size_t chunk, std::uint32_t schedule) {
  text::StopMatcher stop({"STOP", "<end>", "🛑"}, 8192);
  text::ThinkingParser thinking(8192);
  ToolStreamParser tools(true, 8192);
  ToolConfig config{{"lookup", "next"}, true, true};
  Outcome result;
  tools.StreamDeltas(config, [&](std::size_t i, std::string_view name, std::string_view args) {
    Check(i < result.names.size(), "unbounded tool index");
    result.names[i] += name;
    result.arguments[i] += args;
    return true;
  });
  for (std::size_t p = 0; p < input.size();) {
    schedule = schedule * 1664525U + 1013904223U;
    const auto n = std::min(input.size() - p, chunk ? chunk : 1 + schedule % 127);
    auto part = input.substr(p, n);
    result.stopped += stop.Push(part);
    auto thought = thinking.Push(part);
    result.content += thought.content;
    result.reasoning += thought.reasoning;
    result.visible += tools.Push(part);
    tools.FlushDeltas();
    Check(stop.BufferedBytes() <= 255, "stop retained bytes exceed longest prefix");
    Check(thinking.BufferedBytes() <= 7, "thinking retained bytes exceed marker prefix");
    Check(tools.BufferedBytes() <= 8 * kMaximumToolJson + 16, "unbounded tool retention");
    p += n;
  }
  result.stopped += stop.Finish();
  auto thought = thinking.Finish();
  result.content += thought.content;
  result.reasoning += thought.reasoning;
  result.visible += tools.Finish();
  tools.FlushDeltas();
  result.stop = stop.stopped();
  result.tool_error = tools.error() != nullptr;
  result.incomplete = tools.incomplete();
  result.calls = tools.count();
  Check(result.calls <= 8, "unbounded tool count");
  if (!result.tool_error) {
    for (std::size_t i = 0; i < result.calls; ++i) {
      result.raw[i] = tools.raw(i);
      Check(result.raw[i].size() <= kMaximumToolJson, "unbounded tool JSON");
      if (!result.incomplete || i + 1 < result.calls) {
        auto parsed = ParseGeneratedTool(result.raw[i], config, "call_property", true);
        // Semantic JSON validation is deliberately later than marker framing.
        if (parsed.ok())
          Check(result.names[i] == parsed.value().name &&
                    result.arguments[i] == parsed.value().arguments,
                "incremental tool arguments changed raw bytes");
      }
    }
  }
  // For malformed streams only rejection is invariant: the failing push can
  // consume different amounts before incremental validation runs.
  if (result.tool_error) {
    result.visible.clear();
    result.names = {};
    result.arguments = {};
    result.raw = {};
    result.calls = 0;
    result.incomplete = false;
  }
  return result;
}
} // namespace
void CheckParserProperties(std::string_view input) {
  Check(input.size() <= 131072, "property input bound exceeded");
  const json::ParseLimits limits{32, 4096, 131072};
  if (auto parsed = json::Parse(input, limits); parsed.ok()) {
    auto encoded = json::Stringify(parsed.value());
    auto again = json::Parse(encoded, limits);
    Check(again.ok() && SameJson(parsed.value(), again.value()), "JSON semantic roundtrip failed");
  }
  const auto reference = Run(input, 8192, 0);
  for (auto chunk : {std::size_t{1}, std::size_t{7}, std::size_t{0}})
    Check(Run(input, chunk, 0x50414452U) == reference, "chunk boundaries changed parser semantics");
  // The byte-token vocabulary exercises every UTF-8 continuation boundary.
  auto loaded = text::Tokenizer::Load({}, {});
  Check(loaded.ok(), "synthetic tokenizer unavailable");
  text::StreamDecoder decoder(*loaded.value());
  std::string decoded;
  for (unsigned char c : input) {
    decoded += decoder.Push(128 + c);
    Check(decoder.BufferedBytes() <= 3, "decoder retained more than three UTF-8 bytes");
  }
  decoded += decoder.Finish();
  text::StreamDecoder grouped(*loaded.value());
  std::string other;
  for (std::size_t p = 0; p < input.size();) {
    bool matched = false;
    for (int id : {2, 3, 4}) {
      auto token = loaded.value()->RawToken(id);
      if (input.substr(p).starts_with(token)) {
        other += grouped.Push(id);
        p += token.size();
        matched = true;
        break;
      }
    }
    if (!matched)
      other += grouped.Push(128 + static_cast<unsigned char>(input[p++]));
    Check(grouped.BufferedBytes() <= 3, "grouped decoder exceeded UTF-8 carry bound");
  }
  other += grouped.Finish();
  Check(other == decoded, "token boundaries changed UTF-8 decoding");
  auto quoted = json::Parse(json::Quote(decoded));
  Check(quoted.ok() && quoted.value().as_string() == decoded, "decoder emitted invalid UTF-8");
}
} // namespace mellum16::server::test
