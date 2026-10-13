#include "server/tool_protocol.h"
#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>
namespace mellum16::server {
namespace {
Status Invalid(const char *s) {
  return {StatusCode::kInvalidArgument, s};
}
bool Space(char c) {
  return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}
void Skip(std::string_view s, std::size_t &p) {
  while (p < s.size() && Space(s[p]))
    ++p;
}
std::size_t StringEnd(std::string_view s, std::size_t p) {
  bool escape = false;
  for (++p; p < s.size(); ++p) {
    if (escape) {
      escape = false;
      continue;
    }
    if (s[p] == '\\') {
      escape = true;
      continue;
    }
    if (s[p] == '"')
      return p + 1;
  }
  return s.size() + 1;
}
std::size_t ValueEnd(std::string_view s, std::size_t p) {
  if (p >= s.size())
    return s.size();
  if (s[p] == '"')
    return std::min(StringEnd(s, p), s.size());
  if (s[p] == '{' || s[p] == '[') {
    int depth = 0;
    bool text = false, escape = false;
    for (; p < s.size(); ++p) {
      const char c = s[p];
      if (text) {
        if (escape)
          escape = false;
        else if (c == '\\')
          escape = true;
        else if (c == '"')
          text = false;
        continue;
      }
      if (c == '"')
        text = true;
      else if (c == '{' || c == '[')
        ++depth;
      else if (c == '}' || c == ']') {
        if (--depth == 0)
          return p + 1;
      }
    }
    return s.size();
  }
  for (; p < s.size() && s[p] != ',' && s[p] != '}' && !Space(s[p]); ++p) {
  }
  return p;
}
bool ValidId(std::string_view id) {
  return !id.empty() && id.size() <= 128 && std::all_of(id.begin(), id.end(), [](unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' ||
           c == '-';
  });
}
Status Args(const json::Value *args) {
  if (!args || !args->is_string() || args->as_string().size() > kMaximumToolJson)
    return Invalid("function arguments must be bounded JSON-object string");
  auto v = json::Parse(args->as_string());
  if (!v.ok() || !v.value().is_object())
    return Invalid("function arguments must contain a valid JSON object");
  return {};
}
} // namespace
bool ValidFunctionName(std::string_view name) {
  return ValidId(name) && name.size() <= 64;
}
Result<JsonFieldSpan> FindJsonField(std::string_view s, std::string_view wanted, bool partial) {
  std::size_t p = 0;
  Skip(s, p);
  if (p == s.size() || s[p++] != '{')
    return Invalid("raw JSON field requires object");
  while (p < s.size()) {
    Skip(s, p);
    if (p == s.size() || s[p] == '}')
      break;
    if (s[p] != '"')
      return Invalid("invalid raw object key");
    const auto begin = p, end = StringEnd(s, p);
    if (end > s.size())
      return Invalid("incomplete raw object key");
    auto key = json::Parse(s.substr(begin, end - begin));
    if (!key.ok() || !key.value().is_string())
      return Invalid("invalid raw object key encoding");
    p = end;
    Skip(s, p);
    if (p == s.size() || s[p++] != ':')
      return Invalid("missing raw field separator");
    Skip(s, p);
    const auto value = p, finish = ValueEnd(s, p);
    if (key.value().as_string() == wanted) {
      auto following = finish;
      Skip(s, following);
      return JsonFieldSpan{begin, value, finish, following};
    }
    p = finish;
    Skip(s, p);
    if (p == s.size())
      break;
    if (s[p++] == ',')
      continue;
    break;
  }
  return Status(partial ? StatusCode::kUnsupported : StatusCode::kInvalidArgument,
                "JSON field absent or incomplete");
}
Result<std::string> OmitJsonField(std::string_view s, std::string_view name) {
  auto field = FindJsonField(s, name);
  if (!field.ok())
    return field.status();
  const auto f = field.value();
  auto begin = f.key_begin, end = f.following;
  if (end < s.size() && s[end] == ',')
    ++end;
  else {
    auto prior = begin;
    while (prior && Space(s[prior - 1]))
      --prior;
    if (prior && s[prior - 1] == ',')
      begin = prior - 1;
  }
  std::string result(s.substr(0, begin));
  result += s.substr(end);
  auto valid = json::Parse(result);
  if (!valid.ok())
    return valid.status();
  return result;
}
Result<ToolConfig> ValidateTools(const json::Value &root) {
  ToolConfig config;
  const auto *tools = root.find("tools");
  std::set<std::string> seen;
  if (tools && !tools->is_null()) {
    if (!tools->is_array() || tools->as_array().size() > 16)
      return Invalid("tools must be an array of at most16functions");
    for (const auto &t : tools->as_array()) {
      if (!t.is_object() || t.as_object().size() != 2 || !t.find("type") ||
          !t.find("type")->is_string() || t.find("type")->as_string() != "function" ||
          !t.find("function") || !t.find("function")->is_object())
        return Invalid("only function tools are implemented");
      const auto &f = *t.find("function");
      for (const auto &[k, v] : f.as_object()) {
        (void)v;
        if (k != "name" && k != "description" && k != "parameters" && k != "strict")
          return Invalid("unsupported function definition field");
      }
      auto *name = f.find("name");
      if (!name || !name->is_string() || !ValidFunctionName(name->as_string()) ||
          !seen.insert(name->as_string()).second)
        return Invalid("invalid/duplicate function name");
      if (auto *d = f.find("description");
          d && !d->is_null() && (!d->is_string() || d->as_string().size() > 8192))
        return Invalid("invalid function description");
      if (auto *p = f.find("parameters"); p && !p->is_null()) {
        if (!p->is_object())
          return Invalid("function parameters must be schema object");
        if (auto *type = p->find("type");
            type && (!type->is_string() || type->as_string() != "object"))
          return Invalid("function arguments require object schema");
      }
      if (auto *strict = f.find("strict"); strict && !strict->is_null()) {
        if (!strict->is_bool())
          return Invalid("strict must be boolean");
        if (strict->as_bool())
          return Status(StatusCode::kUnsupported,
                        "strict JSON-schema constrained decoding is not implemented");
      }
      config.names.push_back(name->as_string());
    }
  }
  if (auto *parallel = root.find("parallel_tool_calls"); parallel && !parallel->is_null()) {
    if (!parallel->is_bool())
      return Invalid("parallel_tool_calls must be boolean");
    config.parallel = parallel->as_bool();
  }
  std::string choice = config.names.empty() ? "none" : "auto";
  if (auto *c = root.find("tool_choice"); c && !c->is_null()) {
    if (!c->is_string())
      return Status(StatusCode::kUnsupported, "forced named tool choice is not implemented");
    choice = c->as_string();
    if (choice != "auto" && choice != "none")
      return Status(
          StatusCode::kUnsupported,
          "required/forced tool choice has no constrained enforcement and is unsupported");
  }
  if (choice == "auto" && config.names.empty())
    return Invalid("tool_choice=auto requires at least one function");
  config.parse = choice == "auto";
  return config;
}
Status ValidateToolHistory(const json::Value::Array &messages) {
  std::set<std::string> seen, pending;
  for (const auto &m : messages) {
    if (!m.is_object())
      return Invalid("message must be object");
    for (const auto &[key, v] : m.as_object()) {
      (void)v;
      if (key != "role" && key != "content" && key != "reasoning_content" && key != "tool_calls" &&
          key != "tool_call_id")
        return Invalid("unsupported message field");
    }
    auto *role = m.find("role");
    if (!role || !role->is_string())
      return Invalid("message role required");
    const auto &r = role->as_string();
    if (r != "system" && r != "user" && r != "assistant" && r != "tool")
      return Status(StatusCode::kUnsupported, "unsupported text role");
    if (r != "tool" && !pending.empty())
      return Invalid("tool responses must resolve all preceding calls before next message");
    auto *calls = m.find("tool_calls");
    const bool has = calls && !calls->is_null();
    if (has && (r != "assistant" || !calls->is_array() || calls->as_array().empty() ||
                calls->as_array().size() > 8))
      return Invalid("tool_calls requires assistant array1..8");
    auto *content = m.find("content");
    if (!content || (!content->is_string() && !(r == "assistant" && has && content->is_null())))
      return Status(StatusCode::kUnsupported,
                    "message content requires text, or null for assistant tool calls");
    if (auto *reason = m.find("reasoning_content");
        reason && (!reason->is_string() || r != "assistant"))
      return Invalid("reasoning_content requires assistant string");
    if (r == "tool") {
      auto *id = m.find("tool_call_id");
      if (!id || !id->is_string() || !pending.erase(id->as_string()))
        return Invalid("unknown/duplicate tool response call ID");
    } else if (m.find("tool_call_id"))
      return Invalid("tool_call_id requires tool role");
    if (has)
      for (const auto &c : calls->as_array()) {
        if (!c.is_object() || c.as_object().size() != 3)
          return Invalid("tool call requires id/type/function");
        auto *id = c.find("id");
        auto *type = c.find("type");
        auto *f = c.find("function");
        if (!id || !id->is_string() || !ValidId(id->as_string()) ||
            !seen.insert(id->as_string()).second || !type || !type->is_string() ||
            type->as_string() != "function" || !f || !f->is_object() ||
            f->as_object().size() != 2 || !f->find("name") || !f->find("name")->is_string() ||
            !ValidFunctionName(f->find("name")->as_string()))
          return Invalid("invalid function call or duplicate ID");
        auto status = Args(f->find("arguments"));
        if (!status.ok())
          return status;
        pending.insert(id->as_string());
      }
  }
  if (!pending.empty())
    return Invalid("generation requires the outstanding tool responses");
  return {};
}
ToolStreamParser::ToolStreamParser(bool enabled, std::size_t maximum, std::size_t calls)
    : enabled_(enabled), maximum_calls_(calls) {
  if (maximum > 8192 || calls < 1 || calls > 8)
    throw std::invalid_argument("invalid tool stream bound");
  visible_.reserve(maximum + 32);
  if (enabled)
    for (auto &raw : raw_)
      raw.reserve(kMaximumToolJson);
}
void ToolStreamParser::Emit(char c) {
  if (visible_.size() == visible_.capacity()) {
    Fail("tool visible fragment exceeded");
    return;
  }
  visible_.push_back(c);
}
void ToolStreamParser::Plain(char c) {
  constexpr std::string_view open = "<tool_call>";
  held_[held_size_++] = c;
  while (held_size_ && !open.starts_with(std::string_view(held_.data(), held_size_))) {
    Emit(held_[0]);
    std::memmove(held_.data(), held_.data() + 1, --held_size_);
  }
  if (held_size_ == open.size()) {
    held_size_ = 0;
    if (count_ == maximum_calls_) {
      Fail("too many/parallel-disallowed model tool calls");
      return;
    }
    inside_ = true;
    started_ = ended_ = in_string_ = escaped_ = false;
    depth_ = 0;
  }
}
std::string_view ToolStreamParser::Push(std::string_view input) {
  if (!enabled_)
    return input;
  visible_.clear();
  if (error_)
    return {};
  constexpr std::string_view close = "</tool_call>";
  for (char c : input) {
    if (!inside_) {
      Plain(c);
      if (error_)
        break;
      continue;
    }
    if (ended_) {
      if (!held_size_ && Space(c))
        continue;
      if (held_size_ >= close.size() || c != close[held_size_]) {
        Fail("invalid tool call closing marker");
        break;
      }
      held_[held_size_++] = c;
      if (held_size_ == close.size()) {
        held_size_ = 0;
        inside_ = false;
        ++count_;
      }
      continue;
    }
    auto &raw = raw_[count_];
    if (raw.size() == kMaximumToolJson) {
      Fail("model tool JSON exceeds64KiB");
      break;
    }
    raw.push_back(c);
    if (!started_) {
      if (Space(c))
        continue;
      if (c != '{') {
        Fail("model tool call must be JSON object");
        break;
      }
      started_ = true;
      depth_ = 1;
      continue;
    }
    if (in_string_) {
      if (escaped_)
        escaped_ = false;
      else if (c == '\\')
        escaped_ = true;
      else if (c == '"')
        in_string_ = false;
    } else if (c == '"')
      in_string_ = true;
    else if (c == '{' || c == '[')
      ++depth_;
    else if (c == '}' || c == ']') {
      if (--depth_ == 0)
        ended_ = true;
    }
  }
  return visible_;
}
std::string_view ToolStreamParser::Finish() {
  visible_.clear();
  if (!enabled_ || error_ || inside_)
    return {};
  for (std::size_t i = 0; i < held_size_; ++i)
    Emit(held_[i]);
  held_size_ = 0;
  return visible_;
}
Result<ToolCall> ParseGeneratedTool(std::string_view raw, const ToolConfig &config, std::string id,
                                    bool complete) {
  if (raw.size() > kMaximumToolJson)
    return Invalid("model tool JSON exceeds bound");
  auto name = FindJsonField(raw, "name", !complete),
       args = FindJsonField(raw, "arguments", !complete);
  if (!complete) {
    std::string prefix, argument;
    if (name.ok()) {
      auto text =
          raw.substr(name.value().value_begin, name.value().value_end - name.value().value_begin);
      auto decoded = json::Parse(text);
      if (decoded.ok()) {
        if (!decoded.value().is_string())
          return Invalid("partial tool name must be string");
        prefix = decoded.value().as_string();
      } else if (!text.empty() && text.front() == '"') {
        prefix = std::string(text.substr(1));
        if (!prefix.empty() && !ValidFunctionName(prefix))
          return Invalid("malformed partial function-name prefix");
      } else if (!text.empty())
        return Invalid("partial function name must begin with quote");
    }
    if (std::none_of(config.names.begin(), config.names.end(),
                     [&](const auto &offered) { return offered.starts_with(prefix); }))
      return Invalid("unoffered partial function-name prefix");
    if (args.ok()) {
      argument = std::string(
          raw.substr(args.value().value_begin, args.value().value_end - args.value().value_begin));
      if (!argument.empty() && argument.front() != '{')
        return Invalid("partial tool arguments require object prefix");
    }
    return ToolCall{std::move(id), std::move(prefix), std::move(argument), false};
  }
  if (!name.ok() || !args.ok())
    return Invalid("model tool name/arguments absent or malformed");
  auto decoded = json::Parse(
      raw.substr(name.value().value_begin, name.value().value_end - name.value().value_begin));
  if (!decoded.ok() || !decoded.value().is_string() ||
      std::find(config.names.begin(), config.names.end(), decoded.value().as_string()) ==
          config.names.end())
    return Invalid("model requested unoffered function");
  const auto span = args.value();
  auto text = raw.substr(span.value_begin, span.value_end - span.value_begin);
  if (complete) {
    auto whole = json::Parse(raw);
    auto argument = json::Parse(text);
    if (!whole.ok() || !whole.value().is_object() || whole.value().as_object().size() != 2 ||
        !argument.ok() || !argument.value().is_object())
      return Invalid("model tool arguments require valid JSON object");
  }
  return ToolCall{std::move(id), decoded.value().as_string(), std::string(text), complete};
}
std::string ToolCallsJson(std::span<const ToolCall> calls) {
  std::string s = "[";
  for (std::size_t i = 0; i < calls.size(); ++i) {
    if (i)
      s += ',';
    const auto &c = calls[i];
    s += "{\"id\":" + json::Quote(c.id) +
         ",\"type\":\"function\",\"function\":{\"name\":" + json::Quote(c.name) +
         ",\"arguments\":" + json::Quote(c.arguments) + "}}";
  }
  return s + "]";
}
} // namespace mellum16::server
