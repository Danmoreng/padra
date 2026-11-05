#include "server/tool_grammar.h"
#include <algorithm>
#include <array>
#include <bitset>
#include <map>
#include <set>
#include <stdexcept>

namespace mellum16::server {
namespace {
using Sequence = std::vector<int>;
using Stacks = std::vector<Sequence>;
// Byte CFG: terminals are negative character-set indexes, rules nonnegative.
// All recursion consumes a byte before recurring. No arbitrary client grammar
// or left recursion is accepted. UTF-8 and JSON escapes span token boundaries.
struct Grammar {
  std::vector<std::vector<Sequence>> rules;
  std::vector<std::bitset<256>> terminals;
  int ws, string, number, integer, value, object, array, boolean, null;
  int Rule(std::vector<Sequence> alternatives = {}) {
    if (rules.size() >= 8192)
      throw std::invalid_argument("tool grammar exceeds rule budget");
    rules.push_back(std::move(alternatives));
    return static_cast<int>(rules.size() - 1);
  }
  int Set(std::string_view chars) {
    if (terminals.size() >= 65536)
      throw std::invalid_argument("tool grammar exceeds terminal budget");
    std::bitset<256> bits;
    for (unsigned char c : chars)
      bits.set(c);
    terminals.push_back(bits);
    return -static_cast<int>(terminals.size());
  }
  int Range(int a, int b) {
    std::string chars;
    for (int c = a; c <= b; ++c)
      chars += static_cast<char>(c);
    return Set(chars);
  }
  int Literal(std::string_view text) {
    Sequence seq;
    for (char c : text)
      seq.push_back(Set(std::string_view(&c, 1)));
    return Rule({std::move(seq)});
  }
  Grammar() {
    ws = Rule();
    rules[ws] = {{}, {Set(" \n\r\t"), ws}};
    const int digit = Range('0', '9'), hex = Set("0123456789abcdefABCDEF");
    const int digits = Rule();
    rules[digits] = {{}, {digit, digits}};
    const int sign = Rule({{}, {Literal("-")}});
    integer = Rule({{sign, Literal("0")}, {sign, Range('1', '9'), digits}});
    const int frac = Rule({{}, {Literal("."), digit, digits}});
    const int exponent = Rule({{}, {Set("eE"), Rule({{}, {Set("+-")}}), digit, digits}});
    number = Rule({{integer, frac, exponent}});
    boolean = Rule({{Literal("true")}, {Literal("false")}});
    null = Literal("null");
    const int cont = Range(0x80, 0xbf);
    const int unicode = Rule({{Range(0xc2, 0xdf), cont},
                              {Range(0xe1, 0xec), cont, cont},
                              {Range(0xee, 0xef), cont, cont},
                              {Range(0xe0, 0xe0), Range(0xa0, 0xbf), cont},
                              {Range(0xed, 0xed), Range(0x80, 0x9f), cont},
                              {Range(0xf0, 0xf0), Range(0x90, 0xbf), cont, cont},
                              {Range(0xf1, 0xf3), cont, cont, cont},
                              {Range(0xf4, 0xf4), Range(0x80, 0x8f), cont, cont}});
    const int unicode_escape = Rule({{Set("0123456789abcefABCEF"), hex, hex, hex},
                                     {Set("dD"), Set("01234567"), hex, hex},
                                     {Set("dD"), Set("89abAB"), hex, hex, Literal("\\u"), Set("dD"),
                                      Set("cdefCDEF"), hex, hex}});
    std::string ascii;
    for (int c = 32; c < 127; ++c)
      if (c != '"' && c != '\\')
        ascii += static_cast<char>(c);
    const int character = Rule({{Set(ascii)},
                                {unicode},
                                {Literal("\\"), Set("\"\\/bfnrt")},
                                {Literal("\\u"), unicode_escape}});
    const int chars = Rule();
    rules[chars] = {{}, {character, chars}};
    string = Rule({{Literal("\""), chars, Literal("\"")}});
    value = Rule();
    object = Rule();
    array = Rule();
    const int members = Rule(), elements = Rule();
    rules[members] = {{}, {Literal(","), ws, string, ws, Literal(":"), ws, value, ws, members}};
    rules[object] = {
        {Literal("{"), ws, Literal("}")},
        {Literal("{"), ws, string, ws, Literal(":"), ws, value, ws, members, Literal("}")}};
    rules[elements] = {{}, {Literal(","), ws, value, ws, elements}};
    rules[array] = {{Literal("["), ws, Literal("]")},
                    {Literal("["), ws, value, ws, elements, Literal("]")}};
    rules[value] = {{string}, {number}, {object}, {array}, {boolean}, {null}};
  }
  int Schema(const json::Value &schema, int depth = 0, bool force_object = false) {
    if (depth > 32)
      throw std::invalid_argument("tool grammar schema nesting exceeds32");
    if (!schema.is_object())
      return force_object ? object : value;
    // Unsupported schema assertions are not advertised as enforced. The
    // strict=false contract permits structural fallback, never invalid JSON.
    if (schema.find("$ref") || schema.find("allOf") || schema.find("not"))
      return force_object ? object : value;
    if (!force_object) {
      if (auto *c = schema.find("const"))
        return Literal(json::Stringify(*c));
      if (auto *e = schema.find("enum"); e && e->is_array() && !e->as_array().empty()) {
        std::vector<Sequence> alternatives;
        for (const auto &v : e->as_array())
          alternatives.push_back({Literal(json::Stringify(v))});
        return Rule(std::move(alternatives));
      }
      for (auto key : {"anyOf", "oneOf"}) {
        if (auto *a = schema.find(key); a && a->is_array() && !a->as_array().empty()) {
          std::vector<Sequence> alternatives;
          for (const auto &v : a->as_array())
            alternatives.push_back({Schema(v, depth + 1)});
          return Rule(std::move(alternatives));
        }
      }
    }
    auto *type = schema.find("type");
    const auto name = force_object ? "object" : type && type->is_string() ? type->as_string() : "";
    if (name == "string")
      return string;
    if (name == "number")
      return number;
    if (name == "integer")
      return integer;
    if (name == "boolean")
      return boolean;
    if (name == "null")
      return null;
    if (name == "array") {
      const auto *items = schema.find("items");
      const int item = items ? Schema(*items, depth + 1) : value;
      const int rest = Rule();
      rules[rest] = {{}, {Literal(","), ws, item, ws, rest}};
      return Rule(
          {{Literal("["), ws, Literal("]")}, {Literal("["), ws, item, ws, rest, Literal("]")}});
    }
    if (name != "object")
      return value;
    const auto *props = schema.find("properties"), *required = schema.find("required");
    if (!props || !props->is_object())
      return object;
    if (props->as_object().size() > 128)
      throw std::invalid_argument("too many tool grammar properties");
    std::vector<std::string> keys;
    std::set<std::string> mandatory;
    if (required && required->is_array())
      for (const auto &key : required->as_array()) {
        if (!key.is_string())
          throw std::invalid_argument("required tool property must be a string");
        if (mandatory.insert(key.as_string()).second)
          keys.push_back(key.as_string());
      }
    const auto required_count = keys.size();
    for (const auto &[key, _] : props->as_object())
      if (!mandatory.contains(key))
        keys.push_back(key);
    std::vector<int> fields;
    for (const auto &key : keys) {
      const auto *property = props->find(key);
      fields.push_back(Rule({{Literal(json::Quote(key)), ws, Literal(":"), ws,
                              property ? Schema(*property, depth + 1) : value, ws}}));
    }
    // Ordered subset: required fields in required[] order, optional fields in
    // lexical order. Each optional suffix has variants with/without a comma.
    int rest_empty = Rule({{}}), rest_nonempty = rest_empty;
    for (std::size_t i = fields.size(); i > required_count; --i) {
      const int next_empty = Rule({{rest_empty}, {fields[i - 1], rest_nonempty}});
      const int next_nonempty =
          Rule({{rest_nonempty}, {Literal(","), ws, fields[i - 1], rest_nonempty}});
      rest_empty = next_empty;
      rest_nonempty = next_nonempty;
    }
    Sequence seq{Literal("{"), ws};
    for (std::size_t i = 0; i < required_count; ++i) {
      if (i) {
        seq.push_back(Literal(","));
        seq.push_back(ws);
      }
      seq.push_back(fields[i]);
    }
    seq.push_back(required_count ? rest_nonempty : rest_empty);
    seq.push_back(Literal("}"));
    return Rule({std::move(seq)});
  }
  void Expand(Sequence stack, Stacks &out) const {
    if (stack.size() > 1024)
      return; // Reject this over-deep candidate, preserving other continuations.
    if (out.size() > 1024)
      throw std::runtime_error("tool grammar state budget exceeded");
    if (stack.empty() || stack.back() < 0) {
      out.push_back(std::move(stack));
      return;
    }
    const int r = stack.back();
    stack.pop_back();
    for (const auto &alternative : rules[r]) {
      auto next = stack;
      next.insert(next.end(), alternative.rbegin(), alternative.rend());
      Expand(std::move(next), out);
    }
  }
  Stacks Start(int rule) const {
    Stacks out;
    Expand({rule}, out);
    return out;
  }
  bool Byte(Stacks &state, unsigned char c) const {
    Stacks next;
    for (auto stack : state)
      if (!stack.empty() && terminals[-stack.back() - 1][c]) {
        stack.pop_back();
        Expand(std::move(stack), next);
      }
    std::sort(next.begin(), next.end());
    next.erase(std::unique(next.begin(), next.end()), next.end());
    state = std::move(next);
    return !state.empty();
  }
};

struct State {
  enum Phase { Detect, Thinking, Text, Tool } phase = Detect;
  std::string held;
  Stacks stacks;
  int calls = 0;
};
class ToolGrammar final : public TokenConstraint {
  Grammar grammar_;
  int root_, maximum_;
  const text::Tokenizer &tokenizer_;
  State state_;
  std::map<std::string, std::vector<std::uint8_t>> masks_;
  bool Byte(State &s, unsigned char c) const {
    constexpr std::string_view think = "<think>", end = "</think>", tool = "<tool_call>";
    if (s.phase == State::Tool) {
      if (!grammar_.Byte(s.stacks, c))
        return false;
      if (std::any_of(s.stacks.begin(), s.stacks.end(), [](const auto &v) { return v.empty(); })) {
        s.phase = State::Text;
        s.stacks.clear();
        ++s.calls;
      }
      return true;
    }
    s.held += static_cast<char>(c);
    if (s.phase == State::Detect) {
      if (think.starts_with(s.held)) {
        if (s.held == think) {
          s.phase = State::Thinking;
          s.held.clear();
        }
        return true;
      }
      auto pending = std::move(s.held);
      s.held.clear();
      s.phase = State::Text;
      for (unsigned char b : pending)
        if (!Byte(s, b))
          return false;
      return true;
    }
    const auto marker = s.phase == State::Thinking ? end : tool;
    if (s.held == marker) {
      s.held.clear();
      if (s.phase == State::Thinking)
        s.phase = State::Text;
      else {
        if (s.calls >= maximum_)
          return false;
        s.phase = State::Tool;
        s.stacks = grammar_.Start(root_);
      }
    } else
      while (!s.held.empty() && !marker.starts_with(s.held))
        s.held.erase(0, 1);
    return true;
  }
  bool Token(State &state, int id) const {
    if (id == 28)
      return state.phase != State::Tool;
    const auto piece = tokenizer_.RawToken(id);
    if (piece.empty())
      return state.phase != State::Tool;
    for (unsigned char c : piece)
      if (!Byte(state, c))
        return false;
    return true;
  }
  std::string Key() const {
    std::string key =
        std::to_string(state_.phase) + ":" + std::to_string(state_.calls) + ":" + state_.held + ":";
    for (const auto &stack : state_.stacks) {
      for (int symbol : stack) {
        key += std::to_string(symbol);
        key += ',';
      }
      key += ';';
    }
    return key;
  }

public:
  ToolGrammar(const json::Value &request, const text::Tokenizer &tokenizer)
      : maximum_(request.find("parallel_tool_calls") &&
                         request.find("parallel_tool_calls")->is_bool() &&
                         !request.find("parallel_tool_calls")->as_bool()
                     ? 1
                     : 8),
        tokenizer_(tokenizer) {
    std::vector<Sequence> calls;
    const auto *tools = request.find("tools");
    if (!tools || !tools->is_array())
      throw std::invalid_argument("missing tools for grammar");
    for (const auto &tool : tools->as_array()) {
      const auto *fn = tool.find("function"), *name = fn ? fn->find("name") : nullptr;
      if (!name || !name->is_string())
        throw std::invalid_argument("invalid tool grammar definition");
      const auto *schema = fn->find("parameters");
      const int args = schema ? grammar_.Schema(*schema, 0, true) : grammar_.object;
      calls.push_back({grammar_.Literal("{"), grammar_.ws, grammar_.Literal("\"name\""),
                       grammar_.ws, grammar_.Literal(":"), grammar_.ws,
                       grammar_.Literal(json::Quote(name->as_string())), grammar_.ws,
                       grammar_.Literal(","), grammar_.ws, grammar_.Literal("\"arguments\""),
                       grammar_.ws, grammar_.Literal(":"), grammar_.ws, args, grammar_.ws,
                       grammar_.Literal("}")});
    }
    if (calls.empty())
      throw std::invalid_argument("empty tool grammar");
    const int call = grammar_.Rule(std::move(calls));
    root_ = grammar_.Rule({{grammar_.ws, call, grammar_.ws, grammar_.Literal("</tool_call>")}});
  }
  std::span<const std::uint8_t> Mask() override {
    auto key = Key();
    if (auto found = masks_.find(key); found != masks_.end())
      return found->second;
    // Bounded per-request cache. Repeated string-body states dominate code tools.
    if (masks_.size() >= 128)
      masks_.clear();
    std::vector<std::uint8_t> mask(tokenizer_.VocabularySize());
    bool any = false;
    for (std::size_t i = 0; i < mask.size(); ++i) {
      auto candidate = state_;
      mask[i] = Token(candidate, static_cast<int>(i));
      any |= mask[i] != 0;
    }
    if (!any)
      throw std::runtime_error("tool grammar has no valid token");
    return masks_.emplace(std::move(key), std::move(mask)).first->second;
  }
  void Accept(int token) override {
    auto next = state_;
    if (token < 0 || static_cast<std::size_t>(token) >= tokenizer_.VocabularySize() ||
        !Token(next, token))
      throw std::runtime_error("selected token violates tool grammar");
    state_ = std::move(next);
  }
};
} // namespace
Result<std::shared_ptr<TokenConstraint>> MakeToolGrammar(const json::Value &request,
                                                         const text::Tokenizer &tokenizer) {
  try {
    return std::shared_ptr<TokenConstraint>(std::make_shared<ToolGrammar>(request, tokenizer));
  } catch (const std::exception &e) {
    return Status(StatusCode::kInvalidArgument, e.what());
  }
}
} // namespace mellum16::server
