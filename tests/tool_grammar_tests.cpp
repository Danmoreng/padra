#include "server/tool_grammar.h"
#include "server/tool_protocol.h"
#include <iostream>
#include <stdexcept>
using namespace mellum16;
using namespace mellum16::server;
namespace {
void Require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
const char *request =
    R"({"tools":[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]})";
auto Make(const text::Tokenizer &tokenizer, std::string_view body = request) {
  auto parsed = json::Parse(body);
  Require(parsed.ok(), "request fixture");
  auto grammar = MakeToolGrammar(parsed.value(), tokenizer);
  Require(grammar.ok(), "compile grammar");
  return std::move(grammar).value();
}
void Feed(TokenConstraint &grammar, const text::Tokenizer &tokenizer, std::string_view text) {
  auto ids = tokenizer.Encode(text);
  Require(ids.ok(), "encode fixture");
  for (int id : ids.value()) {
    Require(grammar.Mask()[id], "valid token was masked");
    grammar.Accept(id);
  }
}
} // namespace
int main() {
  try {
    auto loaded = text::Tokenizer::Load({}, "");
    Require(loaded.ok(), "test tokenizer");
    const auto &tokenizer = *loaded.value();
    auto grammar = Make(tokenizer);
    Feed(*grammar, tokenizer, "<think>example <tool_call>{invalid]</think>\n");
    const std::string code = "const a = [\"x\"];\nconst s = '\\\\'; // € 😀\n";
    const auto prefix =
        "<tool_call>\n{\"name\":\"write\",\"arguments\":{\"path\":\"game.js\",\"content\":" +
        json::Quote(code);
    Feed(*grammar, tokenizer, prefix);
    Require(!grammar->Mask()[128 + ']'], "extra bracket allowed");
    Require(!grammar->Mask()[28], "EOS allowed in incomplete tool");
    bool rejected = false;
    try {
      grammar->Accept(128 + ']');
    } catch (const std::exception &) {
      rejected = true;
    }
    Require(rejected, "invalid acceptance succeeded");
    Feed(*grammar, tokenizer, "}}\n</tool_call>");
    Require(grammar->Mask()[28], "EOS blocked after complete call");
    Feed(*grammar, tokenizer, prefix + "}}</tool_call>");
    auto fresh = Make(tokenizer);
    Require(fresh->Mask()[28], "request inherited previous state");
    const std::string start = R"(<tool_call>{"name":"write","arguments":{"path":")";
    for (const auto &invalid :
         {std::string("\n"), std::string("\\q"), std::string("\\uDC00"), std::string("\\uD800x"),
          std::string("\xc0\xaf"), std::string("\xed\xa0\x80")}) {
      auto g = Make(tokenizer);
      Feed(*g, tokenizer, start);
      bool blocked = false;
      for (unsigned char c : invalid) {
        if (!g->Mask()[128 + c]) {
          blocked = true;
          break;
        }
        g->Accept(128 + c);
      }
      Require(blocked, "invalid JSON string allowed");
    }
    for (const auto &valid : {"\\uD83D\\uDE00", "\\u0041", "\\n\\t\\\"\\\\", "€😀"}) {
      auto g = Make(tokenizer);
      Feed(*g, tokenizer, start + valid + R"(","content":""}}</tool_call>)");
    }
    auto named = Make(tokenizer);
    Feed(*named, tokenizer, R"(<tool_call>{"name":")");
    Require(!named->Mask()[128 + 'b'], "unoffered name allowed");
    auto typed = Make(tokenizer);
    Feed(*typed, tokenizer, R"(<tool_call>{"name":"write","arguments":{"path":)");
    Require(!typed->Mask()[128 + '1'], "wrong argument type allowed");
    auto single_body = std::string(request);
    single_body.pop_back();
    single_body += R"(,"parallel_tool_calls":false})";
    auto single = Make(tokenizer, single_body);
    Feed(*single, tokenizer, prefix + "}}</tool_call><tool_call");
    Require(!single->Mask()[128 + '>'], "parallel=false allowed second call");
    const char *complex =
        R"({"tools":[{"function":{"name":"edit","parameters":{"type":"object","properties":{"edits":{"type":"array","items":{"type":"object","properties":{"old":{"type":"string"},"new":{"type":"string"}},"required":["old","new"]}},"offset":{"type":"integer"},"mode":{"enum":["a","b"]}},"required":["edits"]}}}]})";
    for (auto args : {R"({"edits":[]})", R"({"edits":[{"old":"a","new":"b"}],"offset":-2})",
                      R"({"edits":[],"mode":"a","offset":0})"}) {
      auto g = Make(tokenizer, complex);
      Feed(*g, tokenizer,
           std::string(R"(<tool_call>{"name":"edit","arguments":)") + args + "}</tool_call>");
    }
    // Pure mask probing (including rejected MTP candidates) cannot advance state.
    auto g = Make(tokenizer);
    Feed(*g, tokenizer, prefix);
    const auto before = g->Mask();
    const std::vector<std::uint8_t> saved(before.begin(), before.end());
    for (int i = 0; i < 8; ++i)
      Require(std::equal(saved.begin(), saved.end(), g->Mask().begin()), "mask mutated state");
    Feed(*g, tokenizer, "}}</tool_call>");
    std::cout << "tool grammar regression checks passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
