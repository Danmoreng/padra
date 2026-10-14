#include "server_test_support.h"
#include <map>

namespace {
std::string Body(bool stream, int budget) {
  return std::string(
             R"({"model":"mellum21-exl3-k4-research","input":"hello","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],"max_output_tokens":)") +
         std::to_string(budget) + ",\"stream\":" + (stream ? "true" : "false") + "}";
}
std::string SemanticOutput(const json::Value &response) {
  auto output = response.find("output")->as_array();
  json::Value::Array normalized;
  for (const auto &item : output) {
    auto fields = item.as_object();
    fields.erase("id");
    fields.erase("call_id");
    normalized.emplace_back(std::move(fields));
  }
  return json::Stringify(json::Value(std::move(normalized)));
}
void Case(text::Tokenizer &tokenizer, std::string output, bool eos) {
  auto state = std::make_shared<MockState>();
  state->tokens = tokenizer.Encode(output).value();
  if (eos)
    state->tokens.push_back(28);
  state->token_delay = 1ms;
  HttpFixture fixture(tokenizer, state);
  httplib::Client client("127.0.0.1", fixture.port);
  client.set_read_timeout(5);
  const auto budget = static_cast<int>(std::max<std::size_t>(1, state->tokens.size()));
  auto wire = client.Post("/v1/responses", Body(true, budget), "application/json");
  Require(wire && wire->status == 200, "output-order HTTP failure");
  std::map<std::string, std::size_t> indices;
  std::map<std::string, std::string> reconstructed;
  std::map<std::string, int> done;
  std::string terminal;
  for (const auto &event : ReadEvents(wire->body)) {
    const auto &type = event.find("type")->as_string();
    const auto *item = event.find("item");
    const auto *item_id = event.find("item_id");
    const auto *index = event.find("output_index");
    if (type == "response.output_item.added") {
      const auto &id = item->find("id")->as_string();
      Require(!indices.contains(id), "output identity added twice");
      Require(static_cast<std::size_t>(index->as_integer()) == indices.size(),
              "noncontiguous output index");
      indices.emplace(id, indices.size());
    }
    if (index) {
      const auto &id = item ? item->find("id")->as_string() : item_id->as_string();
      Require(indices.at(id) == static_cast<std::size_t>(index->as_integer()),
              "live/done index changed");
      if (const auto *delta = event.find("delta"))
        reconstructed[id] += delta->as_string();
      if (type == "response.output_item.done") {
        ++done[id];
        const auto expected = item->find("type")->as_string() == "function_call"
                                  ? item->find("arguments")->as_string()
                                  : item->find("content")->as_array()[0].find("text")->as_string();
        Require(reconstructed[id] == expected, "live bytes differ from done item");
      }
    }
    if (type == "response.completed" || type == "response.incomplete")
      terminal = json::Stringify(*event.find("response"));
  }
  Require(!terminal.empty(), "missing terminal response");
  auto final = json::Parse(terminal).value();
  const auto &items = final.find("output")->as_array();
  Require(items.size() == indices.size(), "live/final item counts differ");
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto &id = items[i].find("id")->as_string();
    Require(indices.at(id) == i && done.at(id) == 1, "live/final output index mismatch");
  }
  auto stored = JsonResponse(client.Get("/v1/responses/" + final.find("id")->as_string()));
  Require(json::Stringify(stored) == terminal, "terminal and stored GET differ");
  auto nonstream =
      JsonResponse(client.Post("/v1/responses", Body(false, budget), "application/json"));
  Require(SemanticOutput(nonstream) == SemanticOutput(final),
          "stream/nonstream order or content differs");
}
} // namespace
int main() {
  try {
    auto tokenizer = text::Tokenizer::Load({}, {});
    const std::string call =
        R"(<tool_call>{"name":"lookup","arguments":{"text":"€","nested":[1,{"x":2}]}}</tool_call>)";
    for (const auto &output :
         {call + "after", "before" + call, "<think>reason</think>" + call + "after",
          call + call + "after", "before" + call + "after", std::string(""), std::string("text")})
      for (bool eos : {false, true})
        Case(*tokenizer.value(), output, eos);
    std::cout << "PASS canonical mixed output identities, indices, bytes and retention\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "TEST ERROR: " << error.what() << '\n';
    return 1;
  }
}
