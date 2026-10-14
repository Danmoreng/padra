#include "compiler/sha256.h"
#include "server_test_support.h"
namespace {
std::string Body(int budget, bool stream, bool store) {
  return std::string(
             R"({"model":"mellum21-exl3-k4-research","input":"hello","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],"max_output_tokens":)") +
         std::to_string(budget) + ",\"stream\":" + (stream ? "true" : "false") +
         ",\"store\":" + (store ? "true" : "false") + "}";
}
json::Value Final(const httplib::Result &result, bool stream) {
  if (!stream)
    return JsonResponse(result);
  Require(result && result->status == 200, "partial stream HTTP failure");
  for (const auto &event : ReadEvents(result->body))
    if (const auto *response = event.find("response");
        response && event.find("type")->as_string() == "response.incomplete")
      return *response;
  throw std::runtime_error("missing incomplete response");
}
void Case(text::Tokenizer &tokenizer, const std::string &prefix, bool partial) {
  for (bool stream : {false, true})
    for (bool store : {false, true}) {
      auto state = std::make_shared<MockState>();
      // Real budget truncation: the backend has more tokens than it may emit.
      state->tokens = tokenizer.Encode(prefix + "not generated").value();
      state->tokens.push_back(28);
      state->token_delay = 1ms;
      const auto budget = static_cast<int>(tokenizer.Encode(prefix).value().size());
      HttpFixture fixture(tokenizer, state);
      httplib::Client client("127.0.0.1", fixture.port);
      client.set_read_timeout(5);
      auto final = Final(
          client.Post("/v1/responses", Body(budget, stream, store), "application/json"), stream);
      Require(final.find("status")->as_string() == "incomplete" &&
                  final.find("usage")->find("output_tokens")->as_integer() == budget,
              "test did not reach actual output budget");
      const auto id = final.find("id")->as_string();
      auto get = client.Get("/v1/responses/" + id);
      Require(get && get->status == (store ? 200 : 404), "partial storage policy lost GET");
      if (store)
        Require(json::Stringify(JsonResponse(get)) == json::Stringify(final),
                "partial GET changed envelope");
      const auto follow =
          std::string(
              R"({"model":"mellum21-exl3-k4-research","input":"next","max_output_tokens":2,"previous_response_id":)") +
          json::Quote(id) + "}";
      auto next = client.Post("/v1/responses", follow, "application/json");
      const auto expected = !store ? 404 : partial ? 400 : 200;
      auto parsed = JsonResponse(next, expected);
      if (partial && store)
        Require(parsed.find("error")->find("code")->as_string() == "response_not_continuable",
                "partial history produced generic parse error instead of state error");
    }
  // The explicit-session path must preserve exactly the same history rule.
  for (bool stream : {false, true}) {
    auto state = std::make_shared<MockState>();
    state->tokens = tokenizer.Encode(prefix + "not generated").value();
    state->token_delay = 1ms;
    HttpFixture fixture(tokenizer, state);
    const std::string credential = "anonymous-loopback";
    const auto owner = compiler::Sha256Hex(credential.data(), credential.size());
    auto session = fixture.coordinator.Sessions().Create(owner).value();
    const auto budget = tokenizer.Encode(prefix).value().size();
    auto body =
        std::string(
            R"({"model":"mellum21-exl3-k4-research","messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"max_completion_tokens":)") +
        std::to_string(budget) + ",\"stream\":" + (stream ? "true" : "false") +
        ",\"session_id\":" + json::Quote(session.id) + "}";
    httplib::Client client("127.0.0.1", fixture.port);
    auto reply = client.Post("/v1/chat/completions", body, "application/json");
    Require(reply && reply->status == 200, "partial explicit-session response failed");
    Eventually(
        [&] { return !fixture.coordinator.Sessions().Get(session.id, owner).value().active; },
        "partial explicit-session lease stuck");
    const auto history = fixture.coordinator.Sessions().Get(session.id, owner).value().history;
    Require(partial ? history == "[]" : history != "[]",
            "session and response history rules diverge");
  }
}
} // namespace
int main() {
  try {
    auto tokenizer = text::Tokenizer::Load({}, {});
    const std::string valid = R"(<tool_call>{"name":"lookup","arguments":{"x":1}}</tool_call>)";
    const std::string args = R"(<tool_call>{"name":"lookup","arguments":{"text":")";
    for (const auto &prefix : std::vector<std::string>{
             R"(<tool_call>{"name":"loo)", args + "unfinished", args + "\\", args + "\\uD83",
             args + std::string("€").substr(0, 1), valid + args + "second"})
      Case(*tokenizer.value(), prefix, true);
    Case(*tokenizer.value(), "valid truncated text", false);
    std::cout << "PASS incomplete tool retrieval and explicit continuation/history policy\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "TEST ERROR: " << error.what() << '\n';
    return 1;
  }
}
