#include "server_test_support.h"
#include <fstream>
#include <iterator>
namespace {
void ResponseCompatibility(text::Tokenizer &tokenizer) {
  const auto body =
      R"({"model":"mellum21-exl3-k4-research","input":[{"role":"developer","content":"instructions"},{"role":"user","content":[{"type":"input_text","text":"hello"}]}],"store":false,"prompt_cache_key":"pi-session","prompt_cache_retention":"in-memory","reasoning":{"effort":"none","summary":"auto"},"include":["reasoning.encrypted_content"]})";
  auto parsed = ParseResponseRequest(body);
  Require(parsed.ok(), "Pi Responses controls were rejected");
  auto chat = ParseChatRequest(parsed.value().chat_body, tokenizer, 8192, "pi-response");
  Require(chat.ok() && !json::Parse(chat.value().body).value().find("enable_thinking")->as_bool(),
          "Responses effort did not reach model thinking control");
  auto history = json::Parse(parsed.value().history).value().as_array();
  Require(history[0].find("role")->as_string() == "system",
          "developer instruction lost template role");
  GenerationResult result;
  result.reasoning = "why €";
  result.output_count = 1;
  result.output_order[0] = {OutputKind::Reasoning, 0};
  auto envelope = json::Parse(ResponseEnvelope(parsed.value(), &result)).value();
  const auto &reasoning = envelope.find("output")->as_array()[0];
  Require(reasoning.find("encrypted_content")->is_null() &&
              reasoning.find("summary")->as_array().empty() &&
              reasoning.find("content")->as_array()[0].find("text")->as_string() == "why €",
          "Responses misrepresented plaintext as encryption or a summary");
  Require(envelope.find("prompt_cache_key")->as_string() == "pi-session" &&
              envelope.find("reasoning")->find("effort")->as_string() == "none",
          "Responses lost accepted option metadata");
  auto replay = json::Parse(body).value().as_object();
  replay.insert_or_assign("input",
                          json::Value(json::Value::Array{
                              reasoning, json::Value(json::Value::Object{
                                             {"role", json::Value(std::string("user"))},
                                             {"content", json::Value(std::string("next"))}})}));
  auto continued = ParseResponseRequest(json::Stringify(json::Value(replay)));
  Require(continued.ok() && continued.value().history.find("why €") != std::string::npos,
          "stateless replay lost native reasoning");
  // A routing hint must not imply retained responses or create explicit sessions.
  Require(!parsed.value().store && chat.value().session_id.empty(),
          "prompt cache hint created server-owned conversation state");
}
void TextNormalization(text::Tokenizer &tokenizer) {
  const auto plain =
      R"({"model":"mellum21-exl3-k4-research","messages":[{"role":"system","content":"Follow instructions"},{"role":"user","content":"hello €"}],"max_tokens":8,"enable_thinking":false})";
  const auto parts =
      R"({"model":"mellum21-exl3-k4-research","messages":[{"role":"developer","content":[{"type":"text","text":"Follow "},{"type":"text","text":"instructions"}]},{"role":"user","content":[{"type":"text","text":"hello "},{"type":"text","text":"€"}]}],"max_tokens":8,"reasoning_effort":"none","store":false})";
  auto reference = ParseChatRequest(plain, tokenizer, 8192, "normalization");
  auto normalized = ParseChatRequest(parts, tokenizer, 8192, "normalization");
  Require(reference.ok() && normalized.ok(), "OpenAI text forms failed admission");
  Require(reference.value().input_ids == normalized.value().input_ids,
          "text part concatenation or developer instructions changed prompt bytes");
  Require(!json::Parse(normalized.value().body).value().find("enable_thinking")->as_bool(),
          "reasoning_effort=none did not disable model thinking");
  for (const auto *effort : {"minimal", "low", "medium", "high", "xhigh", "max"}) {
    auto body = json::Parse(plain).value().as_object();
    body.erase("enable_thinking");
    body.emplace("reasoning_effort", json::Value(std::string(effort)));
    auto enabled = ParseChatRequest(json::Stringify(json::Value(body)), tokenizer, 8192, "effort");
    Require(enabled.ok() &&
                json::Parse(enabled.value().body).value().find("enable_thinking")->as_bool(),
            "nonzero reasoning effort did not enable model thinking");
  }
  auto attached_reference =
      AttachSession(reference.value(), tokenizer, "sess-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
  auto attached_parts =
      AttachSession(normalized.value(), tokenizer, "sess-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
  Require(attached_reference.ok() && attached_parts.ok() &&
              reference.value().checkpoint_anchor == normalized.value().checkpoint_anchor,
          "session checkpoint did not use canonical text history");
  const auto schema =
      R"({"model":"mellum21-exl3-k4-research","messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"z":{"type":"string"},"a":{"type":"number"}}}}}]})";
  auto ordered = ParseChatRequest(schema, tokenizer, 8192, "schema-order");
  Require(ordered.ok() && ordered.value().body.find(
                              R"("properties":{"z":{"type":"string"},"a":{"type":"number"}})") !=
                              std::string::npos,
          "content normalization reordered original function schema bytes");
}
void SamplingForwarding(text::Tokenizer &tokenizer) {
  auto state = std::make_shared<MockState>();
  state->tokens = {0, 28};
  HttpFixture fixture(tokenizer, state);
  httplib::Client client("127.0.0.1", fixture.port);
  client.set_read_timeout(3);
  for (bool responses : {false, true}) {
    auto base = json::Parse(responses ? ResponseBody : ChatBody).value().as_object();
    auto fields =
        json::Parse(
            R"({"temperature":0.75,"top_p":0.9,"top_k":42,"seed":-1,"frequency_penalty":0.25,"presence_penalty":-0.5,"repetition_penalty":1.125})")
            .value()
            .as_object();
    for (const auto &[key, value] : fields)
      base.insert_or_assign(key, value);
    auto reply = client.Post(responses ? "/v1/responses" : "/v1/chat/completions",
                             json::Stringify(json::Value(base)), "application/json");
    Require(reply && reply->status == 200, "sampling request failed");
    std::lock_guard lock(state->sampling_mutex);
    const auto &sampling = state->sampling;
    Require(sampling.temperature == .75f && sampling.top_p == .9f && sampling.top_k == 42 &&
                sampling.seed == UINT64_MAX && sampling.frequency_penalty == .25f &&
                sampling.presence_penalty == -.5f && sampling.repetition_penalty == 1.125f,
            "API adapter lost sampling fields before runtime preparation");
  }
}
} // namespace
int main(int argc, char **argv) {
  try {
    Require(argc == 2, "expected normative contract JSON");
    std::ifstream file(argv[1]);
    std::string input(std::istreambuf_iterator<char>(file), {});
    auto matrix = json::Parse(input);
    Require(matrix.ok() && matrix.value().find("version")->as_integer() == 1,
            "contract matrix invalid");
    auto tokenizer = text::Tokenizer::Load({}, {});
    Require(tokenizer.ok(), "contract tokenizer unavailable");
    auto state = std::make_shared<MockState>();
    state->tokens = {0, 28};
    HttpFixture fixture(*tokenizer.value(), state);
    httplib::Client client("127.0.0.1", fixture.port);
    client.set_read_timeout(3);
    std::size_t checked = 0;
    for (const auto &test : matrix.value().find("cases")->as_array()) {
      const bool responses = test.find("route")->as_string() == "responses";
      auto body = json::Parse(responses ? ResponseBody : ChatBody).value().as_object();
      if (auto *remove = test.find("remove"))
        for (const auto &key : remove->as_array())
          body.erase(key.as_string());
      for (const auto &[key, value] : test.find("patch")->as_object())
        body.insert_or_assign(key, value);
      auto reply = client.Post(responses ? "/v1/responses" : "/v1/chat/completions",
                               json::Stringify(json::Value(body)), "application/json");
      if (!reply || reply->status != test.find("status")->as_integer())
        throw std::runtime_error(
            "contract status: " + test.find("route")->as_string() + "/" +
            test.find("name")->as_string() + " actual=" +
            (reply ? std::to_string(reply->status) + " " + reply->body : "no response"));
      Require(reply->get_header_value("X-Request-Id").starts_with("req_"),
              "contract error/success lacked request ID");
      if (reply->status != 200) {
        auto error = json::Parse(reply->body);
        Require(error.ok() && error.value().find("error"), "contract error was not JSON");
        if (error.value().find("error")->find("code")->as_string() !=
            test.find("code")->as_string())
          throw std::runtime_error("contract error code: " + test.find("name")->as_string());
      }
      ++checked;
    }
    for (const auto *route :
         {"/v1/embeddings", "/v1/completions", "/v1/audio/speech", "/v1/images/generations",
          "/v1/files", "/v1/batches", "/v1/assistants", "/v1/threads",
          "/v1/responses/resp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/cancel"}) {
      auto reply = client.Post(route, "{}", "application/json");
      Require(reply && reply->status == 404, "unsupported endpoint silently accepted");
    }
    SamplingForwarding(*tokenizer.value());
    TextNormalization(*tokenizer.value());
    ResponseCompatibility(*tokenizer.value());
    std::cout
        << checked
        << " normative API field cases, unsupported endpoints and both sampling adapters passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
