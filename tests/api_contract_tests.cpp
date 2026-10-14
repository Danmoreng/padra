#include "server_test_support.h"
#include <fstream>
#include <iterator>
namespace {
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
    std::cout
        << checked
        << " normative API field cases, unsupported endpoints and both sampling adapters passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
