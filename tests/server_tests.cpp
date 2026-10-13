#include "server/coordinator.h"
#include "server/http_server.h"
#include "server/job_error.h"
#include "server/response_store.h"
#include "server/response_stream.h"
#include "server/responses.h"
#include "text/stream.h"
#include "util/json.h"
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>
using namespace mellum16;
using namespace mellum16::server;
using namespace std::chrono_literals;

struct MockState {
  std::promise<void> entered, release;
  std::shared_future<void> released = release.get_future().share();
  bool block = false;
  std::atomic<int> calls{0};
  std::vector<int> tokens{0};
  std::chrono::milliseconds token_delay{0};
};
class MockBackend final : public RuntimeBackend {
  std::shared_ptr<MockState> state_;

public:
  explicit MockBackend(std::shared_ptr<MockState> state) : state_(std::move(state)) {}
  Status Prepare(std::span<const int>, int, bool, SamplingOptions) override {
    return {};
  }
  Status Append(std::span<const int>, int, bool, SamplingOptions) override {
    return {};
  }
  Status Reset() override {
    return {};
  }
  void Cancel() override {}
  RuntimeStats Stats() const override {
    return {};
  }
  Result<InferenceResult> Generate(TokenCallback callback, PrefillCallback) override {
    if (state_->calls.fetch_add(1) == 0 && state_->block) {
      state_->entered.set_value();
      state_->released.wait();
    }
    InferenceResult result;
    for (int token : state_->tokens) {
      if (state_->token_delay.count())
        std::this_thread::sleep_for(state_->token_delay);
      result.generated_ids.push_back(token);
      if (callback && !callback(token)) {
        result.cancelled = true;
        break;
      }
    }
    return result;
  }
};
struct ReleaseGuard {
  std::shared_ptr<MockState> state;
  ~ReleaseGuard() {
    try {
      state->release.set_value();
    } catch (const std::future_error &) {
    }
  }
};
void Require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
ChatRequest Request(std::string id) {
  ChatRequest request;
  request.id = std::move(id);
  request.input_ids = {0};
  request.outputs = 1;
  return request;
}
std::shared_ptr<Job> Submit(Coordinator &coordinator, ChatRequest request) {
  auto result = coordinator.Submit(std::move(request));
  Require(result.ok(), "unexpected admission failure");
  return std::move(result).value();
}
void Wait(const std::shared_ptr<Job> &job) {
  std::unique_lock lock(job->mutex);
  Require(job->changed.wait_for(lock, 2s, [&] { return job->complete; }),
          "test completion timeout");
}
std::string Drain(const std::shared_ptr<Job> &job) {
  EventFrame frame;
  std::string output;
  while (job->Pop(frame, 0ms))
    output.append(frame.data.data(), frame.size);
  return output;
}
std::unique_ptr<ModelRuntime> Runtime(std::shared_ptr<MockState> state) {
  return std::make_unique<ModelRuntime>(std::make_unique<MockBackend>(std::move(state)));
}

void TestQueueDeadline(text::Tokenizer &tokenizer) {
  auto state = std::make_shared<MockState>();
  state->block = true;
  ServerLimits limits;
  limits.queue_timeout = 40ms;
  Coordinator coordinator(Runtime(state), tokenizer, limits);
  ReleaseGuard release{state};
  auto first = Submit(coordinator, Request("first"));
  Require(state->entered.get_future().wait_for(2s) == std::future_status::ready,
          "backend did not block");
  auto session = coordinator.Sessions().Create("owner");
  Require(session.ok(), "session setup failed");
  auto request = Request("second");
  request.session_id = session.value().id;
  request.session_owner = "owner";
  request.stream = true;
  auto second = Submit(coordinator, std::move(request));
  Wait(second);
  Require(!second->failure.ok() && second->cancellation.load() == CancelCause::QueueDeadline,
          "queue timeout did not fail independently");
  {
    std::lock_guard lock(first->mutex);
    Require(!first->complete, "active job unexpectedly completed");
  }
  auto metrics = coordinator.Metrics();
  Require(metrics.queued == 0 && metrics.queued_bytes == 0 && metrics.queued_tokens == 0,
          "expired queue resources remain admitted");
  Require(!coordinator.Sessions().Get(session.value().id, "owner").value().active,
          "expired session lease remains active");
  Require(Drain(second).find("queue_deadline") != std::string::npos,
          "SSE queue error classification missing");
  Require(state->calls.load() == 1, "expired job reached runtime");
}
void TestQueuedCancellation(text::Tokenizer &tokenizer, CancelCause cause) {
  auto state = std::make_shared<MockState>();
  state->block = true;
  Coordinator coordinator(Runtime(state), tokenizer);
  ReleaseGuard release{state};
  auto first = Submit(coordinator, Request("active"));
  Require(state->entered.get_future().wait_for(2s) == std::future_status::ready,
          "backend did not block");
  auto session = coordinator.Sessions().Create("owner");
  Require(session.ok(), "session setup failed");
  auto request = Request("queued");
  request.session_id = session.value().id;
  request.session_owner = "owner";
  auto second = Submit(coordinator, std::move(request));
  if (cause == CancelCause::Shutdown)
    coordinator.BeginDrain();
  else if (cause == CancelCause::SessionDiscard)
    Require(coordinator.DiscardSession(session.value().id, "owner").ok(), "discard failed");
  else
    second->Cancel(cause);
  Wait(second);
  Require(second->cancellation.load() == cause && !second->failure.ok(),
          "queued cancellation not completed");
  Require(coordinator.Metrics().queued == 0, "cancelled job remains queued");
  if (cause != CancelCause::SessionDiscard)
    Require(!coordinator.Sessions().Get(session.value().id, "owner").value().active,
            "cancelled session lease remains active");
  Require(state->calls.load() == 1, "cancelled queued job reached runtime");
}
void TestEndingPublishedAfterPollTimeout() {
  auto request = Request("completion-after-poll");
  request.stream = true;
  Job job(std::move(request), 4);
  EventFrame frame;
  Require(!job.Pop(frame, 0ms), "unexpected frame before publication");
  constexpr std::string_view ending = "data: [DONE]\n\n";
  Require(job.PushEnding(std::span(ending.data(), ending.size())), "ending publication failed");
  {
    std::lock_guard lock(job.mutex);
    job.complete = true;
    Require(!job.StreamDrained(), "poll timeout allowed unsent completion to close stream");
  }
  Require(job.Pop(frame, 0ms), "completion after timeout disappeared");
  {
    std::lock_guard lock(job.mutex);
    Require(job.StreamDrained(), "stream did not finish after delivering ending");
  }
}
void TestStreamingEndings(text::Tokenizer &tokenizer) {
  for (int ending = 0; ending < 3; ++ending)
    for (bool usage : {false, true}) {
      auto state = std::make_shared<MockState>();
      state->tokens.assign(63, 0);
      if (ending == 0)
        state->tokens.push_back(1);
      else if (ending == 1)
        state->tokens.push_back(28);
      Coordinator coordinator(Runtime(state), tokenizer);
      auto request = Request("full-ring");
      request.stream = true;
      request.include_usage = usage;
      request.outputs = 64;
      if (ending == 0)
        request.stops = {"STOP"};
      auto job = Submit(coordinator, std::move(request));
      Wait(job);
      Require(job->failure.ok() && job->cancellation.load() == CancelCause::None,
              "normal ending treated as cancellation");
      auto frames = Drain(job);
      const auto finish =
          frames.find(ending == 2 ? "\"finish_reason\":\"length\"" : "\"finish_reason\":\"stop\"");
      const auto done = frames.find("data: [DONE]");
      Require(finish != std::string::npos && done > finish && done != std::string::npos,
              "full ring lost ordered terminal records");
      if (usage) {
        auto at = frames.find("\"usage\":{\"prompt_tokens\"");
        Require(at > finish && at < done, "usage frame missing/out of order");
      }
      job->Cancel(CancelCause::Disconnect);
      Require(job->cancellation.load() == CancelCause::Disconnect,
              "normal stop masked later disconnect");
    }
  auto state = std::make_shared<MockState>();
  state->tokens.assign(65, 0);
  Coordinator coordinator(Runtime(state), tokenizer);
  auto request = Request("overflow");
  request.stream = true;
  request.outputs = 65;
  auto job = Submit(coordinator, std::move(request));
  Wait(job);
  Require(!job->failure.ok() && job->cancellation.load() == CancelCause::Backpressure,
          "delta overflow falsely reported success");
  auto frames = Drain(job);
  Require(frames.find("stream_backpressure") != std::string::npos &&
              frames.find("[DONE]") == std::string::npos,
          "overflow lacks explicit terminal error");
  Require(coordinator.Metrics().ready, "transport failure poisoned runtime readiness");
}
void TestControlCases(text::Tokenizer &tokenizer) {
  Require(!json::Parse("{\"x\":1,\"x\":2}").ok(), "duplicate key accepted");
  Require(!json::Parse("\"\\ud800\"").ok(), "unpaired surrogate accepted");
  text::StopMatcher stop({"END"}, 32);
  Require(stop.Push("abcE") == "abc", "stop prefix lost");
  Require(stop.Push("NDtail").empty() && stop.stopped(), "cross-chunk stop failed");
  text::ThinkingParser thinking(32);
  auto output = thinking.Push("<thi");
  Require(output.content.empty(), "thinking prefix leaked");
  output = thinking.Push("nk>reason</think>\nanswer");
  Require(output.reasoning == "reason" && output.content == "answer", "thinking split failed");
  text::StreamDecoder decoder(tokenizer);
  Require(decoder.Push(128 + 0xe2).empty(), "partial UTF8 leaked");
  Require(decoder.Push(128 + 0x82).empty(), "partial UTF8 leaked");
  Require(decoder.Push(128 + 0xac) == "€", "UTF8 decoding failed");
  auto sessions = std::make_shared<SessionStore>();
  auto session = sessions->Create("owner-A");
  Require(session.ok(), "session create");
  Require(!sessions->Get(session.value().id, "owner-B").ok(), "owner isolation failed");
  auto lease = sessions->Acquire(session.value().id, "owner-A", 128);
  Require(lease.ok(), "lease create");
  Require(!sessions->Acquire(session.value().id, "owner-A", 128).ok(),
          "overlapping lease accepted");
}

class HttpFixture {
public:
  std::shared_ptr<MockState> state;
  Coordinator coordinator;
  HttpServer http;
  int port;
  std::thread listener;
  HttpFixture(text::Tokenizer &tokenizer, std::shared_ptr<MockState> input,
              ServerLimits limits = {})
      : state(input), coordinator(Runtime(input), tokenizer, limits),
        http(coordinator, tokenizer, 0), port(http.Bind()) {
    Require(port > 0, "HTTP bind failed");
    listener = std::thread([this] { http.Listen(); });
    bool ready = false;
    for (int i = 0; i < 100; ++i) {
      httplib::Client client("127.0.0.1", port);
      client.set_read_timeout(1);
      auto result = client.Get("/healthz");
      if (result && result->status == 200) {
        ready = true;
        break;
      }
      std::this_thread::sleep_for(5ms);
    }
    if (!ready) {
      http.Stop();
      listener.join();
      throw std::runtime_error("HTTP startup timeout");
    }
  }
  ~HttpFixture() {
    try {
      state->release.set_value();
    } catch (const std::future_error &) {
    }
    http.Stop();
    if (listener.joinable())
      listener.join();
  }
};
constexpr auto ChatBody =
    R"({"model":"mellum21-exl3-k4-research","messages":[{"role":"user","content":"hello"}],"max_completion_tokens":2})";
constexpr auto ResponseBody =
    R"({"model":"mellum21-exl3-k4-research","input":"hello","max_output_tokens":2})";
json::Value JsonResponse(const httplib::Result &response, int expected = 200) {
  Require(response && response->status == expected, "unexpected HTTP status");
  auto parsed = json::Parse(response->body);
  Require(parsed.ok(), "invalid HTTP JSON");
  return std::move(parsed).value();
}
void TestHttpRetention(text::Tokenizer &tokenizer) {
  HttpFixture fixture(tokenizer, std::make_shared<MockState>());
  httplib::Client client("127.0.0.1", fixture.port);
  client.set_read_timeout(3);
  std::vector<std::string> ids;
  for (int i = 0; i < 40; ++i) {
    auto response = client.Post("/v1/responses", ResponseBody, "application/json");
    ids.push_back(JsonResponse(response).find("id")->as_string());
  }
  Require(fixture.coordinator.Sessions().Stats().first == 32,
          "root retention count exceeded capacity");
  auto old = client.Get("/v1/responses/" + ids[0]);
  Require(old && old->status == 404, "oldest response not evicted");
  auto last = client.Get("/v1/responses/" + ids.back());
  Require(last && last->status == 200, "latest response missing");
  auto child = client.Post("/v1/responses",
                           "{\"model\":\"mellum21-exl3-k4-research\",\"input\":\"next\",\"max_"
                           "output_tokens\":2,\"previous_response_id\":" +
                               json::Quote(ids.back()) + "}",
                           "application/json");
  auto child_id = JsonResponse(child).find("id")->as_string();
  auto removed = client.Delete("/v1/responses/" + ids.back());
  Require(removed && removed->status == 200, "parent deletion failed");
  Require(client.Get("/v1/responses/" + child_id)->status == 200,
          "parent deletion destroyed child");
  Require(client.Delete("/v1/responses/" + child_id)->status == 200, "child deletion failed");
  auto size = fixture.coordinator.Sessions().Stats().first;
  auto invalid = client.Post(
      "/v1/responses", R"({"model":"mellum21-exl3-k4-research","input":"hello","temperature":3})",
      "application/json");
  Require(invalid && invalid->status == 400 && fixture.coordinator.Sessions().Stats().first == size,
          "invalid request mutated root retention");
}
void TestHttpQueueErrors(text::Tokenizer &tokenizer) {
  auto state = std::make_shared<MockState>();
  state->block = true;
  ServerLimits limits;
  limits.queue_timeout = 40ms;
  HttpFixture fixture(tokenizer, state, limits);
  auto first = Submit(fixture.coordinator, Request("blocker"));
  Require(state->entered.get_future().wait_for(2s) == std::future_status::ready,
          "backend did not block");
  httplib::Client client("127.0.0.1", fixture.port);
  client.set_read_timeout(2);
  for (auto route : {"/v1/chat/completions", "/v1/responses"}) {
    auto response =
        client.Post(route, std::string_view(route) == "/v1/responses" ? ResponseBody : ChatBody,
                    "application/json");
    auto parsed = JsonResponse(response, 503);
    Require(parsed.find("error")->find("code")->as_string() == "queue_deadline",
            "HTTP queue error code differs");
    Require(response->get_header_value("Retry-After") == "1", "Retry-After missing");
  }
  Require(fixture.coordinator.Sessions().Stats().first == 0,
          "failed Responses admission leaked root");
  {
    std::lock_guard lock(first->mutex);
    Require(!first->complete, "HTTP queue error waited for GPU completion");
  }
}
void TestHttpStreams(text::Tokenizer &tokenizer) {
  for (bool responses : {false, true}) {
    auto state = std::make_shared<MockState>();
    state->tokens = {0, 28};
    HttpFixture fixture(tokenizer, state);
    httplib::Client client("127.0.0.1", fixture.port);
    client.set_read_timeout(2);
    auto body = std::string(responses ? ResponseBody : ChatBody);
    body.insert(body.size() - 1, ",\"stream\":true");
    auto reply =
        client.Post(responses ? "/v1/responses" : "/v1/chat/completions", body, "application/json");
    Require(reply && reply->status == 200 &&
                reply->get_header_value("Content-Type").starts_with("text/event-stream"),
            "SSE HTTP response failed");
    Require(reply->body.find(responses ? "response.completed" : "data: [DONE]") !=
                std::string::npos,
            "HTTP stream lacks completion");
    Require(reply->body.find(responses ? "response.output_text.delta" : "\"content\":\"x\"") !=
                std::string::npos,
            "HTTP text delta missing");
  }
}
void TestGracefulDrain(text::Tokenizer &tokenizer) {
  for (bool expires : {false, true}) {
    auto state = std::make_shared<MockState>();
    state->block = true;
    ServerLimits limits;
    limits.drain_timeout = 40ms;
    Coordinator coordinator(Runtime(state), tokenizer, limits);
    ReleaseGuard release{state};
    auto first = Submit(coordinator, Request("drain-active"));
    Require(state->entered.get_future().wait_for(2s) == std::future_status::ready,
            "backend did not block");
    auto second = Submit(coordinator, Request("drain-queued"));
    coordinator.BeginDrain(true);
    Wait(second);
    Require(!coordinator.Metrics().ready && coordinator.Metrics().draining &&
                !coordinator.Submit(Request("new")).ok(),
            "drain accepted new work");
    Require(first->cancellation.load() == CancelCause::None,
            "graceful drain immediately cancelled active work");
    if (expires) {
      const auto until = std::chrono::steady_clock::now() + 1s;
      while (first->cancellation.load() == CancelCause::None &&
             std::chrono::steady_clock::now() < until)
        std::this_thread::sleep_for(2ms);
      Require(first->cancellation.load() == CancelCause::Shutdown,
              "drain grace not enforced independently");
    }
    state->release.set_value();
    Wait(first);
    Require(first->failure.ok() != expires, "graceful drain outcome incorrect");
  }
}
void TestStoreLifetimes() {
  using Clock = std::chrono::steady_clock;
  auto now = Clock::now();
  SessionLimits session_limits;
  session_limits.entries = 2;
  session_limits.ttl = 40ms;
  ResponseStoreLimits response_limits;
  response_limits.entries = 2;
  response_limits.ttl = 40ms;
  auto sessions = std::make_shared<SessionStore>(session_limits, [&] { return now; });
  auto store = std::make_shared<ResponseStore>(response_limits, [&] { return now; });
  auto session = sessions->Create("owner");
  Require(session.ok(), "root session setup failed");
  auto root = std::make_shared<ResponseRoot>();
  root->session_id = session.value().id;
  root->owner = "owner";
  root->discard = [sessions, id = root->session_id] { (void)sessions->Delete(id, "owner"); };
  auto id = [](char c) { return "resp_" + std::string(32, c); };
  for (char c : {'a', 'b'}) {
    auto ticket = store->Reserve(id(c), "owner", 1024);
    Require(ticket.ok(), "shared-root reservation failed");
    Require(store->Commit(ticket.value(), "{}", "[]", root).ok(), "shared-root commit failed");
  }
  root.reset();
  auto pinned = store->Get(id('a'), "owner");
  Require(pinned.ok(), "pin failed");
  Require(!store->Get(id('a'), "other").ok(), "response namespace leaked");
  auto lease = sessions->Acquire(session.value().id, "owner", 128);
  Require(lease.ok(), "active root lease failed");
  auto reserved = store->Reserve(id('c'), "owner", 1024);
  Require(reserved.ok(), "eviction with pinned parent failed");
  Require(store->Get(id('a'), "owner").ok() && !store->Get(id('b'), "owner").ok(),
          "pinned response evicted");
  Require(store->Delete(id('a'), "owner").ok(), "pinned deletion failed");
  pinned = Status{StatusCode::kNotFound, "release pin"};
  Require(store->Stats().first == 1, "deleted pin not pruned");
  Require(!sessions->Get(session.value().id, "owner").ok(), "deleted root still visible");
  lease = Status{StatusCode::kNotFound, "release lease"};
  Require(sessions->Stats().first == 0, "deleted active root not released after lease");
  reserved = Status{StatusCode::kNotFound, "rollback"};
  Require(store->Stats() == std::pair<std::size_t, std::size_t>{0, 0},
          "reservation rollback leaked bytes");
  // Parallel reservations must honor both entry and byte limits.
  std::vector<std::future<bool>> futures;
  std::promise<void> finish;
  auto released = finish.get_future().share();
  for (char c : {'1', '2', '3', '4'})
    futures.push_back(std::async(std::launch::async, [&, c] {
      auto ticket = store->Reserve(id(c), "owner", 1024);
      released.wait();
      return ticket.ok();
    }));
  const auto until = Clock::now() + 1s;
  while (store->Stats().first < 2 && Clock::now() < until)
    std::this_thread::sleep_for(1ms);
  Require(store->Stats().first == 2, "parallel reservations failed to fill capacity");
  finish.set_value();
  for (auto &future : futures)
    (void)future.get();
  Require(store->Stats().first == 0, "parallel reservation rollback failed");
  auto ttl_ticket = store->Reserve(id('d'), "owner", 1024);
  Require(ttl_ticket.ok(), "TTL reservation failed");
  Require(store->Commit(ttl_ticket.value(), "{}", "[]", {}).ok(), "TTL commit failed");
  now += 41ms;
  Require(!store->Get(id('d'), "owner").ok() && store->Stats().first == 0,
          "expired response retained");
}
void TestToolFrames(text::Tokenizer &tokenizer) {
  const auto request_body =
      R"({"model":"mellum21-exl3-k4-research","messages":[{"role":"user","content":"use lookup"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"text":{"type":"string"}}}}}],"max_completion_tokens":512,"stream":true})";
  for (bool overflow : {false, true}) {
    auto state = std::make_shared<MockState>();
    state->tokens.clear();
    auto output = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"text\":\"" +
                  std::string(overflow ? 65000 : 4, 'x') + "\"}}</tool_call>";
    auto encoded = tokenizer.Encode(output);
    state->tokens = std::move(encoded).value();
    state->tokens.push_back(28);
    auto request = ParseChatRequest(request_body, tokenizer, 8192, "tool-stream");
    Require(request.ok(), "tool request invalid");
    request.value().outputs = overflow ? 70000 : 512;
    ServerLimits limits;
    limits.queued_tokens = 1000000;
    Coordinator coordinator(Runtime(state), tokenizer, limits);
    auto job = Submit(coordinator, std::move(request).value());
    Wait(job);
    auto frames = Drain(job);
    if (overflow)
      Require(!job->failure.ok() && job->cancellation.load() == CancelCause::Backpressure &&
                  frames.find("stream_backpressure") != std::string::npos,
              "tool burst overflow falsely succeeded");
    else
      Require(job->failure.ok() &&
                  frames.find("\"finish_reason\":\"tool_calls\"") != std::string::npos &&
                  frames.find("lookup") != std::string::npos &&
                  frames.find("[DONE]") != std::string::npos,
              "tool stream ending incomplete");
  }
}
void TestMetricsAndIdentity(text::Tokenizer &tokenizer) {
  HttpFixture fixture(tokenizer, std::make_shared<MockState>());
  httplib::Client client("127.0.0.1", fixture.port);
  auto models = JsonResponse(client.Get("/v1/models"));
  const auto &model = models.find("data")->as_array()[0];
  Require(model.find("checkpoint_sha256")->is_null() && model.find("manifest_sha256")->is_null(),
          "mock advertises hardcoded artifact hashes");
  Require(model.find("build_source_sha256")->as_string().size() == 64 &&
              !model.find("build_version")->as_string().empty(),
          "build identity missing");
  auto job = Submit(fixture.coordinator, Request("metric"));
  Wait(job);
  job->Cancel(CancelCause::Disconnect);
  job->Cancel(CancelCause::Disconnect);
  auto metrics = fixture.coordinator.Metrics();
  Require(metrics.disconnects == 1 && metrics.output_tokens == 1 &&
              metrics.queue_latency.count == 1 && metrics.lifetime_latency.count == 1,
          "metric counts incorrect");
  auto response = client.Get("/metrics");
  Require(response && response->body.find("mellum_queue_wait_seconds_bucket{le=\"+Inf\"} 1") !=
                          std::string::npos,
          "latency histogram missing");
}

void TestHttpDisconnect(text::Tokenizer &tokenizer) {
  for (bool responses : {false, true}) {
    auto state = std::make_shared<MockState>();
    state->tokens.assign(500, 0);
    state->token_delay = 2ms;
    HttpFixture fixture(tokenizer, state);
    httplib::Client client("127.0.0.1", fixture.port);
    client.set_read_timeout(2);
    const auto body =
        responses
            ? std::string(
                  R"({"model":"mellum21-exl3-k4-research","input":"hello","max_output_tokens":512,"stream":true})")
            : std::string(
                  R"({"model":"mellum21-exl3-k4-research","messages":[{"role":"user","content":"hello"}],"max_completion_tokens":512,"stream":true})");
    auto result =
        client.Post(responses ? "/v1/responses" : "/v1/chat/completions", httplib::Headers{}, body,
                    "application/json", [](const char *, std::size_t) { return false; });
    Require(!result, "disconnect client unexpectedly consumed full stream");
    const auto until = std::chrono::steady_clock::now() + 2s;
    while (fixture.coordinator.Metrics().active && std::chrono::steady_clock::now() < until)
      std::this_thread::sleep_for(2ms);
    const auto metrics = fixture.coordinator.Metrics();
    Require(!metrics.active && metrics.cancelled == 1 && metrics.disconnects == 1 &&
                metrics.output_tokens < 500,
            "HTTP disconnect did not promptly cancel generation");
    Require(fixture.coordinator.Sessions().Stats().first == 0, "disconnected response leaked root");
  }
}

void TestHttpGenerationDeadline(text::Tokenizer &tokenizer) {
  auto state = std::make_shared<MockState>();
  state->token_delay = 10ms;
  ServerLimits limits;
  limits.generation_timeout = 1ms;
  HttpFixture fixture(tokenizer, state, limits);
  httplib::Client client("127.0.0.1", fixture.port);
  for (bool responses : {false, true}) {
    auto result = client.Post(responses ? "/v1/responses" : "/v1/chat/completions",
                              responses ? ResponseBody : ChatBody, "application/json");
    auto parsed = JsonResponse(result, 504);
    Require(parsed.find("error")->find("code")->as_string() == "generation_deadline",
            "generation deadline code differs");
    Require(!result->has_header("Retry-After"), "generation deadline advertised retry");
  }
  Require(fixture.coordinator.Sessions().Stats().first == 0, "deadline response leaked root");
}
int main() {
  try {
    auto loaded = text::Tokenizer::Load({}, {});
    Require(loaded.ok(), "synthetic fixture failed");
    auto &tokenizer = *loaded.value();
    TestQueueDeadline(tokenizer);
    std::cout << "PASS independent queue expiry and resource release\n";
    for (auto cause : {CancelCause::Disconnect, CancelCause::Shutdown, CancelCause::SessionDiscard})
      TestQueuedCancellation(tokenizer, cause);
    std::cout << "PASS queued disconnect, shutdown and session discard\n";
    TestEndingPublishedAfterPollTimeout();
    TestStreamingEndings(tokenizer);
    std::cout << "PASS full-ring stop/EOS/length/usage, disconnect and backpressure\n";
    TestControlCases(tokenizer);
    std::cout << "PASS JSON, Unicode, stop/thinking and exclusive namespaces/leases\n";
    TestHttpRetention(tokenizer);
    std::cout << "PASS HTTP 40 stored roots, lineage, eviction, deletion and invalid requests\n";
    TestHttpQueueErrors(tokenizer);
    std::cout << "PASS HTTP Chat/Responses queue deadlines and rollback\n";
    TestHttpStreams(tokenizer);
    std::cout << "PASS HTTP Chat/Responses streams\n";
    TestGracefulDrain(tokenizer);
    TestStoreLifetimes();
    TestToolFrames(tokenizer);
    TestMetricsAndIdentity(tokenizer);
    TestHttpDisconnect(tokenizer);
    TestHttpGenerationDeadline(tokenizer);
    std::cout << "PASS bounded drain, pinned/shared/expired roots, concurrent rollback, tool "
                 "frames and metrics/identity\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "TEST ERROR: " << error.what() << '\n';
    return 1;
  }
}
