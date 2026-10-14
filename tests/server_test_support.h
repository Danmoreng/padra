#pragma once
#include "server/coordinator.h"
#include "server/chat_stream.h"
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
  bool prepare_failure = false, reset_failure = false, generate_failure = false,
       prepare_block = false;
  int pause_after = 0;
  std::atomic<int> calls{0};
  std::mutex sampling_mutex;
  SamplingOptions sampling;
  std::vector<int> tokens{0};
  std::chrono::milliseconds token_delay{0};
};
class MockBackend final : public RuntimeBackend {
  std::shared_ptr<MockState> state_;
  int outputs_ = 0;

public:
  explicit MockBackend(std::shared_ptr<MockState> state) : state_(std::move(state)) {}
  Status Prepare(std::span<const int>, int outputs, bool, SamplingOptions sampling) override {
    {
      std::lock_guard lock(state_->sampling_mutex);
      state_->sampling = sampling;
    }
    if (state_->prepare_block) {
      state_->entered.set_value();
      state_->released.wait();
    }
    if (state_->prepare_failure)
      return {StatusCode::kInternal, "injected prepare failure"};
    outputs_ = outputs;
    return {};
  }
  Status Append(std::span<const int>, int outputs, bool, SamplingOptions sampling) override {
    {
      std::lock_guard lock(state_->sampling_mutex);
      state_->sampling = sampling;
    }
    outputs_ = outputs;
    return {};
  }
  Status Reset() override {
    if (state_->reset_failure)
      return {StatusCode::kInternal, "injected reset failure"};
    return {};
  }
  void Cancel() override {}
  RuntimeStats Stats() const override {
    return {};
  }
  Result<InferenceResult> Generate(TokenCallback callback, PrefillCallback) override {
    const bool first = state_->calls.fetch_add(1) == 0;
    if (first && state_->block) {
      state_->entered.set_value();
      state_->released.wait();
    }
    if (state_->generate_failure)
      return Status{StatusCode::kInternal, "injected generation failure"};
    InferenceResult result;
    for (int token : state_->tokens) {
      if (result.generated_ids.size() == static_cast<std::size_t>(outputs_))
        break;
      if (state_->token_delay.count())
        std::this_thread::sleep_for(state_->token_delay);
      result.generated_ids.push_back(token);
      if (callback && !callback(token)) {
        result.cancelled = true;
        break;
      }
      if (first && state_->pause_after == static_cast<int>(result.generated_ids.size())) {
        state_->entered.set_value();
        state_->released.wait();
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
inline void Require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
inline ChatRequest Request(std::string id) {
  ChatRequest request;
  request.id = std::move(id);
  request.input_ids = {0};
  request.outputs = 1;
  return request;
}
inline std::shared_ptr<Job> Submit(Coordinator &coordinator, ChatRequest request) {
  auto result = coordinator.Submit(std::move(request));
  Require(result.ok(), "unexpected admission failure");
  return std::move(result).value();
}
inline void Wait(const std::shared_ptr<Job> &job) {
  std::unique_lock lock(job->mutex);
  Require(job->changed.wait_for(lock, 2s, [&] { return job->complete; }),
          "test completion timeout");
}
inline std::string Drain(const std::shared_ptr<Job> &job) {
  GenerationEvent frame;
  std::string output;
  ChatStream encoder(*job);
  while (job->Pop(frame, 0ms)) {
    auto wire = encoder.Encode(frame);
    output.append(wire.data(), wire.size());
  }
  return output;
}
inline std::unique_ptr<ModelRuntime> Runtime(std::shared_ptr<MockState> state) {
  return std::make_unique<ModelRuntime>(std::make_unique<MockBackend>(std::move(state)));
}

class HttpFixture {
public:
  std::shared_ptr<MockState> state;
  Coordinator coordinator;
  HttpServer http;
  int port;
  std::thread listener;
  HttpFixture(text::Tokenizer &tokenizer, std::shared_ptr<MockState> input,
              ServerLimits limits = {}, HttpLimits http_limits = {})
      : state(input), coordinator(Runtime(input), tokenizer, limits),
        http(coordinator, tokenizer, 0, 8192, http_limits), port(http.Bind()) {
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
inline json::Value JsonResponse(const httplib::Result &response, int expected = 200) {
  Require(response && response->status == expected, "unexpected HTTP status");
  auto parsed = json::Parse(response->body);
  Require(parsed.ok(), "invalid HTTP JSON");
  return std::move(parsed).value();
}
template <class Predicate> void Eventually(Predicate predicate, const char *message) {
  const auto until = std::chrono::steady_clock::now() + 2s;
  while (!predicate()) {
    Require(std::chrono::steady_clock::now() < until, message);
    std::this_thread::yield();
  }
}
inline std::vector<json::Value> ReadEvents(std::string_view wire) {
  std::vector<json::Value> out;
  while (!wire.empty()) {
    const auto end = wire.find("\n\n");
    Require(end != wire.npos, "unterminated SSE block");
    const auto block = wire.substr(0, end);
    wire.remove_prefix(end + 2);
    const auto at = block.find("data: ");
    if (at == block.npos)
      continue;
    const auto data = block.substr(at + 6);
    if (data == "[DONE]")
      continue;
    auto parsed = json::Parse(data);
    Require(parsed.ok(), "invalid SSE JSON");
    out.push_back(std::move(parsed).value());
  }
  return out;
}
