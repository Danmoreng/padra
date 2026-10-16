// Deterministic, synthetic model/text fixture; never opens a model artifact.
// All HTTP, retention, streaming and request validation are production code.
#include "server/http_server.h"
#include <charconv>
#include <csignal>
#include <iostream>
#include <condition_variable>
#include <mutex>
using namespace mellum16;
namespace {
volatile std::sig_atomic_t interrupted = 0;
void Signal(int) {
  interrupted = 1;
}
class ToolGate {
public:
  void Arm() {
    std::lock_guard lock(mutex_);
    released_ = false;
  }
  void Release() {
    std::lock_guard lock(mutex_);
    released_ = true;
    changed_.notify_all();
  }
  void Stop() {
    std::lock_guard lock(mutex_);
    stopped_ = true;
    changed_.notify_all();
  }
  bool Wait() {
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [&] { return released_ || stopped_; });
    return !stopped_;
  }

private:
  std::mutex mutex_;
  std::condition_variable changed_;
  bool released_ = false, stopped_ = false;
};
class FixtureBackend final : public RuntimeBackend {
public:
  FixtureBackend(const text::Tokenizer &tokenizer, std::shared_ptr<ToolGate> gate)
      : tokenizer_(tokenizer), gate_(std::move(gate)) {}
  Status Prepare(std::span<const int> ids, int outputs, bool, SamplingOptions) override {
    auto text = tokenizer_.Decode(ids);
    if (!text.ok())
      return text.status();
    outputs_ = outputs;
    pause_after_ = 0;
    delay_ = std::chrono::milliseconds(0);
    std::string output = "hello";
    // Real Pi CLI exercises its read tool and returns the result in a follow-up
    // request. This remains synthetic inference with production HTTP parsing.
    if (text.value().find("pi-read-fixture") != std::string::npos &&
        text.value().find("\"role\":\"tool\"") == std::string::npos)
      output = R"(<tool_call>{"name":"read","arguments":{"path":"probe.txt"}}</tool_call>)";
    if (text.value().find("tool-valid") != std::string::npos ||
        text.value().find("tool-slow") != std::string::npos ||
        text.value().find("tool-mixed") != std::string::npos) {
      output =
          R"(<tool_call>{"name":"lookup","arguments":{"text":"hello €","nested":[1,{"ok":true}]}}</tool_call>)";
      delay_ = std::chrono::milliseconds(2);
    }
    if (text.value().find("tool-mixed") != std::string::npos)
      output += "after";
    if (text.value().find("tool-reverse") != std::string::npos)
      output = R"(<tool_call>{"arguments":{"text":"reverse"},"name":"lookup"}</tool_call>)";
    if (text.value().find("tool-malformed") != std::string::npos)
      output = R"(<tool_call>{"name":"lookup","arguments":{"text":}}</tool_call>)";
    if (text.value().find("slow-text") != std::string::npos) {
      output = std::string(1000, 'x');
      delay_ = std::chrono::milliseconds(2);
    }
    if (text.value().find("queue-blocker") != std::string::npos) {
      output = std::string(12, 'x');
      delay_ = std::chrono::milliseconds(50);
    }
    if (text.value().find("reasoning-text") != std::string::npos)
      output = "<think>why €</think>\nanswer";
    if (text.value().find("utf8-stop") != std::string::npos)
      output = "€ hello STOP tail";
    if (text.value().find("pi-thinking-read-fixture") != std::string::npos) {
      if (text.value().find("\"role\":\"tool\"") == std::string::npos)
        output =
            R"(<think>inspect € probe</think><tool_call>{"name":"read","arguments":{"path":"probe.txt"}}</tool_call>)";
      else if (text.value().find("\"reasoning_content\":\"inspect € probe\"") == std::string::npos)
        return {StatusCode::kInvalidArgument,
                "Pi fixture lost full reasoning before tool continuation"};
    }
    if (text.value().find("pi-history-fixture") != std::string::npos) {
      if (text.value().find("pi-history-check") == std::string::npos)
        output = "<think>retained € trace</think>first";
      else {
        if (text.value().find("\"reasoning_content\":\"retained € trace\"") == std::string::npos)
          return {StatusCode::kInvalidArgument, "Pi fixture lost reasoning between user turns"};
        output = "history verified";
      }
    }
    if (text.value().find("slow-reader") != std::string::npos) {
      output = std::string(5000, 'x');
      delay_ = std::chrono::milliseconds(1);
    }
    if (text.value().find("tool-early") != std::string::npos) {
      const std::string prefix = R"(<tool_call>{"name":"lookup","arguments":{"text":"early)";
      output = prefix + R"( complete"}}</tool_call>)";
      pause_after_ = tokenizer_.Encode(prefix).value().size();
      gate_->Arm();
    }
    auto encoded = tokenizer_.Encode(output);
    if (!encoded.ok())
      return encoded.status();
    tokens_ = std::move(encoded).value();
    tokens_.push_back(28);
    return {};
  }
  Status Append(std::span<const int> ids, int outputs, bool logits,
                SamplingOptions sampling) override {
    return Prepare(ids, outputs, logits, sampling);
  }
  Result<InferenceResult> Generate(TokenCallback callback, PrefillCallback) override {
    InferenceResult result;
    for (std::size_t i = 0; i < tokens_.size() && i < static_cast<std::size_t>(outputs_); ++i) {
      if (delay_.count())
        std::this_thread::sleep_for(delay_);
      result.generated_ids.push_back(tokens_[i]);
      if (callback && !callback(tokens_[i])) {
        result.cancelled = true;
        break;
      }
      if (pause_after_ && i + 1 == pause_after_ && !gate_->Wait()) {
        result.cancelled = true;
        break;
      }
    }
    return result;
  }
  Status Reset() override {
    return {};
  }
  void Cancel() override {}
  RuntimeStats Stats() const override {
    return {};
  }

private:
  const text::Tokenizer &tokenizer_;
  std::vector<int> tokens_;
  int outputs_ = 1;
  std::size_t pause_after_ = 0;
  std::shared_ptr<ToolGate> gate_;
  std::chrono::milliseconds delay_{0};
};
} // namespace
int main(int argc, char **argv) {
  try {
    int port = 0;
    if (argc > 3 || (argc == 3 && std::string_view(argv[2]) != "--control-stdin"))
      throw std::runtime_error("usage: padra-sdk-fixture [PORT] [--control-stdin]");
    if (argc >= 2) {
      auto [end, error] =
          std::from_chars(argv[1], argv[1] + std::char_traits<char>::length(argv[1]), port);
      if (error != std::errc{} || *end || port < 0 || port > 65535)
        throw std::runtime_error("invalid fixture port");
    }
    auto loaded = text::Tokenizer::Load({}, {});
    if (!loaded.ok())
      throw std::runtime_error(loaded.status().message());
    auto tokenizer = std::move(loaded).value();
    server::ServerLimits limits;
    limits.queue_timeout = std::chrono::milliseconds(75);
    auto gate = std::make_shared<ToolGate>();
    auto runtime =
        std::make_unique<ModelRuntime>(std::make_unique<FixtureBackend>(*tokenizer, gate));
    server::Coordinator coordinator(std::move(runtime), *tokenizer, limits);
    server::HttpServer http(coordinator, *tokenizer, port);
    port = http.Bind();
    if (port < 1)
      throw std::runtime_error("fixture bind failed");
    std::signal(SIGINT, Signal);
    std::signal(SIGTERM, Signal);
    std::atomic<bool> finished{false};
    std::jthread stop([&](std::stop_token token) {
      while (!token.stop_requested() && !finished && !interrupted)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      if (interrupted) {
        gate->Stop();
        http.Stop();
      }
    });
    std::jthread control;
    if (argc == 3)
      control = std::jthread([gate] {
        std::string line;
        while (std::getline(std::cin, line))
          if (line == "release-tool")
            gate->Release();
        gate->Stop();
      });
    std::cout << "{\"port\":" << port << ",\"synthetic\":true}" << std::endl;
    const bool listened = http.Listen();
    finished = true;
    stop.join();
    coordinator.Shutdown();
    return listened ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
