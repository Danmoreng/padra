#include "compiler/sha256.h"
#include "server_test_support.h"
#include <utility>
namespace {
struct Gate {
  std::promise<void> entered, release;
  std::shared_future<void> released = release.get_future().share();
};
struct ReleaseGate {
  std::shared_ptr<Gate> gate;
  ~ReleaseGate() {
    try {
      gate->release.set_value();
    } catch (const std::future_error &) {
    }
  }
};
struct CacheState {
  mutable std::mutex mutex;
  SessionState session{8192};
  KVCheckpointLedger ledger{8192, 37, KVCacheFormat::Q8Token512Block32F16ScaleRotatedV2};
  std::vector<int> prompt, tokens{0, 28};
  int outputs = 0, anchor = 0, cached = 0, resets = 0, attempts = 0, restores = 0;
  bool stale = false, enabled = true;
  std::shared_ptr<Gate> gate;
};
// The fake GPU uses the real prefix/identity ledger and lifecycle. In particular,
// Restore cannot succeed just because the coordinator decided a cache might fit.
class CacheBackend final : public RuntimeBackend {
public:
  explicit CacheBackend(std::shared_ptr<CacheState> state) : state_(std::move(state)) {}
  Status Prepare(std::span<const int> ids, int outputs, bool, SamplingOptions) override {
    return PrepareCheckpoint(ids, outputs, 0, false, {});
  }
  Status Append(std::span<const int>, int, bool, SamplingOptions) override {
    return {StatusCode::kUnsupported, "test does not support append"};
  }
  Status PrepareCheckpoint(std::span<const int> ids, int outputs, int anchor, bool,
                           SamplingOptions) override {
    std::lock_guard lock(state_->mutex);
    auto status = state_->session.Prepare(ids, outputs);
    if (!status.ok())
      return status;
    state_->ledger.Invalidate();
    state_->prompt.assign(ids.begin(), ids.end());
    state_->outputs = outputs;
    state_->anchor = CheckpointCaptureAnchor(anchor, 256, true);
    state_->cached = 0;
    return {};
  }
  Status RestoreCheckpoint(std::span<const int> ids, int outputs, const KVCheckpointHandle &handle,
                           int next, bool, SamplingOptions) override {
    std::lock_guard lock(state_->mutex);
    ++state_->attempts;
    if (state_->stale) {
      state_->ledger.Invalidate();
      state_->stale = false;
    }
    auto base = state_->ledger.RestoreBase(ids, handle);
    if (!base.ok())
      return base.status();
    auto status = state_->session.PrepareCheckpointRestore(ids, outputs, base.value());
    if (!status.ok())
      return status;
    ++state_->restores;
    state_->prompt.assign(ids.begin(), ids.end());
    state_->outputs = outputs;
    state_->anchor = CheckpointCaptureAnchor(next, 256, true);
    state_->cached = base.value();
    return {};
  }
  std::optional<KVCheckpointHandle> Checkpoint() const override {
    std::lock_guard lock(state_->mutex);
    return state_->ledger.handle();
  }
  RuntimeStats Stats() const override {
    std::lock_guard lock(state_->mutex);
    RuntimeStats stats;
    stats.checkpoint_enabled = state_->enabled;
    stats.cached_tokens = state_->cached;
    stats.phase = state_->session.phase();
    return stats;
  }
  Status Reset() override {
    std::lock_guard lock(state_->mutex);
    ++state_->resets;
    state_->ledger.Invalidate();
    state_->cached = 0;
    return state_->session.Reset();
  }
  void Cancel() override {}
  Result<InferenceResult> Generate(TokenCallback callback, PrefillCallback prefill) override {
    std::shared_ptr<Gate> gate;
    std::vector<int> tokens;
    int outputs;
    {
      std::lock_guard lock(state_->mutex);
      Require(state_->session.Begin().ok(), "invalid fake GPU Begin");
      Require(state_->session.CommitPrefill().ok(), "invalid fake GPU prefill");
      if (state_->anchor)
        Require(state_->ledger.Capture(state_->prompt, state_->anchor).ok(),
                "invalid fake GPU capture");
      gate = std::exchange(state_->gate, {});
      tokens = state_->tokens;
      outputs = state_->outputs;
    }
    if (gate) {
      gate->entered.set_value();
      gate->released.wait();
    }
    InferenceResult result;
    if (prefill && !prefill(1)) {
      std::lock_guard lock(state_->mutex);
      state_->session.Cancel();
      result.cancelled = true;
      return result;
    }
    for (int token : tokens) {
      if (result.generated_ids.size() == static_cast<std::size_t>(outputs))
        break;
      {
        std::lock_guard lock(state_->mutex);
        Require(state_->session.Emit(token).ok(), "invalid fake GPU token");
      }
      result.generated_ids.push_back(token);
      if (callback && !callback(token)) {
        std::lock_guard lock(state_->mutex);
        state_->session.Cancel();
        result.cancelled = true;
        break;
      }
      if (token == 28)
        break;
    }
    std::lock_guard lock(state_->mutex);
    state_->session.Finish();
    return result;
  }

private:
  std::shared_ptr<CacheState> state_;
};
std::unique_ptr<ModelRuntime> Runtime(std::shared_ptr<CacheState> state) {
  return std::make_unique<ModelRuntime>(std::make_unique<CacheBackend>(std::move(state)));
}
struct CacheHttp {
  std::shared_ptr<CacheState> state = std::make_shared<CacheState>();
  Coordinator coordinator;
  HttpServer http;
  int port;
  std::thread listener;
  explicit CacheHttp(text::Tokenizer &tokenizer)
      : coordinator(Runtime(state), tokenizer), http(coordinator, tokenizer, 0), port(http.Bind()),
        listener([this] { http.Listen(); }) {
    Require(port > 0, "cache fixture bind failed");
    Eventually(
        [&] {
          httplib::Client client("127.0.0.1", port);
          auto result = client.Get("/healthz");
          return result && result->status == 200;
        },
        "cache fixture startup failed");
  }
  ~CacheHttp() {
    http.Stop();
    listener.join();
  }
};
std::string Body(bool responses, char first = 'a', const std::string &session = {}) {
  std::string content(800, 'a');
  content[0] = first;
  auto body = std::string(R"({"model":"mellum21-exl3-k4-research",")");
  body = "{\"model\":\"mellum21-exl3-k4-research\",";
  body += responses
              ? "\"input\":" + json::Quote(content) + ",\"store\":false,\"max_output_tokens\":2"
              : "\"messages\":[{\"role\":\"user\",\"content\":" + json::Quote(content) +
                    "}],\"max_tokens\":2";
  if (!session.empty())
    body += ",\"session_id\":" + json::Quote(session);
  return body + "}";
}
void HttpOwnership(text::Tokenizer &tokenizer) {
  CacheHttp fixture(tokenizer);
  httplib::Client client("127.0.0.1", fixture.port);
  client.set_read_timeout(3);
  const auto post = [&](bool responses, const std::string &owner, char first = 'a',
                        const std::string &session = {}) {
    auto response = client.Post(responses ? "/v1/responses" : "/v1/chat/completions",
                                {{"Authorization", "Bearer " + owner}},
                                Body(responses, first, session), "application/json");
    auto parsed = JsonResponse(response);
    if (session.empty())
      Require(!parsed.find("session_id"), "anonymous cache exposed a session ID");
    const auto *usage = parsed.find("usage");
    const int cached =
        static_cast<int>(usage->find(responses ? "input_tokens_details" : "prompt_tokens_details")
                             ->find("cached_tokens")
                             ->as_integer());
    if (responses)
      Require(client.Get("/v1/responses/" + parsed.find("id")->as_string(),
                         {{"Authorization", "Bearer " + owner}})
                      ->status == 404,
              "store:false anonymously retained a response");
    return cached;
  };
  Require(post(false, "A") == 0 && post(false, "A") == 768,
          "stateless Chat did not reuse a complete expert chunk prefix");
  auto followup = json::Parse(Body(false)).value().as_object();
  auto messages = followup.at("messages").as_array();
  for (const auto *message :
       {R"({"role":"assistant","content":"x"})", R"({"role":"user","content":"Continue."})"})
    messages.push_back(json::Parse(message).value());
  followup.insert_or_assign("messages", json::Value(std::move(messages)));
  auto continued = JsonResponse(client.Post("/v1/chat/completions", {{"Authorization", "Bearer A"}},
                                            json::Stringify(json::Value(std::move(followup))),
                                            "application/json"));
  Require(
      continued.find("usage")->find("prompt_tokens_details")->find("cached_tokens")->as_integer() ==
          768,
      "full-history follow-up failed to reuse the stable prompt prefix");
  Require(post(false, "B") == 0 && post(false, "B") == 768 && post(false, "A") == 0,
          "anonymous cache crossed credential namespaces");
  Require(post(false, "A", 'b') == 0 && post(false, "A", 'b') == 768,
          "changed canonical prefix reused stale KV");
  Require(post(true, "A", 'c') == 0 && post(true, "A", 'c') == 768,
          "stateless Responses did not reuse its canonical prefix");
  Require(post(true, "B", 'c') == 0, "Responses cache crossed credential namespaces");
  Require(fixture.coordinator.Sessions().Stats().first == 0,
          "automatic cache created logical sessions");
  const std::string credential = "Bearer A";
  const auto owner = compiler::Sha256Hex(credential.data(), credential.size());
  auto session = fixture.coordinator.Sessions().Create(owner).value();
  Require(post(false, "A", 'a', session.id) == 0 && post(false, "A", 'a', session.id) == 768,
          "explicit session cache regressed");
  Require(post(false, "A", 'a') == 0 && post(false, "A", 'a', session.id) == 0,
          "explicit and anonymous physical cache namespaces collided");
  Require(fixture.coordinator.DiscardSession(session.id, owner).ok(), "session discard failed");
}
ChatRequest CachedRequest(const text::Tokenizer &tokenizer, std::string id) {
  auto parsed = ParseChatRequest(Body(false), tokenizer, 8192, std::move(id));
  Require(parsed.ok(), "cache request parse failed");
  auto request = std::move(parsed).value();
  request.automatic_prefix_cache = true;
  request.session_owner = "owner";
  return request;
}
void Lifecycle(text::Tokenizer &tokenizer) {
  auto state = std::make_shared<CacheState>();
  Coordinator coordinator(Runtime(state), tokenizer);
  const auto run = [&](std::string id, bool hit) {
    auto job = Submit(coordinator, CachedRequest(tokenizer, std::move(id)));
    Wait(job);
    Require(job->failure.ok() && job->cached_tokens == (hit ? 768 : 0),
            "wrong cache lifecycle outcome");
  };
  run("cold", false);
  run("hit", true);
  {
    std::lock_guard lock(state->mutex);
    state->stale = true;
  }
  run("stale-fallback", false);
  run("after-stale", true);
  auto changed = CachedRequest(tokenizer, "canonical-change");
  changed.input_ids[0] ^= 1;
  auto changed_job = Submit(coordinator, std::move(changed));
  Wait(changed_job);
  Require(changed_job->failure.ok() && changed_job->cached_tokens == 0,
          "different canonical tools/system prefix reused KV");
  run("original-after-change", false);
  for (bool cancel_active : {false, true}) {
    auto gate = std::make_shared<Gate>();
    ReleaseGate release{gate};
    {
      std::lock_guard lock(state->mutex);
      state->gate = gate;
    }
    auto active = Submit(coordinator, CachedRequest(tokenizer, "active"));
    Require(gate->entered.get_future().wait_for(2s) == std::future_status::ready,
            "cache cancellation fixture did not enter generation");
    if (cancel_active)
      active->Cancel(CancelCause::Disconnect);
    else {
      auto queued = Submit(coordinator, CachedRequest(tokenizer, "queued"));
      queued->Cancel(CancelCause::Disconnect);
      Wait(queued);
    }
    gate->release.set_value();
    Wait(active);
    Require(active->failure.ok() != cancel_active, "wrong active cancellation outcome");
    run("after-cancel", !cancel_active);
  }
  const auto partial =
      tokenizer.Encode(R"(<tool_call>{"name":"lookup","arguments":{"text":"x)").value();
  for (bool malformed : {false, true}) {
    auto request = CachedRequest(tokenizer, "invalid-tool");
    request.tools = {{"lookup"}, true, true};
    auto tokens = partial;
    if (malformed) {
      const auto suffix = tokenizer.Encode("\"]}").value();
      tokens.insert(tokens.end(), suffix.begin(), suffix.end());
    }
    request.outputs = static_cast<int>(tokens.size());
    {
      std::lock_guard lock(state->mutex);
      state->tokens = tokens;
    }
    auto job = Submit(coordinator, std::move(request));
    Wait(job);
    Require(malformed ? !job->failure.ok() : job->partial_tool_call,
            "test did not produce malformed/partial tool output");
    {
      std::lock_guard lock(state->mutex);
      state->tokens = {0, 28};
    }
    run("after-invalid-tool", false);
  }
}
} // namespace
int main() {
  try {
    auto tokenizer = text::Tokenizer::Load({}, {});
    Require(tokenizer.ok(), "synthetic tokenizer failed");
    HttpOwnership(*tokenizer.value());
    Lifecycle(*tokenizer.value());
    std::cout
        << "PASS stateless prefix ownership, canonical identity, bounded retention and recovery\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "TEST ERROR: " << error.what() << '\n';
    return 1;
  }
}
