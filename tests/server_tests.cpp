#include "compiler/sha256.h"
#include "server_test_support.h"

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

void TestHttpDisconnect(text::Tokenizer &tokenizer, bool queued) {
  // Responses owns a temporary root; Chat also exercises an explicit session.
  for (int scenario : {0, 1, 2}) {
    const bool responses = scenario == 0, explicit_session = scenario == 2;
    auto state = std::make_shared<MockState>();
    state->block = true;
    state->tokens = {0, 28};
    HttpFixture fixture(tokenizer, state);
    ReleaseGuard release{state};
    auto entered = state->entered.get_future();
    std::shared_ptr<Job> predecessor;
    if (queued) {
      predecessor = Submit(fixture.coordinator, Request("blocked-predecessor"));
      Require(entered.wait_for(2s) == std::future_status::ready, "queue predecessor not started");
    }
    auto body = std::string(responses ? ResponseBody : ChatBody);
    body.insert(body.size() - 1, ",\"stream\":true");
    std::string session;
    const std::string credential = "anonymous-loopback";
    const auto owner = compiler::Sha256Hex(credential.data(), credential.size());
    if (explicit_session) {
      auto created = fixture.coordinator.Sessions().Create(owner);
      Require(created.ok(), "disconnect session setup failed");
      session = created.value().id;
      body.insert(body.size() - 1, ",\"session_id\":" + json::Quote(session));
    }
    httplib::Client client("127.0.0.1", fixture.port);
    client.set_read_timeout(3);
    auto response = std::async(std::launch::async, [&] {
      return client.Post(responses ? "/v1/responses" : "/v1/chat/completions", httplib::Headers{},
                         body, "application/json", [](const char *, std::size_t) { return true; });
    });
    ReleaseGuard release_after_client{state};
    if (queued) {
      Eventually([&] { return fixture.coordinator.Metrics().queued == 1; },
                 "disconnect request never entered queue");
    } else {
      Require(entered.wait_for(2s) == std::future_status::ready,
              "active disconnect backend did not start");
    }
    client.stop();
    Eventually([&] { return fixture.coordinator.Metrics().disconnects == 1; },
               "server did not observe socket disconnect");
    if (queued) {
      Eventually(
          [&] {
            const auto m = fixture.coordinator.Metrics();
            return m.cancelled == 1 && m.queued == 0 && m.queued_bytes == 0 && m.queued_tokens == 0;
          },
          "queued disconnect did not complete independently of predecessor");
      Require(state->calls == 1, "cancelled queued job reached backend");
    }
    state->release.set_value();
    Require(response.wait_for(2s) == std::future_status::ready, "disconnected HTTP client hung");
    Require(!response.get(), "disconnected client unexpectedly completed response");
    if (predecessor)
      Wait(predecessor);
    Eventually(
        [&] {
          const auto m = fixture.coordinator.Metrics();
          return !m.active && m.queued == 0 && m.queued_bytes == 0 && m.queued_tokens == 0 &&
                 m.cancelled == 1 && m.completed == (queued ? 1u : 0u) && m.failed == 0 &&
                 m.active_transports == 0 && m.write_failures == 1;
        },
        "disconnect lifecycle or transport accounting incomplete");
    if (explicit_session) {
      auto lease = fixture.coordinator.Sessions().Acquire(session, owner, 128);
      Require(lease.ok(), "disconnect leaked explicit session lease/reservation");
    } else {
      Eventually([&] { return fixture.coordinator.Sessions().Stats().first == 0; },
                 "disconnect leaked temporary root");
    }
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
struct CollectedTool {
  std::string id, name, arguments;
  int identities = 0;
};
std::array<CollectedTool, 8> CollectTools(std::string_view wire) {
  std::array<CollectedTool, 8> calls;
  for (std::size_t begin = 0; begin < wire.size();) {
    const auto end = wire.find("\n\n", begin);
    if (end == wire.npos)
      break;
    auto record = wire.substr(begin, end - begin);
    begin = end + 2;
    if (!record.starts_with("data: "))
      continue;
    record.remove_prefix(6);
    if (record == "[DONE]")
      continue;
    auto parsed = json::Parse(record);
    Require(parsed.ok(), "invalid SSE JSON");
    auto *choices = parsed.value().find("choices");
    if (!choices)
      continue;
    for (const auto &choice : choices->as_array()) {
      auto *delta = choice.find("delta");
      auto *tools = delta ? delta->find("tool_calls") : nullptr;
      if (!tools)
        continue;
      for (const auto &tool : tools->as_array()) {
        const auto i = tool.find("index")->as_integer();
        Require(i >= 0 && i < 8, "invalid tool index");
        auto &call = calls[static_cast<std::size_t>(i)];
        auto *function = tool.find("function");
        Require(function && function->is_object(), "missing tool function");
        if (auto *id = tool.find("id")) {
          Require(call.id.empty(), "duplicate streamed identity");
          call.id = id->as_string();
          ++call.identities;
        }
        if (auto *name = function->find("name"))
          call.name += name->as_string();
        if (auto *arguments = function->find("arguments"))
          call.arguments += arguments->as_string();
      }
    }
  }
  return calls;
}
void TestIncrementalToolParser() {
  ToolConfig config;
  config.parse = true;
  config.names = {"lookup", "next"};
  const std::vector<std::string> raw_calls = {
      R"({"name":"lookup","arguments":{"text":"a\"b\\c","list":[1,true,null,{"x":"<tool_call>"}]}})",
      R"({"arguments":{"nested":{"n":-2.5e+3}},"na\u006de":"loo\u006bup"})",
      R"({"name":"next","arguments":{}})"};
  for (const auto &raw : raw_calls) {
    const auto wire = "before<tool_call>" + raw + "</tool_call>after";
    for (std::size_t split = 0; split <= wire.size(); ++split) {
      ToolStreamParser parser(true, 4096);
      std::array<CollectedTool, 8> calls;
      parser.StreamDeltas(config,
                          [&](std::size_t i, std::string_view name, std::string_view arguments) {
                            if (!name.empty()) {
                              ++calls[i].identities;
                              calls[i].name += name;
                            }
                            calls[i].arguments += arguments;
                            return true;
                          });
      std::string visible(parser.Push(std::string_view(wire).substr(0, split)));
      parser.FlushDeltas();
      visible += parser.Push(std::string_view(wire).substr(split));
      parser.FlushDeltas();
      visible += parser.Finish();
      Require(!parser.error() && !parser.incomplete() && parser.count() == 1 &&
                  visible == "beforeafter",
              "split tool parsing failed");
      auto expected = ParseGeneratedTool(parser.raw(0), config, "call_test", true);
      Require(expected.ok(), "valid tool fixture rejected");
      Require(calls[0].identities == 1 && calls[0].name == expected.value().name &&
                  calls[0].arguments == expected.value().arguments,
              "incremental bytes differ from completed call");
    }
  }
  ToolStreamParser parallel(true, 4096);
  std::array<CollectedTool, 8> calls;
  parallel.StreamDeltas(config,
                        [&](std::size_t i, std::string_view name, std::string_view arguments) {
                          if (!name.empty()) {
                            ++calls[i].identities;
                            calls[i].name = name;
                          }
                          calls[i].arguments += arguments;
                          return true;
                        });
  for (const auto &raw : raw_calls) {
    parallel.Push("<tool_call>" + raw + "</tool_call>");
    parallel.FlushDeltas();
  }
  Require(!parallel.error() && parallel.count() == 3 && calls[2].name == "next" &&
              calls[1].arguments == "{\"nested\":{\"n\":-2.5e+3}}",
          "parallel tool indexes mixed");
  for (auto raw :
       {R"({"name":"missing","arguments":{}})", R"({"name":"lookup","name":"next","arguments":{}})",
        R"({"name":"lookup","arguments":[1]})", R"({"name":"lookup","arguments":{"x":]}})",
        R"({"name":"lookup","arguments":{},})"}) {
    ToolStreamParser parser(true, 4096);
    parser.StreamDeltas(config,
                        [](std::size_t, std::string_view, std::string_view) { return true; });
    parser.Push(std::string("<tool_call>") + raw + "</tool_call>");
    parser.FlushDeltas();
    Require(parser.error() != nullptr, "malformed/unoffered/duplicate streamed tool accepted");
  }
}
void TestToolDeltasBeforeCompletion(text::Tokenizer &tokenizer) {
  auto state = std::make_shared<MockState>();
  const auto prefix = R"(<tool_call>{"name":"lookup","arguments":{"text":"first)",
             suffix = R"( € second"}}</tool_call>)";
  auto encoded = tokenizer.Encode(std::string(prefix) + suffix);
  Require(encoded.ok(), "tool token fixture failed");
  state->tokens = std::move(encoded).value();
  state->tokens.push_back(28);
  state->pause_after = static_cast<int>(std::string_view(prefix).size());
  Coordinator coordinator(Runtime(state), tokenizer);
  ReleaseGuard release{state};
  auto request = Request("incremental-tool");
  request.stream = true;
  request.outputs = 256;
  request.tools.parse = true;
  request.tools.names = {"lookup"};
  auto job = Submit(coordinator, std::move(request));
  Require(state->entered.get_future().wait_for(2s) == std::future_status::ready,
          "tool generation did not pause");
  {
    std::lock_guard lock(job->mutex);
    Require(!job->complete, "tool finished before early delta check");
  }
  const auto early = Drain(job);
  auto calls = CollectTools(early);
  Require(calls[0].name == "lookup" && calls[0].arguments == R"({"text":"first)",
          "tool arguments were buffered until completion");
  state->release.set_value();
  Wait(job);
  Require(job->failure.ok(), "incremental tool generation failed");
  calls = CollectTools(early + Drain(job));
  Require(calls[0].identities == 1 && calls[0].id == job->tool_calls[0].id &&
              calls[0].arguments == job->tool_calls[0].arguments &&
              calls[0].arguments == R"({"text":"first € second"})",
          "tool stream identity/UTF8/arguments not preserved");
}
void TestHttpToolDeltasBeforeCompletion(text::Tokenizer &tokenizer) {
  for (bool responses : {false, true}) {
    auto state = std::make_shared<MockState>();
    const std::string prefix = R"(<tool_call>{"name":"lookup","arguments":{"text":"first)";
    auto encoded = tokenizer.Encode(prefix + R"( second"}}</tool_call>)");
    Require(encoded.ok(), "HTTP tool token fixture failed");
    state->tokens = std::move(encoded).value();
    state->tokens.push_back(28);
    state->pause_after = static_cast<int>(prefix.size());
    HttpFixture fixture(tokenizer, state);
    std::promise<void> observed;
    auto arrived = observed.get_future();
    auto client = std::async(std::launch::async, [&] {
      httplib::Client http("127.0.0.1", fixture.port);
      http.set_read_timeout(2);
      std::string wire;
      bool notified = false;
      const std::string body =
          responses
              ? R"({"model":"mellum21-exl3-k4-research","input":"use lookup","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],"max_output_tokens":256,"stream":true,"store":false})"
              : R"({"model":"mellum21-exl3-k4-research","messages":[{"role":"user","content":"use lookup"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"max_completion_tokens":256,"stream":true})";
      auto result = http.Post(
          responses ? "/v1/responses" : "/v1/chat/completions", httplib::Headers{}, body,
          "application/json", [&](const char *data, std::size_t size) {
            wire.append(data, size);
            if (!notified && wire.find(responses ? "response.function_call_arguments.delta"
                                                 : R"("arguments":"{)") != wire.npos) {
              notified = true;
              observed.set_value();
            }
            return true;
          });
      Require(result && result->status == 200, "HTTP incremental tool request failed");
      return wire;
    });
    ReleaseGuard release{state};
    Require(state->entered.get_future().wait_for(2s) == std::future_status::ready,
            "HTTP tool backend did not pause");
    Require(arrived.wait_for(1s) == std::future_status::ready,
            "HTTP tool arguments not delivered until generation completed");
    Require(client.wait_for(0ms) == std::future_status::timeout,
            "HTTP tool stream ended before backend resume");
    state->release.set_value();
    const auto wire = client.get();
    Require(wire.find(responses ? "response.completed" : "[DONE]") != wire.npos,
            "HTTP early tool stream lost completion");
  }
}
void TestPartialAndInvalidStreamedTools(text::Tokenizer &tokenizer) {
  for (const auto &output : std::vector<std::string>{
           R"(<tool_call>{"name":"lookup","arguments":{"text":"unfinished)",
           R"(<tool_call>{"name":"loo)",
           R"(<tool_call>{"name":"lookup","arguments":{"text":}}</tool_call>)"}) {
    auto state = std::make_shared<MockState>();
    auto tokens = tokenizer.Encode(output);
    state->tokens = std::move(tokens).value();
    const bool invalid = output.ends_with("</tool_call>");
    if (invalid)
      state->tokens.push_back(28);
    Coordinator coordinator(Runtime(state), tokenizer);
    auto session = coordinator.Sessions().Create("owner");
    Require(session.ok(), "partial tool session setup");
    auto request = Request("partial-tool");
    request.stream = true;
    request.outputs = 256;
    request.tools.parse = true;
    request.tools.names = {"lookup"};
    request.session_id = session.value().id;
    request.session_owner = "owner";
    auto job = Submit(coordinator, std::move(request));
    Wait(job);
    auto wire = Drain(job);
    if (invalid)
      Require(!job->failure.ok() && job->cancellation.load() == CancelCause::ToolInvalid &&
                  wire.find("invalid_tool_output") != wire.npos && wire.find("[DONE]") == wire.npos,
              "invalid streamed arguments reported success");
    else {
      auto calls = CollectTools(wire);
      Require(job->failure.ok() && job->partial_tool_call && job->finish_reason == "length" &&
                  job->tool_calls.size() == 1,
              "length-truncated tool state lost");
      Require(calls[0].identities == 1 && calls[0].arguments == job->tool_calls[0].arguments &&
                  calls[0].name == job->tool_calls[0].name,
              "partial tool identity/arguments duplicated or lost");
    }
    Require(coordinator.Sessions().Get(session.value().id, "owner").value().history == "[]",
            "partial/invalid tool committed session history");
  }
}
void TestTransportMetricAccounting() {
  auto counters = std::make_shared<TransportCounters>();
  {
    Job job(Request("write-accounting"), 4);
    job.transport = counters;
    job.StartTransport();
    job.StartTransport();
    job.RecordWrite(10, false);
    Require(counters->active == 1 && counters->first_visible_write_latency.count == 0,
            "protocol metadata counted as visible write");
    job.RecordWrite(20, true);
    job.RecordWrite(30, true);
    job.FinishTransport(true);
    job.FinishTransport(false);
    Require(counters->active == 0 && counters->responses_written == 1 &&
                counters->write_failures == 0 && counters->body_bytes == 60 &&
                counters->first_visible_write_latency.count == 1 &&
                counters->response_write_latency.count == 1,
            "transport accounting duplicated/lost writes");
  }
  {
    Job abandoned(Request("abandoned"), 4);
    abandoned.transport = counters;
    abandoned.StartTransport();
  }
  Require(counters->active == 0 && counters->write_failures == 1 &&
              counters->response_write_latency.count == 2,
          "abandoned transport leaked accounting");
}
void TestHttpWrittenMetrics(text::Tokenizer &tokenizer) {
  for (bool responses : {false, true})
    for (bool stream : {false, true})
      for (bool visible : {false, true}) {
        auto state = std::make_shared<MockState>();
        state->tokens = visible ? std::vector<int>{0, 28} : std::vector<int>{28};
        state->token_delay = 2ms;
        HttpFixture fixture(tokenizer, state);
        httplib::Client client("127.0.0.1", fixture.port);
        auto body = std::string(responses ? ResponseBody : ChatBody);
        if (stream)
          body.insert(body.size() - 1, ",\"stream\":true");
        auto result = client.Post(responses ? "/v1/responses" : "/v1/chat/completions", body,
                                  "application/json");
        Require(result && result->status == 200, "written-metric HTTP request failed");
        const auto until = std::chrono::steady_clock::now() + 1s;
        while (fixture.coordinator.Metrics().active_transports &&
               std::chrono::steady_clock::now() < until)
          std::this_thread::sleep_for(1ms);
        const auto metrics = fixture.coordinator.Metrics();
        Require(metrics.active_transports == 0 && metrics.responses_written == 1 &&
                    metrics.write_failures == 0 &&
                    metrics.response_body_bytes == result->body.size() &&
                    metrics.response_write_latency.count == 1,
                "HTTP write metric does not match sent response");
        Require(metrics.first_visible_write_latency.count == (visible ? 1u : 0u),
                "empty/protocol response misclassified as visible");
        if (visible)
          Require(metrics.first_visible_write_latency.sum >= .002 &&
                      metrics.response_write_latency.sum >= metrics.first_visible_write_latency.sum,
                  "written latency omits generation or exceeds full response");
      }
}
void TestHttpGracefulDrain(text::Tokenizer &tokenizer) {
  for (bool responses : {false, true})
    for (bool stream : {false, true}) {
      auto state = std::make_shared<MockState>();
      state->block = true;
      state->tokens = {0, 28};
      HttpFixture fixture(tokenizer, state);
      auto entered = state->entered.get_future();
      auto client = std::async(std::launch::async, [&] {
        httplib::Client http("127.0.0.1", fixture.port);
        http.set_read_timeout(2);
        auto body = std::string(responses ? ResponseBody : ChatBody);
        if (stream)
          body.insert(body.size() - 1, ",\"stream\":true");
        return http.Post(responses ? "/v1/responses" : "/v1/chat/completions", body,
                         "application/json");
      });
      auto stopped = std::async(std::launch::async, [&] {
        Require(entered.wait_for(2s) == std::future_status::ready,
                "HTTP drain backend never started");
        fixture.http.Stop();
      });
      ReleaseGuard release{state};
      const auto until = std::chrono::steady_clock::now() + 2s;
      while (!fixture.coordinator.Metrics().draining && std::chrono::steady_clock::now() < until)
        std::this_thread::sleep_for(1ms);
      Require(fixture.coordinator.Metrics().draining &&
                  stopped.wait_for(0ms) == std::future_status::timeout,
              "HTTP listener stopped without allowing active transport grace");
      state->release.set_value();
      auto result = client.get();
      Require(result && result->status == 200, "HTTP graceful drain lost active response");
      if (stream)
        Require(result->body.find(responses ? "response.completed" : "[DONE]") != result->body.npos,
                "HTTP graceful drain truncated stream completion");
      stopped.get();
      // Fixture teardown and independent shutdown callers may repeat Stop while
      // the listener thread is still leaving its task queue.
      auto repeated = std::async(std::launch::async, [&] { fixture.http.Stop(); });
      fixture.http.Stop();
      repeated.get();
      const auto metrics = fixture.coordinator.Metrics();
      Require(metrics.responses_written == 1 && metrics.write_failures == 0 &&
                  metrics.active_transports == 0,
              "drain ended before final transport write");
    }
}
void TestHttpDrainDeadline(text::Tokenizer &tokenizer) {
  auto state = std::make_shared<MockState>();
  state->block = true;
  state->tokens = {0, 28};
  ServerLimits limits;
  limits.drain_timeout = 30ms;
  HttpFixture fixture(tokenizer, state, limits);
  auto entered = state->entered.get_future();
  auto client = std::async(std::launch::async, [&] {
    httplib::Client http("127.0.0.1", fixture.port);
    http.set_read_timeout(2);
    return http.Post("/v1/chat/completions", ChatBody, "application/json");
  });
  ReleaseGuard release{state};
  Require(entered.wait_for(2s) == std::future_status::ready, "drain-deadline backend not entered");
  auto stopped = std::async(std::launch::async, [&] { fixture.http.Stop(); });
  Require(stopped.wait_for(1s) == std::future_status::ready,
          "HTTP stop exceeded grace waiting for a blocked backend");
  stopped.get();
  state->release.set_value();
  auto result = client.get();
  // After the grace deadline, shutdown may close the socket before an error can
  // be delivered. It must never deliver an apparently successful generation.
  Require(!result ||
              (result->status == 503 && result->body.find("server_shutdown") != std::string::npos),
          "drain deadline reported successful generation");
  const auto until = std::chrono::steady_clock::now() + 1s;
  ServerMetrics metrics;
  do {
    metrics = fixture.coordinator.Metrics();
    if (!metrics.active && !metrics.active_transports)
      break;
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < until);
  Require(!metrics.active && metrics.cancelled == 1 && metrics.active_transports == 0 &&
              metrics.responses_written + metrics.write_failures == 1,
          "drain deadline leaked job or transport accounting");
}
int main(int argc, char **argv) {
  try {
    auto loaded = text::Tokenizer::Load({}, {});
    Require(loaded.ok(), "synthetic fixture failed");
    auto &tokenizer = *loaded.value();
    if (argc == 2 && std::string_view(argv[1]) == "--transport") {
      TestTransportMetricAccounting();
      TestHttpWrittenMetrics(tokenizer);
      TestHttpGracefulDrain(tokenizer);
      TestHttpDrainDeadline(tokenizer);
      std::cout << "PASS HTTP transport accounting, graceful drain, deadline and repeated stop\n";
      return 0;
    }
    if (argc == 2 && (std::string_view(argv[1]) == "--disconnect-queue" ||
                      std::string_view(argv[1]) == "--disconnect-active")) {
      TestHttpDisconnect(tokenizer, std::string_view(argv[1]) == "--disconnect-queue");
      std::cout << "PASS " << argv[1] << '\n';
      return 0;
    }
    if (argc != 1)
      throw std::runtime_error("unknown test filter");
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
    TestIncrementalToolParser();
    TestToolDeltasBeforeCompletion(tokenizer);
    TestHttpToolDeltasBeforeCompletion(tokenizer);
    TestPartialAndInvalidStreamedTools(tokenizer);
    TestToolFrames(tokenizer);
    TestMetricsAndIdentity(tokenizer);
    TestHttpDisconnect(tokenizer, true);
    TestHttpDisconnect(tokenizer, false);
    TestHttpGenerationDeadline(tokenizer);
    TestTransportMetricAccounting();
    TestHttpWrittenMetrics(tokenizer);
    TestHttpGracefulDrain(tokenizer);
    TestHttpDrainDeadline(tokenizer);
    std::cout << "PASS bounded drain, pinned/shared/expired roots, concurrent rollback, tool "
                 "frames and metrics/identity\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "TEST ERROR: " << error.what() << '\n';
    return 1;
  }
}
