#include "server/secure_id.h"
#include "server_test_support.h"
#include "tokenization_gate.h"
#include <set>
using namespace mellum16::server::test;

const char *Route(bool responses) {
  return responses ? "/v1/responses" : "/v1/chat/completions";
}
const char *Body(bool responses) {
  return responses ? ResponseBody : ChatBody;
}
ApiRoute RouteKind(bool responses) {
  return responses ? ApiRoute::Responses : ApiRoute::Chat;
}
void Expect(const httplib::Result &result, int status, std::string_view code, bool retry) {
  auto parsed = JsonResponse(result, status);
  Require(parsed.find("error")->find("code")->as_string() == code, "admission error code differs");
  Require(retry ? result->get_header_value("Retry-After") == "1"
                : !result->has_header("Retry-After"),
          "admission retry contract differs");
  const auto id = result->get_header_value("X-Request-Id");
  static std::set<std::string> ids;
  Require(id.starts_with("req_") && ids.insert(id).second, "request identity missing/reused");
}
void Count(Coordinator &coordinator, ApiRoute route, AdmissionCause cause, std::uint64_t expected) {
  const auto metrics = coordinator.Metrics();
  Require(metrics.rejections[static_cast<std::size_t>(route)][static_cast<std::size_t>(cause)] ==
              expected,
          "route/cause rejection counter differs");
  std::uint64_t sum = 0;
  for (const auto &row : metrics.rejections)
    for (auto value : row)
      sum += value;
  Require(sum == metrics.rejected, "total and bounded rejection counters diverged");
}
void WaitForTokenization(const std::shared_ptr<TokenizationBarrier> &barrier, unsigned count) {
  std::unique_lock lock(barrier->mutex);
  Require(barrier->changed.wait_for(lock, 2s, [&] { return barrier->entered >= count; }),
          "preparation barrier not reached");
}

void DrainRejections(text::Tokenizer &tokenizer) {
  HttpFixture fixture(tokenizer, std::make_shared<MockState>());
  fixture.coordinator.BeginDrain(true);
  httplib::Client client("127.0.0.1", fixture.port);
  for (bool responses : {false, true}) {
    auto result = client.Post(responses ? "/v1/responses" : "/v1/chat/completions",
                              responses ? ResponseBody : ChatBody, "application/json");
    auto body = JsonResponse(result, 503);
    Require(body.find("error")->find("code")->as_string() == "server_shutdown",
            "drain misclassified as queue capacity");
    Require(result->get_header_value("Retry-After") == "1" && result->has_header("X-Request-Id"),
            "early rejection missing retry/request identity");
    Count(fixture.coordinator, RouteKind(responses), AdmissionCause::Draining, 1);
  }
  Require(fixture.coordinator.Metrics().rejected == 2, "early rejection not counted exactly once");
  auto metrics = client.Get("/metrics");
  Require(metrics &&
              metrics->body.find(
                  "mellum_admission_rejected_total{route=\"chat\",cause=\"draining\"} 1\n") !=
                  std::string::npos &&
              metrics->body.find(
                  "mellum_admission_rejected_total{route=\"responses\",cause=\"draining\"} 1\n") !=
                  std::string::npos,
          "early rejections not exported with bounded labels");
}
void NotReady(text::Tokenizer &tokenizer) {
  auto state = std::make_shared<MockState>();
  state->prepare_failure = true;
  HttpFixture fixture(tokenizer, state);
  auto failed = Submit(fixture.coordinator, Request("readiness-failure"));
  Wait(failed);
  Require(!fixture.coordinator.CheckAdmission().allowed(),
          "runtime failure did not remove readiness");
  httplib::Client client("127.0.0.1", fixture.port);
  for (bool responses : {false, true}) {
    Expect(client.Post(Route(responses), Body(responses), "application/json"), 503,
           "model_not_ready", true);
    Count(fixture.coordinator, RouteKind(responses), AdmissionCause::NotReady, 1);
  }
  Require(fixture.coordinator.Metrics().rejected == 2 && fixture.coordinator.Metrics().failed == 1,
          "generation failure double-counted as admission rejection");
}
void FullQueue(text::Tokenizer &tokenizer) {
  auto state = std::make_shared<MockState>();
  state->block = true;
  ServerLimits limits;
  limits.queued_requests = 1;
  HttpFixture fixture(tokenizer, state, limits);
  ReleaseGuard release{state};
  auto first = Submit(fixture.coordinator, Request("active"));
  Require(state->entered.get_future().wait_for(2s) == std::future_status::ready,
          "queue blocker not entered");
  auto queued = Submit(fixture.coordinator, Request("queued"));
  httplib::Client client("127.0.0.1", fixture.port);
  for (bool responses : {false, true}) {
    Expect(client.Post(Route(responses), Body(responses), "application/json"), 503, "queue_full",
           true);
    Count(fixture.coordinator, RouteKind(responses), AdmissionCause::QueueFull, 1);
  }
  Require(fixture.coordinator.Metrics().rejected == 2,
          "full queue rejection counted twice/not counted");
}
void ResourceBudgets(text::Tokenizer &tokenizer) {
  for (bool bytes : {false, true}) {
    ServerLimits limits;
    if (bytes)
      limits.queued_bytes = 1;
    else
      limits.queued_tokens = 1;
    HttpFixture fixture(tokenizer, std::make_shared<MockState>(), limits);
    httplib::Client client("127.0.0.1", fixture.port);
    for (bool responses : {false, true}) {
      Expect(client.Post(Route(responses), Body(responses), "application/json"), 503,
             bytes ? "queue_bytes_exhausted" : "queue_tokens_exhausted", true);
      Count(fixture.coordinator, RouteKind(responses),
            bytes ? AdmissionCause::QueueBytes : AdmissionCause::QueueTokens, 1);
    }
    Require(fixture.coordinator.Metrics().rejected == 2 &&
                fixture.coordinator.Sessions().Stats().first == 0,
            "final byte/token rejection leaked reservation/root or double-counted");
  }
}
void PreparationLimit(text::Tokenizer &tokenizer, bool responses) {
  HttpFixture fixture(tokenizer, std::make_shared<MockState>());
  auto barrier = std::make_shared<TokenizationBarrier>();
  SetTokenizationBarrier(barrier);
  std::vector<std::future<httplib::Result>> requests;
  ReleaseTokenization release{barrier}; // destroyed before futures on an assertion
  for (int i = 0; i < 4; ++i)
    requests.push_back(std::async(std::launch::async, [&] {
      httplib::Client client("127.0.0.1", fixture.port);
      client.set_read_timeout(5);
      return client.Post(Route(responses), Body(responses), "application/json");
    }));
  WaitForTokenization(barrier, 4);
  httplib::Client client("127.0.0.1", fixture.port);
  Expect(client.Post(Route(responses), Body(responses), "application/json"), 503,
         "preparation_limited", true);
  Count(fixture.coordinator, RouteKind(responses), AdmissionCause::PreparationLimit, 1);
  barrier->Release();
  for (auto &request : requests)
    Require(JsonResponse(request.get()).is_object(), "prepared request failed");
  Require(fixture.coordinator.Metrics().rejected == 1, "preparation rejection counted twice");
}
void LateLoss(text::Tokenizer &tokenizer, bool responses, bool drain) {
  auto state = std::make_shared<MockState>();
  state->block = true;
  ServerLimits limits;
  limits.queued_requests = 1;
  HttpFixture fixture(tokenizer, state, limits);
  ReleaseGuard backend_guard{state};
  auto first = Submit(fixture.coordinator, Request("active"));
  Require(state->entered.get_future().wait_for(2s) == std::future_status::ready,
          "late-loss blocker not entered");
  auto barrier = std::make_shared<TokenizationBarrier>();
  SetTokenizationBarrier(barrier);
  auto request = std::async(std::launch::async, [&] {
    httplib::Client client("127.0.0.1", fixture.port);
    client.set_read_timeout(5);
    return client.Post(Route(responses), Body(responses), "application/json");
  });
  ReleaseTokenization release{barrier};
  WaitForTokenization(barrier, 1); // preflight succeeded; final admission still pending
  auto queued = Submit(fixture.coordinator, Request("fill-queue"));
  if (drain)
    fixture.coordinator.BeginDrain(true);
  barrier->Release();
  Expect(request.get(), 503, drain ? "server_shutdown" : "queue_full", true);
  Count(fixture.coordinator, RouteKind(responses),
        drain ? AdmissionCause::Draining : AdmissionCause::QueueFull, 1);
  Require(fixture.coordinator.Metrics().rejected == 1 &&
              fixture.coordinator.Sessions().Stats().first == 0,
          "late admission loss double-counted or leaked new root");
}
void SessionConflict(text::Tokenizer &tokenizer, bool responses, bool queued) {
  auto state = std::make_shared<MockState>();
  HttpFixture fixture(tokenizer, state);
  httplib::Client client("127.0.0.1", fixture.port);
  std::string body;
  if (responses) {
    const auto base = JsonResponse(client.Post(Route(true), ResponseBody, "application/json"));
    body =
        std::string(
            R"({"model":"mellum21-exl3-k4-research","input":"next","max_output_tokens":2,"previous_response_id":)") +
        json::Quote(base.find("id")->as_string()) + "}";
  } else {
    const auto session = JsonResponse(client.Post(
        "/v1/sessions", R"({"model":"mellum21-exl3-k4-research"})", "application/json"));
    body = ChatBody;
    body.insert(body.size() - 1, ",\"session_id\":" + json::Quote(session.find("id")->as_string()));
  }
  state->calls = 0;
  state->block = true;
  ReleaseGuard backend_guard{state};
  auto entered = state->entered.get_future();
  std::shared_ptr<Job> predecessor;
  if (queued) {
    predecessor = Submit(fixture.coordinator, Request("predecessor"));
    Require(entered.wait_for(2s) == std::future_status::ready,
            "session queue predecessor not entered");
  }
  auto first = std::async(std::launch::async, [&] {
    httplib::Client http("127.0.0.1", fixture.port);
    http.set_read_timeout(5);
    return http.Post(Route(responses), body, "application/json");
  });
  ReleaseGuard release_after_client{state};
  if (queued)
    Eventually([&] { return fixture.coordinator.Metrics().queued == 1; },
               "session not leased in queue");
  else
    Require(entered.wait_for(2s) == std::future_status::ready, "session active job not entered");
  Expect(client.Post(Route(responses), body, "application/json"), 409, "session_busy", true);
  Count(fixture.coordinator, RouteKind(responses), AdmissionCause::SessionBusy, 1);
  Require(fixture.coordinator.Metrics().rejected == 1, "session conflict double-counted");
  state->release.set_value();
  Require(JsonResponse(first.get()).is_object(), "leased session request failed");
}
void AdmittedDeadline(text::Tokenizer &tokenizer) {
  for (bool responses : {false, true})
    for (bool stream : {false, true}) {
      auto state = std::make_shared<MockState>();
      state->block = true;
      ServerLimits limits;
      limits.queue_timeout = 30ms;
      HttpFixture fixture(tokenizer, state, limits);
      ReleaseGuard release{state};
      auto first = Submit(fixture.coordinator, Request("deadline-predecessor"));
      Require(state->entered.get_future().wait_for(2s) == std::future_status::ready,
              "deadline blocker not entered");
      httplib::Client client("127.0.0.1", fixture.port);
      auto body = std::string(Body(responses));
      if (stream)
        body.insert(body.size() - 1, ",\"stream\":true");
      auto reply = client.Post(Route(responses), body, "application/json");
      Require(reply && reply->status == (stream ? 200 : 503) &&
                  reply->body.find("queue_deadline") != std::string::npos,
              "admitted deadline contract lost");
      Require(fixture.coordinator.Metrics().rejected == 0 &&
                  fixture.coordinator.Metrics().cancelled == 1,
              "admitted timeout counted as new admission rejection");
    }
}
void MissingSession(text::Tokenizer &tokenizer) {
  HttpFixture fixture(tokenizer, std::make_shared<MockState>());
  httplib::Client client("127.0.0.1", fixture.port);
  auto body = std::string(ChatBody);
  body.insert(body.size() - 1, ",\"session_id\":" + json::Quote(MakeSecureId("sess-").value()));
  Expect(client.Post(Route(false), body, "application/json"), 404, "session_not_found", false);
  Count(fixture.coordinator, ApiRoute::Chat, AdmissionCause::SessionMissing, 1);
}
void StoreDecisions() {
  auto sessions = std::make_shared<SessionStore>(SessionLimits{1, 1024, 1024});
  auto session = sessions->Create("owner").value();
  AdmissionDecision decision;
  auto lease = sessions->Acquire(session.id, "owner", 1024, &decision).value();
  Require(decision.allowed(), "successful lease retained stale rejection");
  Require(!sessions->Acquire(session.id, "owner", 1024, &decision).ok() &&
              decision.cause == AdmissionCause::SessionBusy,
          "session busy not structured");
  lease.reset();
  Require(!sessions->Acquire(session.id, "owner", 1025, &decision).ok() &&
              decision.cause == AdmissionCause::SessionBudget,
          "session byte limit confused with session busy");
  auto responses = std::make_shared<ResponseStore>(ResponseStoreLimits{1, 2048, 1024});
  const auto reserve = [&] {
    return responses->Reserve(MakeSecureId("resp_").value(), "owner", 1024, sessions, true, {},
                              &decision);
  };
  Require(!reserve().ok() && decision.cause == AdmissionCause::RootCapacity,
          "root capacity not structured");
  Require(sessions->Delete(session.id, "owner").ok(), "manual deletion failed");
  auto pending = reserve().value();
  Require(decision.allowed(), "successful reservation retained stale rejection");
  auto full = responses->Reserve(MakeSecureId("resp_").value(), "owner", 1024, sessions, false, {},
                                 &decision);
  Require(!full.ok() && decision.cause == AdmissionCause::ResponseStoreFull,
          "pending record quota not structured");
}
int main() {
  try {
    auto tokenizer = text::Tokenizer::Load({}, {});
    DrainRejections(*tokenizer.value());
    NotReady(*tokenizer.value());
    FullQueue(*tokenizer.value());
    ResourceBudgets(*tokenizer.value());
    for (bool responses : {false, true}) {
      PreparationLimit(*tokenizer.value(), responses);
      for (bool drain : {false, true})
        LateLoss(*tokenizer.value(), responses, drain);
      for (bool queued : {false, true})
        SessionConflict(*tokenizer.value(), responses, queued);
    }
    AdmittedDeadline(*tokenizer.value());
    MissingSession(*tokenizer.value());
    StoreDecisions();
    std::cout << "PASS admission error contracts and exactly-once rejection accounting\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "TEST ERROR: " << error.what() << '\n';
    return 1;
  }
}
