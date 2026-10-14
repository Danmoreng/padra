#include "server_test_support.h"
#include "server/stream_heartbeat.h"
#include <barrier>
#ifndef _WIN32
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
namespace {
class RawConnection {
public:
  explicit RawConnection(int port, bool small_receive = false) {
    socket_ = ::socket(AF_INET, SOCK_STREAM, 0);
    Require(socket_ != INVALID_SOCKET, "raw socket creation failed");
    if (small_receive) {
      const int bytes = 1024;
      (void)setsockopt(socket_, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char *>(&bytes),
                       sizeof(bytes));
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<unsigned short>(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(socket_, reinterpret_cast<sockaddr *>(&address), sizeof(address))) {
      Close();
      throw std::runtime_error("raw socket connection failed");
    }
  }
  ~RawConnection() {
    Close();
  }
  RawConnection(const RawConnection &) = delete;
  void Send(std::string_view bytes) {
    while (!bytes.empty()) {
#ifdef MSG_NOSIGNAL
      constexpr int flags = MSG_NOSIGNAL;
#else
      constexpr int flags = 0;
#endif
      const auto count = ::send(socket_, bytes.data(), static_cast<int>(bytes.size()), flags);
      Require(count > 0, "raw socket write failed");
      bytes.remove_prefix(static_cast<std::size_t>(count));
    }
  }

private:
  void Close() {
    if (socket_ == INVALID_SOCKET)
      return;
#ifdef _WIN32
    ::shutdown(socket_, SD_BOTH);
    ::closesocket(socket_);
#else
    ::shutdown(socket_, SHUT_RDWR);
    ::close(socket_);
#endif
    socket_ = INVALID_SOCKET;
  }
  socket_t socket_ = INVALID_SOCKET;
};
std::string Headers(int port, std::string_view route, std::size_t length) {
  return "POST " + std::string(route) + " HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(port) +
         "\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(length) +
         "\r\n\r\n";
}
int HttpStatus(httplib::Result result) {
  Require(static_cast<bool>(result), "HTTP operation returned no response");
  return result->status;
}
void HeartbeatClock() {
  using Clock = StreamHeartbeat::Clock;
  const auto zero = Clock::time_point{};
  StreamHeartbeat heartbeat(25ms, zero);
  Require(!heartbeat.Due(zero + 24ms) && heartbeat.Due(zero + 25ms),
          "heartbeat due boundary wrong");
  heartbeat.Wrote(zero + 25ms);
  Require(!heartbeat.Due(zero + 49ms) && heartbeat.Due(zero + 50ms), "heartbeat write reset wrong");
}
void HeartbeatDuringPreparation(text::Tokenizer &tokenizer, bool responses) {
  auto state = std::make_shared<MockState>();
  state->prepare_block = true;
  state->tokens = {0, 28};
  HttpLimits http;
  http.heartbeat = 20ms;
  http.poll = 2ms;
  HttpFixture fixture(tokenizer, state, {}, http);
  auto entered = state->entered.get_future();
  std::promise<void> heartbeat;
  auto received = heartbeat.get_future();
  httplib::Client client("127.0.0.1", fixture.port);
  client.set_read_timeout(3);
  std::string body(responses ? ResponseBody : ChatBody);
  body.insert(body.size() - 1, ",\"stream\":true");
  auto reply = std::async(std::launch::async, [&] {
    bool signalled = false;
    std::string wire;
    return client.Post(responses ? "/v1/responses" : "/v1/chat/completions", httplib::Headers{},
                       body, "application/json", [&](const char *bytes, std::size_t size) {
                         wire.append(bytes, size);
                         if (!signalled && wire.find(": heartbeat\n\n") != std::string::npos) {
                           signalled = true;
                           heartbeat.set_value();
                         }
                         return true;
                       });
  });
  ReleaseGuard release_after_client{state};
  Require(entered.wait_for(2s) == std::future_status::ready, "preparation did not reach barrier");
  Require(received.wait_for(2s) == std::future_status::ready,
          "blocked preparation received no heartbeat");
  Require(fixture.coordinator.Metrics().active && fixture.coordinator.Metrics().output_tokens == 0,
          "heartbeat test did not hold generation before first token");
  httplib::Client admin("127.0.0.1", fixture.http.AdminPort());
  admin.set_read_timeout(1);
  Require(HttpStatus(admin.Get("/readyz")) == 200, "preparation blocked admin readiness");
  state->release.set_value();
  Require(reply.wait_for(2s) == std::future_status::ready, "heartbeat request did not complete");
  auto result = reply.get();
  Require(result && result->status == 200, "heartbeat transport failed after release");
}
void SlowBodiesAndAdmin(text::Tokenizer &tokenizer) {
  auto state = std::make_shared<MockState>();
  HttpFixture fixture(tokenizer, state);
  std::vector<std::unique_ptr<RawConnection>> sockets;
  for (int i = 0; i < 16; ++i) {
    auto connection = std::make_unique<RawConnection>(fixture.port);
    connection->Send(Headers(fixture.port, "/v1/chat/completions", 1000) + "{");
    sockets.push_back(std::move(connection));
  }
  Eventually([&] { return fixture.http.Connections().active == 16; },
             "slow bodies did not fill API workers");
  httplib::Client api("127.0.0.1", fixture.port);
  api.set_read_timeout(0, 100000);
  Require(!api.Get("/healthz"), "starvation fixture failed to occupy shared API pool");
  for (int i = 0; i < 48; ++i) {
    auto connection = std::make_unique<RawConnection>(fixture.port);
    // Accepted excess sockets may already be closed; their connection alone
    // exercises bounded enqueue/rejection, without relying on a send race.
    sockets.push_back(std::move(connection));
  }
  Eventually(
      [&] {
        auto c = fixture.http.Connections();
        return c.rejected > 0 && c.queued == 32;
      },
      "connection queue did not enforce its configured limit");
  httplib::Client admin("127.0.0.1", fixture.http.AdminPort());
  admin.set_read_timeout(0, 300000);
  Require(HttpStatus(admin.Get("/healthz")) == 200 && HttpStatus(admin.Get("/readyz")) == 200,
          "API slow bodies displaced reserved admin workers");
  const auto config = JsonResponse(admin.Get("/config"));
  Require(config.find("server_config")->find("admin_port")->as_integer() ==
                  fixture.http.AdminPort() &&
              config.find("build_source_sha256")->as_string().size() == 64,
          "resolved config/build identity missing from admin listener");
  auto metrics = admin.Get("/metrics");
  Require(metrics && metrics->body.find("padra_http_active_connections{listener=\"api\"} 16") !=
                         std::string::npos,
          "connection accounting missing under saturated API pool");
  fixture.coordinator.BeginDrain(true);
  Require(HttpStatus(admin.Get("/readyz")) == 503 && HttpStatus(admin.Get("/healthz")) == 200,
          "drain readiness/liveness contract failed under connection load");
  Require(HttpStatus(admin.Get("/healthz", {{"Host", "outside.example"}})) == 403,
          "admin listener omitted local Host policy");
  Require(HttpStatus(admin.Post("/v1/responses", ResponseBody, "application/json")) == 403,
          "admin listener accepted a generation route");
  sockets.clear();
  Eventually(
      [&] {
        auto c = fixture.http.Connections();
        return c.active == 0 && c.queued == 0;
      },
      "closed slow bodies/queued connections did not release workers");
  Require(fixture.coordinator.Metrics().submitted == 0, "incomplete bodies reached admission");
}
void SlowReader(text::Tokenizer &tokenizer, bool responses) {
  auto state = std::make_shared<MockState>();
  state->tokens.assign(5000, 0);
  state->token_delay = 1ms;
  HttpLimits limits;
  limits.write_timeout = 100ms;
  limits.send_buffer_bytes = 1024;
  HttpFixture fixture(tokenizer, state, {}, limits);
  std::string body =
      responses
          ? R"({"model":"mellum21-exl3-k4-research","input":"hello","stream":true,"max_output_tokens":5000})"
          : R"({"model":"mellum21-exl3-k4-research","messages":[{"role":"user","content":"hello"}],"stream":true,"max_completion_tokens":5000})";
  RawConnection client(fixture.port, true);
  client.Send(
      Headers(fixture.port, responses ? "/v1/responses" : "/v1/chat/completions", body.size()) +
      body);
  Eventually([&] { return fixture.coordinator.Metrics().submitted == 1; },
             "slow reader was not admitted");
  httplib::Client admin("127.0.0.1", fixture.http.AdminPort());
  admin.set_read_timeout(0, 300000);
  Require(HttpStatus(admin.Get("/healthz")) == 200 && HttpStatus(admin.Get("/metrics")) == 200,
          "slow reader blocked admin listener");
  Eventually(
      [&] {
        auto m = fixture.coordinator.Metrics();
        return !m.active && !m.active_transports && m.write_failures == 1 &&
               m.completed + m.cancelled + m.failed == 1;
      },
      "nonreading peer did not release generation and transport");
  Eventually([&] { return fixture.coordinator.Sessions().Stats().first == 0; },
             "slow reader retained an unfinished response root");
}
void StartupAndRestart(text::Tokenizer &tokenizer) {
  {
    HttpListener occupied;
    const int port = occupied.Bind(0);
    Require(port > 0, "occupied-port setup failed");
    HttpListener duplicate;
    Require(duplicate.Bind(port) < 0, "two listeners shared an exclusive live port");
    auto state = std::make_shared<MockState>();
    Coordinator coordinator(Runtime(state), tokenizer);
    HttpLimits limits;
    limits.admin_port = port;
    HttpServer failed(coordinator, tokenizer, 0, 8192, limits);
    Require(failed.Bind() < 0, "occupied admin listener was silently ignored");
  }
  for (bool bound : {false, true}) {
    auto state = std::make_shared<MockState>();
    Coordinator coordinator(Runtime(state), tokenizer);
    HttpServer http(coordinator, tokenizer, 0);
    int port = -1, admin = -1;
    if (bound) {
      port = http.Bind();
      admin = http.AdminPort();
      Require(port > 0, "bind failed");
    }
    http.Stop();
    http.Stop();
    Require(http.Listen() && http.Bind() < 0, "stop-before-listen started or reported failure");
    if (bound) {
      HttpListener api_probe, admin_probe;
      Require(api_probe.Bind(port) == port && admin_probe.Bind(admin) == admin,
              "bound-but-unstarted shutdown leaked a listening socket");
    }
  }
  for (int i = 0; i < 10; ++i) {
    auto state = std::make_shared<MockState>();
    Coordinator coordinator(Runtime(state), tokenizer);
    HttpServer http(coordinator, tokenizer, 0);
    if (i % 2)
      Require(http.Bind() > 0, "startup-race bind failed");
    std::barrier gate(3);
    bool listened = false;
    std::thread listener([&] {
      gate.arrive_and_wait();
      listened = http.Listen();
    });
    std::thread stopper([&] {
      gate.arrive_and_wait();
      http.Stop();
    });
    gate.arrive_and_wait();
    listener.join();
    stopper.join();
    Require(listened, "concurrent start/stop reported a failed listener");
  }
  int port = 0;
  std::string previous;
  for (int run = 0; run < 2; ++run) {
    auto state = std::make_shared<MockState>();
    state->tokens = {0, 28};
    Coordinator coordinator(Runtime(state), tokenizer);
    HttpServer http(coordinator, tokenizer, port);
    port = http.Bind();
    Require(port > 0, "restart did not rebind API port");
    std::thread listener([&] { http.Listen(); });
    struct Stop {
      HttpServer &http;
      std::thread &thread;
      ~Stop() {
        http.Stop();
        thread.join();
      }
    } stop{http, listener};
    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(1);
    Eventually(
        [&] {
          auto reply = client.Get("/healthz");
          return reply && reply->status == 200;
        },
        "restart health did not become reachable");
    if (!previous.empty())
      Require(HttpStatus(client.Get("/v1/responses/" + previous)) == 404,
              "restart retained volatile response state");
    auto reply = JsonResponse(client.Post("/v1/responses", ResponseBody, "application/json"));
    previous = reply.find("id")->as_string();
  }
}
void DeadlineOverlap(text::Tokenizer &tokenizer) {
  auto state = std::make_shared<MockState>();
  state->prepare_block = true;
  ServerLimits limits;
  limits.queue_timeout = 50ms;
  limits.generation_timeout = 20ms;
  Coordinator coordinator(Runtime(state), tokenizer, limits);
  ReleaseGuard release{state};
  auto entered = state->entered.get_future();
  auto active = Submit(coordinator, Request("prefill_deadline"));
  Require(entered.wait_for(2s) == std::future_status::ready,
          "deadline preparation barrier not entered");
  auto queued = Submit(coordinator, Request("queued_deadline"));
  Wait(queued);
  Require(queued->failure_cause == CancelCause::QueueDeadline,
          "overlapping queue deadline classification wrong");
  state->release.set_value();
  Wait(active);
  Require(active->failure_cause == CancelCause::Deadline,
          "expired generation did not cancel at cooperative boundary");
  auto metrics = coordinator.Metrics();
  Require(metrics.cancelled == 2 && metrics.rejected == 0 && !metrics.active &&
              !metrics.queued_bytes,
          "overlapping deadline accounting/release wrong");
}
} // namespace
int main(int argc, char **argv) {
  try {
    auto loaded = text::Tokenizer::Load({}, {});
    Require(loaded.ok(), "operation tokenizer unavailable");
    auto &tokenizer = *loaded.value();
    if (argc != 2)
      throw std::runtime_error("expected operation case");
    const std::string_view mode(argv[1]);
    if (mode == "heartbeat") {
      HeartbeatClock();
      for (bool r : {false, true})
        HeartbeatDuringPreparation(tokenizer, r);
    } else if (mode == "slow-bodies")
      SlowBodiesAndAdmin(tokenizer);
    else if (mode == "slow-reader") {
      for (bool r : {false, true})
        SlowReader(tokenizer, r);
    } else if (mode == "restart")
      StartupAndRestart(tokenizer);
    else if (mode == "deadlines")
      DeadlineOverlap(tokenizer);
    else
      throw std::runtime_error("unknown operation case");
    std::cout << "PASS HTTP operation " << mode << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
