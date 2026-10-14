#include "server_test_support.h"
namespace {
void Case(text::Tokenizer &tokenizer, int boundary) {
  auto state = std::make_shared<MockState>();
  state->reset_failure = boundary == 0;
  state->prepare_failure = boundary == 1;
  state->generate_failure = boundary == 2;
  if (boundary == 3)
    state->tokens =
        tokenizer.Encode(R"(<tool_call>{"name":"lookup","arguments":{"x":]}}</tool_call>)").value();
  Coordinator coordinator(Runtime(state), tokenizer);
  auto session = coordinator.Sessions().Create("owner");
  Require(session.ok(), "boundary session setup failed");
  auto request = Request("failure_boundary");
  request.stream = true;
  request.outputs = 128;
  request.session_id = session.value().id;
  request.session_owner = "owner";
  request.body = boundary == 4   ? "injected invalid normalized history"
                 : boundary == 5 ? "{}"
                                 : ChatBody;
  if (boundary == 3)
    request.tools = {{"lookup"}, true, true};
  auto admitted = coordinator.Submit(std::move(request), {true, ApiRoute::Chat});
  Require(admitted.ok(), "boundary submission failed");
  auto job = std::move(admitted).value();
  Wait(job);
  Require(!job->failure.ok() && !job->session_lease, "terminal boundary retained generation lease");
  auto snapshot = coordinator.Sessions().Get(session.value().id, "owner");
  Require(snapshot.ok() && !snapshot.value().active && snapshot.value().history == "[]",
          "failure committed invalid history or kept active session");
  const auto metrics = coordinator.Metrics();
  Require(!metrics.active && metrics.queued == 0 && metrics.queued_bytes == 0 &&
              metrics.queued_tokens == 0 && metrics.failed + metrics.cancelled == 1 &&
              metrics.completed == 0 && metrics.active_transports == 1,
          "generation cleanup confused transport and generation completion");
  Require(metrics.ready == (boundary >= 3), "boundary failure readiness classification changed");
  const auto frames = Drain(job);
  Require(frames.find("\"error\"") != std::string::npos &&
              frames.find("[DONE]") == std::string::npos,
          "failure published a normal ending");
  job->FinishTransport(false);
  Require(coordinator.Metrics().active_transports == 0 && coordinator.Metrics().write_failures == 1,
          "failed transport was not released once");
  // Fill the exact 32 MiB logical history reservation budget. Any leaked
  // reservation from the failed job makes the fourth lease fail.
  std::vector<std::shared_ptr<SessionLease>> leases;
  for (int i = 0; i < 4; ++i) {
    const auto id = i == 0 ? session.value().id : coordinator.Sessions().Create("owner").value().id;
    auto lease = coordinator.Sessions().Acquire(id, "owner", 8 * 1024 * 1024);
    Require(lease.ok(), "failure leaked session reservation bytes");
    leases.push_back(std::move(lease).value());
  }
  leases.clear();
  Require(coordinator.Sessions().Stats() == std::pair<std::size_t, std::size_t>{4, 8},
          "logical reservation probe leaked retained history");
}
} // namespace
int main() {
  try {
    auto tokenizer = text::Tokenizer::Load({}, {});
    Require(tokenizer.ok(), "test tokenizer unavailable");
    for (int boundary = 0; boundary < 6; ++boundary)
      Case(*tokenizer.value(), boundary);
    std::cout << "PASS reset/preparation/generation/tool/history failure release boundaries\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
