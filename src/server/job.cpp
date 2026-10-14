#include "server/job.h"
#include "runtime/size_checked.h"
#include <stdexcept>
namespace mellum16::server {
namespace {
using Clock = std::chrono::steady_clock;
std::uint64_t Created() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                                        std::chrono::system_clock::now().time_since_epoch())
                                        .count());
}
} // namespace
Job::Job(ChatRequest input, std::size_t maximum)
    : request(std::move(input)), created_at(Created()) {
  prompt_tokens = request.input_ids.size();
  output_budget = request.outputs;
  temperature = request.sampling.temperature;
  top_p = request.sampling.top_p;
  if (request.outputs < 1 || request.outputs > 131072 || maximum > 8192)
    throw std::invalid_argument("invalid request output allocation contract");
  const auto output_budget = admission::Sum(
      {admission::Bytes(request.outputs, admission::Bytes(admission::Sum({maximum, 3}), 3)), 1024});
  content.reserve(output_budget);
  reasoning.reserve(output_budget);
  tool_calls.reserve(8);
  admitted_bytes = admission::Sum({sizeof(Job), request.body.size(), request.adapter_bytes,
                                   admission::Bytes(request.input_ids.size(), sizeof(int)),
                                   admission::Bytes(output_budget, 2),
                                   request.tools.parse ? 16 * kMaximumToolJson + 8192 : 0});
}
void Job::Cancel(CancelCause cause) {
  if (transport) {
    if (cause == CancelCause::Disconnect && !disconnect_recorded.exchange(true))
      ++transport->disconnects;
    if (cause == CancelCause::Backpressure && !backpressure_recorded.exchange(true))
      ++transport->backpressure;
  }
  auto expected = CancelCause::None;
  cancellation.compare_exchange_strong(expected, cause);
  changed.notify_all();
}
void Job::ObserveOutput(OutputKind kind, std::size_t tool) {
  for (std::size_t i = 0; i < output_count; ++i)
    if (output_order[i].kind == kind && output_order[i].tool == tool)
      return;
  if (output_count == output_order.size() || (kind == OutputKind::Tool && tool >= 8))
    throw std::logic_error("bounded logical output count exceeded");
  output_order[output_count++] = {kind, tool};
}
void Job::StartTransport() {
  if (!transport || transport_started.exchange(true))
    return;
  std::lock_guard lock(transport->mutex);
  ++transport->active;
}
void Job::RecordWrite(std::size_t bytes, bool visible) {
  if (!transport || !transport_started.load())
    return;
  const auto seconds = std::chrono::duration<double>(Clock::now() - request.submitted).count();
  std::lock_guard lock(transport->mutex);
  transport->body_bytes += bytes;
  if (visible && !visible_written.exchange(true))
    transport->first_visible_write_latency.Observe(seconds);
}
void Job::FinishTransport(bool success) {
  if (!transport || !transport_started.load() || transport_finished.exchange(true))
    return;
  const auto seconds = std::chrono::duration<double>(Clock::now() - request.submitted).count();
  std::lock_guard lock(transport->mutex);
  --transport->active;
  if (success)
    ++transport->responses_written;
  else
    ++transport->write_failures;
  transport->response_write_latency.Observe(seconds);
  transport->changed.notify_all();
}
bool Job::Push(const GenerationEvent &frame) {
  if (!request.stream)
    return true;
  std::lock_guard lock(mutex);
  if (!frame.Valid() || frame.Terminal() || count == events.size()) {
    Cancel(CancelCause::Backpressure);
    return false;
  }
  auto &slot = events[tail];
  slot = frame;
  tail = (tail + 1) % events.size();
  ++count;
  changed.notify_all();
  return true;
}
bool Job::PushEnding(const GenerationEvent &frame) {
  if (!request.stream)
    return true;
  std::lock_guard lock(mutex);
  if (!frame.Valid() || !frame.Terminal() || ending_count == endings.size()) {
    Cancel(CancelCause::Backpressure);
    return false;
  }
  auto &slot = endings[ending_count++];
  slot = frame;
  return true;
}
bool Job::Pop(GenerationEvent &frame, std::chrono::milliseconds wait) {
  std::unique_lock lock(mutex);
  changed.wait_for(lock, wait, [&] {
    return count || complete || cancellation.load() == CancelCause::Disconnect;
  });
  if (count) {
    frame = events[head];
    head = (head + 1) % events.size();
    --count;
    return true;
  }
  if (complete && ending_head < ending_count) {
    frame = endings[ending_head++];
    return true;
  }
  return false;
}
void Job::Fail(std::string_view message) {
  failure = {StatusCode::kInternal, std::string(message)};
  if (message.size() > 1024)
    message = "generation failed";
  auto event = GenerationEvent::Make(EventKind::Error, message);
  std::lock_guard lock(mutex);
  failure_cause = cancellation.load();
  ending_head = 0;
  ending_count = 1;
  endings[0] = event;
}
} // namespace mellum16::server
