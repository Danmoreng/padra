#include "server/coordinator.h"
#include "runtime/size_checked.h"
#include "server/job_error.h"
#include "server/secure_id.h"
#include "text/stream.h"
#include "util/json.h"
#include <chrono>
#include <cstring>
#include <stdexcept>
namespace mellum16::server {
namespace {
using Clock = std::chrono::steady_clock;
std::uint64_t Created() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                                        std::chrono::system_clock::now().time_since_epoch())
                                        .count());
}
class Events {
public:
  explicit Events(Job &job) : job_(job) {}
  void Begin() {
    Send(GenerationEvent::Make(EventKind::Begin));
  }
  void Delta(std::string_view content, std::string_view reasoning) {
    if (!reasoning.empty())
      job_.ObserveOutput(OutputKind::Reasoning);
    if (!content.empty())
      job_.ObserveOutput(OutputKind::Message);
    Parts(EventKind::Reasoning, reasoning);
    Parts(EventKind::Text, content);
  }
  void Terminal() {
    job_.PushEnding(GenerationEvent::Make(EventKind::Finish));
    if (job_.request.include_usage)
      job_.PushEnding(GenerationEvent::Make(EventKind::Usage));
    job_.PushEnding(GenerationEvent::Make(EventKind::Done));
  }
  void ToolBegin(std::size_t index, std::string_view id, std::string_view name) {
    job_.ObserveOutput(OutputKind::Tool, index);
    Send(GenerationEvent::Make(EventKind::ToolStart, {}, index, id, name));
  }
  void ToolArguments(std::size_t index, std::string_view args) {
    Parts(EventKind::ToolArguments, args, index);
  }
  void Tool(std::size_t index, const ToolCall &call) {
    ToolBegin(index, call.id, call.name);
    ToolArguments(index, call.arguments);
  }

private:
  void Send(const GenerationEvent &event) {
    if (job_.request.stream && !job_.Push(event))
      job_.Cancel(CancelCause::Backpressure);
  }
  void Parts(EventKind kind, std::string_view text, std::size_t index = 0) {
    if (!job_.request.stream)
      return;
    while (!text.empty() && job_.cancellation.load() == CancelCause::None) {
      auto size = std::min<std::size_t>(1024, text.size());
      while (size < text.size() && (static_cast<unsigned char>(text[size]) & 0xc0) == 0x80)
        --size;
      Send(GenerationEvent::Make(kind, text.substr(0, size), index));
      text.remove_prefix(size);
    }
  }
  Job &job_;
};
void SetTerminalError(Job &job, std::string_view message) {
  if (message.size() > 1024)
    message = "generation failed";
  auto event = GenerationEvent::Make(EventKind::Error, message);
  std::lock_guard lock(job.mutex);
  job.failure_cause = job.cancellation.load();
  job.ending_head = 0;
  job.ending_count = 1;
  job.endings[0] = event;
}

} // namespace
std::string ChatUsage(const GenerationResult &job) {
  return "{\"prompt_tokens\":" + std::to_string(job.prompt_tokens) +
         ",\"completion_tokens\":" + std::to_string(job.completion_tokens) +
         ",\"total_tokens\":" + std::to_string(job.prompt_tokens + job.completion_tokens) +
         ",\"prompt_tokens_details\":{\"cached_tokens\":" + std::to_string(job.cached_tokens) +
         "}}";
}
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
Coordinator::Coordinator(std::unique_ptr<ModelRuntime> runtime, const text::Tokenizer &tokenizer,
                         ServerLimits limits)
    : runtime_(std::move(runtime)), tokenizer_(tokenizer), limits_(limits) {
  if (!runtime_ || !limits_.queued_requests || !limits_.queued_bytes || !limits_.queued_tokens ||
      limits_.queue_timeout.count() < 1 || limits_.generation_timeout.count() < 1 ||
      limits_.drain_timeout.count() < 0)
    throw std::invalid_argument("invalid coordinator limits/runtime");
  expiry_worker_ = std::thread([this] { ExpireQueued(); });
  try {
    worker_ = std::thread([this] { Work(); });
  } catch (...) {
    BeginDrain();
    expiry_worker_.join();
    throw;
  }
}
Coordinator::~Coordinator() {
  Shutdown();
}
Result<std::shared_ptr<Job>> Coordinator::Submit(ChatRequest request, SubmissionContext context) {
  if (context.decision)
    *context.decision = {};
  if (request.outputs < 1 || request.outputs > 131072) {
    if (context.decision)
      context.decision->cause = AdmissionCause::InvalidRequest;
    RecordRejection(AdmissionCause::InvalidRequest, context.route);
    return Status(StatusCode::kInvalidArgument, "invalid requested output budget");
  }
  const auto output_budget = admission::Sum(
      {admission::Bytes(request.outputs,
                        admission::Bytes(admission::Sum({tokenizer_.MaxRawTokenBytes(), 3}), 3)),
       1024});
  const auto predicted = admission::Sum({sizeof(Job), request.body.size(), request.adapter_bytes,
                                         admission::Bytes(request.input_ids.size(), sizeof(int)),
                                         admission::Bytes(output_budget, 2),
                                         request.tools.parse ? 16 * kMaximumToolJson + 8192 : 0});
  const auto tokens =
      admission::Sum({request.input_ids.size(), static_cast<std::size_t>(request.outputs)});
  std::lock_guard lock(mutex_);
  const auto reject = [&](AdmissionCause cause, Status status) -> Result<std::shared_ptr<Job>> {
    if (context.decision)
      context.decision->cause = cause;
    RecordRejectionLocked(cause, context.route);
    return status;
  };
  auto decision = CheckAdmissionLocked();
  if (decision.allowed() && predicted > limits_.queued_bytes - bytes_)
    decision.cause = AdmissionCause::QueueBytes;
  if (decision.allowed() && tokens > limits_.queued_tokens - tokens_)
    decision.cause = AdmissionCause::QueueTokens;
  if (!decision.allowed())
    return reject(decision.cause, Status(StatusCode::kResourceExhausted,
                                         "bounded generation admission unavailable"));
  std::shared_ptr<SessionLease> lease;
  if (!request.session_id.empty()) {
    const auto history_budget = admission::Sum(
        {request.body.size(),
         admission::Bytes(request.outputs,
                          admission::Bytes(admission::Sum({tokenizer_.MaxRawTokenBytes(), 3}), 18)),
         1024});
    auto acquired =
        sessions_->Acquire(request.session_id, request.session_owner, history_budget, &decision);
    if (!acquired.ok())
      return reject(decision.cause, acquired.status());
    lease = std::move(acquired).value();
  }
  auto job = std::make_shared<Job>(std::move(request), tokenizer_.MaxRawTokenBytes());
  job->session_lease = std::move(lease);
  job->transport = transport_;
  job->admitted_at = Clock::now();
  job->queue_deadline = job->admitted_at + limits_.queue_timeout;
  // Publishing may allocate. Charge resources only once it succeeds, while the
  // worker is still excluded by mutex_, so an exception leaves no admission debt.
  queue_.push_back(job);
  if (context.track_transport)
    job->StartTransport();
  bytes_ += job->admitted_bytes;
  tokens_ += tokens;
  ++metrics_.submitted;
  available_.notify_all();
  return job;
}
void Coordinator::BeginDrain(bool graceful) {
  std::lock_guard lock(mutex_);
  if (!metrics_.draining)
    drain_deadline_ =
        Clock::now() + (graceful ? limits_.drain_timeout : std::chrono::milliseconds(0));
  metrics_.draining = true;
  metrics_.ready = false;
  for (auto &job : queue_)
    job->Cancel(CancelCause::Shutdown);
  if (!graceful)
    drain_deadline_ = Clock::now();
  if (active_ && Clock::now() >= *drain_deadline_)
    active_->Cancel(CancelCause::Shutdown);
  available_.notify_all();
}
bool Coordinator::WaitForDrain() {
  Clock::time_point deadline;
  {
    std::unique_lock lock(mutex_);
    if (!metrics_.draining || !drain_deadline_)
      return false;
    deadline = *drain_deadline_;
    if (!available_.wait_until(lock, deadline, [&] { return !active_ && queue_.empty(); }))
      return false;
  }
  std::unique_lock lock(transport_->mutex);
  return transport_->changed.wait_until(lock, deadline, [&] { return transport_->active == 0; });
}
void Coordinator::Shutdown() {
  BeginDrain();
  std::lock_guard shutdown(shutdown_mutex_);
  if (worker_.joinable())
    worker_.join();
  if (expiry_worker_.joinable())
    expiry_worker_.join();
}
ServerMetrics Coordinator::Metrics() const {
  std::lock_guard lock(mutex_);
  auto result = metrics_;
  result.queued = queue_.size();
  result.queued_bytes = bytes_;
  result.queued_tokens = tokens_;
  result.active = bool(active_);
  {
    std::lock_guard transport_lock(transport_->mutex);
    result.active_transports = transport_->active;
    result.responses_written = transport_->responses_written;
    result.write_failures = transport_->write_failures;
    result.response_body_bytes = transport_->body_bytes;
    result.first_visible_write_latency = transport_->first_visible_write_latency;
    result.response_write_latency = transport_->response_write_latency;
  }
  result.disconnects = transport_->disconnects.load();
  result.backpressure = transport_->backpressure.load();
  return result;
}
AdmissionDecision Coordinator::CheckAdmissionLocked() const {
  if (metrics_.draining)
    return {AdmissionCause::Draining};
  if (!metrics_.ready)
    return {AdmissionCause::NotReady};
  if (queue_.size() >= limits_.queued_requests)
    return {AdmissionCause::QueueFull};
  return {};
}
AdmissionDecision Coordinator::CheckAdmission() const {
  std::lock_guard lock(mutex_);
  return CheckAdmissionLocked();
}
void Coordinator::RecordRejectionLocked(AdmissionCause cause, ApiRoute route) {
  if (cause == AdmissionCause::None)
    throw std::logic_error("rejection needs a cause");
  ++metrics_.rejected;
  ++metrics_.rejections.at(static_cast<std::size_t>(route)).at(static_cast<std::size_t>(cause));
}
void Coordinator::RecordRejection(AdmissionCause cause, ApiRoute route) {
  std::lock_guard lock(mutex_);
  RecordRejectionLocked(cause, route);
}
bool Coordinator::SessionCacheAvailable(const std::string &id, const std::string &owner) const {
  std::lock_guard lock(mutex_);
  return id == gpu_session_ && owner == gpu_owner_ && gpu_checkpoint_ &&
         sessions_->Get(id, owner).ok() && runtime_->Checkpoint() == gpu_checkpoint_;
}
Status Coordinator::DiscardSession(const std::string &id, const std::string &owner) {
  auto status = sessions_->Delete(id, owner);
  std::lock_guard lock(mutex_);
  if (gpu_session_ == id && gpu_owner_ == owner) {
    gpu_session_.clear();
    gpu_owner_.clear();
    gpu_checkpoint_.reset();
  }
  if (active_ && active_->request.session_id == id && active_->request.session_owner == owner)
    active_->Cancel(CancelCause::SessionDiscard);
  return status;
}
void Coordinator::Work() {
  for (;;) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock lock(mutex_);
      available_.wait(lock, [&] { return metrics_.draining || !queue_.empty(); });
      if (queue_.empty())
        return;
      job = queue_.front();
      queue_.pop_front();
      bytes_ -= job->admitted_bytes;
      tokens_ -= job->request.input_ids.size() + job->request.outputs;
      active_ = job;
    }
    Execute(job);
  }
}
void Coordinator::CompleteJob(const std::shared_ptr<Job> &job) {
  job->lifetime_seconds =
      std::chrono::duration<double>(Clock::now() - job->request.submitted).count();
  job->session_lease.reset();
  {
    std::lock_guard lock(mutex_);
    RecordOutcome(*job);
    if (active_ == job)
      active_.reset();
    available_.notify_all();
  }
  {
    std::lock_guard lock(job->mutex);
    job->complete = true;
    job->changed.notify_all();
  }
}
void Coordinator::RecordOutcome(const Job &job) {
  const auto cause = job.cancellation.load();
  if (job.failure.ok())
    ++metrics_.completed;
  else if (cause != CancelCause::None)
    ++metrics_.cancelled;
  else
    ++metrics_.failed;
  metrics_.tokenize_seconds += job.request.tokenize_seconds;
  metrics_.queue_seconds += job.queue_seconds;
  metrics_.prefill_seconds += job.prefill_seconds;
  metrics_.decode_seconds += job.decode_seconds;
  metrics_.first_token_seconds += job.first_token_seconds;
  metrics_.visible_seconds += job.visible_seconds;
  metrics_.output_tokens += job.completion_tokens;
  metrics_.queue_latency.Observe(job.queue_seconds);
  metrics_.tokenization_latency.Observe(job.request.tokenize_seconds);
  metrics_.lifetime_latency.Observe(job.lifetime_seconds);
  if (job.completion_tokens) {
    metrics_.prefill_latency.Observe(job.prefill_seconds);
    metrics_.decode_latency.Observe(job.decode_seconds);
    metrics_.first_token_latency.Observe(job.first_token_seconds);
  }
  if (job.visible_seconds > 0)
    metrics_.output_available_latency.Observe(job.visible_seconds);
}
void Coordinator::ExpireQueued() {
  for (;;) {
    std::vector<std::shared_ptr<Job>> expired;
    {
      std::unique_lock lock(mutex_);
      const auto now = Clock::now();
      auto wake = now + std::chrono::milliseconds(20);
      if (metrics_.draining && active_ && drain_deadline_) {
        if (now >= *drain_deadline_)
          active_->Cancel(CancelCause::Shutdown);
        else
          wake = std::min(wake, *drain_deadline_);
      }
      for (auto it = queue_.begin(); it != queue_.end();) {
        const auto &job = *it;
        if (now >= job->queue_deadline)
          job->Cancel(CancelCause::QueueDeadline);
        if (job->session_lease && job->session_lease->record->deleted.load())
          job->Cancel(CancelCause::SessionDiscard);
        if (job->cancellation.load() != CancelCause::None) {
          bytes_ -= job->admitted_bytes;
          tokens_ -= job->request.input_ids.size() + job->request.outputs;
          expired.push_back(job);
          it = queue_.erase(it);
        } else {
          wake = std::min(wake, job->queue_deadline);
          ++it;
        }
      }
      if (expired.empty()) {
        if (metrics_.draining && queue_.empty() && !active_)
          return;
        if (queue_.empty() && !metrics_.draining)
          available_.wait(lock);
        else
          available_.wait_until(lock, wake);
        continue;
      }
    }
    // The expiry worker never touches the GPU. Release leases before publishing
    // completion, even if HTTP still retains the Job object.
    for (auto &job : expired) {
      job->queue_seconds = std::chrono::duration<double>(Clock::now() - job->admitted_at).count();
      job->failure = {StatusCode::kInternal, "request cancelled while queued"};
      SetTerminalError(*job, job->failure.message());
      CompleteJob(job);
    }
  }
}
void Coordinator::Execute(const std::shared_ptr<Job> &job) {
  const auto started = Clock::now();
  job->queue_seconds = std::chrono::duration<double>(started - job->admitted_at).count();
  const auto deadline = started + limits_.generation_timeout;
  auto lease = job->session_lease;
  bool runtime_touched = false;
  try {
    {
      std::lock_guard lock(mutex_);
      if (!metrics_.ready)
        job->Cancel(CancelCause::Shutdown);
    }
    if (started >= job->queue_deadline)
      job->Cancel(CancelCause::QueueDeadline);
    if (job->cancellation.load() != CancelCause::None)
      throw std::runtime_error("request cancelled before generation");
    if (lease && lease->record->deleted.load()) {
      job->Cancel(CancelCause::SessionDiscard);
      throw std::runtime_error("session discarded before generation");
    }
    std::optional<KVCheckpointHandle> retained;
    {
      std::lock_guard lock(mutex_);
      if (lease && gpu_session_ == job->request.session_id &&
          gpu_owner_ == job->request.session_owner)
        retained = gpu_checkpoint_;
    }
    Status status;
    bool restored = false;
    if (retained) {
      runtime_touched = true;
      status = runtime_->PrepareCheckpointRestore(job->request.input_ids, job->request.outputs,
                                                  *retained, job->request.checkpoint_anchor, false,
                                                  job->request.sampling);
      restored = status.ok();
      if (!restored && status.code() != StatusCode::kInvalidArgument &&
          status.code() != StatusCode::kUnsupported)
        throw std::runtime_error(status.message());
    }
    if (!restored) {
      runtime_touched = true;
      {
        std::lock_guard lock(mutex_);
        gpu_session_.clear();
        gpu_owner_.clear();
        gpu_checkpoint_.reset();
      }
      status = runtime_->Reset();
      if (!status.ok()) {
        std::lock_guard lock(mutex_);
        metrics_.ready = false;
        throw std::runtime_error(status.message());
      }
      status = lease && job->request.checkpoint_anchor && runtime_->Stats().checkpoint_enabled
                   ? runtime_->PrepareWithCheckpoint(job->request.input_ids, job->request.outputs,
                                                     job->request.checkpoint_anchor, false,
                                                     job->request.sampling)
                   : runtime_->PrepareRequestShape(job->request.input_ids, job->request.outputs,
                                                   false, job->request.sampling);
      if (!status.ok()) {
        std::lock_guard lock(mutex_);
        metrics_.ready = false;
        throw std::runtime_error(status.message());
      }
    }
    job->cached_tokens = runtime_->Stats().cached_tokens;
    text::StreamDecoder decoder(tokenizer_);
    const auto maximum = (tokenizer_.MaxRawTokenBytes() + 3) * 3;
    text::StopMatcher stops(job->request.stops, maximum);
    text::ThinkingParser thinking(maximum + 256);
    ToolStreamParser tools(job->request.tools.parse, maximum + 256,
                           job->request.tools.parallel ? 8 : 1);
    std::array<std::string, 8> call_ids;
    if (job->request.tools.parse)
      for (auto &id : call_ids) {
        auto generated = MakeSecureId("call_");
        if (!generated.ok())
          throw std::runtime_error(generated.status().message());
        id = std::move(generated).value();
      }
    Events events(*job);
    if (job->request.stream)
      events.Begin();
    if (job->request.tools.parse)
      tools.StreamDeltas(job->request.tools, [&](std::size_t index, std::string_view name,
                                                 std::string_view arguments) {
        if (job->visible_seconds == 0)
          job->visible_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        if (!name.empty())
          events.ToolBegin(index, call_ids[index], name);
        if (!arguments.empty())
          events.ToolArguments(index, arguments);
        return job->cancellation.load() == CancelCause::None;
      });
    const auto accept_output = [&](text::ThinkingOutput output) {
      auto visible = tools.Push(output.content);
      if (!visible.empty() && job->visible_seconds == 0)
        job->visible_seconds = std::chrono::duration<double>(Clock::now() - started).count();
      job->content.append(visible);
      job->reasoning.append(output.reasoning);
      events.Delta(visible, output.reasoning);
      tools.FlushDeltas();
      if (tools.delivery_failed())
        job->Cancel(CancelCause::Backpressure);
      if (tools.error())
        job->Cancel(CancelCause::ToolInvalid);
    };
    const auto accept = [&](std::string_view text) {
      accept_output(thinking.Push(stops.Push(text)));
    };
    const auto allowed = [&] {
      if (Clock::now() > deadline)
        job->Cancel(CancelCause::Deadline);
      if (lease && lease->record->deleted.load())
        job->Cancel(CancelCause::SessionDiscard);
      return !stops.stopped() && job->cancellation.load() == CancelCause::None;
    };
    auto result = runtime_->Generate(
        [&](int token) {
          if (job->completion_tokens++ == 0)
            job->first_token_seconds =
                std::chrono::duration<double>(Clock::now() - started).count();
          const bool was_reasoning = thinking.in_reasoning();
          accept(decoder.Push(token));
          if (was_reasoning || thinking.in_reasoning())
            ++job->reasoning_tokens;
          return allowed();
        },
        [&](int) { return allowed(); });
    if (!result.ok()) {
      {
        std::lock_guard lock(mutex_);
        metrics_.ready = false;
      }
      throw std::runtime_error(result.status().message());
    }
    job->prefill_seconds = result.value().prefill_seconds;
    job->decode_seconds = result.value().decode_seconds;
    if (tools.error())
      throw std::runtime_error(tools.error());
    const auto cause = job->cancellation.load();
    if (cause != CancelCause::None)
      throw std::runtime_error("generation cancelled/deadline/disconnect/backpressure");
    accept(decoder.Finish());
    accept_output(thinking.Push(stops.Finish()));
    accept_output(thinking.Finish());
    auto visible = tools.Finish();
    job->content.append(visible);
    if (!visible.empty() && job->visible_seconds == 0)
      job->visible_seconds = std::chrono::duration<double>(Clock::now() - started).count();
    events.Delta(visible, {});
    if (tools.error())
      throw std::runtime_error(tools.error());
    job->finish_reason = (stops.stopped() || (!result.value().generated_ids.empty() &&
                                              result.value().generated_ids.back() == 28))
                             ? "stop"
                             : "length";
    if (tools.incomplete() && job->finish_reason != "length") {
      job->Cancel(CancelCause::ToolInvalid);
      throw std::runtime_error("model ended/stopped with incomplete tool call");
    }
    for (std::size_t i = 0; i < tools.count(); ++i) {
      const bool complete = !(tools.incomplete() && i + 1 == tools.count());
      auto parsed =
          ParseGeneratedTool(tools.raw(i), job->request.tools, std::move(call_ids[i]), complete);
      if (!parsed.ok()) {
        job->Cancel(CancelCause::ToolInvalid);
        throw std::runtime_error(parsed.status().message());
      }
      job->tool_calls.push_back(std::move(parsed).value());
    }
    job->partial_tool_call = tools.incomplete();
    if (!job->tool_calls.empty() && !job->partial_tool_call)
      job->finish_reason = "tool_calls";
    if (!job->tool_calls.empty() && job->visible_seconds == 0)
      job->visible_seconds = std::chrono::duration<double>(Clock::now() - started).count();
    for (std::size_t i = 0; i < job->tool_calls.size(); ++i)
      if (!tools.identity_sent(i))
        events.Tool(i, job->tool_calls[i]);
    if (!job->output_count && job->finish_reason != "length")
      job->ObserveOutput(OutputKind::Message);
    if (job->cancellation.load() != CancelCause::None)
      throw std::runtime_error("generation cancelled during output finalization");
    if (lease && !job->partial_tool_call) {
      auto body = json::Parse(job->request.body);
      if (!body.ok())
        throw std::runtime_error("validated session body became invalid");
      auto messages = body.value().find("messages")->as_array();
      json::Value::Object assistant;
      assistant.emplace("role", json::Value(std::string("assistant")));
      assistant.emplace("content", job->content.empty() && !job->tool_calls.empty()
                                       ? json::Value(nullptr)
                                       : json::Value(job->content));
      assistant.emplace("reasoning_content", json::Value(job->reasoning));
      if (!job->tool_calls.empty()) {
        auto calls = json::Parse(ToolCallsJson(job->tool_calls));
        if (!calls.ok())
          throw std::runtime_error("validated calls became invalid");
        assistant.emplace("tool_calls", std::move(calls).value());
      }
      messages.emplace_back(std::move(assistant));
      status = sessions_->Commit(lease, json::Stringify(json::Value(std::move(messages))));
      if (!status.ok())
        throw std::runtime_error(status.message());
      {
        std::lock_guard lock(mutex_);
        gpu_session_ = job->request.session_id;
        gpu_owner_ = job->request.session_owner;
        gpu_checkpoint_ = runtime_->Checkpoint();
      }
    }
    if (job->partial_tool_call) {
      std::lock_guard lock(mutex_);
      gpu_session_.clear();
      gpu_owner_.clear();
      gpu_checkpoint_.reset();
    }
    if (job->request.stream) {
      events.Terminal();
      if (job->cancellation.load() != CancelCause::None)
        throw std::runtime_error("terminal stream backpressure");
    }
  } catch (const std::exception &error) {
    if (runtime_touched) {
      {
        std::lock_guard lock(mutex_);
        gpu_session_.clear();
        gpu_owner_.clear();
        gpu_checkpoint_.reset();
      }
      if (runtime_->Stats().phase != SessionPhase::Poisoned)
        (void)runtime_->Reset();
    }
    (void)(job->failure = Status{StatusCode::kInternal, error.what()});
    SetTerminalError(*job, error.what());
  }
  lease.reset();
  CompleteJob(job);
}
std::string CompletionJson(const Job &job) {
  return "{\"id\":" + json::Quote(job.request.id) +
         ",\"object\":\"chat.completion\",\"created\":" + std::to_string(Created()) +
         ",\"model\":" + json::Quote(kModelId) +
         (job.request.session_id.empty()
              ? ""
              : ",\"session_id\":" + json::Quote(job.request.session_id)) +
         ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":" +
         (job.content.empty() && !job.tool_calls.empty() ? "null" : json::Quote(job.content)) +
         ",\"reasoning_content\":" + json::Quote(job.reasoning) +
         (job.tool_calls.empty() ? "" : ",\"tool_calls\":" + ToolCallsJson(job.tool_calls)) +
         "},\"finish_reason\":" + json::Quote(job.finish_reason) +
         ",\"logprobs\":null}],\"usage\":" + ChatUsage(job) + "}";
}
} // namespace mellum16::server
