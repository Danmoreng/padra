#include "server/coordinator.h"
#include "runtime/size_checked.h"
#include <stdexcept>
#include <algorithm>
namespace mellum16::server {
namespace {
using Clock = std::chrono::steady_clock;
}
namespace {
std::vector<std::unique_ptr<ModelRuntime>> OneRuntime(std::unique_ptr<ModelRuntime> runtime) {
  std::vector<std::unique_ptr<ModelRuntime>> result;
  result.push_back(std::move(runtime));
  return result;
}
} // namespace
Coordinator::Coordinator(std::unique_ptr<ModelRuntime> runtime, const text::Tokenizer &tokenizer,
                         ServerLimits limits)
    : Coordinator(OneRuntime(std::move(runtime)), tokenizer, limits) {}
Coordinator::Coordinator(std::vector<std::unique_ptr<ModelRuntime>> runtimes,
                         const text::Tokenizer &tokenizer, ServerLimits limits)
    : tokenizer_(tokenizer), limits_(ValidateServerLimits(limits)) {
  if (runtimes.size() != limits_.running_sequences)
    throw std::invalid_argument("runtime count must match configured running sequences");
  for (auto &runtime : runtimes) {
    if (!runtime)
      throw std::invalid_argument("invalid coordinator runtime");
    auto lane = std::make_unique<Lane>();
    lane->runtime = std::move(runtime);
    lanes_.push_back(std::move(lane));
  }
  expiry_worker_ = std::thread([this] { ExpireQueued(); });
  try {
    for (auto &lane : lanes_)
      lane->worker = std::thread([this, ptr = lane.get()] { Work(*ptr); });
  } catch (...) {
    BeginDrain();
    for (auto &lane : lanes_)
      if (lane->worker.joinable())
        lane->worker.join();
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
  if (Clock::now() >= *drain_deadline_)
    for (auto &lane : lanes_)
      if (lane->active)
        lane->active->Cancel(CancelCause::Shutdown);
  available_.notify_all();
}
bool Coordinator::WaitForDrain() {
  Clock::time_point deadline;
  {
    std::unique_lock lock(mutex_);
    if (!metrics_.draining || !drain_deadline_)
      return false;
    deadline = *drain_deadline_;
    if (!available_.wait_until(lock, deadline,
                               [&] { return !AnyActiveLocked() && queue_.empty(); }))
      return false;
  }
  std::unique_lock lock(transport_->mutex);
  return transport_->changed.wait_until(lock, deadline, [&] { return transport_->active == 0; });
}
void Coordinator::Shutdown() {
  BeginDrain();
  std::lock_guard shutdown(shutdown_mutex_);
  for (auto &lane : lanes_)
    if (lane->worker.joinable())
      lane->worker.join();
  if (expiry_worker_.joinable())
    expiry_worker_.join();
}
ServerMetrics Coordinator::Metrics() const {
  std::lock_guard lock(mutex_);
  auto result = metrics_;
  result.queued = queue_.size();
  result.queued_bytes = bytes_;
  result.queued_tokens = tokens_;
  result.active = 0;
  for (const auto &lane : lanes_)
    result.active += bool(lane->active);
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
bool Coordinator::AnyActiveLocked() const {
  for (const auto &lane : lanes_)
    if (lane->active)
      return true;
  return false;
}
bool Coordinator::CanRunLocked(const Lane &lane, const Job &job) const {
  if (job.request.session_id.empty())
    return true;
  for (const auto &other : lanes_)
    if (other.get() != &lane && other->gpu_session == job.request.session_id &&
        other->gpu_owner == job.request.session_owner && other->gpu_checkpoint)
      return false;
  return true;
}
bool Coordinator::SessionCacheAvailable(const std::string &id, const std::string &owner) const {
  std::lock_guard lock(mutex_);
  if (!sessions_->Get(id, owner).ok())
    return false;
  for (const auto &lane : lanes_)
    if (id == lane->gpu_session && owner == lane->gpu_owner && lane->gpu_checkpoint &&
        lane->runtime->Checkpoint() == lane->gpu_checkpoint)
      return true;
  return false;
}
Status Coordinator::DiscardSession(const std::string &id, const std::string &owner) {
  auto status = sessions_->Delete(id, owner);
  std::lock_guard lock(mutex_);
  for (auto &lane : lanes_) {
    if (lane->gpu_session == id && lane->gpu_owner == owner) {
      lane->gpu_session.clear();
      lane->gpu_owner.clear();
      lane->gpu_checkpoint.reset();
    }
    if (lane->active && lane->active->request.session_id == id &&
        lane->active->request.session_owner == owner)
      lane->active->Cancel(CancelCause::SessionDiscard);
  }
  available_.notify_all();
  return status;
}
void Coordinator::Work(Lane &lane) {
  for (;;) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock lock(mutex_);
      auto candidate = [&] {
        return std::find_if(queue_.begin(), queue_.end(),
                            [&](const auto &item) { return CanRunLocked(lane, *item); });
      };
      available_.wait(lock, [&] { return metrics_.draining || candidate() != queue_.end(); });
      if (metrics_.draining && candidate() == queue_.end())
        return;
      auto it = candidate();
      job = *it;
      queue_.erase(it);
      bytes_ -= job->admitted_bytes;
      tokens_ -= job->request.input_ids.size() + job->request.outputs;
      lane.active = job;
    }
    Execute(lane, job);
  }
}
void Coordinator::CompleteJob(const std::shared_ptr<Job> &job) {
  job->lifetime_seconds =
      std::chrono::duration<double>(Clock::now() - job->request.submitted).count();
  job->session_lease.reset();
  {
    std::lock_guard lock(mutex_);
    RecordOutcome(*job);
    for (auto &lane : lanes_)
      if (lane->active == job)
        lane->active.reset();
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
      if (metrics_.draining && AnyActiveLocked() && drain_deadline_) {
        if (now >= *drain_deadline_) {
          for (auto &lane : lanes_)
            if (lane->active)
              lane->active->Cancel(CancelCause::Shutdown);
        } else {
          wake = std::min(wake, *drain_deadline_);
        }
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
        if (metrics_.draining && queue_.empty() && !AnyActiveLocked())
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
      job->Fail("request cancelled while queued");
      CompleteJob(job);
    }
  }
}
} // namespace mellum16::server
