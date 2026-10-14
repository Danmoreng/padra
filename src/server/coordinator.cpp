#include "server/coordinator.h"
#include "runtime/size_checked.h"
#include <stdexcept>
namespace mellum16::server {
namespace {
using Clock = std::chrono::steady_clock;
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
      job->Fail("request cancelled while queued");
      CompleteJob(job);
    }
  }
}
} // namespace mellum16::server
