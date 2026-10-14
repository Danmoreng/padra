#include "server/coordinator.h"
#include "server/output_processing.h"
#include <stdexcept>
namespace mellum16::server {
namespace {
using Clock = std::chrono::steady_clock;
}
void Coordinator::PrepareExecution(Job &job, Clock::time_point started, bool &runtime_touched) {
  const auto lease = job.session_lease;
  {
    std::lock_guard lock(mutex_);
    if (!metrics_.ready)
      job.Cancel(CancelCause::Shutdown);
  }
  if (started >= job.queue_deadline)
    job.Cancel(CancelCause::QueueDeadline);
  if (job.cancellation.load() != CancelCause::None)
    throw std::runtime_error("request cancelled before generation");
  if (lease && lease->record->deleted.load()) {
    job.Cancel(CancelCause::SessionDiscard);
    throw std::runtime_error("session discarded before generation");
  }
  std::optional<KVCheckpointHandle> retained;
  {
    std::lock_guard lock(mutex_);
    if (lease && gpu_session_ == job.request.session_id && gpu_owner_ == job.request.session_owner)
      retained = gpu_checkpoint_;
  }
  Status status;
  bool restored = false;
  if (retained) {
    runtime_touched = true;
    status = runtime_->PrepareCheckpointRestore(job.request.input_ids, job.request.outputs,
                                                *retained, job.request.checkpoint_anchor, false,
                                                job.request.sampling);
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
    status = lease && job.request.checkpoint_anchor && runtime_->Stats().checkpoint_enabled
                 ? runtime_->PrepareWithCheckpoint(job.request.input_ids, job.request.outputs,
                                                   job.request.checkpoint_anchor, false,
                                                   job.request.sampling)
                 : runtime_->PrepareRequestShape(job.request.input_ids, job.request.outputs, false,
                                                 job.request.sampling);
    if (!status.ok()) {
      std::lock_guard lock(mutex_);
      metrics_.ready = false;
      throw std::runtime_error(status.message());
    }
  }
  job.cached_tokens = runtime_->Stats().cached_tokens;
}
void Coordinator::CommitHistory(Job &job) {
  const auto lease = job.session_lease;
  if (lease && !job.partial_tool_call) {
    auto body = json::Parse(job.request.body);
    const auto *validated_messages = body.ok() ? body.value().find("messages") : nullptr;
    if (!validated_messages || !validated_messages->is_array())
      throw std::runtime_error("validated session body became invalid");
    auto messages = validated_messages->as_array();
    json::Value::Object assistant;
    assistant.emplace("role", json::Value(std::string("assistant")));
    assistant.emplace("content", job.content.empty() && !job.tool_calls.empty()
                                     ? json::Value(nullptr)
                                     : json::Value(job.content));
    assistant.emplace("reasoning_content", json::Value(job.reasoning));
    if (!job.tool_calls.empty()) {
      auto calls = json::Parse(ToolCallsJson(job.tool_calls));
      if (!calls.ok())
        throw std::runtime_error("validated calls became invalid");
      assistant.emplace("tool_calls", std::move(calls).value());
    }
    messages.emplace_back(std::move(assistant));
    auto status = sessions_->Commit(lease, json::Stringify(json::Value(std::move(messages))));
    if (!status.ok())
      throw std::runtime_error(status.message());
    {
      std::lock_guard lock(mutex_);
      gpu_session_ = job.request.session_id;
      gpu_owner_ = job.request.session_owner;
      gpu_checkpoint_ = runtime_->Checkpoint();
    }
  }
  if (job.partial_tool_call) {
    std::lock_guard lock(mutex_);
    gpu_session_.clear();
    gpu_owner_.clear();
    gpu_checkpoint_.reset();
  }
}
void Coordinator::Execute(const std::shared_ptr<Job> &job) {
  const auto started = Clock::now();
  job->queue_seconds = std::chrono::duration<double>(started - job->admitted_at).count();
  const auto deadline = started + limits_.generation_timeout;
  auto lease = job->session_lease;
  bool runtime_touched = false, runtime_failed = false;
  try {
    PrepareExecution(*job, started, runtime_touched);
    GenerateOutput(*runtime_, tokenizer_, *job, started, deadline, runtime_failed);
    CommitHistory(*job);
    PublishTerminal(*job);
  } catch (const std::exception &error) {
    {
      std::lock_guard lock(mutex_);
      if (runtime_failed)
        metrics_.ready = false;
      if (runtime_touched) {
        gpu_session_.clear();
        gpu_owner_.clear();
        gpu_checkpoint_.reset();
      }
    }
    if (runtime_touched && runtime_->Stats().phase != SessionPhase::Poisoned)
      (void)runtime_->Reset();
    job->Fail(error.what());
  }
  lease.reset();
  CompleteJob(job);
}
} // namespace mellum16::server
