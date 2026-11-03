#include "server/coordinator.h"
#include "server/output_processing.h"
#include <stdexcept>
namespace mellum16::server {
namespace {
using Clock = std::chrono::steady_clock;
}
void Coordinator::PrepareExecution(Lane &lane, Job &job, Clock::time_point started,
                                   bool &runtime_touched) {
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
    if (lease && lane.gpu_session == job.request.session_id &&
        lane.gpu_owner == job.request.session_owner)
      retained = lane.gpu_checkpoint;
  }
  Status status;
  bool restored = false;
  if (retained) {
    runtime_touched = true;
    status = lane.runtime->PrepareCheckpointRestore(job.request.input_ids, job.request.outputs,
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
      lane.gpu_session.clear();
      lane.gpu_owner.clear();
      lane.gpu_checkpoint.reset();
    }
    status = lane.runtime->Reset();
    if (!status.ok()) {
      std::lock_guard lock(mutex_);
      metrics_.ready = false;
      throw std::runtime_error(status.message());
    }
    status = lease && job.request.checkpoint_anchor && lane.runtime->Stats().checkpoint_enabled
                 ? lane.runtime->PrepareWithCheckpoint(job.request.input_ids, job.request.outputs,
                                                       job.request.checkpoint_anchor, false,
                                                       job.request.sampling)
                 : lane.runtime->PrepareRequestShape(job.request.input_ids, job.request.outputs,
                                                     false, job.request.sampling);
    if (!status.ok()) {
      std::lock_guard lock(mutex_);
      metrics_.ready = false;
      throw std::runtime_error(status.message());
    }
  }
  job.cached_tokens = lane.runtime->Stats().cached_tokens;
}
void Coordinator::CommitHistory(Lane &lane, Job &job) {
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
      lane.gpu_session = job.request.session_id;
      lane.gpu_owner = job.request.session_owner;
      lane.gpu_checkpoint = lane.runtime->Checkpoint();
    }
  }
  if (job.partial_tool_call) {
    std::lock_guard lock(mutex_);
    lane.gpu_session.clear();
    lane.gpu_owner.clear();
    lane.gpu_checkpoint.reset();
  }
}
void Coordinator::Execute(Lane &lane, const std::shared_ptr<Job> &job) {
  const auto started = Clock::now();
  job->queue_seconds = std::chrono::duration<double>(started - job->admitted_at).count();
  const auto deadline = started + limits_.generation_timeout;
  auto lease = job->session_lease;
  bool runtime_touched = false, runtime_failed = false;
  try {
    PrepareExecution(lane, *job, started, runtime_touched);
    GenerateOutput(*lane.runtime, tokenizer_, *job, started, deadline, runtime_failed);
    CommitHistory(lane, *job);
    PublishTerminal(*job);
  } catch (const std::exception &error) {
    {
      std::lock_guard lock(mutex_);
      if (runtime_failed)
        metrics_.ready = false;
      if (runtime_touched) {
        lane.gpu_session.clear();
        lane.gpu_owner.clear();
        lane.gpu_checkpoint.reset();
      }
    }
    if (runtime_touched && lane.runtime->Stats().phase != SessionPhase::Poisoned) {
      auto cleanup = lane.runtime->Reset();
      if (!cleanup.ok()) {
        std::lock_guard lock(mutex_);
        metrics_.ready = false;
      }
    }
    job->Fail(error.what());
  }
  lease.reset();
  CompleteJob(job);
}
} // namespace mellum16::server
