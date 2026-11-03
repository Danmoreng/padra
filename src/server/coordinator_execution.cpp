#include "server/coordinator.h"
#include "server/output_processing.h"
#include <algorithm>
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
    if (lease && limits_.session_cache_entries) {
      std::shared_ptr<const KVSessionSnapshot> snapshot;
      {
        std::lock_guard lock(mutex_);
        for (auto it = session_cache_.begin(); it != session_cache_.end(); ++it) {
          if (it->id == job.request.session_id && it->owner == job.request.session_owner) {
            snapshot = it->snapshot;
            importing_snapshot_ = snapshot;
            session_cache_.splice(session_cache_.begin(), session_cache_, it);
            break;
          }
        }
      }
      if (snapshot) {
        auto release_snapshot = [&] {
          std::lock_guard lock(mutex_);
          const bool indexed = std::any_of(session_cache_.begin(), session_cache_.end(),
                                           [&](const auto &entry) { return entry.snapshot == snapshot; });
          if (!indexed)
            session_cache_bytes_ -= snapshot->Size();
          importing_snapshot_.reset();
          snapshot.reset();
        };
        try {
          status = lane.runtime->LoadSession(*snapshot);
        } catch (...) {
          release_snapshot();
          throw;
        }
        release_snapshot();
        if (status.ok()) {
          if (lease->record->deleted.load())
            throw std::runtime_error("session discarded during snapshot import");
          auto handle = lane.runtime->Checkpoint();
          if (!handle)
            throw std::runtime_error("imported snapshot has no checkpoint");
          status = lane.runtime->PrepareCheckpointRestore(job.request.input_ids, job.request.outputs,
                                                      *handle, job.request.checkpoint_anchor, false,
                                                      job.request.sampling);
          restored = status.ok();
        }
        if (!restored) {
          {
            std::lock_guard lock(mutex_);
            RemoveCached(job.request.session_id, job.request.session_owner);
          }
          if (status.code() != StatusCode::kInvalidArgument &&
              status.code() != StatusCode::kUnsupported)
            throw std::runtime_error(status.message());
          status = lane.runtime->Reset();
          if (!status.ok())
            throw std::runtime_error(status.message());
        }
      }
    }
    if (!restored)
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
    if (limits_.session_cache_entries) {
      auto handle = lane.runtime->Checkpoint();
      if (handle) {
        const std::size_t needed = KVCheckpointLedger::LocalSnapshotBytes(KVCacheFormat::BF16) +
                                   7ULL * handle->anchor * 2048 +
                                   static_cast<std::size_t>(handle->anchor) * sizeof(int);
        std::size_t available = 0;
        {
          std::lock_guard lock(mutex_);
          RemoveCached(job.request.session_id, job.request.session_owner);
          while (!session_cache_.empty() &&
                 (session_cache_.size() >= limits_.session_cache_entries ||
                  needed > limits_.session_cache_bytes - session_cache_bytes_)) {
            session_cache_bytes_ -= session_cache_.back().snapshot->Size();
            session_cache_.pop_back();
          }
          available = limits_.session_cache_bytes - session_cache_bytes_;
        }
        if (needed <= available) {
          auto saved = lane.runtime->SaveSession(available);
          if (saved.ok()) {
            std::lock_guard lock(mutex_);
            if (!lease->record->deleted.load()) {
              auto snapshot = std::make_shared<KVSessionSnapshot>(std::move(saved).value());
              const auto bytes = snapshot->Size();
              session_cache_.push_front(
                  {job.request.session_owner, job.request.session_id, std::move(snapshot)});
              session_cache_bytes_ += bytes;
            }
          } else if (saved.status().code() != StatusCode::kResourceExhausted &&
                     saved.status().code() != StatusCode::kUnsupported) {
            throw std::runtime_error(saved.status().message());
          }
        }
      }
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
