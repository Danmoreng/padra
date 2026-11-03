#pragma once
#include "runtime/model_runtime.h"
#include "server/admission.h"
#include "server/chat_request.h"
#include "server/latency_histogram.h"
#include "server/session_store.h"
#include "server/job.h"
#include "server/server_config.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
namespace mellum16::server {
struct ServerMetrics {
  RejectionCounters rejections{};
  std::uint64_t output_tokens = 0;
  std::uint64_t active_transports = 0, responses_written = 0, write_failures = 0,
                response_body_bytes = 0;
  LatencyHistogram first_visible_write_latency, response_write_latency;
  LatencyHistogram queue_latency, tokenization_latency, prefill_latency, decode_latency,
      first_token_latency, output_available_latency, lifetime_latency;
  std::uint64_t submitted = 0, completed = 0, failed = 0, cancelled = 0, rejected = 0,
                disconnects = 0, backpressure = 0;
  double tokenize_seconds = 0, queue_seconds = 0, prefill_seconds = 0, decode_seconds = 0,
         first_token_seconds = 0, visible_seconds = 0;
  std::size_t queued = 0, queued_bytes = 0, queued_tokens = 0;
  std::size_t active = 0;
  bool ready = true, draining = false;
};
class Coordinator {
public:
  Coordinator(std::unique_ptr<ModelRuntime>, const text::Tokenizer &, ServerLimits = {});
  Coordinator(std::vector<std::unique_ptr<ModelRuntime>>, const text::Tokenizer &,
              ServerLimits = {});
  ~Coordinator();
  Result<std::shared_ptr<Job>> Submit(ChatRequest, SubmissionContext = {});
  void BeginDrain(bool graceful = false);
  bool WaitForDrain();
  void Shutdown();
  ServerMetrics Metrics() const;
  const ServerLimits &Limits() const {
    return limits_;
  }
  AdmissionDecision CheckAdmission() const;
  void RecordRejection(AdmissionCause, ApiRoute);
  RuntimeStats RuntimeMetrics() const {
    return lanes_.front()->runtime->Stats();
  }
  std::vector<RuntimeStats> RuntimeLaneMetrics() const {
    std::vector<RuntimeStats> result;
    for (const auto &lane : lanes_)
      result.push_back(lane->runtime->Stats());
    return result;
  }
  std::size_t RuntimeCount() const {
    return lanes_.size();
  }
  SessionStore &Sessions() {
    return *sessions_;
  }
  Status DiscardSession(const std::string &, const std::string &);
  bool SessionCacheAvailable(const std::string &, const std::string &) const;

private:
  AdmissionDecision CheckAdmissionLocked() const;
  void RecordRejectionLocked(AdmissionCause, ApiRoute);
  struct Lane {
    std::unique_ptr<ModelRuntime> runtime;
    std::shared_ptr<Job> active;
    std::string gpu_session, gpu_owner;
    std::optional<KVCheckpointHandle> gpu_checkpoint;
    std::thread worker;
  };
  bool AnyActiveLocked() const;
  bool CanRunLocked(const Lane &, const Job &) const;
  void Work(Lane &);
  void ExpireQueued();
  void Execute(Lane &, const std::shared_ptr<Job> &);
  void PrepareExecution(Lane &, Job &, std::chrono::steady_clock::time_point,
                        bool &runtime_touched);
  void CommitHistory(Lane &, Job &);
  void CompleteJob(const std::shared_ptr<Job> &);
  void RecordOutcome(const Job &); // Caller holds the coordinator mutex.
  std::vector<std::unique_ptr<Lane>> lanes_;
  const text::Tokenizer &tokenizer_;
  const ServerLimits limits_;
  std::shared_ptr<SessionStore> sessions_ = std::make_shared<SessionStore>();
  mutable std::mutex mutex_;
  std::condition_variable available_;
  std::deque<std::shared_ptr<Job>> queue_;
  std::mutex shutdown_mutex_;
  std::optional<std::chrono::steady_clock::time_point> drain_deadline_;
  std::shared_ptr<TransportCounters> transport_ = std::make_shared<TransportCounters>();
  std::size_t bytes_ = 0, tokens_ = 0;
  ServerMetrics metrics_;
  std::thread expiry_worker_;
};
} // namespace mellum16::server
