#pragma once
#include "runtime/model_runtime.h"
#include "server/chat_request.h"
#include "server/latency_histogram.h"
#include "server/session_store.h"
#include "server/sse_chunk.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
namespace mellum16::server {
enum class CancelCause {
  None,
  Disconnect,
  Deadline,
  QueueDeadline,
  Backpressure,
  Shutdown,
  SessionDiscard,
  ToolInvalid
};
struct EventFrame {
  std::array<char, 8192> data{};
  std::size_t size = 0;
  bool visible = false;
};
struct TransportCounters {
  std::atomic<std::uint64_t> disconnects{0}, backpressure{0};
  std::mutex mutex;
  std::condition_variable changed;
  std::uint64_t active = 0, responses_written = 0, write_failures = 0, body_bytes = 0;
  LatencyHistogram first_visible_write_latency, response_write_latency;
};
enum class OutputKind { Reasoning, Message, Tool };
struct OutputKey {
  OutputKind kind;
  std::size_t tool = 0;
};
struct Job {
  explicit Job(ChatRequest request, std::size_t maximum_token_bytes);
  ~Job() {
    FinishTransport(false);
  }
  std::shared_ptr<SessionLease> session_lease;
  std::shared_ptr<TransportCounters> transport;
  std::atomic<bool> disconnect_recorded{false}, backpressure_recorded{false};
  int cached_tokens = 0;
  std::vector<ToolCall> tool_calls;
  bool partial_tool_call = false;
  // Worker-owned until completion; adapters read only after complete is published.
  // Repeated text/reasoning extends the same item; each tool index is distinct.
  std::array<OutputKey, 10> output_order{};
  std::size_t output_count = 0;
  void ObserveOutput(OutputKind, std::size_t tool = 0);
  ChatRequest request;
  std::atomic<CancelCause> cancellation{CancelCause::None};
  std::mutex mutex;
  std::condition_variable changed;
  std::array<EventFrame, 64> events{};
  std::size_t head = 0, tail = 0, count = 0;
  // Completion has its own bounded lane, drained after ordinary deltas.
  std::array<EventFrame, 3> endings{};
  std::size_t ending_count = 0, ending_head = 0;
  bool complete = false;
  Status failure;
  std::chrono::steady_clock::time_point admitted_at, queue_deadline;
  std::string content, reasoning, finish_reason;
  int completion_tokens = 0, reasoning_tokens = 0;
  std::size_t admitted_bytes = 0;
  double lifetime_seconds = 0;
  double queue_seconds = 0, prefill_seconds = 0, decode_seconds = 0, first_token_seconds = 0,
         visible_seconds = 0;
  bool Push(std::span<const char>, bool visible = false);
  bool PushEnding(std::span<const char>);
  bool Pop(EventFrame &, std::chrono::milliseconds wait);
  void Cancel(CancelCause cause);
  void StartTransport();
  void RecordWrite(std::size_t, bool visible);
  void FinishTransport(bool success);
  std::atomic<bool> transport_started{false}, transport_finished{false}, visible_written{false};
  // Caller holds mutex; include the ending lane when deciding to close SSE.
  bool StreamDrained() const {
    return complete && count == 0 && ending_head == ending_count;
  }
};
struct ServerLimits {
  std::size_t queued_requests = 8, queued_bytes = 64 * 1024 * 1024, queued_tokens = 65536;
  std::chrono::milliseconds queue_timeout{5000}, generation_timeout{30000}, drain_timeout{5000};
};
struct ServerMetrics {
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
  bool active = false, ready = true, draining = false;
};
class Coordinator {
public:
  Coordinator(std::unique_ptr<ModelRuntime>, const text::Tokenizer &, ServerLimits = {});
  ~Coordinator();
  Result<std::shared_ptr<Job>> Submit(ChatRequest, bool track_transport = false);
  void BeginDrain(bool graceful = false);
  bool WaitForDrain();
  void Shutdown();
  ServerMetrics Metrics() const;
  bool AdmissionAvailable() const;
  RuntimeStats RuntimeMetrics() const {
    return runtime_->Stats();
  }
  SessionStore &Sessions() {
    return *sessions_;
  }
  Status DiscardSession(const std::string &, const std::string &);
  bool SessionCacheAvailable(const std::string &, const std::string &) const;

private:
  void Work();
  void ExpireQueued();
  void Execute(const std::shared_ptr<Job> &);
  void CompleteJob(const std::shared_ptr<Job> &);
  void RecordOutcome(const Job &); // Caller holds the coordinator mutex.
  std::unique_ptr<ModelRuntime> runtime_;
  const text::Tokenizer &tokenizer_;
  ServerLimits limits_;
  std::shared_ptr<SessionStore> sessions_ = std::make_shared<SessionStore>();
  std::string gpu_session_, gpu_owner_;
  std::optional<KVCheckpointHandle> gpu_checkpoint_;
  mutable std::mutex mutex_;
  std::condition_variable available_;
  std::deque<std::shared_ptr<Job>> queue_;
  std::shared_ptr<Job> active_;
  std::mutex shutdown_mutex_;
  std::optional<std::chrono::steady_clock::time_point> drain_deadline_;
  std::shared_ptr<TransportCounters> transport_ = std::make_shared<TransportCounters>();
  std::size_t bytes_ = 0, tokens_ = 0;
  ServerMetrics metrics_;
  std::thread worker_, expiry_worker_;
};
std::string CompletionJson(const Job &);
} // namespace mellum16::server
