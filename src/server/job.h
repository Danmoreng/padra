#pragma once
#include "server/chat_request.h"
#include "server/generation_event.h"
#include "server/generation_result.h"
#include "server/latency_histogram.h"
#include "server/session_store.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
namespace mellum16::server {
struct TransportCounters {
  std::atomic<std::uint64_t> disconnects{0}, backpressure{0};
  std::mutex mutex;
  std::condition_variable changed;
  std::uint64_t active = 0, responses_written = 0, write_failures = 0, body_bytes = 0;
  LatencyHistogram first_visible_write_latency, response_write_latency;
};
struct Job : GenerationResult {
  explicit Job(ChatRequest request, std::size_t maximum_token_bytes);
  ~Job() {
    FinishTransport(false);
  }
  std::shared_ptr<SessionLease> session_lease;
  std::shared_ptr<TransportCounters> transport;
  std::atomic<bool> disconnect_recorded{false}, backpressure_recorded{false};
  void ObserveOutput(OutputKind, std::size_t tool = 0);
  ChatRequest request;
  const std::uint64_t created_at;
  std::atomic<CancelCause> cancellation{CancelCause::None};
  std::mutex mutex;
  std::condition_variable changed;
  std::array<GenerationEvent, 64> events{};
  std::size_t head = 0, tail = 0, count = 0;
  // Completion has its own bounded lane, drained after ordinary deltas.
  std::array<GenerationEvent, 3> endings{};
  std::size_t ending_count = 0, ending_head = 0;
  bool complete = false;
  std::chrono::steady_clock::time_point admitted_at, queue_deadline;
  std::size_t admitted_bytes = 0;
  double lifetime_seconds = 0;
  double queue_seconds = 0, prefill_seconds = 0, decode_seconds = 0, first_token_seconds = 0,
         visible_seconds = 0;
  bool Push(const GenerationEvent &);
  bool PushEnding(const GenerationEvent &);
  bool Pop(GenerationEvent &, std::chrono::milliseconds wait);
  void Cancel(CancelCause cause);
  void Fail(std::string_view); // Worker only, before completion publication.
  void StartTransport();
  void RecordWrite(std::size_t, bool visible);
  void FinishTransport(bool success);
  std::atomic<bool> transport_started{false}, transport_finished{false}, visible_written{false};
  // Caller holds mutex; include the ending lane when deciding to close SSE.
  bool StreamDrained() const {
    return complete && count == 0 && ending_head == ending_count;
  }
};
} // namespace mellum16::server
