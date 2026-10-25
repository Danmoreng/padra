// Host-side owner of the live existing-forward frame. No CUDA calls from API
// callers; commands hand off at synchronized, non-captured worker boundaries.
#pragma once
#include "runtime/model_runtime.h"
#include "runtime/chunk_graph_contract.h"
#include "runtime/mtp_memory.h"
#include <condition_variable>
#include <mutex>
#include <thread>
#include <stdexcept>
namespace mellum16 {
inline std::uint64_t NewCheckpointRuntimeBinding() {
  static std::atomic<std::uint64_t> next{1};
  auto value = next.load();
  do {
    if (value == UINT64_MAX)
      throw std::overflow_error("checkpoint runtime identity exhausted");
  } while (!next.compare_exchange_weak(value, value + 1));
  return value;
}
class CudaModelWorker final : public RuntimeBackend {
public:
  explicit CudaModelWorker(InferenceOptions profile)
      : options(std::move(profile)), state(options.capacity), prefix(options.capacity) {
    stats.capacity = options.capacity;
    stats.mtp_depth = options.mtp.artifact.empty() ? 0 : options.mtp.depth;
    if (stats.mtp_depth)
      stats.mtp_resident_bytes =
          MtpDraftResidentBytes(options.mtp.fp8_experts, options.mtp.exl3_experts) +
          kMtpVerifyResidentBytes + MtpHiddenBytes(options.prefill_chunk) +
          (options.server_kv_checkpoint ? kMtpCheckpointBytes : 0);
    stats.expert_rate = options.exl3_rate;
    stats.prefill_chunk = options.prefill_chunk;
    stats.device_budget_gib = options.server_device_budget_gib;
    stats.kv_format = options.kv_cache_format;
    stats.controlled_chunks = options.controlled_chunk_graphs;
    stats.checkpoint_enabled = options.server_kv_checkpoint;
    stats.checkpoint_bytes = options.server_kv_checkpoint
                                 ? KVCheckpointLedger::LocalSnapshotBytes(options.kv_cache_format)
                                 : 0;
    if (options.server_kv_checkpoint && !options.mtp.artifact.empty())
      stats.checkpoint_bytes += kMtpCheckpointBytes;
    if (options.server_kv_checkpoint)
      checkpoint.emplace(options.capacity, NewCheckpointRuntimeBinding(), options.kv_cache_format);
    options.input_ids.resize(options.capacity, 0);
    options.generate = 0;
    options.capture_logits = true;
    options.cancellation_requested = &cancel;
  }
  ~CudaModelWorker() override {
    cancel.store(true);
    {
      std::lock_guard lock(mutex);
      command = Command::Shutdown;
      cv.notify_all();
    }
    if (thread.joinable())
      thread.join();
  }
  Status Prepare(std::span<const int> ids, int outputs, bool logits,
                 SamplingOptions sampling = {}) override {
    return PrepareImpl(ids, outputs, 0, logits, sampling);
  }
  Status PrepareCheckpoint(std::span<const int> ids, int outputs, int anchor, bool logits,
                           SamplingOptions sampling = {}) override {
    if (!checkpoint)
      return {StatusCode::kUnsupported, "checkpoint profile not enabled"};
    if (anchor < 1)
      return {StatusCode::kInvalidArgument, "checkpoint anchor must be positive"};
    return PrepareImpl(ids, outputs, anchor, logits, sampling);
  }
  Status PrepareImpl(std::span<const int> ids, int outputs, int anchor, bool logits,
                     SamplingOptions sampling) {
    if (options.diagnostic_decode_token >= 0 && (outputs > 2 || sampling.selection_graph()))
      return {StatusCode::kUnsupported,
              "fixed teacher diagnostic permits at most2 greedy outputs, no sampling"};
    std::unique_lock api(api_mutex, std::try_to_lock);
    if (!api.owns_lock())
      return Busy();
    std::unique_lock lock(mutex);
    if (failed)
      return fault;
    if (auto status = ValidateSampling(sampling); !status.ok())
      return status;
    if (anchor < 0 || static_cast<std::size_t>(anchor) > ids.size())
      return {StatusCode::kInvalidArgument, "checkpoint anchor outside canonical prompt"};
    if (anchor)
      anchor = CheckpointCaptureAnchor(anchor, options.prefill_chunk);
    if (request_epoch == UINT32_MAX)
      return {StatusCode::kResourceExhausted, "request epoch exhausted; recreate runtime"};
    auto status = state.Prepare(ids, outputs);
    if (!status.ok())
      return status;
    if (checkpoint)
      checkpoint->Invalidate();
    checkpoint_anchor = anchor;
    checkpoint_restore_pending = false;
    options.input_ids.assign(ids.begin(), ids.end());
    options.generate = outputs;
    options.capture_logits = logits;
    options.sampling = sampling;
    ++request_epoch;
    append_base = 0;
    cancel.store(false);
    prepared = false;
    completed = false;
    command = Command::Prepare;
    cv.notify_all();
    cv.wait(lock, [&] { return prepared || failed; });
    return failed ? fault : Status{};
  }
  Status RestoreCheckpoint(std::span<const int> ids, int outputs, const KVCheckpointHandle &handle,
                           int next_anchor, bool logits, SamplingOptions sampling = {}) override {
    std::unique_lock api(api_mutex, std::try_to_lock);
    if (!api.owns_lock())
      return Busy();
    std::unique_lock lock(mutex);
    if (failed)
      return fault;
    if (!checkpoint)
      return {StatusCode::kUnsupported, "checkpoint restore requires enabled runtime"};
    if (options.diagnostic_decode_token >= 0 && (outputs > 2 || sampling.selection_graph()))
      return {StatusCode::kUnsupported,
              "fixed teacher checkpoint restore permits at most2 greedy outputs, no sampling"};
    auto base = checkpoint->RestoreBase(ids, handle);
    if (!base.ok())
      return base.status();
    if (next_anchor < 0 || (next_anchor && (next_anchor < base.value() ||
                                            static_cast<std::size_t>(next_anchor) > ids.size())))
      return {StatusCode::kInvalidArgument, "next checkpoint anchor must extend the saved prefix"};
    if (next_anchor)
      next_anchor = CheckpointCaptureAnchor(next_anchor, options.prefill_chunk);
    if (auto status = ValidateSampling(sampling); !status.ok())
      return status;
    if (request_epoch == UINT32_MAX)
      return {StatusCode::kResourceExhausted, "request epoch exhausted; recreate runtime"};
    auto status = state.PrepareCheckpointRestore(ids, outputs, base.value());
    if (!status.ok())
      return status;
    ++request_epoch;
    append_base = base.value();
    checkpoint_anchor = next_anchor == base.value() ? 0 : next_anchor;
    checkpoint_restore_pending = true;
    options.input_ids.assign(ids.begin(), ids.end());
    options.generate = outputs;
    options.capture_logits = logits;
    options.sampling = sampling;
    cancel.store(false);
    prepared = false;
    completed = false;
    command = Command::Prepare;
    cv.notify_all();
    cv.wait(lock, [&] { return prepared || failed; });
    return failed ? fault : Status{};
  }
  std::optional<KVCheckpointHandle> Checkpoint() const override {
    std::lock_guard lock(mutex);
    return checkpoint ? checkpoint->handle() : std::optional<KVCheckpointHandle>{};
  }
  void BeginCheckpointCapture() {
    std::lock_guard lock(mutex);
    if (!checkpoint || !checkpoint_anchor)
      throw std::runtime_error("checkpoint capture without requested binding/anchor");
    checkpoint->Invalidate();
  }
  void CompleteCheckpointCapture() {
    std::lock_guard lock(mutex);
    auto status = checkpoint->Capture(options.input_ids, checkpoint_anchor);
    if (!status.ok())
      throw std::runtime_error(status.message());
    ++stats.checkpoint_captures;
  }
  void RecordCheckpointRestore() {
    std::lock_guard lock(mutex);
    if (!checkpoint_restore_pending || !checkpoint || !checkpoint->handle())
      throw std::runtime_error("checkpoint restore lacks completed snapshot");
    checkpoint_restore_pending = false;
    ++stats.checkpoint_restores;
  }
  Status Append(std::span<const int> ids, int outputs, bool logits,
                SamplingOptions sampling = {}) override {
    if (options.diagnostic_decode_token >= 0)
      return {StatusCode::kUnsupported,
              "fixed teacher diagnostic cannot reuse a canonical generated prefix"};
    std::unique_lock api(api_mutex, std::try_to_lock);
    if (!api.owns_lock())
      return Busy();
    std::unique_lock lock(mutex);
    if (failed)
      return fault;
    if (!options.controlled_chunk_graphs)
      return {StatusCode::kUnsupported, "Append requires controlled chunk graphs"};
    auto base = prefix.AppendBase(ids);
    if (!base.ok())
      return base.status();
    if (auto status = ValidateSampling(sampling); !status.ok())
      return status;
    if (request_epoch == UINT32_MAX)
      return {StatusCode::kResourceExhausted, "request epoch exhausted; recreate runtime"};
    auto status = state.PrepareAppend(ids, outputs, static_cast<int>(base.value()));
    if (!status.ok())
      return status;
    checkpoint_anchor = 0;
    checkpoint_restore_pending = false;
    ++request_epoch;
    append_base = static_cast<int>(base.value());
    options.input_ids.assign(ids.begin(), ids.end());
    options.generate = outputs;
    options.capture_logits = logits;
    options.sampling = sampling;
    cancel.store(false);
    prepared = false;
    completed = false;
    command = Command::Prepare;
    cv.notify_all();
    cv.wait(lock, [&] { return prepared || failed; });
    return failed ? fault : Status{};
  }
  void CommitChunk(int count, bool eager) {
    {
      std::lock_guard lock(mutex);
      ++stats.prefill_chunks;
      if (eager)
        ++stats.eager_tail_chunks;
      auto status = state.CommitPrefix(count);
      if (!status.ok())
        throw std::runtime_error(status.message());
    }
    if (progress_callback && !progress_callback(count))
      cancel.store(true);
  }
  Result<InferenceResult> Generate(TokenCallback callback, PrefillCallback progress = {}) override {
    std::unique_lock api(api_mutex, std::try_to_lock);
    if (!api.owns_lock())
      return Busy();
    std::unique_lock lock(mutex);
    if (failed)
      return fault;
    auto status = state.Begin(cancel.load());
    if (!status.ok())
      return status;
    token_callback = std::move(callback);
    progress_callback = std::move(progress);
    command = Command::Generate;
    cv.notify_all();
    cv.wait(lock, [&] { return completed || failed; });
    token_callback = {};
    progress_callback = {};
    if (failed)
      return fault;
    return output;
  }
  Status Reset() override {
    std::unique_lock api(api_mutex, std::try_to_lock);
    if (!api.owns_lock())
      return Busy();
    std::unique_lock lock(mutex);
    if (failed)
      return fault;
    if (state.phase() == SessionPhase::Prepared) {
      discarded = false;
      command = Command::Discard;
      cv.notify_all();
      cv.wait(lock, [&] { return discarded || failed; });
      if (failed)
        return fault;
    }
    auto status = state.Reset();
    if (status.ok()) {
      prefix.Reset();
      if (checkpoint)
        checkpoint->Invalidate();
      checkpoint_anchor = 0;
      checkpoint_restore_pending = false;
      append_base = 0;
    }
    return status;
  }
  void Cancel() override {
    cancel.store(true);
  }
  RuntimeStats Stats() const override {
    std::lock_guard lock(mutex);
    auto result = stats;
    result.phase = state.phase();
    result.committed = state.committed();
    result.pending = state.pending();
    result.checkpoint_available = checkpoint && checkpoint->handle().has_value();
    result.cached_tokens = append_base;
    return result;
  }
  bool AwaitPrepare() {
    std::unique_lock lock(mutex);
    loaded = true;
    cv.notify_all();
    cv.wait(lock, [&] { return command == Command::Prepare || command == Command::Shutdown; });
    if (command == Command::Shutdown)
      return false;
    command = Command::None;
    return true;
  }
  bool AwaitGenerate() {
    std::unique_lock lock(mutex);
    prepared = true;
    cv.notify_all();
    cv.wait(lock, [&] {
      return command == Command::Generate || command == Command::Discard ||
             command == Command::Shutdown;
    });
    if (command == Command::Shutdown)
      return false;
    dropped = command == Command::Discard;
    command = Command::None;
    if (dropped) {
      discarded = true;
      cv.notify_all();
    }
    return true;
  }
  bool Emit(int token, bool first) {
    {
      std::lock_guard lock(mutex);
      if (first) {
        auto status = state.CommitPrefill();
        if (!status.ok())
          throw std::runtime_error(status.message());
      }
      if (options.generate > 0) {
        auto status = state.Emit(token);
        if (!status.ok())
          throw std::runtime_error(status.message());
      }
    }
    // Callback is never a graph node. Reentrant API methods return busy; Cancel
    // and Stats remain available. Callback exceptions poison via worker catch.
    if (options.generate > 0 && token_callback && !token_callback(token))
      cancel.store(true);
    return !cancel.load();
  }
  void Publish(const InferenceResult &result) {
    std::lock_guard lock(mutex);
    stats.mtp_groups += result.mtp_groups;
    stats.mtp_draft_calls += result.mtp_draft_calls;
    stats.mtp_proposed += result.mtp_proposed;
    stats.mtp_accepted += result.mtp_accepted;
    stats.mtp_ordinary_tails += result.mtp_ordinary_tails;
    if (result.cancelled)
      state.Cancel();
    else
      state.Finish();
    auto status = prefix.Capture(options.input_ids, result.generated_ids,
                                 static_cast<std::size_t>(state.committed()), state.pending());
    if (!status.ok())
      throw std::runtime_error(status.message());
    output = result;
    completed = true;
    ++stats.requests;
    cv.notify_all();
  }
  void Fail(Status status) {
    std::lock_guard lock(mutex);
    (void)(fault = std::move(status));
    failed = true;
    state.Poison();
    if (checkpoint)
      checkpoint->Invalidate();
    cv.notify_all();
  }
  void RecordAdmission(std::size_t total, std::size_t free, std::size_t available,
                       std::size_t required) {
    std::lock_guard lock(mutex);
    stats.device_total_bytes = total;
    stats.device_free_before_bytes = free;
    stats.admission_available_bytes = available;
    stats.required_bytes = required;
  }
  void RecordGraphAdmission(std::size_t available) {
    std::lock_guard lock(mutex);
    stats.post_graph_available_bytes = available;
  }
  void RecordLoad(std::uint64_t bytes) {
    std::lock_guard lock(mutex);
    ++stats.model_loads;
    ++stats.weight_uploads;
    ++stats.static_initializations;
    stats.resident_bytes = bytes;
  }
  void RecordIdentity(const std::string &math, const std::string &kernel,
                      const std::string &weights, const std::string &manifest) {
    std::lock_guard lock(mutex);
    stats.math_profile = math;
    stats.kernel_profile = kernel;
    stats.weights_sha256 = weights;
    stats.manifest_sha256 = manifest;
  }
  void RecordCapture() {
    std::lock_guard lock(mutex);
    ++stats.graph_captures;
  }
  static Status Busy() {
    return {StatusCode::kResourceExhausted, "one runtime worker/session already in use"};
  }
  InferenceOptions options;
  std::atomic<bool> cancel{false};
  std::thread thread;
  mutable std::mutex mutex;
  std::condition_variable cv;
  bool loaded = false, prepared = false, completed = false, failed = false, discarded = false,
       dropped = false;
  Status fault;
  RuntimeStats stats;
  SessionState state;
  SessionPrefix prefix;
  int append_base = 0;
  std::uint32_t request_epoch = 0;
  std::optional<KVCheckpointLedger> checkpoint;
  int checkpoint_anchor = 0;
  bool checkpoint_restore_pending = false;

private:
  enum class Command { None, Prepare, Generate, Discard, Shutdown };
  Command command = Command::None;
  std::mutex api_mutex;
  TokenCallback token_callback;
  PrefillCallback progress_callback;
  InferenceResult output;
};
} // namespace mellum16
