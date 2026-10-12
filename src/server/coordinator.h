#pragma once
#include "runtime/model_runtime.h"
#include "server/chat_request.h"
#include "server/sse_chunk.h"
#include "server/session_store.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
namespace mellum16::server {
enum class CancelCause {None,Stop,Disconnect,Deadline,QueueDeadline,Backpressure,Shutdown,SessionDiscard,ToolInvalid};
struct EventFrame {std::array<char,8192> data{};std::size_t size=0;};
struct Job {
 explicit Job(ChatRequest request,std::size_t maximum_token_bytes);
 std::shared_ptr<SessionLease> session_lease;int cached_tokens=0;
 std::vector<ToolCall> tool_calls;bool partial_tool_call=false;
 ChatRequest request;std::atomic<CancelCause> cancellation{CancelCause::None};
 std::mutex mutex;std::condition_variable changed;
 std::array<EventFrame,64> events{};std::size_t head=0,tail=0,count=0;
 EventFrame terminal;bool terminal_sent=false,complete=false;Status failure;
 std::string content,reasoning,finish_reason;
 int completion_tokens=0,reasoning_tokens=0;std::size_t admitted_bytes=0;
 double queue_seconds=0,prefill_seconds=0,decode_seconds=0,first_token_seconds=0,visible_seconds=0;
 bool Push(std::span<const char>);
 bool Pop(EventFrame&,std::chrono::milliseconds wait);
 void Cancel(CancelCause cause);
};
struct ServerLimits {
 std::size_t queued_requests=8,queued_bytes=64*1024*1024,queued_tokens=65536;
 std::chrono::milliseconds queue_timeout{5000},generation_timeout{30000};
};
struct ServerMetrics {
 std::uint64_t submitted=0,completed=0,failed=0,cancelled=0,rejected=0,disconnects=0,backpressure=0;
 double tokenize_seconds=0,queue_seconds=0,prefill_seconds=0,decode_seconds=0,first_token_seconds=0,visible_seconds=0;
 std::size_t queued=0,queued_bytes=0,queued_tokens=0;bool active=false,ready=true,draining=false;
};
class Coordinator {
 public:
 Coordinator(std::unique_ptr<ModelRuntime>,const text::Tokenizer&,ServerLimits={});
 ~Coordinator();
 Result<std::shared_ptr<Job>> Submit(ChatRequest);
 void BeginDrain();void Shutdown();
 ServerMetrics Metrics()const;
 RuntimeStats RuntimeMetrics()const{return runtime_->Stats();}
 SessionStore& Sessions(){return *sessions_;}
 Status DiscardSession(const std::string&,const std::string&);
 bool SessionCacheAvailable(const std::string&,const std::string&)const;
 private:
 void Work();void Execute(const std::shared_ptr<Job>&);
 std::unique_ptr<ModelRuntime> runtime_;const text::Tokenizer& tokenizer_;ServerLimits limits_;
 std::shared_ptr<SessionStore> sessions_=std::make_shared<SessionStore>();
 std::string gpu_session_,gpu_owner_;std::optional<KVCheckpointHandle> gpu_checkpoint_;
 mutable std::mutex mutex_;std::condition_variable available_;std::deque<std::shared_ptr<Job>> queue_;std::shared_ptr<Job> active_;
 std::mutex shutdown_mutex_;
 std::size_t bytes_=0,tokens_=0;ServerMetrics metrics_;std::thread worker_;
};
std::string CompletionJson(const Job&);
}
