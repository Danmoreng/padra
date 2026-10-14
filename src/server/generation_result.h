#pragma once
#include "server/tool_protocol.h"
#include <array>
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
enum class OutputKind { Reasoning, Message, Tool };
struct OutputKey {
  OutputKind kind;
  std::size_t tool = 0;
};
// Solely written by the generation/expiry worker. Job::complete publishes this
// result under the job mutex; protocol encoders and retention then read const
// references. Transport cancellation after publication cannot rewrite it.
struct GenerationResult {
  std::string content, reasoning, finish_reason;
  std::vector<ToolCall> tool_calls;
  std::array<OutputKey, 10> output_order{};
  std::size_t output_count = 0, prompt_tokens = 0;
  int completion_tokens = 0, reasoning_tokens = 0, cached_tokens = 0, output_budget = 0;
  float temperature = 0, top_p = 1;
  bool partial_tool_call = false;
  Status failure;
  CancelCause failure_cause = CancelCause::None;
};
} // namespace mellum16::server
