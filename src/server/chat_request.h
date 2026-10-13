#pragma once
#include "mellum16/status.h"
#include "runtime/sampling.h"
#include "server/tool_protocol.h"
#include "text/tokenizer.h"
#include <chrono>
#include <string>
#include <string_view>
#include <vector>
namespace mellum16::server {
inline constexpr const char *kModelId = "mellum21-exl3-k4-research";
inline std::size_t RequestBodyLimit(int capacity) {
  return capacity <= 8192 ? 512 * 1024 : 2 * 1024 * 1024;
}
struct ChatRequest {
  std::string body, id, session_id, session_owner;
  std::vector<int> input_ids;
  std::vector<std::string> stops;
  int checkpoint_anchor = 0;
  std::size_t adapter_bytes = 0; // Internal bounded Responses context; never a client field.
  ToolConfig tools;
  SamplingOptions sampling;
  int outputs = 256;
  bool stream = false, include_usage = false;
  std::chrono::steady_clock::time_point submitted = std::chrono::steady_clock::now();
  double tokenize_seconds = 0;
};
Result<ChatRequest> ParseChatRequest(std::string_view, const text::Tokenizer &, int capacity,
                                     std::string id);
// Attach a retained root after request validation and response-store admission.
Status AttachSession(ChatRequest &, const text::Tokenizer &, const std::string &);
std::string ErrorJson(std::string_view message, std::string_view code = "invalid_request_error",
                      std::string_view type = "invalid_request_error");
} // namespace mellum16::server
