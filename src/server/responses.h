#pragma once
#include "util/json.h"
#include "server/chat_request.h"
#include <array>
#include <string>
namespace mellum16::server {
struct GenerationResult;
struct ResponseRequest {
  std::string model = kModelId;
  std::string id, previous_id, chat_body, history, instructions = "null", metadata = "{}",
                                                   tools = "[]", tool_choice = "\"none\"";
  std::string reasoning_id, message_id;
  std::string reasoning = "{\"effort\":null,\"summary\":null}";
  // Optional cache-routing hints. They do not create or retain KV state.
  std::string prompt_cache_key, prompt_cache_retention;
  std::array<std::string, 8> function_ids;
  std::int64_t created_at = 0;
  bool store = true, stream = false, parallel = true;
  bool include_encrypted_reasoning = false;
  int outputs = 256;
  float temperature = 0, top_p = 1;
};
// Previous history excludes the previous request's top-level instructions.
// Original nested tool/schema order is retained in chat_body.
Result<ResponseRequest> ParseResponseRequest(std::string_view body,
                                             std::string_view previous_history = "[]",
                                             std::string_view selected_model = kModelId);
std::string ResponseOutputItems(const ResponseRequest &, const GenerationResult &);
std::string ResponseEnvelope(const ResponseRequest &, const GenerationResult *job = nullptr,
                             std::string_view status = "in_progress");
std::string ResponseCompletedHistory(const ResponseRequest &, const GenerationResult &);
} // namespace mellum16::server
