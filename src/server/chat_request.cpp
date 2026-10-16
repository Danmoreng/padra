#include "server/chat_request.h"
#include "runtime/kv_checkpoint.h"
#include "util/json.h"
#include <set>
namespace mellum16::server {
namespace {
Status Invalid(std::string message) {
  return {StatusCode::kInvalidArgument, std::move(message)};
}
bool Present(const json::Value &root, const char *key) {
  auto *v = root.find(key);
  return v && !v->is_null();
}
Result<int> Budget(const json::Value *value, int fallback) {
  if (!value || value->is_null())
    return fallback;
  if (!value->is_integer() || value->as_integer() < 1 || value->as_integer() > 131072)
    return Invalid("completion budget must be integer1..131072");
  return static_cast<int>(value->as_integer());
}
} // namespace
std::string ErrorJson(std::string_view message, std::string_view code, std::string_view type) {
  return "{\"error\":{\"message\":" + json::Quote(message) + ",\"type\":" + json::Quote(type) +
         ",\"param\":null,\"code\":" + json::Quote(code) + "}}";
}
Result<ChatRequest> ParseChatRequest(std::string_view body, const text::Tokenizer &tokenizer,
                                     int capacity, std::string id,
                                     std::string_view selected_model) {
  if (body.size() > RequestBodyLimit(capacity))
    return Status(StatusCode::kResourceExhausted,
                  "request body exceeds selected capacity profile byte limit");
  auto parsed = json::Parse(body);
  if (!parsed.ok())
    return parsed.status();
  const auto &root = parsed.value();
  if (!root.is_object())
    return Invalid("request body must be object");
  static const std::set<std::string, std::less<>> allowed{"model",
                                                          "messages",
                                                          "max_completion_tokens",
                                                          "max_tokens",
                                                          "stream",
                                                          "stream_options",
                                                          "temperature",
                                                          "top_p",
                                                          "top_k",
                                                          "seed",
                                                          "frequency_penalty",
                                                          "presence_penalty",
                                                          "repetition_penalty",
                                                          "n",
                                                          "stop",
                                                          "enable_thinking",
                                                          "session_id",
                                                          "tools",
                                                          "tool_choice",
                                                          "parallel_tool_calls"};
  for (const auto &[key, value] : root.as_object()) {
    (void)value;
    if (!allowed.contains(key))
      return Status(StatusCode::kUnsupported, "unsupported request parameter: " + key);
  }
  auto *model = root.find("model");
  if (!model || !model->is_string() || model->as_string() != selected_model)
    return Status(StatusCode::kNotFound, "unknown model; use " + std::string(selected_model));
  auto *messages = root.find("messages");
  if (!messages || !messages->is_array() || messages->as_array().empty() ||
      messages->as_array().size() > 256)
    return Invalid("messages must contain1..256 entries");
  auto history_status = ValidateToolHistory(messages->as_array());
  if (!history_status.ok())
    return history_status;
  auto tool_config = ValidateTools(root);
  if (!tool_config.ok())
    return tool_config.status();
  SamplingOptions sampling;
  for (const auto *key : {"temperature", "top_p"})
    if (auto *v = root.find(key); v && !v->is_null()) {
      if (!v->is_number())
        return Invalid(std::string(key) + " must be number or null");
      const auto number = v->as_number();
      if (!std::isfinite(number) ||
          (std::string_view(key) == "temperature"
               ? (number < 0 || number > 2 || (number > 0 && number < 0.0001))
               : (number <= 0 || number > 1)))
        return Invalid(std::string(key) + " outside admitted range");
      if (std::string_view(key) == "temperature")
        sampling.temperature = static_cast<float>(v->as_number());
      else
        sampling.top_p = static_cast<float>(v->as_number());
    }
  if (auto *v = root.find("top_k"); v && !v->is_null()) {
    if (!v->is_integer() || v->as_integer() < 0 || v->as_integer() > 98304)
      return Invalid("top_k must be integer0..98304");
    sampling.top_k = static_cast<int>(v->as_integer());
  }
  // Omitted seed derives from the OS-random completion ID, never timing/worker order.
  sampling.seed = 14695981039346656037ULL;
  for (unsigned char c : id)
    sampling.seed = (sampling.seed ^ c) * 1099511628211ULL;
  if (auto *v = root.find("seed"); v && !v->is_null()) {
    if (!v->is_integer())
      return Invalid("seed must be signed64-bit integer");
    sampling.seed = static_cast<std::uint64_t>(v->as_integer());
  }
  for (const auto *key : {"frequency_penalty", "presence_penalty", "repetition_penalty"})
    if (auto *v = root.find(key); v && !v->is_null()) {
      if (!v->is_number())
        return Invalid(std::string(key) + " must be number or null");
      const double value = v->as_number();
      const bool repetition = std::string_view(key) == "repetition_penalty";
      if (!std::isfinite(value) ||
          (repetition ? (value < 0.01 || value > 100) : (value < -2 || value > 2)))
        return Invalid(std::string(key) + " outside admitted range");
      if (repetition)
        sampling.repetition_penalty = static_cast<float>(value);
      else if (std::string_view(key) == "frequency_penalty")
        sampling.frequency_penalty = static_cast<float>(value);
      else
        sampling.presence_penalty = static_cast<float>(value);
    }
  auto sample = ValidateSampling(sampling);
  if (!sample.ok())
    return sample;
  if (auto *n = root.find("n"); n && !n->is_null() && (!n->is_integer() || n->as_integer() != 1))
    return Status(StatusCode::kUnsupported, "only n=1 is implemented");
  if (Present(root, "max_tokens") && Present(root, "max_completion_tokens"))
    return Invalid("max_tokens and max_completion_tokens conflict");
  auto budget = Budget(
      root.find(Present(root, "max_completion_tokens") ? "max_completion_tokens" : "max_tokens"),
      256);
  if (!budget.ok())
    return budget.status();
  ChatRequest result;
  result.model = selected_model;
  result.outputs = budget.value();
  result.sampling = sampling;
  result.id = std::move(id);
  result.body = body;
  result.tools = std::move(tool_config).value();
  if (auto *session = root.find("session_id"); session && !session->is_null()) {
    if (!session->is_string() || session->as_string().size() != 37 ||
        !session->as_string().starts_with("sess-") ||
        session->as_string().find_first_not_of("0123456789abcdef", 5) != std::string::npos)
      return Invalid("invalid opaque session_id");
    result.session_id = session->as_string();
  }
  if (auto *v = root.find("stream"); v && !v->is_null()) {
    if (!v->is_bool())
      return Invalid("stream must be boolean");
    result.stream = v->as_bool();
  }
  if (auto *v = root.find("enable_thinking"); v && !v->is_bool())
    return Invalid("enable_thinking must be boolean");
  if (auto *options = root.find("stream_options"); options && !options->is_null()) {
    if (!result.stream || !options->is_object())
      return Invalid("stream_options requires stream=true and object");
    for (const auto &[key, value] : options->as_object()) {
      if (key != "include_usage")
        return Status(StatusCode::kUnsupported, "unsupported stream option: " + key);
      if (!value.is_bool())
        return Invalid("include_usage must be boolean");
      result.include_usage = value.as_bool();
    }
  }
  if (auto *stop = root.find("stop"); stop && !stop->is_null()) {
    if (stop->is_string())
      result.stops.push_back(stop->as_string());
    else if (stop->is_array()) {
      if (stop->as_array().size() > 4)
        return Invalid("at most4 stop strings");
      for (const auto &item : stop->as_array()) {
        if (!item.is_string())
          return Invalid("stop items must be strings");
        result.stops.push_back(item.as_string());
      }
    } else
      return Invalid("stop must be string/array/null");
    for (const auto &s : result.stops)
      if (s.empty() || s.size() > 256)
        return Invalid("stop strings must contain1..256 UTF8 bytes");
  }
  const auto begin = std::chrono::steady_clock::now();
  std::string render_body;
  if (!result.tools.parse && root.find("tools")) {
    auto omitted = OmitJsonField(body, "tools");
    if (!omitted.ok())
      return omitted.status();
    render_body = std::move(omitted).value();
  } else
    render_body = body;
  auto rendered = text::RenderChat(render_body);
  if (!rendered.ok())
    return rendered.status();
  auto encoded = tokenizer.Encode(rendered.value());
  if (!encoded.ok())
    return encoded.status();
  if (encoded.value().empty() || encoded.value().size() > static_cast<std::size_t>(capacity) ||
      static_cast<std::size_t>(result.outputs) >
          static_cast<std::size_t>(capacity) - encoded.value().size())
    return Invalid("prompt plus completion budget exceeds admitted context capacity");
  result.input_ids = std::move(encoded).value();
  if (!result.session_id.empty()) {
    auto status = AttachSession(result, tokenizer, result.session_id);
    if (!status.ok())
      return status;
  }

  result.tokenize_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  return result;
}
Status AttachSession(ChatRequest &request, const text::Tokenizer &tokenizer,
                     const std::string &id) {
  if (id.size() != 37 || !id.starts_with("sess-") ||
      id.find_first_not_of("0123456789abcdef", 5) != std::string::npos)
    return Invalid("invalid opaque session_id");
  const auto begin = std::chrono::steady_clock::now();
  std::string body = request.body;
  if (request.session_id.empty()) {
    const auto end = body.find_last_not_of(" \t\r\n");
    body.insert(end, ",\"session_id\":" + json::Quote(id));
  }
  std::string render_body = body;
  if (!request.tools.parse) {
    auto parsed = json::Parse(body);
    if (!parsed.ok())
      return parsed.status();
    if (parsed.value().find("tools")) {
      auto omitted = OmitJsonField(body, "tools");
      if (!omitted.ok())
        return omitted.status();
      render_body = std::move(omitted).value();
    }
  }
  // Preserve nested tool/schema order while adding the renderer-only option.
  const auto end = render_body.find_last_not_of(" \t\r\n");
  std::string history(render_body.substr(0, end));
  history += ",\"add_generation_prompt\":false}";
  auto stable = text::RenderChat(history);
  if (!stable.ok())
    return stable.status();
  auto stable_ids = tokenizer.Encode(stable.value());
  if (!stable_ids.ok())
    return stable_ids.status();
  auto anchor = PlanCheckpointAnchor(request.input_ids, stable_ids.value());
  if (!anchor.ok())
    return anchor.status();
  request.body = std::move(body);
  request.session_id = id;
  request.checkpoint_anchor = anchor.value();
  request.tokenize_seconds +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  return {};
}

} // namespace mellum16::server
