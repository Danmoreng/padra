#include "server/responses.h"
#include "server/chat_request.h"
#include "server/generation_result.h"
#include "server/job_error.h"
#include "server/secure_id.h"
#include <algorithm>
#include <chrono>
#include <set>
#include <stdexcept>
namespace mellum16::server {
namespace {
Status Invalid(std::string s) {
  return {StatusCode::kInvalidArgument, std::move(s)};
}
Status Unsupported(std::string s) {
  return {StatusCode::kUnsupported, std::move(s)};
}
bool NonNull(const json::Value &v, const char *key) {
  auto *p = v.find(key);
  return p && !p->is_null();
}
Status Fields(const json::Value &v, std::initializer_list<std::string_view> allowed) {
  if (!v.is_object())
    return Invalid("Responses item must be object");
  for (const auto &[k, value] : v.as_object()) {
    (void)value;
    if (std::find(allowed.begin(), allowed.end(), k) == allowed.end())
      return Unsupported("unsupported Responses field: " + k);
  }
  return {};
}
Result<std::string> TextParts(const json::Value &content, bool assistant) {
  if (content.is_string())
    return content.as_string();
  if (!content.is_array())
    return Unsupported("Responses text content requires string or text parts");
  std::string text;
  for (const auto &p : content.as_array()) {
    auto fields = Fields(p, {"type", "text", "annotations", "logprobs"});
    if (!fields.ok())
      return fields;
    auto *type = p.find("type");
    auto *value = p.find("text");
    if (!type || !type->is_string() || !value || !value->is_string() ||
        (type->as_string() != "input_text" && !(assistant && type->as_string() == "output_text")))
      return Unsupported("only input_text and assistant output_text are implemented");
    for (auto k : {"annotations", "logprobs"})
      if (auto *a = p.find(k); a && (!a->is_array() || !a->as_array().empty()))
        return Unsupported("nonempty annotations/logprobs unsupported");
    text += value->as_string();
  }
  return text;
}
json::Value Message(std::string role, std::string content) {
  return json::Value(json::Value::Object{{"role", json::Value(std::move(role))},
                                         {"content", json::Value(std::move(content))}});
}
// Values are already validated by json::Parse. Scan only to preserve raw schema order.
std::vector<std::string_view> RawArray(std::string_view raw) {
  std::vector<std::string_view> result;
  std::size_t start = 1;
  int depth = 0;
  bool quoted = false, escape = false;
  for (std::size_t p = 1; p < raw.size(); ++p) {
    const char c = raw[p];
    if (quoted) {
      if (escape)
        escape = false;
      else if (c == '\\')
        escape = true;
      else if (c == '"')
        quoted = false;
      continue;
    }
    if (c == '"') {
      quoted = true;
      continue;
    }
    if (c == '{' || c == '[')
      ++depth;
    else if (c == '}' || (c == ']' && depth))
      --depth;
    if (!depth && (c == ',' || c == ']')) {
      auto item = raw.substr(start, p - start);
      auto first = item.find_first_not_of(" \t\r\n");
      if (first != std::string_view::npos) {
        item.remove_prefix(first);
        item = item.substr(0, item.find_last_not_of(" \t\r\n") + 1);
        result.push_back(item);
      }
      start = p + 1;
    }
  }
  return result;
}
Result<std::string> RawField(std::string_view raw, std::string_view name) {
  auto f = FindJsonField(raw, name);
  if (!f.ok())
    return f.status();
  return std::string(
      raw.substr(f.value().value_begin, f.value().value_end - f.value().value_begin));
}
Status AppendItems(json::Value::Array &messages, const json::Value &input) {
  if (input.is_string()) {
    messages.push_back(Message("user", input.as_string()));
    return {};
  }
  if (!input.is_array() || input.as_array().size() > 256)
    return Invalid("Responses input requires text or up to256 text/function/reasoning items");
  // Group one reasoning+message+parallel functions output into a single assistant
  // turn for Mellum's original renderer. A new tool/user message ends the group.
  std::optional<std::size_t> assistant;
  auto ensure = [&]() -> json::Value::Object {
    if (!assistant) {
      messages.push_back(Message("assistant", ""));
      assistant = messages.size() - 1;
    }
    return messages[*assistant].as_object();
  };
  for (const auto &item : input.as_array()) {
    if (!item.is_object())
      return Invalid("Responses input item must be object");
    auto *type = item.find("type");
    if (type && !type->is_string())
      return Invalid("Responses item type requires string");
    const std::string t = type ? type->as_string() : "message";
    if (t == "message") {
      auto fields = Fields(item, {"type", "role", "content", "id", "status"});
      if (!fields.ok())
        return fields;
      auto *role = item.find("role");
      auto *content = item.find("content");
      if (!role || !role->is_string() || !content)
        return Invalid("Responses message needs role/content");
      const auto r = role->as_string() == "developer" ? std::string("system") : role->as_string();
      if (r != "user" && r != "system" && r != "assistant")
        return Unsupported("Responses role is not in original Mellum template");
      auto text = TextParts(*content, r == "assistant");
      if (!text.ok())
        return text.status();
      if (r == "assistant") {
        auto a = ensure();
        auto existing = a.find("content");
        if (existing != a.end() && existing->second.is_string()) {
          auto merged = existing->second.as_string() + text.value();
          a.insert_or_assign("content", json::Value(std::move(merged)));
        } else
          a.insert_or_assign("content", json::Value(text.value()));
        messages[*assistant] = json::Value(std::move(a));
      } else {
        assistant.reset();
        messages.push_back(Message(r, std::move(text).value()));
      }
    } else if (t == "reasoning") {
      auto fields =
          Fields(item, {"type", "id", "status", "summary", "content", "encrypted_content"});
      if (!fields.ok())
        return fields;
      if (NonNull(item, "encrypted_content"))
        return Unsupported("encrypted reasoning is not implemented");
      if (auto *summary = item.find("summary");
          summary && (!summary->is_array() || !summary->as_array().empty()))
        return Unsupported("summary cannot replace full Mellum reasoning");
      auto *parts = item.find("content");
      if (!parts || !parts->is_array())
        return Invalid("reasoning input requires full plaintext content");
      std::string text;
      for (const auto &p : parts->as_array()) {
        auto f = Fields(p, {"type", "text"});
        if (!f.ok())
          return f;
        if (!p.find("type") || !p.find("type")->is_string() ||
            p.find("type")->as_string() != "reasoning_text" || !p.find("text") ||
            !p.find("text")->is_string())
          return Invalid("reasoning content requires reasoning_text");
        text += p.find("text")->as_string();
      }
      auto a = ensure();
      if (a.contains("reasoning_content"))
        return Invalid("duplicate reasoning in one assistant turn");
      a.emplace("reasoning_content", json::Value(std::move(text)));
      messages[*assistant] = json::Value(std::move(a));
    } else if (t == "function_call") {
      auto fields = Fields(item, {"type", "id", "status", "call_id", "name", "arguments"});
      if (!fields.ok())
        return fields;
      for (auto *key : {"call_id", "name", "arguments"})
        if (!item.find(key) || !item.find(key)->is_string())
          return Invalid("function_call requires call_id/name/arguments strings");
      auto a = ensure();
      json::Value::Array calls;
      if (auto it = a.find("tool_calls"); it != a.end())
        calls = it->second.as_array();
      calls.emplace_back(json::Value::Object{
          {"id", *item.find("call_id")},
          {"type", json::Value(std::string("function"))},
          {"function", json::Value(json::Value::Object{{"name", *item.find("name")},
                                                       {"arguments", *item.find("arguments")}})}});
      a.insert_or_assign("tool_calls", json::Value(std::move(calls)));
      if (a.at("content").is_null() || a.at("content").as_string().empty())
        a.insert_or_assign("content", json::Value(nullptr));
      messages[*assistant] = json::Value(std::move(a));
    } else if (t == "function_call_output") {
      auto fields = Fields(item, {"type", "id", "status", "call_id", "output"});
      if (!fields.ok())
        return fields;
      if (!item.find("call_id") || !item.find("call_id")->is_string() || !item.find("output") ||
          !item.find("output")->is_string())
        return Unsupported("function_call_output requires call_id and text output");
      assistant.reset();
      messages.emplace_back(json::Value::Object{{"role", json::Value(std::string("tool"))},
                                                {"tool_call_id", *item.find("call_id")},
                                                {"content", *item.find("output")}});
    } else
      return Unsupported("unsupported Responses input item: " + t);
    if (auto *status = item.find("status");
        status && (!status->is_string() ||
                   (status->as_string() != "completed" && !((t == "message" || t == "reasoning") &&
                                                            status->as_string() == "incomplete"))))
      return Invalid("input items must be completed, or incomplete text/reasoning");
    if (auto *id = item.find("id"); id && !id->is_string())
      return Invalid("item ID must be string");
    if (messages.size() > 256)
      return Invalid("Responses canonical history exceeds256messages");
  }
  return {};
}
} // namespace
Result<ResponseRequest> ParseResponseRequest(std::string_view body, std::string_view previous,
                                             std::string_view selected_model) {
  if (body.size() > 2 * 1024 * 1024 || previous.size() > 8 * 1024 * 1024)
    return Status(StatusCode::kResourceExhausted, "Responses body/history byte limit exceeded");
  auto parsed = json::Parse(body);
  if (!parsed.ok())
    return parsed.status();
  const auto &root = parsed.value();
  auto fields = Fields(root, {"model",
                              "input",
                              "instructions",
                              "previous_response_id",
                              "store",
                              "stream",
                              "tools",
                              "tool_choice",
                              "parallel_tool_calls",
                              "max_output_tokens",
                              "temperature",
                              "top_p",
                              "top_k",
                              "seed",
                              "frequency_penalty",
                              "presence_penalty",
                              "repetition_penalty",
                              "stop",
                              "enable_thinking",
                              "metadata",
                              "prompt_cache_key",
                              "prompt_cache_retention",
                              "reasoning",
                              "text",
                              "truncation",
                              "background",
                              "include",
                              "stream_options"});
  if (!fields.ok())
    return fields;
  auto *model = root.find("model");
  if (!model || !model->is_string() || model->as_string() != selected_model)
    return Status(StatusCode::kNotFound, "unknown model");
  ResponseRequest result;
  result.model = selected_model;
  if (auto *key = root.find("prompt_cache_key"); key && !key->is_null()) {
    if (!key->is_string() || key->as_string().empty() || key->as_string().size() > 64)
      return Invalid("prompt_cache_key requires 1..64 UTF8 bytes");
    result.prompt_cache_key = key->as_string();
  }
  if (auto *retention = root.find("prompt_cache_retention"); retention && !retention->is_null()) {
    if (!retention->is_string() ||
        (retention->as_string() != "in-memory" && retention->as_string() != "24h"))
      return Invalid("prompt_cache_retention requires in-memory or 24h");
    result.prompt_cache_retention = retention->as_string();
  }
  if (auto *v = root.find("previous_response_id"); v && !v->is_null()) {
    if (!v->is_string() || v->as_string().size() != 37 || !v->as_string().starts_with("resp_") ||
        v->as_string().find_first_not_of("0123456789abcdef", 5) != std::string::npos)
      return Invalid("invalid previous_response_id");
    result.previous_id = v->as_string();
  }
  if (result.previous_id.empty() && previous != "[]")
    return Invalid("previous history requires previous_response_id");
  for (auto *key : {"store", "stream", "parallel_tool_calls"})
    if (auto *v = root.find(key); v && !v->is_null()) {
      if (!v->is_bool())
        return Invalid(std::string(key) + " must be boolean");
      if (std::string_view(key) == "store")
        result.store = v->as_bool();
      else if (std::string_view(key) == "stream")
        result.stream = v->as_bool();
      else
        result.parallel = v->as_bool();
    }
  if (auto *v = root.find("instructions"); v && !v->is_null()) {
    if (!v->is_string())
      return Unsupported("instructions requires text");
    result.instructions = json::Quote(v->as_string());
  }
  if (auto *v = root.find("metadata"); v && !v->is_null()) {
    if (!v->is_object() || v->as_object().size() > 16)
      return Invalid("metadata needs at most16string pairs");
    for (const auto &[k, value] : v->as_object())
      if (k.size() > 64 || !value.is_string() || value.as_string().size() > 512)
        return Invalid("metadata key/value length exceeded");
    result.metadata = json::Stringify(*v);
  }
  if (auto *v = root.find("truncation");
      v && !v->is_null() && (!v->is_string() || v->as_string() != "disabled"))
    return Unsupported("only truncation=disabled is implemented");
  if (auto *v = root.find("background"); v && !v->is_null() && (!v->is_bool() || v->as_bool()))
    return Unsupported("background responses are not implemented");
  if (auto *v = root.find("include")) {
    if (!v->is_array() || v->as_array().size() > 1)
      return Unsupported("include supports only reasoning.encrypted_content");
    for (const auto &field : v->as_array()) {
      if (!field.is_string() || field.as_string() != "reasoning.encrypted_content")
        return Unsupported("include supports only reasoning.encrypted_content");
      result.include_encrypted_reasoning = true;
    }
  }
  if (auto *v = root.find("reasoning"); v && !v->is_null()) {
    auto f = Fields(*v, {"effort", "summary"});
    if (!f.ok())
      return f;
    if (auto *effort = v->find("effort"); effort && !effort->is_null()) {
      auto thinking = ThinkingForEffort(*effort);
      if (!thinking.ok())
        return thinking.status();
      if (auto *explicit_thinking = root.find("enable_thinking");
          explicit_thinking &&
          (!explicit_thinking->is_bool() || explicit_thinking->as_bool() != thinking.value()))
        return Invalid("reasoning.effort conflicts with enable_thinking");
    }
    // Automatic presentation uses Mellum's full reasoning_text content. There
    // is no separate summarizer; summary stays empty and optional encryption null.
    if (auto *summary = v->find("summary");
        summary && !summary->is_null() && (!summary->is_string() || summary->as_string() != "auto"))
      return Unsupported("only automatic reasoning presentation is implemented");
    result.reasoning = json::Stringify(*v);
  }
  if (auto *v = root.find("text"); v && !v->is_null()) {
    auto f = Fields(*v, {"format"});
    if (!f.ok())
      return f;
    if (auto *format = v->find("format")) {
      auto g = Fields(*format, {"type"});
      if (!g.ok())
        return g;
      if (!format->find("type") || !format->find("type")->is_string() ||
          format->find("type")->as_string() != "text")
        return Unsupported("only text format is implemented");
    }
  }
  if (auto *v = root.find("stream_options"); v && !v->is_null()) {
    if (!result.stream)
      return Invalid("stream_options requires stream");
    auto f = Fields(*v, {"include_obfuscation"});
    if (!f.ok())
      return f;
    if (auto *ob = v->find("include_obfuscation"); ob && (!ob->is_bool() || ob->as_bool()))
      return Unsupported("stream obfuscation is not implemented");
  }
  auto old = json::Parse(previous);
  if (!old.ok() || !old.value().is_array())
    return Invalid("stored response history is invalid");
  auto messages = old.value().as_array();
  auto *input = root.find("input");
  if (!input)
    return Invalid("Responses input is required");
  auto append = AppendItems(messages, *input);
  if (!append.ok())
    return append;
  // Outstanding function calls in the parent must be resolved by this input.
  auto valid = ValidateToolHistory(messages);
  if (!valid.ok())
    return valid;
  result.history = json::Stringify(json::Value(messages));
  if (result.instructions != "null")
    messages.insert(messages.begin(), Message("system", root.find("instructions")->as_string()));
  std::string chat = "{\"model\":" + json::Quote(selected_model) +
                     ",\"messages\":" + json::Stringify(json::Value(std::move(messages)));
  if (auto *reasoning = root.find("reasoning"); reasoning && NonNull(*reasoning, "effort"))
    chat += ",\"reasoning_effort\":" + json::Stringify(*reasoning->find("effort"));
  for (const auto *key :
       {"temperature", "top_p", "top_k", "seed", "frequency_penalty", "presence_penalty",
        "repetition_penalty", "stop", "enable_thinking", "tool_choice", "parallel_tool_calls"})
    if (root.find(key)) {
      auto raw = RawField(body, key);
      if (!raw.ok())
        return raw.status();
      chat += "," + json::Quote(key) + ":" + raw.value();
      if (std::string_view(key) == "tool_choice")
        result.tool_choice = raw.value();
    }
  if (root.find("max_output_tokens")) {
    auto raw = RawField(body, "max_output_tokens");
    if (!raw.ok())
      return raw.status();
    chat += ",\"max_completion_tokens\":" + raw.value();
  }
  if (auto *tools = root.find("tools"); tools && !tools->is_null()) {
    if (!tools->is_array() || tools->as_array().size() > 16)
      return Invalid("Responses tools require at most16function definitions");
    auto raw = RawField(body, "tools");
    if (!raw.ok())
      return raw.status();
    result.tools = raw.value();
    std::string mapped = "[";
    bool comma = false;
    for (auto item : RawArray(raw.value())) {
      auto t = json::Parse(item);
      if (!t.ok())
        return t.status();
      auto f = Fields(t.value(), {"type", "name", "description", "parameters", "strict"});
      if (!f.ok())
        return f;
      if (!t.value().find("type") || !t.value().find("type")->is_string() ||
          t.value().find("type")->as_string() != "function")
        return Unsupported("only Responses function tools are implemented");
      auto fn = OmitJsonField(item, "type");
      if (!fn.ok())
        return fn.status();
      if (comma)
        mapped += ',';
      mapped += "{\"type\":\"function\",\"function\":" + fn.value() + "}";
      comma = true;
    }
    chat += ",\"tools\":" + mapped + "]";
    if (!root.find("tool_choice") && !tools->as_array().empty())
      result.tool_choice = "\"auto\"";
  }
  chat += ",\"stream\":" + std::string(result.stream ? "true" : "false");
  if (result.stream)
    chat += ",\"stream_options\":{\"include_usage\":true}";
  chat += '}';
  auto config = json::Parse(chat);
  if (!config.ok())
    return config.status();
  auto validated = ValidateTools(config.value());
  if (!validated.ok())
    return validated.status();
  result.tool_choice = validated.value().parse ? "\"auto\"" : "\"none\"";
  result.chat_body = std::move(chat);
  auto id = MakeSecureId("resp_");
  if (!id.ok())
    return id.status();
  result.id = std::move(id).value();
  auto reason = MakeSecureId("rs_");
  if (!reason.ok())
    return reason.status();
  result.reasoning_id = std::move(reason).value();
  auto message = MakeSecureId("msg_");
  if (!message.ok())
    return message.status();
  result.message_id = std::move(message).value();
  for (auto &item : result.function_ids) {
    auto fn = MakeSecureId("fc_");
    if (!fn.ok())
      return fn.status();
    item = std::move(fn).value();
  }
  result.created_at = std::chrono::duration_cast<std::chrono::seconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
  return result;
}
std::string ResponseOutputItems(const ResponseRequest &r, const GenerationResult &j) {
  const std::string status =
      j.finish_reason == "length" || !j.failure.ok() ? "incomplete" : "completed";
  std::string out = "[";
  bool comma = false;
  for (std::size_t n = 0; n < j.output_count; ++n) {
    const auto key = j.output_order[n];
    // Failed validation may leave an announced tool without a valid final call.
    if (key.kind == OutputKind::Tool && key.tool >= j.tool_calls.size())
      continue;
    if (comma)
      out += ',';
    if (key.kind == OutputKind::Reasoning) {
      out += "{\"id\":" + json::Quote(r.reasoning_id) +
             ",\"type\":\"reasoning\",\"summary\":[],\"content\":[{\"type\":\"reasoning_text\","
             "\"text\":" +
             json::Quote(j.reasoning) + "}],\"status\":" + json::Quote(status) +
             (r.include_encrypted_reasoning ? ",\"encrypted_content\":null}" : "}");
    } else if (key.kind == OutputKind::Message) {
      out += "{\"id\":" + json::Quote(r.message_id) +
             ",\"type\":\"message\",\"role\":\"assistant\",\"status\":" + json::Quote(status) +
             ",\"content\":[{\"type\":\"output_text\",\"text\":" + json::Quote(j.content) +
             ",\"annotations\":[],\"logprobs\":[]}]}";
    } else {
      const auto &c = j.tool_calls[key.tool];
      out += "{\"id\":" + json::Quote(r.function_ids[key.tool]) +
             ",\"type\":\"function_call\",\"call_id\":" + json::Quote(c.id) +
             ",\"name\":" + json::Quote(c.name) + ",\"arguments\":" + json::Quote(c.arguments) +
             ",\"status\":" + json::Quote(c.complete ? "completed" : "incomplete") + "}";
    }
    comma = true;
  }
  return out + "]";
}
std::string ResponseEnvelope(const ResponseRequest &r, const GenerationResult *j,
                             std::string_view requested) {
  std::string status(requested);
  if (j && requested == "in_progress")
    status =
        j->failure.ok() ? (j->finish_reason == "length" ? "incomplete" : "completed") : "failed";
  std::string usage = "null";
  if (j)
    usage =
        "{\"input_tokens\":" + std::to_string(j->prompt_tokens) +
        ",\"output_tokens\":" + std::to_string(j->completion_tokens) +
        ",\"total_tokens\":" + std::to_string(j->prompt_tokens + j->completion_tokens) +
        ",\"input_tokens_details\":{\"cached_tokens\":" + std::to_string(j->cached_tokens) +
        "},\"output_tokens_details\":{\"reasoning_tokens\":" + std::to_string(j->reasoning_tokens) +
        "}}";
  return "{\"id\":" + json::Quote(r.id) +
         ",\"object\":\"response\",\"created_at\":" + std::to_string(r.created_at) +
         ",\"status\":" + json::Quote(status) + ",\"error\":" +
         (j && !j->failure.ok()
              ? "{\"code\":" + json::Quote(ClassifyJobError(j->failure_cause).code) +
                    ",\"message\":" + json::Quote(j->failure.message()) + "}"
              : "null") +
         ",\"incomplete_details\":" +
         (status == "incomplete" ? "{\"reason\":\"max_output_tokens\"}" : "null") +
         ",\"instructions\":" + r.instructions + ",\"metadata\":" + r.metadata +
         ",\"model\":" + json::Quote(r.model) +
         ",\"output\":" + (j ? ResponseOutputItems(r, *j) : "[]") +
         ",\"parallel_tool_calls\":" + (r.parallel ? "true" : "false") +
         ",\"previous_response_id\":" +
         (r.previous_id.empty() ? "null" : json::Quote(r.previous_id)) +
         ",\"store\":" + (r.store ? "true" : "false") + ",\"tools\":" + r.tools +
         (r.prompt_cache_key.empty() ? ""
                                     : ",\"prompt_cache_key\":" + json::Quote(r.prompt_cache_key)) +
         (r.prompt_cache_retention.empty()
              ? ""
              : ",\"prompt_cache_retention\":" + json::Quote(r.prompt_cache_retention)) +
         ",\"tool_choice\":" + r.tool_choice + ",\"temperature\":" +
         (j ? std::to_string(j->temperature) : std::to_string(r.temperature)) +
         ",\"top_p\":" + (j ? std::to_string(j->top_p) : std::to_string(r.top_p)) +
         ",\"max_output_tokens\":" +
         (j ? std::to_string(j->output_budget) : std::to_string(r.outputs)) +
         ",\"reasoning\":" + r.reasoning +
         ",\"text\":{\"format\":{\"type\":"
         "\"text\"}},\"truncation\":\"disabled\",\"background\":false,\"usage\":" +
         usage + "}";
}
std::string ResponseCompletedHistory(const ResponseRequest &r, const GenerationResult &j) {
  if (j.partial_tool_call)
    throw std::logic_error("incomplete tool output cannot become continuation history");
  auto old = json::Parse(r.history);
  if (!old.ok() || !old.value().is_array())
    throw std::logic_error("response history lost validated form");
  auto messages = old.value().as_array();
  json::Value::Object a{{"role", json::Value(std::string("assistant"))},
                        {"content", j.content.empty() && !j.tool_calls.empty()
                                        ? json::Value(nullptr)
                                        : json::Value(j.content)},
                        {"reasoning_content", json::Value(j.reasoning)}};
  if (!j.tool_calls.empty()) {
    auto calls = json::Parse(ToolCallsJson(j.tool_calls));
    if (!calls.ok())
      throw std::logic_error("validated tool calls malformed");
    a.emplace("tool_calls", std::move(calls).value());
  }
  messages.emplace_back(std::move(a));
  return json::Stringify(json::Value(std::move(messages)));
}
} // namespace mellum16::server
