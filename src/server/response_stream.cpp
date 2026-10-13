#include "server/response_stream.h"
#include "server/coordinator.h"
namespace mellum16::server {
ResponseStream::ResponseStream(const ResponseRequest &request, Send send)
    : request_(request), send_(std::move(send)) {
  items_.reserve(10);
}
bool ResponseStream::Event(std::string_view type, std::string_view fields) {
  return send_("event: " + std::string(type) + "\ndata: {\"type\":" + json::Quote(type) +
               ",\"sequence_number\":" + std::to_string(sequence_++) +
               (fields.empty() ? "" : "," + std::string(fields)) + "}\n\n");
}
bool ResponseStream::Begin() {
  return Event("response.created", "\"response\":" + ResponseEnvelope(request_)) &&
         Event("response.in_progress", "\"response\":" + ResponseEnvelope(request_));
}
bool ResponseStream::Fail(std::string_view message) {
  return Event("error",
               "\"code\":\"server_error\",\"message\":" + json::Quote(message) + ",\"param\":null");
}
Result<std::size_t> ResponseStream::Ensure(std::string type, std::string id, std::string item,
                                           std::string part) {
  for (const auto &i : items_)
    if (i.id == id)
      return i.index;
  const auto index = items_.size();
  if (index >= 10)
    return Status(StatusCode::kResourceExhausted, "too many response output items");
  if (!Event("response.output_item.added",
             "\"output_index\":" + std::to_string(index) + ",\"item\":" + item))
    return Status(StatusCode::kCancelled, "Responses network send failed");
  items_.push_back({std::move(id), std::move(type), index});
  if (!part.empty() &&
      !Event("response.content_part.added", "\"output_index\":" + std::to_string(index) +
                                                ",\"item_id\":" + json::Quote(items_.back().id) +
                                                ",\"content_index\":0,\"part\":" + part))
    return Status(StatusCode::kCancelled, "Responses part send failed");
  return index;
}
bool ResponseStream::Delta(std::string_view type, std::string_view id, std::string_view text) {
  const bool reasoning = type == "reasoning";
  std::string part =
      reasoning ? "{\"type\":\"reasoning_text\",\"text\":\"\"}"
                : "{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[],\"logprobs\":[]}";
  std::string item =
      "{\"id\":" + json::Quote(id) + ",\"type\":" + json::Quote(type) +
      ",\"status\":\"in_progress\"," +
      (reasoning ? "\"summary\":[],\"content\":[]}" : "\"role\":\"assistant\",\"content\":[]}");
  auto index = Ensure(std::string(type), std::string(id), std::move(item), std::move(part));
  if (!index.ok())
    return false;
  return Event(reasoning ? "response.reasoning_text.delta" : "response.output_text.delta",
               "\"output_index\":" + std::to_string(index.value()) + ",\"item_id\":" +
                   json::Quote(id) + ",\"content_index\":0,\"delta\":" + json::Quote(text) +
                   (reasoning ? "" : ",\"logprobs\":[]"));
}
bool ResponseStream::ChatFrame(std::string_view frame) {
  if (!frame.starts_with("data: "))
    return false;
  frame.remove_prefix(6);
  while (!frame.empty() && (frame.back() == '\n' || frame.back() == '\r'))
    frame.remove_suffix(1);
  if (frame == "[DONE]")
    return true;
  auto parsed = json::Parse(frame);
  if (!parsed.ok())
    return false;
  const auto &root = parsed.value();
  if (root.find("error"))
    return true;
  auto *choices = root.find("choices");
  if (!choices || !choices->is_array())
    return false;
  for (const auto &choice : choices->as_array()) {
    auto *d = choice.find("delta");
    if (!d || !d->is_object())
      continue;
    if (auto *reason = d->find("reasoning_content");
        reason && reason->is_string() && !reason->as_string().empty())
      if (!Delta("reasoning", request_.reasoning_id, reason->as_string()))
        return false;
    if (auto *text = d->find("content"); text && text->is_string() && !text->as_string().empty())
      if (!Delta("message", request_.message_id, text->as_string()))
        return false;
    if (auto *calls = d->find("tool_calls"); calls && calls->is_array())
      for (const auto &c : calls->as_array()) {
        auto *idx = c.find("index");
        auto *fn = c.find("function");
        if (!idx || !idx->is_integer() || idx->as_integer() < 0 || idx->as_integer() > 7 || !fn ||
            !fn->is_object())
          return false;
        const auto i = static_cast<std::size_t>(idx->as_integer());
        const auto &id = request_.function_ids[i];
        std::optional<std::size_t> output;
        for (const auto &found : items_)
          if (found.id == id)
            output = found.index;
        if (auto *call = c.find("id"); call && call->is_string()) {
          auto *name = fn->find("name");
          if (!name || !name->is_string())
            return false;
          auto added =
              Ensure("function_call", id,
                     "{\"id\":" + json::Quote(id) +
                         ",\"type\":\"function_call\",\"status\":\"in_progress\",\"call_id\":" +
                         json::Quote(call->as_string()) +
                         ",\"name\":" + json::Quote(name->as_string()) + ",\"arguments\":\"\"}");
          if (!added.ok())
            return false;
          output = added.value();
        }
        if (!output)
          return false;
        if (auto *args = fn->find("arguments");
            args && args->is_string() && !args->as_string().empty())
          if (!Event("response.function_call_arguments.delta",
                     "\"output_index\":" + std::to_string(*output) + ",\"item_id\":" +
                         json::Quote(id) + ",\"delta\":" + json::Quote(args->as_string())))
            return false;
      }
  }
  return true;
}
bool ResponseStream::Finish(const Job &job) {
  auto parsed = json::Parse(ResponseOutputItems(request_, job));
  if (!parsed.ok())
    return false;
  for (const auto &item : parsed.value().as_array()) {
    const auto &id = item.find("id")->as_string();
    const auto &type = item.find("type")->as_string();
    std::optional<std::size_t> index;
    for (const auto &found : items_)
      if (found.id == id)
        index = found.index;
    if (!index) {
      auto initial = item.as_object();
      initial.insert_or_assign("status", json::Value(std::string("in_progress")));
      std::string part;
      if (type == "function_call")
        initial.insert_or_assign("arguments", json::Value(std::string()));
      else {
        initial.insert_or_assign("content", json::Value(json::Value::Array{}));
        auto empty = item.find("content")->as_array()[0].as_object();
        empty.insert_or_assign("text", json::Value(std::string()));
        part = json::Stringify(json::Value(std::move(empty)));
      }
      auto added =
          Ensure(type, id, json::Stringify(json::Value(std::move(initial))), std::move(part));
      if (!added.ok())
        return false;
      index = added.value();
    }
    std::string fields =
        "\"output_index\":" + std::to_string(*index) + ",\"item_id\":" + json::Quote(id);
    if (type == "function_call") {
      if (!Event("response.function_call_arguments.done",
                 fields + ",\"arguments\":" + json::Quote(item.find("arguments")->as_string()) +
                     ",\"name\":" + json::Quote(item.find("name")->as_string())))
        return false;
    } else {
      const auto &part = item.find("content")->as_array()[0];
      if (!Event(type == "reasoning" ? "response.reasoning_text.done" : "response.output_text.done",
                 fields + ",\"content_index\":0,\"text\":" +
                     json::Quote(part.find("text")->as_string()) +
                     (type == "reasoning" ? "" : ",\"logprobs\":[]")))
        return false;
      if (!Event("response.content_part.done",
                 fields + ",\"content_index\":0,\"part\":" + json::Stringify(part)))
        return false;
    }
    if (!Event("response.output_item.done",
               "\"output_index\":" + std::to_string(*index) + ",\"item\":" + json::Stringify(item)))
      return false;
  }
  return Event(!job.failure.ok()               ? "response.failed"
               : job.finish_reason == "length" ? "response.incomplete"
                                               : "response.completed",
               "\"response\":" + ResponseEnvelope(request_, &job));
}
} // namespace mellum16::server
