#include "server/http_server.h"
#include "server/http_request.h"
#include "server/job_error.h"
#include "util/json.h"
namespace mellum16::server {
void HttpServer::StoredResponses(const httplib::Request &request, httplib::Response &response,
                                 bool remove) {
  auto owner = http_detail::SessionOwner(request);
  if (!owner.ok()) {
    Error(response, 400, owner.status().message());
    return;
  }
  const auto id = request.matches[1].str();
  response.set_header("Cache-Control", "no-store");
  if (remove) {
    auto deleted = responses_->Delete(id, owner.value());
    if (!deleted.ok()) {
      Error(response, 404, deleted.message(), "response_not_found");
      return;
    }
    response.set_content("{\"id\":" + json::Quote(id) +
                             ",\"object\":\"response.deleted\",\"deleted\":true}",
                         "application/json");
    return;
  }
  auto record = responses_->Get(id, owner.value());
  if (!record.ok()) {
    Error(response, 404, record.status().message(), "response_not_found");
    return;
  }
  response.set_content(record.value()->envelope, "application/json");
}
void HttpServer::Session(const httplib::Request &request, httplib::Response &response,
                         std::string_view operation) {
  auto owner = http_detail::SessionOwner(request);
  if (!owner.ok()) {
    Error(response, 400, owner.status().message());
    return;
  }
  auto &store = coordinator_.Sessions();
  const auto id = operation == "create" ? std::string() : request.matches[1].str();
  if (operation == "delete") {
    auto status = coordinator_.DiscardSession(id, owner.value());
    if (!status.ok()) {
      Error(response, 404, status.message(), "session_not_found");
      return;
    }
    response.set_content("{\"id\":" + json::Quote(id) +
                             ",\"object\":\"mellum.session.deleted\",\"deleted\":true}",
                         "application/json");
    return;
  }
  Result<SessionSnapshot> snapshot =
      Status{StatusCode::kInternal, "session operation not completed"};
  if (operation == "create") {
    auto body = json::Parse(request.body);
    if (!body.ok() || !body.value().is_object()) {
      Error(response, 400, "session creation requires JSON object");
      return;
    }
    auto *model = body.value().find("model");
    if (!model || !model->is_string() || model->as_string() != kModelId ||
        body.value().as_object().size() != 1) {
      Error(response, 400, "session creation supports exactly the selected model field");
      return;
    }
    snapshot = store.Create(owner.value());
  } else
    snapshot = store.Get(id, owner.value());
  if (!snapshot.ok()) {
    Error(response, snapshot.status().code() == StatusCode::kNotFound ? 404 : 429,
          snapshot.status().message(), "session_unavailable");
    return;
  }
  const auto &s = snapshot.value();
  response.set_header("Cache-Control", "no-store");
  response.set_content(
      "{\"id\":" + json::Quote(s.id) + ",\"object\":\"mellum.session\",\"model\":" +
          json::Quote(kModelId) + ",\"ttl_seconds\":" + std::to_string(store.ttl_seconds()) +
          ",\"active\":" + (s.active ? "true" : "false") +
          ",\"stored_history_bytes\":" + std::to_string(s.bytes) + ",\"kv_cache_available\":" +
          (coordinator_.SessionCacheAvailable(s.id, owner.value()) ? "true" : "false") +
          ",\"messages\":" + s.history + "}",
      "application/json");
}
} // namespace mellum16::server
