#include "server/http_server.h"
#include "server/chat_stream.h"
#include "server/http_request.h"
#include "server/job_error.h"
#include "util/json.h"
#include "runtime/size_checked.h"
#include "server/secure_id.h"
namespace mellum16::server {
void HttpServer::Responses(const httplib::Request &request, httplib::Response &response) {
  const auto received = std::chrono::steady_clock::now();
  auto decision = coordinator_.CheckAdmission();
  if (!decision.allowed()) {
    RejectAdmission(response, decision.cause, ApiRoute::Responses);
    return;
  }
  http_detail::Preparation preparation(preparations_, config_.http().preparation_requests);
  if (!preparation.acquired()) {
    RejectAdmission(response, AdmissionCause::PreparationLimit, ApiRoute::Responses);
    return;
  }

  auto owner = http_detail::SessionOwner(request);
  if (!owner.ok()) {
    Error(response, 400, owner.status().message());
    return;
  }
  std::shared_ptr<const StoredResponse> parent;
  // First validate shape before resolving an owned previous ID. No GPU mutation.
  auto raw = json::Parse(request.body);
  if (!raw.ok() || !raw.value().is_object()) {
    Error(response, 400, "Responses body must be JSON object");
    return;
  }
  if (auto *previous = raw.value().find("previous_response_id"); previous && !previous->is_null()) {
    if (!previous->is_string()) {
      Error(response, 400, "previous_response_id must be string or null");
      return;
    }
    auto stored = responses_->Get(previous->as_string(), owner.value());
    if (!stored.ok()) {
      Error(response, 404, stored.status().message(), "response_not_found");
      return;
    }
    parent = std::move(stored).value();
    if (!parent->continuable) {
      Error(response, 400,
            "response contains an incomplete tool call; retry the original request or use the "
            "last valid previous_response_id with a sufficient output budget",
            "response_not_continuable");
      return;
    }
  }
  auto parsed = ParseResponseRequest(request.body, parent ? parent->history : "[]", model_id_);
  if (!parsed.ok()) {
    Error(response,
          parsed.status().code() == StatusCode::kNotFound            ? 404
          : parsed.status().code() == StatusCode::kResourceExhausted ? 413
                                                                     : 400,
          parsed.status().message());
    return;
  }
  auto spec = std::make_shared<ResponseRequest>(std::move(parsed).value());
  std::shared_ptr<ResponseRoot> root = parent ? parent->root : nullptr;
  if (root && !coordinator_.Sessions().Get(root->session_id, root->owner).ok())
    root.reset();
  auto chat = ParseChatRequest(spec->chat_body, tokenizer_, capacity_, spec->id, model_id_);
  if (!chat.ok()) {
    Error(response, chat.status().code() == StatusCode::kResourceExhausted ? 413 : 400,
          chat.status().message());
    return;
  }
  chat.value().session_owner = owner.value();
  chat.value().submitted = received;
  spec->outputs = chat.value().outputs;
  spec->temperature = chat.value().sampling.temperature;
  spec->top_p = chat.value().sampling.top_p;
  std::shared_ptr<ResponseReservation> reservation;
  {
    // Records, bytes and any new physical root are reserved atomically by the
    // two stores. Manual session operations participate through the session lock.
    if (spec->store) {
      const auto budget = admission::Sum(
          {admission::Bytes(admission::Sum({spec->chat_body.size(), 128}), 2),
           admission::Bytes(
               chat.value().outputs,
               admission::Bytes(admission::Sum({tokenizer_.MaxRawTokenBytes(), 3}), 18)),
           131072});
      auto reserved = responses_->Reserve(
          spec->id, owner.value(), budget, coordinator_.Sessions().shared_from_this(), !root,
          [coordinator = &coordinator_](const std::string &id, const std::string &who) {
            (void)coordinator->DiscardSession(id, who);
          },
          &decision);
      if (!reserved.ok()) {
        RejectAdmission(response, decision.cause, ApiRoute::Responses, true,
                        reserved.status().message());
        return;
      }
      reservation = std::move(reserved).value();
      if (!root)
        root = reservation->root;
    }
  }
  if (root) {
    auto attached = AttachSession(chat.value(), tokenizer_, root->session_id);
    if (!attached.ok()) {
      Error(response, 400, attached.message());
      return;
    }
    spec->chat_body = chat.value().body;
  }
  chat.value().adapter_bytes = admission::Sum(
      {sizeof(ResponseRequest), spec->chat_body.size(), spec->history.size(), spec->tools.size(),
       spec->metadata.size(), spec->instructions.size(), spec->reasoning.size(),
       spec->prompt_cache_key.size(), spec->prompt_cache_retention.size(), 2048});
  auto admitted =
      coordinator_.Submit(std::move(chat).value(), {true, ApiRoute::Responses, &decision});
  if (!admitted.ok()) {
    RejectAdmission(response, decision.cause, ApiRoute::Responses, false,
                    admitted.status().message());
    return;
  }
  preparation.Release();
  auto job = std::move(admitted).value();
  TrackTransport(response, job);
  response.set_header("Cache-Control", "no-store");
  ResponseRetention retain(responses_, reservation, spec, root);
  if (!spec->stream) {
    if (!AwaitJob(request, response, job))
      return;
    auto stored = retain.Commit(*job);
    if (!stored.ok()) {
      JobErrorResponse(response, job, 500, stored.message(), "response_store_error");
      return;
    }
    response.set_header("X-Mellum-Cached-Tokens", std::to_string(job->cached_tokens));
    response.set_header("X-Mellum-Prefill-Ms", std::to_string(job->prefill_seconds * 1000));
    response.set_header("X-Mellum-Decode-Ms", std::to_string(job->decode_seconds * 1000));
    SendJobJson(response, job, ResponseEnvelope(*spec, job.get()),
                !job->content.empty() || !job->tool_calls.empty());
    return;
  }
  SendResponseStream(request, response, job, spec, std::move(retain));
}
void HttpServer::Chat(const httplib::Request &request, httplib::Response &response) {
  const auto received = std::chrono::steady_clock::now();
  auto decision = coordinator_.CheckAdmission();
  if (!decision.allowed()) {
    RejectAdmission(response, decision.cause, ApiRoute::Chat);
    return;
  }
  http_detail::Preparation preparation(preparations_, config_.http().preparation_requests);
  if (!preparation.acquired()) {
    RejectAdmission(response, AdmissionCause::PreparationLimit, ApiRoute::Chat);
    return;
  }

  auto id = MakeSecureId("chatcmpl-");
  if (!id.ok()) {
    Error(response, 500, id.status().message(), "id_failure");
    return;
  }
  auto parsed =
      ParseChatRequest(request.body, tokenizer_, capacity_, std::move(id).value(), model_id_);
  if (!parsed.ok()) {
    Error(response,
          parsed.status().code() == StatusCode::kNotFound            ? 404
          : parsed.status().code() == StatusCode::kResourceExhausted ? 413
                                                                     : 400,
          parsed.status().message());
    return;
  }
  if (!parsed.value().session_id.empty()) {
    auto owner = http_detail::SessionOwner(request);
    if (!owner.ok()) {
      Error(response, 400, owner.status().message());
      return;
    }
    parsed.value().session_owner = std::move(owner).value();
  }
  parsed.value().submitted = received;
  const bool stream = parsed.value().stream;
  auto admitted = coordinator_.Submit(std::move(parsed).value(), {true, ApiRoute::Chat, &decision});
  if (!admitted.ok()) {
    RejectAdmission(response, decision.cause, ApiRoute::Chat, false, admitted.status().message());
    return;
  }
  preparation.Release();
  auto job = std::move(admitted).value();
  TrackTransport(response, job);
  response.set_header("Cache-Control", "no-cache");
  if (!stream) {
    if (!AwaitJob(request, response, job))
      return;
    response.set_header("X-Mellum-Queue-Ms", std::to_string(job->queue_seconds * 1000));
    response.set_header("X-Mellum-Tokenization-Ms",
                        std::to_string(job->request.tokenize_seconds * 1000));
    response.set_header("X-Mellum-Prefill-Ms", std::to_string(job->prefill_seconds * 1000));
    response.set_header("X-Mellum-Decode-Ms", std::to_string(job->decode_seconds * 1000));
    response.set_header("X-Mellum-First-Token-Ms", std::to_string(job->first_token_seconds * 1000));
    response.set_header("X-Mellum-First-Visible-Ms", std::to_string(job->visible_seconds * 1000));
    response.set_header("X-Mellum-Cached-Tokens", std::to_string(job->cached_tokens));
    SendJobJson(response, job, CompletionJson(*job),
                !job->content.empty() || !job->tool_calls.empty());
    return;
  }
  SendChatStream(request, response, job);
}
} // namespace mellum16::server
