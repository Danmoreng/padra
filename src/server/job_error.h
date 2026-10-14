#pragma once
#include "server/admission.h"
#include "server/generation_result.h"
#include "util/json.h"
namespace mellum16::server {
struct JobError {
  int status = 500;
  std::string_view code = "generation_error", type = "server_error";
  bool retry = false;
};
inline JobError ClassifyAdmissionError(AdmissionCause cause) {
  switch (cause) {
  case AdmissionCause::NotReady:
    return {503, "model_not_ready", "server_error", true};
  case AdmissionCause::Draining:
    return {503, "server_shutdown", "server_error", true};
  case AdmissionCause::PreparationLimit:
    return {503, "preparation_limited", "server_error", true};
  case AdmissionCause::QueueFull:
    return {503, "queue_full", "server_error", true};
  case AdmissionCause::QueueBytes:
    return {503, "queue_bytes_exhausted", "server_error", true};
  case AdmissionCause::QueueTokens:
    return {503, "queue_tokens_exhausted", "server_error", true};
  case AdmissionCause::SessionMissing:
    return {404, "session_not_found", "invalid_request_error", false};
  case AdmissionCause::SessionBusy:
    return {409, "session_busy", "invalid_request_error", true};
  case AdmissionCause::SessionBudget:
    return {429, "session_history_full", "invalid_request_error", true};
  case AdmissionCause::ResponseStoreFull:
    return {429, "response_store_full", "invalid_request_error", true};
  case AdmissionCause::RootCapacity:
    return {429, "session_unavailable", "invalid_request_error", true};
  case AdmissionCause::InvalidRequest:
    return {400, "invalid_request_error", "invalid_request_error", false};
  case AdmissionCause::None:
  case AdmissionCause::Internal:
  case AdmissionCause::Count:
    return {500, "admission_error", "server_error", false};
  }
  return {};
}
inline JobError ClassifyJobError(CancelCause cause) {
  switch (cause) {
  case CancelCause::QueueDeadline:
    return {503, "queue_deadline", "server_error", true};
  case CancelCause::Deadline:
    return {504, "generation_deadline", "server_error", false};
  case CancelCause::Shutdown:
    return {503, "server_shutdown", "server_error", true};
  case CancelCause::Backpressure:
    return {503, "stream_backpressure", "server_error", true};
  case CancelCause::Disconnect:
    return {499, "client_disconnected", "invalid_request_error", false};
  case CancelCause::SessionDiscard:
    return {404, "session_discarded", "invalid_request_error", false};
  case CancelCause::ToolInvalid:
    return {500, "invalid_tool_output", "server_error", false};
  case CancelCause::None:
    return {};
  }
  return {};
}
} // namespace mellum16::server
