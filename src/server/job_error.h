#pragma once
#include "server/coordinator.h"
namespace mellum16::server {
struct JobError {
  int status = 500;
  std::string_view code = "generation_error", type = "server_error";
  bool retry = false;
};
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
