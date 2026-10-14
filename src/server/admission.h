#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
namespace mellum16::server {
enum class ApiRoute { Internal, Chat, Responses, Count };
enum class AdmissionCause {
  None,
  NotReady,
  Draining,
  PreparationLimit,
  QueueFull,
  QueueBytes,
  QueueTokens,
  SessionMissing,
  SessionBusy,
  SessionBudget,
  ResponseStoreFull,
  RootCapacity,
  InvalidRequest,
  Internal,
  Count
};
inline constexpr std::array<std::string_view, static_cast<std::size_t>(ApiRoute::Count)>
    kApiRouteNames{"internal", "chat", "responses"};
inline constexpr std::array<std::string_view, static_cast<std::size_t>(AdmissionCause::Count)>
    kAdmissionCauseNames{"none",
                         "not_ready",
                         "draining",
                         "preparation_limit",
                         "queue_full",
                         "queue_bytes",
                         "queue_tokens",
                         "session_missing",
                         "session_busy",
                         "session_budget",
                         "response_store_full",
                         "root_capacity",
                         "invalid_request",
                         "internal"};
struct AdmissionDecision {
  AdmissionCause cause = AdmissionCause::None;
  bool allowed() const {
    return cause == AdmissionCause::None;
  }
};
using RejectionCounters =
    std::array<std::array<std::uint64_t, kAdmissionCauseNames.size()>, kApiRouteNames.size()>;
struct SubmissionContext {
  bool track_transport = false;
  ApiRoute route = ApiRoute::Internal;
  AdmissionDecision *decision = nullptr;
};
} // namespace mellum16::server
