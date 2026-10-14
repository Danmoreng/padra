#include "server/server_config.h"
#include "runtime/context_capacity.h"
#include "server/chat_request.h"
#include "util/json.h"
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
namespace mellum16::server {
namespace {
Status Invalid(std::string message) {
  return {StatusCode::kInvalidArgument, std::move(message)};
}
bool Duration(std::chrono::milliseconds d, bool zero = false) {
  return d.count() >= (zero ? 0 : 1) && d.count() <= 600000;
}
template <class T> void Read(const json::Value &object, std::string_view key, T &target) {
  const auto *v = object.find(key);
  if (!v)
    return;
  if (!v->is_integer() || v->as_integer() < 0)
    throw std::invalid_argument(std::string(key) + " must be a nonnegative integer");
  const auto value = v->as_integer();
  if constexpr (requires { target.count(); }) {
    target = T(value);
  } else {
    if (static_cast<std::uint64_t>(value) >
        static_cast<std::uint64_t>(std::numeric_limits<T>::max()))
      throw std::invalid_argument(std::string(key) + " exceeds integer range");
    target = static_cast<T>(value);
  }
}
void Known(const json::Value &object, std::initializer_list<std::string_view> keys) {
  if (!object.is_object())
    throw std::invalid_argument("configuration section must be an object");
  for (const auto &[key, value] : object.as_object()) {
    (void)value;
    if (std::find(keys.begin(), keys.end(), key) == keys.end())
      throw std::invalid_argument("unknown configuration field: " + key);
  }
}
} // namespace
ServerLimits ValidateServerLimits(ServerLimits limits) {
  if (!limits.queued_requests || limits.queued_requests > 1024 || !limits.queued_bytes ||
      limits.queued_bytes > (std::uint64_t{1} << 40) || !limits.queued_tokens ||
      limits.queued_tokens > (std::uint64_t{1} << 31) || !Duration(limits.queue_timeout) ||
      !Duration(limits.generation_timeout) || !Duration(limits.drain_timeout, true))
    throw std::invalid_argument("invalid queue, generation or drain limits");
  return limits;
}
Result<ServerConfig> ServerConfig::Create(ServerOptions options) {
  try {
    options.generation = ValidateServerLimits(options.generation);
    const auto &h = options.http;
    if (options.port < 0 || options.port > 65535 || !AdmitNativeContext(options.capacity).ok())
      return Invalid("invalid listener port or context capacity");
    if (!h.preparation_requests || h.preparation_requests > 64 || !h.threads || h.threads > 64 ||
        !h.queued_connections || h.queued_connections > 1024 || !h.admin_threads ||
        h.admin_threads > 8 || !h.admin_queued_connections || h.admin_queued_connections > 64 ||
        h.admin_port < 0 || h.admin_port > 65535 ||
        (h.admin_port && h.admin_port == options.port) || h.send_buffer_bytes < 1024 ||
        h.send_buffer_bytes > 1024 * 1024 || !h.keep_alive_seconds || h.keep_alive_seconds > 60 ||
        !h.keep_alive_requests || h.keep_alive_requests > 1024 || !Duration(h.read_timeout) ||
        !Duration(h.write_timeout) || !Duration(h.heartbeat) || !Duration(h.poll) ||
        h.poll.count() > 1000 || h.poll > h.heartbeat)
      return Invalid("invalid HTTP, admin or heartbeat limits");
    return ServerConfig(std::move(options));
  } catch (const std::invalid_argument &e) {
    return Invalid(e.what());
  }
}
Result<ServerConfig> ServerConfig::Parse(std::string_view body, ServerOptions defaults) {
  if (body.size() > 65536)
    return Invalid("server configuration exceeds 64 KiB");
  auto parsed = json::Parse(body, {8, 128, 65536});
  if (!parsed.ok())
    return parsed.status();
  try {
    const auto &root = parsed.value();
    Known(root, {"queue", "http"});
    if (const auto *q = root.find("queue")) {
      Known(*q, {"requests", "bytes", "tokens", "timeout_ms", "generation_timeout_ms",
                 "drain_timeout_ms"});
      Read(*q, "requests", defaults.generation.queued_requests);
      Read(*q, "bytes", defaults.generation.queued_bytes);
      Read(*q, "tokens", defaults.generation.queued_tokens);
      Read(*q, "timeout_ms", defaults.generation.queue_timeout);
      Read(*q, "generation_timeout_ms", defaults.generation.generation_timeout);
      Read(*q, "drain_timeout_ms", defaults.generation.drain_timeout);
    }
    if (const auto *h = root.find("http")) {
      Known(*h, {"preparation_requests", "threads", "queued_connections", "admin_threads",
                 "admin_queued_connections", "admin_port", "send_buffer_bytes",
                 "keep_alive_seconds", "keep_alive_requests", "read_timeout_ms", "write_timeout_ms",
                 "heartbeat_ms", "poll_ms"});
      Read(*h, "preparation_requests", defaults.http.preparation_requests);
      Read(*h, "threads", defaults.http.threads);
      Read(*h, "queued_connections", defaults.http.queued_connections);
      Read(*h, "admin_threads", defaults.http.admin_threads);
      Read(*h, "admin_queued_connections", defaults.http.admin_queued_connections);
      Read(*h, "admin_port", defaults.http.admin_port);
      Read(*h, "send_buffer_bytes", defaults.http.send_buffer_bytes);
      Read(*h, "keep_alive_seconds", defaults.http.keep_alive_seconds);
      Read(*h, "keep_alive_requests", defaults.http.keep_alive_requests);
      Read(*h, "read_timeout_ms", defaults.http.read_timeout);
      Read(*h, "write_timeout_ms", defaults.http.write_timeout);
      Read(*h, "heartbeat_ms", defaults.http.heartbeat);
      Read(*h, "poll_ms", defaults.http.poll);
    }
    return Create(std::move(defaults));
  } catch (const std::invalid_argument &e) {
    return Invalid(e.what());
  }
}
std::string ServerConfig::Json(int port, int admin_port) const {
  const auto &q = options_.generation;
  const auto &h = options_.http;
  return "{\"port\":" + std::to_string(port) + ",\"admin_port\":" + std::to_string(admin_port) +
         ",\"context_capacity\":" + std::to_string(options_.capacity) +
         ",\"request_body_bytes\":" + std::to_string(RequestBodyLimit(options_.capacity)) +
         ",\"queue\":{\"requests\":" + std::to_string(q.queued_requests) +
         ",\"bytes\":" + std::to_string(q.queued_bytes) +
         ",\"tokens\":" + std::to_string(q.queued_tokens) +
         ",\"timeout_ms\":" + std::to_string(q.queue_timeout.count()) +
         ",\"generation_timeout_ms\":" + std::to_string(q.generation_timeout.count()) +
         ",\"drain_timeout_ms\":" + std::to_string(q.drain_timeout.count()) +
         "},\"http\":{\"preparation_requests\":" + std::to_string(h.preparation_requests) +
         ",\"threads\":" + std::to_string(h.threads) +
         ",\"queued_connections\":" + std::to_string(h.queued_connections) +
         ",\"admin_threads\":" + std::to_string(h.admin_threads) +
         ",\"admin_queued_connections\":" + std::to_string(h.admin_queued_connections) +
         ",\"send_buffer_bytes\":" + std::to_string(h.send_buffer_bytes) +
         ",\"keep_alive_seconds\":" + std::to_string(h.keep_alive_seconds) +
         ",\"keep_alive_requests\":" + std::to_string(h.keep_alive_requests) +
         ",\"read_timeout_ms\":" + std::to_string(h.read_timeout.count()) +
         ",\"write_timeout_ms\":" + std::to_string(h.write_timeout.count()) +
         ",\"heartbeat_ms\":" + std::to_string(h.heartbeat.count()) +
         ",\"poll_ms\":" + std::to_string(h.poll.count()) + "}}";
}
} // namespace mellum16::server
