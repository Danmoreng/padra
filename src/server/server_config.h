#pragma once
#include "mellum16/status.h"
#include <chrono>
#include <cstddef>
#include <string_view>
namespace mellum16::server {
struct ServerLimits {
  std::size_t queued_requests = 8, queued_bytes = 64 * 1024 * 1024, queued_tokens = 65536;
  std::chrono::milliseconds queue_timeout{5000}, generation_timeout{30000}, drain_timeout{5000};
  bool operator==(const ServerLimits &) const = default;
};
ServerLimits ValidateServerLimits(ServerLimits);
struct HttpLimits {
  unsigned preparation_requests = 4, threads = 16, queued_connections = 32;
  unsigned admin_threads = 2, admin_queued_connections = 8;
  int admin_port = 0, send_buffer_bytes = 16384;
  unsigned keep_alive_seconds = 2, keep_alive_requests = 16;
  std::chrono::milliseconds read_timeout{5000}, write_timeout{2000}, heartbeat{15000}, poll{20};
};
struct ServerOptions {
  ServerLimits generation;
  HttpLimits http;
  int port = 0, capacity = 8192;
};
// Mutable inputs are accepted only by the factories. Running servers retain a
// const validated value; callers get no mutable access to the resolved options.
class ServerConfig {
public:
  static Result<ServerConfig> Create(ServerOptions = {});
  static Result<ServerConfig> Parse(std::string_view json, ServerOptions defaults = {});
  const ServerLimits &generation() const {
    return options_.generation;
  }
  const HttpLimits &http() const {
    return options_.http;
  }
  int port() const {
    return options_.port;
  }
  int capacity() const {
    return options_.capacity;
  }
  std::string Json(int bound_port, int bound_admin_port) const;

private:
  explicit ServerConfig(ServerOptions options) : options_(std::move(options)) {}
  ServerOptions options_;
};
} // namespace mellum16::server
