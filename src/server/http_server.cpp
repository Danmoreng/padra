#include "server/http_server.h"
#include "runtime/build_identity.h"
#include "server/http_policy.h"
#include "server/secure_id.h"
#include "util/json.h"
#include <chrono>
#include <sstream>
#ifndef _WIN32
#include <sys/socket.h>
#endif
namespace mellum16::server {
namespace {
ServerConfig MakeHttpConfig(const Coordinator &coordinator, int port, int capacity,
                            HttpLimits http) {
  auto config = ServerConfig::Create({coordinator.Limits(), http, port, capacity});
  if (!config.ok())
    throw std::invalid_argument(config.status().message());
  return std::move(config).value();
}
} // namespace
HttpServer::HttpServer(Coordinator &coordinator, const text::Tokenizer &tokenizer, int port,
                       int capacity, HttpLimits limits)
    : HttpServer(coordinator, tokenizer, MakeHttpConfig(coordinator, port, capacity, limits)) {}
HttpServer::HttpServer(Coordinator &coordinator, const text::Tokenizer &tokenizer,
                       ServerConfig config)
    : config_(std::move(config)), coordinator_(coordinator), tokenizer_(tokenizer),
      port_(config_.port()), capacity_(config_.capacity()),
      model_id_(ModelIdForRate(coordinator_.RuntimeMetrics().expert_rate)) {
  if (!(config_.generation() == coordinator_.Limits()))
    throw std::invalid_argument("HTTP/coordinator configuration mismatch");
  http_.new_task_queue = [this] {
    return MakeConnectionPool(config_.http().threads, config_.http().queued_connections,
                              connections_);
  };
  admin_.new_task_queue = [this] {
    return MakeConnectionPool(config_.http().admin_threads, config_.http().admin_queued_connections,
                              admin_connections_);
  };
  if (coordinator_.RuntimeMetrics().capacity != capacity_)
    throw std::invalid_argument("HTTP/runtime context capacity mismatch");
  http_.set_payload_max_length(RequestBodyLimit(capacity_));
  http_.set_read_timeout(config_.http().read_timeout);
  http_.set_write_timeout(config_.http().write_timeout);
  http_.set_keep_alive_timeout(config_.http().keep_alive_seconds);
  http_.set_keep_alive_max_count(config_.http().keep_alive_requests);
  http_.set_tcp_nodelay(true);
#ifndef _WIN32
  http_.set_socket_options([this](socket_t socket) {
    ConfigureListenSocket(socket);
    int bytes = config_.http().send_buffer_bytes;
    (void)setsockopt(socket, SOL_SOCKET, SO_SNDBUF, &bytes, sizeof(bytes));
  });
#endif
  http_.set_pre_routing_handler(
      [this](const httplib::Request &request, httplib::Response &response) {
        if (request.method == "POST" &&
            (request.path == "/v1/chat/completions" || request.path == "/v1/responses")) {
          auto id = MakeSecureId("req_");
          if (!id.ok()) {
            RejectAdmission(response, AdmissionCause::Internal,
                            request.path == "/v1/responses" ? ApiRoute::Responses : ApiRoute::Chat);
            return httplib::Server::HandlerResponse::Handled;
          }
          response.set_header("X-Request-Id", id.value());
        }
        auto error = ValidateLocalHttpRequest(request, port_);
        if (!error.empty()) {
          Error(response, 403, error, "http_policy");
          return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
      });
  http_.set_error_handler([this](const httplib::Request &, httplib::Response &response) {
    if (response.get_header_value("Content-Type").find("application/json") != 0)
      Error(response, response.status, "endpoint/request unsupported",
            response.status == 404 ? "not_found" : "http_error");
  });
  http_.set_exception_handler(
      [this](const httplib::Request &, httplib::Response &response, std::exception_ptr error) {
        try {
          if (error)
            std::rethrow_exception(error);
        } catch (const std::exception &e) {
          Error(response, 500, e.what(), "server_error");
        }
      });
  const httplib::Server::Handler health = [](const httplib::Request &,
                                             httplib::Response &response) {
    response.set_content("{\"status\":\"alive\"}", "application/json");
  };
  http_.Get("/healthz", health);
  admin_.Get("/healthz", health);
  const httplib::Server::Handler ready = [this](const httplib::Request &,
                                                httplib::Response &response) {
    auto state = coordinator_.Metrics();
    response.status = state.ready ? 200 : 503;
    response.set_content(std::string("{\"ready\":") + (state.ready ? "true" : "false") +
                             ",\"quality_status\":\"unqualified\"}",
                         "application/json");
  };
  http_.Get("/readyz", ready);
  admin_.Get("/readyz", ready);
  http_.Get("/v1/models", [this](const httplib::Request &, httplib::Response &response) {
    const auto r = coordinator_.RuntimeMetrics();
    std::ostringstream out;
    out << "{\"object\":\"list\",\"data\":[{\"id\":" << json::Quote(model_id_)
        << ",\"object\":\"model\",\"created\":0,\"owned_by\":\"mellum16\",\"context_capacity\":"
        << capacity_
        << ",\"quality_status\":\"unqualified\",\"sampling_profiles\":[\"greedy\",\"gpu-graph-"
           "parameters-penalties-splitmix64-v2\"],\"kv_format\":"
        << json::Quote(KVCacheLayout::Make(r.kv_format, 1).name()) << ",\"execution_profile\":"
        << json::Quote(ServerExecutionProfile(capacity_, true, r.kv_format, r.prefill_chunk,
                                              r.device_budget_gib, r.checkpoint_enabled,
                                              r.expert_rate))
        << ",\"prefill_chunk\":" << r.prefill_chunk
        << ",\"device_budget_gib\":" << r.device_budget_gib
        << ",\"active_gpu_sessions\":1,\"logical_session_limit\":32,\"session_ttl_seconds\":1800,"
           "\"native_kv_reuse\":"
        << (r.checkpoint_enabled && r.model_loads ? "true" : "false")
        << ",\"checkpoint_snapshot_bytes\":" << r.checkpoint_bytes
        << ",\"tokenizer_sha256\":" << json::Quote(tokenizer_.sha256()) << ",\"checkpoint_sha256\":"
        << (r.weights_sha256.empty() ? "null" : json::Quote(r.weights_sha256))
        << ",\"manifest_sha256\":"
        << (r.manifest_sha256.empty() ? "null" : json::Quote(r.manifest_sha256))
        << ",\"server_config\":" << config_.Json(port_, admin_port_)
        << ",\"build_version\":" << json::Quote(kPadraBuildVersion)
        << ",\"build_source_sha256\":" << json::Quote(kPadraBuildSourceSha256)
        << ",\"build_kernel_profile\":" << json::Quote(kPadraBuildKernelProfile)
        << ",\"build_prefill_profile\":" << json::Quote(kPadraBuildPrefillProfile)
        << ",\"kernel_profile\":" << json::Quote(r.kernel_profile)
        << ",\"math_profile\":" << json::Quote(r.math_profile)
        << ",\"test_backend\":" << (r.model_loads == 0 ? "true" : "false") << "}]}";
    response.set_content(out.str(), "application/json");
  });
  const httplib::Server::Handler metrics = [this](const httplib::Request &,
                                                  httplib::Response &response) {
    auto m = coordinator_.Metrics();
    auto r = coordinator_.RuntimeMetrics();
    std::ostringstream out;
    out << "mellum_requests_submitted_total " << m.submitted << "\nmellum_requests_completed_total "
        << m.completed << "\nmellum_requests_failed_total " << m.failed
        << "\nmellum_requests_cancelled_total " << m.cancelled << "\nmellum_queue_rejected_total "
        << m.rejected << "\nmellum_disconnects_total " << m.disconnects
        << "\nmellum_backpressure_total " << m.backpressure << "\nmellum_queue_requests "
        << m.queued << "\nmellum_queue_bytes " << m.queued_bytes << "\nmellum_queue_tokens "
        << m.queued_tokens << "\nmellum_active_requests " << m.active << "\nmellum_ready "
        << m.ready << "\nmellum_model_loads_total " << r.model_loads
        << "\nmellum_weight_uploads_total " << r.weight_uploads << "\nmellum_graph_captures_total "
        << r.graph_captures << "\nmellum_resident_device_bytes " << r.resident_bytes
        << "\nmellum_required_device_bytes " << r.required_bytes
        << "\nmellum_post_graph_available_device_bytes " << r.post_graph_available_bytes
        << "\nmellum_admission_available_device_bytes " << r.admission_available_bytes
        << "\nmellum_device_budget_gib " << r.device_budget_gib << "\nmellum_prefill_chunk "
        << r.prefill_chunk << "\nmellum_tokenization_seconds_total " << m.tokenize_seconds
        << "\nmellum_queue_wait_seconds_total " << m.queue_seconds
        << "\nmellum_prefill_seconds_total " << m.prefill_seconds
        << "\nmellum_decode_seconds_total " << m.decode_seconds
        << "\nmellum_first_token_seconds_total " << m.first_token_seconds
        << "\nmellum_first_visible_seconds_total " << m.visible_seconds << '\n';
    out << "mellum_output_tokens_total " << m.output_tokens << "\nmellum_prefill_chunks_total "
        << r.prefill_chunks << "\nmellum_prefill_eager_chunks_total " << r.eager_tail_chunks
        << '\n';
    m.queue_latency.Write(out, "mellum_queue_wait_seconds");
    m.tokenization_latency.Write(out, "mellum_tokenization_seconds");
    m.prefill_latency.Write(out, "mellum_prefill_seconds");
    m.decode_latency.Write(out, "mellum_decode_seconds");
    m.first_token_latency.Write(out, "mellum_first_token_seconds");
    m.output_available_latency.Write(out, "mellum_output_available_seconds");
    m.lifetime_latency.Write(out, "mellum_job_lifetime_seconds");
    out << "mellum_active_transports " << m.active_transports << "\nmellum_responses_written_total "
        << m.responses_written << "\nmellum_response_write_failures_total " << m.write_failures
        << "\nmellum_response_body_bytes_total " << m.response_body_bytes << '\n';
    m.first_visible_write_latency.Write(out, "mellum_first_visible_write_seconds");
    m.response_write_latency.Write(out, "mellum_response_write_seconds");
    const auto connections = connections_->Snapshot();
    const auto admin_connections = admin_connections_->Snapshot();
    out << "padra_http_active_connections{listener=\"api\"} " << connections.active
        << "\npadra_http_queued_connections{listener=\"api\"} " << connections.queued
        << "\npadra_http_rejected_connections_total{listener=\"api\"} " << connections.rejected
        << "\npadra_http_active_connections{listener=\"admin\"} " << admin_connections.active
        << "\npadra_http_queued_connections{listener=\"admin\"} " << admin_connections.queued
        << "\npadra_http_rejected_connections_total{listener=\"admin\"} "
        << admin_connections.rejected << '\n';
    const auto [records, roots] = responses_->Evictions();
    out << "mellum_response_evicted_records_total " << records
        << "\nmellum_response_evicted_roots_total " << roots << '\n';
    for (std::size_t route = 0; route < kApiRouteNames.size(); ++route)
      for (std::size_t cause = 1; cause < kAdmissionCauseNames.size(); ++cause)
        if (m.rejections[route][cause])
          out << "mellum_admission_rejected_total{route=\"" << kApiRouteNames[route]
              << "\",cause=\"" << kAdmissionCauseNames[cause] << "\"} "
              << m.rejections[route][cause] << '\n';
    response.set_content(out.str(), "text/plain; version=0.0.4");
  };
  http_.Get("/metrics", metrics);
  admin_.Get("/metrics", metrics);
  admin_.set_read_timeout(config_.http().read_timeout);
  admin_.set_write_timeout(config_.http().write_timeout);
  admin_.set_keep_alive_timeout(config_.http().keep_alive_seconds);
  admin_.set_keep_alive_max_count(config_.http().keep_alive_requests);
  admin_.set_payload_max_length(1);
  admin_.set_pre_routing_handler(
      [this](const httplib::Request &request, httplib::Response &response) {
        auto error = ValidateLocalHttpRequest(request, admin_port_);
        if (!error.empty() || (request.method != "GET" && request.method != "HEAD")) {
          Error(response, 403, error.empty() ? "admin listener accepts read-only requests" : error,
                "http_policy");
          return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
      });
  admin_.Get("/config", [this](const httplib::Request &, httplib::Response &response) {
    response.set_content("{\"server_config\":" + config_.Json(port_, admin_port_) +
                             ",\"build_version\":" + json::Quote(kPadraBuildVersion) +
                             ",\"build_source_sha256\":" + json::Quote(kPadraBuildSourceSha256) +
                             ",\"build_kernel_profile\":" + json::Quote(kPadraBuildKernelProfile) +
                             ",\"kernel_profile\":" +
                             json::Quote(coordinator_.RuntimeMetrics().kernel_profile) + "}",
                         "application/json");
  });
  http_.Post("/v1/chat/completions",
             [this](const httplib::Request &request, httplib::Response &response) {
               Chat(request, response);
             });
  http_.Post("/v1/sessions", [this](const httplib::Request &request, httplib::Response &response) {
    Session(request, response, "create");
  });
  http_.Get(R"(/v1/sessions/(sess-[0-9a-f]{32}))",
            [this](const httplib::Request &request, httplib::Response &response) {
              Session(request, response, "get");
            });
  http_.Delete(R"(/v1/sessions/(sess-[0-9a-f]{32}))",
               [this](const httplib::Request &request, httplib::Response &response) {
                 Session(request, response, "delete");
               });
  http_.Post("/v1/responses", [this](const httplib::Request &request, httplib::Response &response) {
    Responses(request, response);
  });
  http_.Get(R"(/v1/responses/(resp_[0-9a-f]{32}))",
            [this](const httplib::Request &request, httplib::Response &response) {
              StoredResponses(request, response, false);
            });
  http_.Delete(R"(/v1/responses/(resp_[0-9a-f]{32}))",
               [this](const httplib::Request &request, httplib::Response &response) {
                 StoredResponses(request, response, true);
               });
}
HttpServer::~HttpServer() {
  Stop();
  (void)http_.Wait();
  (void)admin_.Wait();
}
int HttpServer::Bind() {
  std::lock_guard lock(stop_mutex_);
  if (stopped_)
    return -1;
  if (bound_)
    return port_;
  port_ = http_.Bind(port_);
  if (port_ <= 0)
    return -1;
  admin_port_ = admin_.Bind(config_.http().admin_port);
  if (admin_port_ <= 0) {
    http_.Stop();
    return -1;
  }
  bound_ = true;
  return port_;
}
bool HttpServer::Listen() {
  if (Bind() <= 0) {
    std::lock_guard lock(stop_mutex_);
    return stopped_;
  }
  if (!admin_.Start() || !http_.Start()) {
    Stop();
    return false;
  }
  const bool result = http_.Wait();
  Stop();
  (void)admin_.Wait();
  return result;
}
void HttpServer::Stop() {
  std::lock_guard lock(stop_mutex_);
  if (stopped_)
    return;
  coordinator_.BeginDrain(true);
  if (!coordinator_.WaitForDrain())
    coordinator_.BeginDrain(false);
  http_.Stop();
  admin_.Stop();
  stopped_ = true;
}
} // namespace mellum16::server
