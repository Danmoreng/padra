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
HttpServer::HttpServer(Coordinator &coordinator, const text::Tokenizer &tokenizer, int port,
                       int capacity)
    : coordinator_(coordinator), tokenizer_(tokenizer), port_(port), capacity_(capacity) {
  http_.new_task_queue = [] { return new httplib::ThreadPool(16, 16, 32); };
  if (coordinator_.RuntimeMetrics().capacity != capacity_)
    throw std::invalid_argument("HTTP/runtime context capacity mismatch");
  http_.set_payload_max_length(RequestBodyLimit(capacity_));
  http_.set_read_timeout(5);
  http_.set_write_timeout(2);
  http_.set_keep_alive_timeout(2);
  http_.set_keep_alive_max_count(16);
  http_.set_tcp_nodelay(true);
#ifndef _WIN32
  http_.set_socket_options([](socket_t socket) {
    int bytes = 16384;
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
  http_.Get("/healthz", [](const httplib::Request &, httplib::Response &response) {
    response.set_content("{\"status\":\"alive\"}", "application/json");
  });
  http_.Get("/readyz", [this](const httplib::Request &, httplib::Response &response) {
    auto state = coordinator_.Metrics();
    response.status = state.ready ? 200 : 503;
    response.set_content(std::string("{\"ready\":") + (state.ready ? "true" : "false") +
                             ",\"quality_status\":\"unqualified\"}",
                         "application/json");
  });
  http_.Get("/v1/models", [this](const httplib::Request &, httplib::Response &response) {
    const auto r = coordinator_.RuntimeMetrics();
    std::ostringstream out;
    out << "{\"object\":\"list\",\"data\":[{\"id\":" << json::Quote(kModelId)
        << ",\"object\":\"model\",\"created\":0,\"owned_by\":\"mellum16\",\"context_capacity\":"
        << capacity_
        << ",\"quality_status\":\"unqualified\",\"sampling_profiles\":[\"greedy\",\"gpu-graph-"
           "parameters-penalties-splitmix64-v2\"],\"kv_format\":"
        << json::Quote(KVCacheLayout::Make(r.kv_format, 1).name()) << ",\"execution_profile\":"
        << json::Quote(ServerExecutionProfile(capacity_, true, r.kv_format, r.prefill_chunk,
                                              r.device_budget_gib, r.checkpoint_enabled))
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
        << ",\"build_version\":" << json::Quote(kPadraBuildVersion)
        << ",\"build_source_sha256\":" << json::Quote(kPadraBuildSourceSha256)
        << ",\"build_kernel_profile\":" << json::Quote(kPadraBuildKernelProfile)
        << ",\"kernel_profile\":" << json::Quote(r.kernel_profile)
        << ",\"math_profile\":" << json::Quote(r.math_profile)
        << ",\"test_backend\":" << (r.model_loads == 0 ? "true" : "false") << "}]}";
    response.set_content(out.str(), "application/json");
  });
  http_.Get("/metrics", [this](const httplib::Request &, httplib::Response &response) {
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
int HttpServer::Bind() {
  if (bound_)
    return port_;
  if (port_ == 0) {
    port_ = http_.bind_to_any_port("127.0.0.1");
    bound_ = port_ > 0;
  } else
    bound_ = http_.bind_to_port("127.0.0.1", port_);
  return bound_ ? port_ : -1;
}
bool HttpServer::Listen() {
  return Bind() > 0 && http_.listen_after_bind();
}
void HttpServer::Stop() {
  std::lock_guard lock(stop_mutex_);
  if (stopped_)
    return;
  coordinator_.BeginDrain(true);
  if (!coordinator_.WaitForDrain())
    coordinator_.BeginDrain(false);
  http_.stop();
  stopped_ = true;
}
} // namespace mellum16::server
