#include "server/http_server.h"
#include "server/http_request.h"
#include "server/job_error.h"
#include "util/json.h"
#include "server/chat_stream.h"
#include "server/response_stream.h"
#include "server/stream_heartbeat.h"
#include <iostream>
#include <syncstream>
namespace mellum16::server {
namespace {
struct TransportLifetime {
  explicit TransportLifetime(std::shared_ptr<Job> input) : job(std::move(input)) {
    job->StartTransport();
  }
  ~TransportLifetime() {
    if (!job->transport_finished.load()) {
      job->FinishTransport(false);
      job->Cancel(CancelCause::Disconnect);
    }
  }
  std::shared_ptr<Job> job;
};
struct ConsumedEvent {
  bool success, written;
};
template <class Closed, class Consume, class Complete>
bool PumpStream(const std::shared_ptr<Job> &job, Closed closed, httplib::DataSink &sink,
                Consume consume, Complete complete, const HttpLimits &timing) {
  GenerationEvent frame;
  StreamHeartbeat heartbeat(timing.heartbeat, std::chrono::steady_clock::now());
  for (;;) {
    if (closed()) {
      job->Cancel(CancelCause::Disconnect);
      return false;
    }
    if (job->Pop(frame, timing.poll)) {
      const auto consumed = consume(frame);
      if (!consumed.success) {
        job->Cancel(CancelCause::Disconnect);
        return false;
      }
      if (consumed.written)
        heartbeat.Wrote(std::chrono::steady_clock::now());
      continue;
    }
    if (heartbeat.Due(std::chrono::steady_clock::now())) {
      if (!sink.write(": heartbeat\n\n", 13)) {
        job->Cancel(CancelCause::Disconnect);
        return false;
      }
      job->RecordWrite(13, false);
      heartbeat.Wrote(std::chrono::steady_clock::now());
    }
    bool drained;
    {
      std::lock_guard lock(job->mutex);
      drained = job->StreamDrained();
    }
    // Result is immutable after publication. Store and socket callbacks run
    // without the job lock, including under slow-reader backpressure.
    if (drained)
      return complete();
  }
}
} // namespace
void HttpServer::TrackTransport(httplib::Response &response, const std::shared_ptr<Job> &job) {
  response.user_data.emplace("padra.transport", std::make_shared<TransportLifetime>(job));
}
void HttpServer::Error(httplib::Response &response, int status, std::string_view message,
                       std::string_view code) {
  response.status = status;
  response.set_content(
      ErrorJson(message, code, status >= 500 ? "server_error" : "invalid_request_error"),
      "application/json");
}
void HttpServer::RejectAdmission(httplib::Response &response, AdmissionCause cause, ApiRoute route,
                                 bool count, std::string_view message) {
  if (count)
    coordinator_.RecordRejection(cause, route);
  const auto error = ClassifyAdmissionError(cause);
  const auto name = kAdmissionCauseNames.at(static_cast<std::size_t>(cause));
  const auto text = message.empty() ? "request admission unavailable: " + std::string(name)
                                    : std::string(message);
  Error(response, error.status, text, error.code);
  if (error.retry)
    response.set_header("Retry-After", "1");
  std::osyncstream(std::clog) << "{\"event\":\"admission_rejected\",\"route\":"
                              << json::Quote(kApiRouteNames.at(static_cast<std::size_t>(route)))
                              << ",\"cause\":" << json::Quote(name) << ",\"request_id\":"
                              << json::Quote(response.get_header_value("X-Request-Id")) << "}\n";
}
void HttpServer::SendJobJson(httplib::Response &response, const std::shared_ptr<Job> &job,
                             std::string body, bool visible) {
  auto content = std::make_shared<std::string>(std::move(body));
  response.set_content_provider(
      content->size(), "application/json",
      [job, content, visible](std::size_t offset, std::size_t length, httplib::DataSink &sink) {
        if (offset > content->size() || length > content->size() - offset)
          return false;
        if (!sink.write(content->data() + offset, length))
          return false;
        job->RecordWrite(length, visible && length > 0);
        return true;
      },
      [job, content](bool success) {
        job->FinishTransport(success);
        if (!success)
          job->Cancel(CancelCause::Disconnect);
      });
}
void HttpServer::JobErrorResponse(httplib::Response &response, const std::shared_ptr<Job> &job,
                                  int status, std::string_view message, std::string_view code) {
  Error(response, status, message, code);
  auto body = std::move(response.body);
  SendJobJson(response, job, std::move(body), false);
}
bool HttpServer::AwaitJob(const httplib::Request &request, httplib::Response &response,
                          const std::shared_ptr<Job> &job) {
  std::unique_lock lock(job->mutex);
  while (!job->complete) {
    job->changed.wait_for(lock, config_.http().poll);
    if (request.is_connection_closed()) {
      job->Cancel(CancelCause::Disconnect);
      break;
    }
  }
  if (!job->complete) {
    JobErrorResponse(response, job, 499, "client disconnected", "client_disconnected");
    return false;
  }
  if (!job->failure.ok()) {
    const auto error = ClassifyJobError(job->failure_cause);
    if (error.retry)
      response.set_header("Retry-After", "1");
    JobErrorResponse(response, job, error.status, job->failure.message(), error.code);
    return false;
  }
  return true;
}
void HttpServer::SendChatStream(const httplib::Request &request, httplib::Response &response,
                                const std::shared_ptr<Job> &job) {
  response.set_header("X-Accel-Buffering", "no");
  response.set_chunked_content_provider(
      "text/event-stream",
      [job, timing = config_.http(),
       closed = request.is_connection_closed](std::size_t, httplib::DataSink &sink) {
        try {
          ChatStream encoder(*job);
          return PumpStream(
              job, closed, sink,
              [&](const GenerationEvent &event) {
                auto wire = encoder.Encode(event);
                if (!sink.write(wire.data(), wire.size()))
                  return ConsumedEvent{false, false};
                job->RecordWrite(wire.size(), event.Visible());
                return ConsumedEvent{true, true};
              },
              [&] {
                sink.done();
                return true;
              },
              timing);
        } catch (...) {
          job->Cancel(CancelCause::Disconnect);
          return false;
        }
      },
      [job](bool success) {
        job->FinishTransport(success);
        if (!success)
          job->Cancel(CancelCause::Disconnect);
      });
}
void HttpServer::SendResponseStream(const httplib::Request &request, httplib::Response &response,
                                    const std::shared_ptr<Job> &job,
                                    std::shared_ptr<const ResponseRequest> spec,
                                    ResponseRetention retain) {
  response.set_header("X-Accel-Buffering", "no");
  response.set_chunked_content_provider(
      "text/event-stream",
      [job, spec, retain = std::move(retain), timing = config_.http(),
       closed = request.is_connection_closed](std::size_t, httplib::DataSink &sink) {
        try {
          ResponseStream events(
              *spec,
              [&sink](std::string_view frame) { return sink.write(frame.data(), frame.size()); },
              [job](std::size_t bytes, bool visible) { job->RecordWrite(bytes, visible); });
          if (!events.Begin()) {
            job->Cancel(CancelCause::Disconnect);
            return false;
          }
          return PumpStream(
              job, closed, sink,
              [&](const GenerationEvent &event) {
                const auto before = events.EmittedEvents();
                const bool ok = events.OnEvent(event);
                return ConsumedEvent{ok, ok && events.EmittedEvents() != before};
              },
              [&] {
                if (job->failure.ok()) {
                  auto stored = retain.Commit(*job);
                  if (!stored.ok()) {
                    if (!events.Fail(stored.message()))
                      return false;
                    sink.done();
                    return true;
                  }
                }
                if (!events.Finish(*job)) {
                  job->Cancel(CancelCause::Disconnect);
                  return false;
                }
                sink.done();
                return true;
              },
              timing);
        } catch (...) {
          job->Cancel(CancelCause::Disconnect);
          return false;
        }
      },
      [job](bool success) {
        job->FinishTransport(success);
        if (!success)
          job->Cancel(CancelCause::Disconnect);
      });
}
} // namespace mellum16::server
