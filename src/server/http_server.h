#pragma once
#include "server/coordinator.h"
#include "server/response_store.h"
#include <httplib.h>
namespace mellum16::server {
class HttpServer {
public:
  HttpServer(Coordinator &, const text::Tokenizer &, int port, int capacity = 8192);
  int Bind();
  bool Listen();
  void Stop();

private:
  void Chat(const httplib::Request &, httplib::Response &);
  void SendJobJson(httplib::Response &, const std::shared_ptr<Job> &, std::string, bool visible);
  void JobErrorResponse(httplib::Response &, const std::shared_ptr<Job> &, int, std::string_view,
                        std::string_view);
  void Session(const httplib::Request &, httplib::Response &, std::string_view operation);
  void Responses(const httplib::Request &, httplib::Response &);
  void StoredResponses(const httplib::Request &, httplib::Response &, bool remove);
  void Error(httplib::Response &, int, std::string_view,
             std::string_view = "invalid_request_error");
  std::atomic<unsigned> preparations_{0};
  // Serialize shutdown through the library's socket close, including repeat calls.
  std::mutex stop_mutex_;
  bool stopped_ = false;
  Coordinator &coordinator_;
  const text::Tokenizer &tokenizer_;
  int port_, capacity_;
  bool bound_ = false;
  httplib::Server http_;
  std::shared_ptr<ResponseStore> responses_ = std::make_shared<ResponseStore>();
};
} // namespace mellum16::server
