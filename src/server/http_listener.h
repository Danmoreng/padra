#pragma once
#include <httplib.h>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
namespace mellum16::server {
struct ConnectionSnapshot {
  std::uint64_t active, queued, rejected;
};
struct ConnectionCounters {
  std::atomic<std::uint64_t> active{0}, queued{0}, rejected{0};
  ConnectionSnapshot Snapshot() const {
    return {active.load(), queued.load(), rejected.load()};
  }
};
void ConfigureListenSocket(socket_t);
httplib::TaskQueue *MakeConnectionPool(unsigned threads, unsigned queued,
                                       std::shared_ptr<ConnectionCounters>);
// Owns bound-but-unstarted sockets too. Starting/stopping is serialized against
// the library's running flag; Wait has one owner (HttpServer::Listen).
class HttpListener final : public httplib::Server {
public:
  HttpListener();
  ~HttpListener() override;
  int Bind(int port);
  bool Start();
  bool Wait();
  void Stop();

private:
  void CloseBoundSocket();
  std::mutex mutex_;
  std::condition_variable changed_;
  std::thread thread_;
  int port_ = -1;
  bool started_ = false, stopped_ = false, finished_ = false, success_ = true;
};
} // namespace mellum16::server
