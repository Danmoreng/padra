#include "server/http_listener.h"
namespace mellum16::server {
namespace {
class ConnectionPool final : public httplib::TaskQueue {
public:
  ConnectionPool(unsigned threads, unsigned queued, std::shared_ptr<ConnectionCounters> counters)
      : counters_(std::move(counters)), pool_(threads, threads, queued) {}
  bool enqueue(std::function<void()> task) override {
    ++counters_->queued;
    bool accepted;
    try {
      accepted = pool_.enqueue([counters = counters_, task = std::move(task)] {
        --counters->queued;
        ++counters->active;
        struct Release {
          std::shared_ptr<ConnectionCounters> counters;
          ~Release() {
            --counters->active;
          }
        } release{counters};
        task();
      });
    } catch (...) {
      --counters_->queued;
      throw;
    }
    if (!accepted) {
      --counters_->queued;
      ++counters_->rejected;
    }
    return accepted;
  }
  void shutdown() override {
    pool_.shutdown();
  }

private:
  std::shared_ptr<ConnectionCounters> counters_;
  httplib::ThreadPool pool_;
};
} // namespace
httplib::TaskQueue *MakeConnectionPool(unsigned threads, unsigned queued,
                                       std::shared_ptr<ConnectionCounters> counters) {
  return new ConnectionPool(threads, queued, std::move(counters));
}
void ConfigureListenSocket(socket_t socket) {
  // Rebind after shutdown while keeping each live port exclusive to one server.
#ifdef _WIN32
  (void)httplib::set_socket_opt(socket, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, 1);
#else
  (void)httplib::set_socket_opt(socket, SOL_SOCKET, SO_REUSEADDR, 1);
#ifdef SO_REUSEPORT
  (void)httplib::set_socket_opt(socket, SOL_SOCKET, SO_REUSEPORT, 0);
#endif
#endif
}
HttpListener::HttpListener() {
  set_socket_options(ConfigureListenSocket);
}
HttpListener::~HttpListener() {
  Stop();
  (void)Wait();
}
void HttpListener::CloseBoundSocket() {
  const auto socket = svr_sock_.exchange(INVALID_SOCKET);
  if (socket != INVALID_SOCKET)
    (void)httplib::detail::close_socket(socket);
}
int HttpListener::Bind(int port) {
  std::lock_guard lock(mutex_);
  if (stopped_)
    return -1;
  if (port_ > 0)
    return port_;
  if (!port)
    port_ = bind_to_any_port("127.0.0.1");
  else if (bind_to_port("127.0.0.1", port))
    port_ = port;
  return port_;
}
bool HttpListener::Start() {
  std::unique_lock lock(mutex_);
  if (stopped_)
    return true; // An intentional stop before Listen is successful shutdown.
  if (started_ || port_ <= 0)
    return false;
  started_ = true;
  try {
    thread_ = std::thread([this] {
      bool success;
      try {
        success = listen_after_bind();
      } catch (...) {
        // The library running guard has unwound; this owner closes its socket.
        std::lock_guard lock(mutex_);
        CloseBoundSocket();
        success_ = false;
        finished_ = true;
        changed_.notify_all();
        return;
      }
      std::lock_guard lock(mutex_);
      // The library closes the listening socket on an accept failure.
      if (!success)
        svr_sock_.store(INVALID_SOCKET);
      success_ = success;
      finished_ = true;
      changed_.notify_all();
    });
  } catch (...) {
    started_ = false;
    throw;
  }
  while (!is_running() && !finished_)
    changed_.wait_for(lock, std::chrono::milliseconds(1));
  return !finished_ || success_;
}
bool HttpListener::Wait() {
  if (thread_.joinable())
    thread_.join();
  std::lock_guard lock(mutex_);
  return success_;
}
void HttpListener::Stop() {
  std::unique_lock lock(mutex_);
  if (stopped_)
    return;
  stopped_ = true;
  // stop() ignores a socket until listen_internal has published running=true.
  // Wait for that handoff, or close a socket that will never be listened on.
  while (started_ && !is_running() && !finished_)
    changed_.wait_for(lock, std::chrono::milliseconds(1));
  if (!started_)
    CloseBoundSocket();
  else if (is_running() && svr_sock_.load() != INVALID_SOCKET)
    httplib::Server::stop();
}
} // namespace mellum16::server
