// Isolate Responses translation with an already parsed HTTP tree. The default
// mode emulates the old boundary sequence (parse, then parse again in the
// public adapter); pass "reuse" for the validated-tree path. This benchmark
// excludes native rendering, tokenization, retained-session attachment and GPU.
// Example: build padra_test_server in a Release host-only build, then link this
// file with libpadra_test_server.a, libpadra_http.a and pthread.
#include "server/responses.h"
#include "util/json.h"
#include <chrono>
#include <iostream>
using namespace mellum16;
using namespace mellum16::server;
int main(int argc, char **argv) {
  bool reuse = argc > 1 && std::string_view(argv[1]) == "reuse";
  std::string body = "{\"model\":\"mellum21-exl3-k4-research\",\"input\":[";
  for (int i = 0; i < 200; ++i) {
    if (i)
      body += ',';
    body += "{\"role\":\"user\",\"content\":\"" + std::string(1024, 'x') + "\"}";
  }
  body += "]}";
  constexpr int iterations = 300;
  std::size_t bytes = 0;
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    auto tree = json::Parse(body);
    if (!tree.ok())
      return 1;
    auto result = reuse ? ParseResponseRequestValidated(body, tree.value(), "[]", kModelId)
                        : ParseResponseRequest(body);
    if (!result.ok()) {
      std::cerr << result.status().message() << '\n';
      return 2;
    }
    bytes += result.value().chat_body.size();
  }
  std::cout << "body_bytes=" << body.size() << " iterations=" << iterations << " mean_ms="
            << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                       .count() /
                   iterations
            << " output_bytes=" << bytes << '\n';
}
