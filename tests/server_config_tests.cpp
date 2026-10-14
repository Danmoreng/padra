#include "server_test_support.h"
int main() {
  try {
    ServerOptions options;
    options.port = 8080;
    auto defaults = ServerConfig::Create(options);
    Require(defaults.ok(), "default configuration invalid");
    options.http.threads = 3;
    Require(defaults.value().http().threads == 16 &&
                defaults.value().generation().queue_timeout == 5s &&
                defaults.value().http().heartbeat == 15s,
            "resolved configuration borrowed mutable inputs");
    auto custom = ServerConfig::Parse(
        R"({"queue":{"requests":2,"tokens":128,"timeout_ms":60,"generation_timeout_ms":80,"drain_timeout_ms":0},"http":{"preparation_requests":1,"threads":2,"queued_connections":4,"admin_port":8081,"heartbeat_ms":25,"poll_ms":1,"read_timeout_ms":100,"write_timeout_ms":200}})",
        options);
    Require(custom.ok() && custom.value().generation().queued_requests == 2 &&
                custom.value().generation().generation_timeout == 80ms &&
                custom.value().http().heartbeat == 25ms &&
                custom.value().http().admin_port == 8081 && custom.value().port() == 8080,
            "configuration overrides/defaults were not resolved");
    const auto encoded = json::Parse(custom.value().Json(8080, 8081));
    Require(encoded.ok() &&
                encoded.value().find("queue")->find("generation_timeout_ms")->as_integer() == 80,
            "resolved configuration JSON differs from live limits");
    for (const auto *input : {"null",
                              "[]",
                              "",
                              "{} trailing",
                              R"({"unknown":1})",
                              R"({"queue":null})",
                              R"({"http":{"threads":0}})",
                              R"({"http":{"threads":65}})",
                              R"({"http":{"threads":2.5}})",
                              R"({"http":{"admin_port":8080}})",
                              R"({"http":{"admin_port":65536}})",
                              R"({"http":{"poll_ms":100,"heartbeat_ms":20}})",
                              R"({"http":{"read_timeout_ms":0}})",
                              R"({"queue":{"requests":0}})",
                              R"({"queue":{"bytes":-1}})",
                              R"({"queue":{"tokens":false}})",
                              R"({"queue":{"timeout_ms":600001}})",
                              R"({"queue":{"timeout_ms":20,"timeout_ms":30}})",
                              R"({"http":{"send_buffer_bytes":1}})",
                              R"({"http":{"keep_alive_seconds":0}})",
                              R"({"request_timeout_ms":100})"})
      Require(!ServerConfig::Parse(input, options).ok(), "invalid/unknown configuration accepted");
    Require(!ServerConfig::Parse(std::string(65537, ' '), options).ok(),
            "unbounded config accepted");
    options.capacity = 8193;
    Require(!ServerConfig::Create(options).ok(), "unsupported context configuration accepted");
    std::cout
        << "PASS immutable resolved configuration, overrides, bounds and unknown-field rejection\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
