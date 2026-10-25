#include "runtime/model_runtime.h"
#include "server/http_server.h"
#include "text/tokenizer.h"
#include "util/json.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <csignal>
#include <fstream>
#include <iostream>
#include <thread>
namespace {
volatile std::sig_atomic_t interrupted = 0;
void Signal(int) {
  interrupted = 1;
}
} // namespace
int main(int argc, char **argv) {
  try {
    int positional_argc = argc;
    std::string config_body;
    mellum16::MtpOptions mtp;
    bool head_format_set = false;
    std::optional<bool> q8_head;
    bool depth_set = false, artifact_set = false, config_set = false, expert_dispatch_set = false,
         draft_experts_set = false;
    while (positional_argc >= 3) {
      const std::string_view flag = argv[positional_argc - 2];
      if (flag == "--mtp-artifact") {
        if (artifact_set)
          throw std::runtime_error("duplicate MTP artifact");
        mtp.artifact = argv[positional_argc - 1];
        artifact_set = true;
        if (mtp.artifact.empty())
          throw std::runtime_error("empty MTP artifact path");
      } else if (flag == "--mtp-depth") {
        if (depth_set)
          throw std::runtime_error("duplicate MTP depth");
        const char *value = argv[positional_argc - 1];
        auto [stop, code] =
            std::from_chars(value, value + std::char_traits<char>::length(value), mtp.depth);
        if (code != std::errc{} || *stop || mtp.depth < 1 || mtp.depth > 4)
          throw std::runtime_error("MTP depth must be1..4");
        depth_set = true;
      } else if (flag == "--mtp-expert-dispatch") {
        if (expert_dispatch_set)
          throw std::runtime_error("duplicate MTP expert dispatch");
        const std::string_view value = argv[positional_argc - 1];
        if (value != "full" && value != "compact")
          throw std::runtime_error("MTP expert dispatch must be full or compact");
        mtp.compact_experts = value == "compact";
        expert_dispatch_set = true;
      } else if (flag == "--mtp-draft-experts") {
        if (draft_experts_set)
          throw std::runtime_error("duplicate MTP draft experts");
        const std::string_view value = argv[positional_argc - 1];
        if (value != "bf16" && value != "fp8" && value != "exl3")
          throw std::runtime_error("MTP draft experts must be bf16, fp8 or exl3");
        mtp.fp8_experts = value == "fp8";
        mtp.exl3_experts = value == "exl3";
        draft_experts_set = true;
      } else if (flag == "--head-format") {
        if (head_format_set)
          throw std::runtime_error("duplicate head format");
        const std::string_view value = argv[positional_argc - 1];
        if (value != "bf16" && value != "q8")
          throw std::runtime_error("head format must be bf16 or q8");
        q8_head = value == "q8";
        head_format_set = true;
      } else if (flag == "--config") {
        if (config_set)
          throw std::runtime_error("duplicate server configuration");
        config_set = true;
        std::ifstream file(argv[positional_argc - 1], std::ios::binary);
        if (!file)
          throw std::runtime_error("cannot open server configuration");
        std::array<char, 65537> bytes{};
        file.read(bytes.data(), bytes.size());
        if (file.gcount() > 65536 || (!file.eof() && !file.good()))
          throw std::runtime_error("server configuration read failed or exceeds 64 KiB");
        config_body.assign(bytes.data(), static_cast<std::size_t>(file.gcount()));
        if (config_body.empty())
          throw std::runtime_error("server configuration is empty");
      } else
        break;
      positional_argc -= 2;
    }
    if (depth_set && !artifact_set)
      throw std::runtime_error("MTP depth requires --mtp-artifact");
    if (expert_dispatch_set && !artifact_set)
      throw std::runtime_error("MTP expert dispatch requires --mtp-artifact");
    if (draft_experts_set && !artifact_set)
      throw std::runtime_error("MTP draft experts require --mtp-artifact");
    if (positional_argc < 5 || positional_argc > 10)
      throw std::runtime_error(
          "usage: padra-server ARTIFACT TOKENIZER_JSON TOKENIZER_SHA256 PORT "
          "[GENERATION_TIMEOUT_MS] [CONTEXT_CAPACITY] [PREFILL_CHUNK] "
          "[DEVICE_BUDGET_GIB] [KV_FORMAT=bf16|q8|q8_0] [--mtp-artifact DIRECTORY] [--mtp-depth "
          "1..4] [--mtp-expert-dispatch full|compact] [--mtp-draft-experts bf16|fp8|exl3] "
          "[--head-format bf16|q8] [--config JSON]");
    int port = 0;
    auto [end, error] =
        std::from_chars(argv[4], argv[4] + std::char_traits<char>::length(argv[4]), port);
    if (error != std::errc{} || *end || port < 1024 || port > 65535)
      throw std::runtime_error("port must be1024..65535");
    mellum16::server::ServerLimits limits;
    if (positional_argc > 5) {
      int milliseconds = 0;
      auto [stop, code] =
          std::from_chars(argv[5], argv[5] + std::char_traits<char>::length(argv[5]), milliseconds);
      if (code != std::errc{} || *stop || milliseconds < 1 || milliseconds > 600000)
        throw std::runtime_error("generation timeout must be1..600000ms");
      limits.generation_timeout = std::chrono::milliseconds(milliseconds);
    }
    int capacity = 131072;
    if (positional_argc <= 5)
      limits.generation_timeout = std::chrono::milliseconds(600000);
    if (positional_argc > 6) {
      auto [stop, code] =
          std::from_chars(argv[6], argv[6] + std::char_traits<char>::length(argv[6]), capacity);
      if (code != std::errc{} || *stop)
        throw std::runtime_error("invalid context capacity integer");
    }
    int chunk = 2048, budget = 0;
    for (int argument = 7; argument < std::min(positional_argc, 9); ++argument) {
      int &value = argument == 7 ? chunk : budget;
      auto [stop, code] = std::from_chars(
          argv[argument], argv[argument] + std::char_traits<char>::length(argv[argument]), value);
      if (code != std::errc{} || *stop)
        throw std::runtime_error("invalid explicit chunk/device budget");
    }
    if (!mellum16::ValidServerChunk(chunk) || !mellum16::ValidServerBudget(budget))
      throw std::runtime_error("server requires chunk2048/512/256 and budget0/8/12GiB");
    auto kv_format = mellum16::KVCacheFormat::BF16;
    if (positional_argc > 9) {
      const std::string_view format = argv[9];
      if (format == "q8")
        kv_format = mellum16::KVCacheFormat::Q8TokenHead128F32ScaleV1;
      else if (format == "q8_0")
        kv_format = mellum16::KVCacheFormat::Q8Token512Block32F16ScaleRotatedV2;
      else if (format != "bf16")
        throw std::runtime_error("KV_FORMAT must be bf16, q8 or q8_0");
    }
    auto context = mellum16::AdmitNativeContext(capacity);
    if (!context.ok())
      throw std::runtime_error(context.status().message());
    limits.queued_tokens = static_cast<std::size_t>(capacity) * 8;
    mellum16::server::ServerOptions options{limits, {}, port, capacity};
    auto validated = config_body.empty()
                         ? mellum16::server::ServerConfig::Create(options)
                         : mellum16::server::ServerConfig::Parse(config_body, options);
    if (!validated.ok())
      throw std::runtime_error(validated.status().message());
    auto config = std::move(validated).value();
    limits = config.generation();
    auto text = mellum16::text::Tokenizer::Load(argv[2], argv[3]);
    if (!text.ok())
      throw std::runtime_error(text.status().message());
    auto tokenizer = std::move(text).value();
    std::unique_ptr<mellum16::ModelRuntime> runtime;
    auto loaded = mellum16::ModelRuntime::Load(
        argv[1], tokenizer->sha256(), true, context.value(), kv_format, -1, chunk, budget,
        kv_format == mellum16::KVCacheFormat::BF16, mtp, q8_head);
    if (!loaded.ok())
      throw std::runtime_error(loaded.status().message());
    runtime = std::move(loaded).value();
    // Prepare startup graphs without claiming model inference; discard request.
    const std::array<int, 1> primer{0};
    auto prepared = runtime->PrepareRequestShape(primer, 0);
    if (!prepared.ok())
      throw std::runtime_error(prepared.message());
    auto reset = runtime->Reset();
    if (!reset.ok())
      throw std::runtime_error(reset.message());
    mellum16::server::Coordinator coordinator(std::move(runtime), *tokenizer, limits);
    mellum16::server::HttpServer http(coordinator, *tokenizer, config);
    if (http.Bind() <= 0)
      throw std::runtime_error("API/admin loopback bind failed");
    std::signal(SIGINT, Signal);
    std::signal(SIGTERM, Signal);
    std::atomic<bool> finished{false};
    std::jthread shutdown([&](std::stop_token stop) {
      while (!stop.stop_requested() && !finished.load() && !interrupted)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      if (interrupted)
        http.Stop();
    });
    std::cout << "{\"listener\":" << mellum16::json::Quote("127.0.0.1:" + std::to_string(port))
              << ",\"admin_listener\":"
              << mellum16::json::Quote("127.0.0.1:" + std::to_string(http.AdminPort()))
              << ",\"server_config\":" << config.Json(port, http.AdminPort()) << ",\"model\":"
              << mellum16::json::Quote(
                     mellum16::server::ModelIdForRate(coordinator.RuntimeMetrics().expert_rate))
              << ",\"mtp_depth\":" << coordinator.RuntimeMetrics().mtp_depth
              << ",\"quality_status\":\"unqualified\"}" << std::endl;
    const bool listened = http.Listen();
    finished.store(true);
    shutdown.join();
    coordinator.Shutdown();
    if (!listened)
      throw std::runtime_error("loopback listener failed");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
