#include "runtime/model_runtime.h"
#include "util/json.h"
#include "compiler/sha256.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cuda_profiler_api.h>
namespace mellum16 {
Result<std::unique_ptr<RuntimeBackend>> LoadCudaModelRuntime(InferenceOptions);
}
using namespace mellum16;
json::Value Read(const std::filesystem::path &path) {
  std::ifstream f(path);
  std::string s{std::istreambuf_iterator<char>(f), {}};
  auto r = json::Parse(s);
  if (!r.ok())
    throw std::runtime_error(r.status().message());
  return std::move(r).value();
}
void Check(Status s) {
  if (!s.ok())
    throw std::runtime_error(s.message());
}
int main(int argc, char **argv) {
  try {
    if (argc != 5 && argc != 6)
      throw std::runtime_error("usage padra-mtp-probe TARGET MTP REQUESTS OUTPUT [DEPTH]");
    const auto requests = Read(argv[3]);
    std::ofstream out(argv[4]);
    out << std::setprecision(17);
    const std::string tokenizer =
        "58548a346eb073e5132bf7d8ad17dc6971bca36ade378ca4d2bfbc49bf60da2a";
    const int first = argc == 6 ? std::stoi(argv[5]) : 0;
    const int last = argc == 6 ? first : 4;
    for (int depth = first; depth <= last; ++depth) {
      const auto *rate = requests.as_array().front().find("expert_rate");
      auto options = SelectedServerOptions(argv[1], tokenizer, rate ? int(rate->as_integer()) : 4);
      options.controlled_chunk_graphs = true;
      options.capture_hidden = true;
      options.q8_head = true;
      if (const auto *c = requests.as_array().front().find("q8_head"))
        options.q8_head = c->as_bool();
      if (const auto *c = requests.as_array().front().find("capacity"))
        options.capacity = int(c->as_integer());
      options.server_kv_checkpoint = true;
      if (const auto *c = requests.as_array().front().find("kv_format")) {
        if (c->as_string() == "q8")
          options.kv_cache_format = KVCacheFormat::Q8TokenHead128F32ScaleV1;
        else if (c->as_string() == "q8_0")
          options.kv_cache_format = KVCacheFormat::Q8Token512Block32F16ScaleRotatedV2;
        else if (c->as_string() != "bf16")
          throw std::runtime_error("unknown KV format");
      }
      if (const auto *c = requests.as_array().front().find("checkpoint_enabled"))
        options.server_kv_checkpoint = c->as_bool();
      if (const auto *c = requests.as_array().front().find("device_budget_gib"))
        options.server_device_budget_gib = int(c->as_integer());
      if (depth) {
        options.mtp = {argv[2], depth};
        options.mtp.exl3_experts = true;
      }
      if (const auto *c = requests.as_array().front().find("exl3_experts"))
        options.mtp.exl3_experts = depth && c->as_bool();
      if (const auto *c = requests.as_array().front().find("packed_qkv"))
        options.mtp.packed_qkv = c->as_bool();
      if (const auto *c = requests.as_array().front().find("batched_prime"))
        options.mtp.batched_prime = c->as_bool();
      if (const auto *c = requests.as_array().front().find("prefill_chunk"))
        options.prefill_chunk = int(c->as_integer());
      if (const auto *c = requests.as_array().front().find("packed_o"))
        options.mtp.packed_o = c->as_bool();
      if (const auto *c = requests.as_array().front().find("compact_experts"))
        options.mtp.compact_experts = c->as_bool();
      if (const auto *c = requests.as_array().front().find("fp8_experts"))
        options.mtp.fp8_experts = depth && c->as_bool();
      auto backend = LoadCudaModelRuntime(options);
      if (!backend.ok())
        throw std::runtime_error(backend.status().message());
      ModelRuntime runtime(std::move(backend).value());
      std::vector<int> previous;
      std::optional<KVCheckpointHandle> checkpoint;
      int index = 0;
      for (const auto &request : requests.as_array()) {
        auto fixture = Read(request.find("input")->as_string());
        std::vector<int> ids;
        for (const auto &id : fixture.find("input_ids")->as_array())
          ids.push_back(int(id.as_integer()));
        if (const auto *n = request.find("prefix"))
          ids.resize(n->as_integer());
        const bool append = request.find("append") && request.find("append")->as_bool();
        const bool restore =
            request.find("restore_checkpoint") && request.find("restore_checkpoint")->as_bool();
        if (append) {
          previous.push_back(233);
          ids = previous;
        } else if (!restore)
          Check(runtime.Reset());
        if (const auto *at = request.find("mutate_at"))
          ids.at(at->as_integer()) = 233;
        const int generate = int(request.find("generate")->as_integer());
        SamplingOptions sampling;
        if (const auto *s = request.find("temperature"))
          sampling.temperature = float(s->as_number());
        if (const auto *s = request.find("seed"))
          sampling.seed = s->as_integer();
        if (const auto *s = request.find("top_k"))
          sampling.top_k = int(s->as_integer());
        if (const auto *s = request.find("top_p"))
          sampling.top_p = float(s->as_number());
        if (const auto *s = request.find("frequency_penalty"))
          sampling.frequency_penalty = float(s->as_number());
        if (const auto *s = request.find("presence_penalty"))
          sampling.presence_penalty = float(s->as_number());
        if (const auto *s = request.find("repetition_penalty"))
          sampling.repetition_penalty = float(s->as_number());
        const int anchor = request.find("checkpoint_anchor")
                               ? int(request.find("checkpoint_anchor")->as_integer())
                               : 0;
        const auto preparationStarted = std::chrono::steady_clock::now();
        if (restore) {
          if (!checkpoint)
            throw std::runtime_error("missing checkpoint");
          Check(
              runtime.PrepareCheckpointRestore(ids, generate, *checkpoint, anchor, true, sampling));
        } else if (anchor)
          Check(runtime.PrepareWithCheckpoint(ids, generate, anchor, true, sampling));
        else
          Check(append ? runtime.PrepareAppend(ids, generate, true, sampling)
                       : runtime.PrepareRequestShape(ids, generate, true, sampling));
        const double preparation =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - preparationStarted)
                .count();
        int callbacks = 0;
        const int cancel =
            request.find("cancel_after") ? int(request.find("cancel_after")->as_integer()) : -1;
        const bool profile =
            request.find("profile_decode") && request.find("profile_decode")->as_bool();
        const int cancelPrefill = request.find("cancel_prefill_after")
                                      ? int(request.find("cancel_prefill_after")->as_integer())
                                      : -1;
        const auto generationStarted = std::chrono::steady_clock::now();
        auto result = runtime.Generate(
            [&](int) {
              ++callbacks;
              if (profile && callbacks == 1)
                cudaProfilerStart();
              return cancel < 0 || callbacks < cancel;
            },
            [&](int committed) { return cancelPrefill < 0 || committed < cancelPrefill; });
        if (profile)
          cudaProfilerStop();
        if (!result.ok())
          throw std::runtime_error("D" + std::to_string(depth) + " request" +
                                   std::to_string(index) + ": " + result.status().message());
        const double wall =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - generationStarted)
                .count();
        const auto &r = result.value();
        auto stats = runtime.Stats();
        out << "{\"depth\":" << depth << ",\"request\":" << index++ << ",\"ids\":[";
        for (std::size_t i = 0; i < r.generated_ids.size(); ++i) {
          if (i)
            out << ',';
          out << r.generated_ids[i];
        }
        out << "],\"head_sha256\":"
            << json::Quote(compiler::Sha256Hex(r.logits.data(), r.logits.size() * 4))
            << ",\"hidden_sha256\":"
            << json::Quote(compiler::Sha256Hex(r.hidden.data(), r.hidden.size() * 4))
            << ",\"callbacks\":" << callbacks
            << ",\"cancelled\":" << (r.cancelled ? "true" : "false")
            << ",\"committed\":" << stats.committed
            << ",\"pending\":" << (stats.pending ? std::to_string(*stats.pending) : "null")
            << ",\"prefill_s\":" << r.prefill_seconds << ",\"decode_s\":" << r.decode_seconds
            << ",\"decode_steps\":" << r.decode_steps << ",\"generation_wall_s\":" << wall
            << ",\"capacity\":" << stats.capacity << ",\"prefill_tokens\":" << r.prefill_tokens
            << ",\"prepare_s\":" << preparation << ",\"cached_tokens\":" << stats.cached_tokens
            << ",\"checkpoint_bytes\":" << stats.checkpoint_bytes
            << ",\"checkpoint_captures\":" << stats.checkpoint_captures
            << ",\"checkpoint_restores\":" << stats.checkpoint_restores
            << ",\"checkpoint_available\":" << (stats.checkpoint_available ? "true" : "false")
            << ",\"mtp_bytes\":" << r.mtp_resident_bytes
            << ",\"kernel_profile\":" << json::Quote(stats.kernel_profile)
            << ",\"resident_bytes\":" << stats.resident_bytes
            << ",\"free_initialized\":" << r.device_free_after_initialization
            << ",\"device_total_bytes\":" << stats.device_total_bytes
            << ",\"device_free_before_bytes\":" << stats.device_free_before_bytes
            << ",\"required_bytes\":" << stats.required_bytes
            << ",\"post_graph_available_bytes\":" << stats.post_graph_available_bytes
            << ",\"draft_calls\":" << r.mtp_draft_calls << ",\"groups\":" << r.mtp_groups
            << ",\"proposed\":" << r.mtp_proposed << ",\"accepted\":" << r.mtp_accepted
            << ",\"tails\":" << r.mtp_ordinary_tails << "}\n"
            << std::flush;
        if (auto handle = runtime.Checkpoint())
          checkpoint = handle;
        previous = ids;
        previous.insert(previous.end(), r.generated_ids.begin(), r.generated_ids.end());
      }
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
