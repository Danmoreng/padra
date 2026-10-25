#include "server_test_support.h"
#include "runtime/native_artifacts.h"
#include "cuda/device_matrix.cuh"
#include "runtime/server_memory.h"
int main() {
  try {
    const auto k4 = SelectedServerOptions("unused", "unused");
    const auto k3 = SelectedServerOptions("unused", "unused", 3);
#if PADRA_BUILD_SIMT
    Require(k4.exl3_rate == 4 && !k4.exl3_decode_mma && !k4.exl3_prefill_fp16,
            "SIMT K4 profile must not select Tensor Cores");
    Require(k3.exl3_rate == 3 && !k3.exl3_decode_mma,
            "SIMT selection must retain the rate for explicit runtime admission");
#else
    Require(k4.exl3_rate == 4 && k4.exl3_decode_mma && k4.exl3_packed_activations,
            "K4 selected dispatch changed");
    Require(k3.exl3_rate == 3 && k3.exl3_decode_mma && k3.exl3_decode_dual &&
                k3.exl3_prefill_fp16 && k3.exl3_packed_activations && k3.prefill_graphs &&
                k3.full_decode_graph,
            "K3 must select its own packed decoder with the optimized MMA dispatch");
#endif
    bool unsupported = false;
    try {
      (void)SelectedServerOptions("unused", "unused", 8);
    } catch (const std::invalid_argument &) {
      unsupported = true;
    }
    Require(unsupported, "selected server silently admitted an unsupported expert rate");
    unsupported = false;
    try {
      (void)ModelIdForRate(8);
    } catch (const std::invalid_argument &) {
      unsupported = true;
    }
    Require(unsupported, "unsupported expert rate was mislabeled as a supported model");
    const auto k4memory = PlanServerMemory(7092933120ULL, 8192, 8192, 2048, true);
    const auto k3memory =
        PlanServerMemory(5705667072ULL, 8192, 8192, 2048, true, 0, KVCacheFormat::BF16, false, 3);
    Require(k3memory.buffers.at("expert_schedule") == k4memory.buffers.at("expert_schedule") &&
                k3memory.buffers.at("expert_workspace") ==
                    k4memory.buffers.at("expert_workspace") &&
                k4memory.resident - k3memory.resident == 1387266048ULL,
            "K3 optimized workspace must match K4 apart from its smaller weights");
    Require(ServerExecutionProfile(8192, true, KVCacheFormat::BF16, 2048, 0, false, 3)
                .starts_with("exl3-k3-"),
            "K3 execution identity reports K4");
    internal::CompiledModel model;
    model.resident_weights_bytes = 6656;
    internal::Matrix matrix;
    matrix.format = internal::WeightFormat::Exl3K3;
    matrix.shape = {128, 128};
    matrix.regions = {{"packed", {0, 6144}}, {"suh", {6144, 256}}, {"svh", {6400, 256}}};
    model.tensors.emplace("expert", matrix);
    std::vector<unsigned char> arena(6656);
    const auto bound = BindDeviceMatrix(model, arena.data(), "expert");
    Require(bound.format == 6 && bound.suh && bound.svh &&
                reinterpret_cast<const unsigned char *>(bound.suh) == arena.data() + 6144 &&
                reinterpret_cast<const unsigned char *>(bound.svh) == arena.data() + 6400,
            "K3 device binding lost its FP16 sidecars");
    auto tokenizer = text::Tokenizer::Load("test", "test");
    Require(tokenizer.ok(), "synthetic tokenizer setup failed");
    auto state = std::make_shared<MockState>();
    state->expert_rate = 3;
    HttpFixture fixture(*tokenizer.value(), state);
    httplib::Client client("127.0.0.1", fixture.port);
    client.set_read_timeout(3);
    const auto models = JsonResponse(client.Get("/v1/models"));
    Require(models.find("data")->as_array().at(0).find("id")->as_string() == kK3ModelId,
            "models endpoint did not report selected K3 identity");
    Require(client.Post("/v1/chat/completions", ChatBody, "application/json")->status == 404,
            "K3 server accepted a request for K4");
    Require(client.Post("/v1/responses", ResponseBody, "application/json")->status == 404,
            "K3 Responses accepted a request for K4");
    const auto chat = JsonResponse(client.Post(
        "/v1/chat/completions",
        R"({"model":"mellum21-exl3-k3-research","messages":[{"role":"user","content":"hello"}],"max_completion_tokens":2})",
        "application/json"));
    Require(chat.find("model")->as_string() == kK3ModelId, "K3 Chat output is mislabeled");
    const auto response = JsonResponse(client.Post(
        "/v1/responses",
        R"({"model":"mellum21-exl3-k3-research","input":"hello","max_output_tokens":2})",
        "application/json"));
    Require(response.find("model")->as_string() == kK3ModelId, "K3 Responses output is mislabeled");
    for (bool responses : {false, true}) {
      const auto body =
          responses
              ? R"({"model":"mellum21-exl3-k3-research","input":"hello","max_output_tokens":2,"stream":true})"
              : R"({"model":"mellum21-exl3-k3-research","messages":[{"role":"user","content":"hello"}],"max_completion_tokens":2,"stream":true})";
      const auto reply = client.Post(responses ? "/v1/responses" : "/v1/chat/completions", body,
                                     "application/json");
      Require(reply && reply->status == 200 &&
                  reply->body.find("\"model\":\"mellum21-exl3-k3-research\"") !=
                      std::string::npos &&
                  reply->body.find("mellum21-exl3-k4-research") == std::string::npos,
              "K3 stream omitted or mislabeled its selected model");
    }
    const auto session = JsonResponse(client.Post(
        "/v1/sessions", R"({"model":"mellum21-exl3-k3-research"})", "application/json"));
    Require(session.find("model")->as_string() == kK3ModelId, "K3 session is mislabeled");
    std::cout << "K3 selected artifact and API contract passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
