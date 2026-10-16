// Isolated post-RoPE attention benchmark. No model artifacts or client API.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "compiler/sha256.h"
#include "runtime/prefill_state.h"
#include "runtime/build_identity.h"
#include "util/json.h"
#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
using mellum16::PrefillState;
using mellum16::ValidPrefillState;
constexpr int kW4A8FContextCapacity = 131072, kW4A8FMaxPrefillChunk = 2048;
__device__ float B(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}
__device__ void Finite(float x, int *bad) {
  if (!isfinite(x))
    atomicExch(bad, 1);
}
#include "cuda/attention_mma.cuh"
#include "cuda/attention_prefill_padded_single.cuh"
#include "attention_prefill_frozen.cuh"
namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
  void *pointer = nullptr;
  explicit Buffer(std::size_t bytes) {
    Check(cudaMalloc(&pointer, bytes));
  }
  ~Buffer() {
    cudaFree(pointer);
  }
  template <class T> T *As() const {
    return static_cast<T *>(pointer);
  }
};
struct Stream {
  cudaStream_t value;
  Stream() {
    Check(cudaStreamCreate(&value));
  }
  ~Stream() {
    cudaStreamDestroy(value);
  }
};
struct Event {
  cudaEvent_t value;
  Event() {
    Check(cudaEventCreate(&value));
  }
  ~Event() {
    cudaEventDestroy(value);
  }
};
struct Graph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  ~Graph() {
    if (exec)
      cudaGraphExecDestroy(exec);
    if (graph)
      cudaGraphDestroy(graph);
  }
};
struct Options {
  int base = 1024, tokens = 8, capacity = 131072, repeats = 7, warmup = 3;
  bool local = false, round = false, graph = false, poison = false;
  std::string fixture, dump;
};
Options Parse(int argc, char **argv) {
  Options o;
  const auto integer = [](const std::string &text) {
    int value = 0;
    auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
      throw std::runtime_error("option requires a complete integer");
    return value;
  };
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--local")
      o.local = true;
    else if (key == "--round-output")
      o.round = true;
    else if (key == "--graph")
      o.graph = true;
    else if (key == "--poison-invisible")
      o.poison = true;
    else {
      if (++i == argc)
        throw std::runtime_error("missing option value");
      const std::string v = argv[i];
      if (key == "--base")
        o.base = integer(v);
      else if (key == "--tokens")
        o.tokens = integer(v);
      else if (key == "--capacity")
        o.capacity = integer(v);
      else if (key == "--repeats")
        o.repeats = integer(v);
      else if (key == "--warmup")
        o.warmup = integer(v);
      else if (key == "--fixture")
        o.fixture = v;
      else if (key == "--dump-fixture")
        o.dump = v;
      else
        throw std::runtime_error("unknown option: " + key);
    }
  }
  if (o.capacity < 1 || o.capacity > 131072 || o.tokens < 1 || o.tokens > 2048 || o.base < 0 ||
      o.base > o.capacity - o.tokens || o.repeats < 1 || o.repeats > 100 || o.warmup < 1 ||
      o.warmup > 100)
    throw std::runtime_error("inadmissible attention shape/repetitions");
  return o;
}
template <class T>
void Fixture(std::vector<T> &v, const std::string &input, const std::string &output,
             const char *name, mellum16::compiler::Sha256 &hash) {
  const auto bytes = v.size() * sizeof(T);
  if (!input.empty()) {
    auto path = std::filesystem::path(input) / name;
    if (std::filesystem::file_size(path) != bytes)
      throw std::runtime_error("fixture byte count mismatch");
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char *>(v.data()), bytes))
      throw std::runtime_error("fixture read failed");
  }
  hash.Update(name);
  hash.Update(v.data(), bytes);
  if (!output.empty()) {
    std::filesystem::create_directories(output);
    std::ofstream file(std::filesystem::path(output) / name, std::ios::binary);
    if (!file.write(reinterpret_cast<const char *>(v.data()), bytes))
      throw std::runtime_error("fixture write failed");
  }
}
float Value(std::size_t i, unsigned seed) {
  unsigned x = static_cast<unsigned>(i) ^ seed;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return (static_cast<int>(x & 2047) - 1024) / 4096.f;
}
// FP64 scalar oracle on small fixtures. It isolates indexing/masks/GQA errors;
// BF16 intermediate probability rounding makes this a tolerance comparison.
double Oracle(const Options &o, int cap, const std::vector<float> &q,
              const std::vector<__nv_bfloat16> &k, const std::vector<__nv_bfloat16> &v,
              const std::vector<__nv_bfloat16> &ck, const std::vector<__nv_bfloat16> &cv,
              const std::vector<float> &output) {
  double error = 0;
  if (o.tokens > 16 || o.base + o.tokens > 2048)
    return -1;
  for (int row = 0; row < o.tokens; ++row)
    for (int head = 0; head < 32; ++head) {
      const int end = o.base + row, first = o.local ? std::max(0, end - 1023) : 0;
      std::vector<double> scores;
      auto kv = [&](const auto &cache, const auto &chunk, int key, int d) {
        return double(__bfloat162float(key < o.base
                                           ? cache[(key % cap) * 512 + (head / 8) * 128 + d]
                                           : chunk[(key - o.base) * 512 + (head / 8) * 128 + d]));
      };
      for (int key = first; key <= end; ++key) {
        double dot = 0;
        for (int d = 0; d < 128; ++d)
          dot += __bfloat162float(__float2bfloat16_rn(q[(row * 32 + head) * 128 + d])) *
                 kv(k, ck, key, d);
        scores.push_back(dot * .08838834764831845);
      }
      const double maximum = *std::max_element(scores.begin(), scores.end());
      double sum = 0;
      for (auto &score : scores) {
        score = std::exp(score - maximum);
        sum += score;
      }
      for (int d = 0; d < 128; ++d) {
        double result = 0;
        for (int key = first; key <= end; ++key)
          result += scores[key - first] * kv(v, cv, key, d);
        result /= sum;
        const double delta = std::abs(result - output[(row * 32 + head) * 128 + d]);
        if (delta > .002 + .02 * std::abs(result))
          throw std::runtime_error("FP64 oracle tolerance exceeded");
        error = std::max(error, delta);
      }
    }
  return error;
}
} // namespace
int main(int argc, char **argv) {
  try {
    const auto o = Parse(argc, argv);
    const int cap = o.local ? std::min(1024, o.capacity) : o.capacity;
    const std::size_t nq = std::size_t(o.tokens) * 4096, nk = std::size_t(cap) * 512,
                      nc = std::size_t(o.tokens) * 512;
    const auto required = (2 * nq) * sizeof(float) + (2 * nk + 2 * nc) * sizeof(__nv_bfloat16) +
                          sizeof(int) + sizeof(PrefillState);
    std::size_t free = 0, total = 0;
    Check(cudaMemGetInfo(&free, &total));
    if (free < required + 700ULL * 1024 * 1024)
      throw std::runtime_error("700 MiB reserve not available");
    std::vector<float> q(nq), ref(nq), actual(nq);
    std::vector<__nv_bfloat16> k(nk), v(nk), ck(nc), cv(nc);
    for (std::size_t i = 0; i < nq; ++i)
      q[i] = Value(i, 123);
    for (std::size_t i = 0; i < nk; ++i) {
      k[i] = __float2bfloat16_rn(Value(i, 456));
      v[i] = __float2bfloat16_rn(Value(i, 789));
    }
    for (std::size_t i = 0; i < nc; ++i) {
      ck[i] = __float2bfloat16_rn(Value(i, 101));
      cv[i] = __float2bfloat16_rn(Value(i, 202));
    }
    if (o.poison) {
      std::vector<bool> visible(cap, false);
      for (int key = o.local ? std::max(0, o.base - 1023) : 0; key < o.base; ++key)
        visible[key % cap] = true;
      for (int key = 0; key < cap; ++key)
        if (!visible[key])
          for (int d = 0; d < 512; ++d)
            k[key * 512 + d] = v[key * 512 + d] = __float2bfloat16_rn(NAN);
    }
    mellum16::compiler::Sha256 hash;
    for (int value : {o.base, o.tokens, o.capacity, int(o.local), int(o.round)})
      hash.Update(&value, sizeof(value));
    Fixture(q, o.fixture, o.dump, "q.f32", hash);
    Fixture(k, o.fixture, o.dump, "k.bf16", hash);
    Fixture(v, o.fixture, o.dump, "v.bf16", hash);
    Fixture(ck, o.fixture, o.dump, "chunk-k.bf16", hash);
    Fixture(cv, o.fixture, o.dump, "chunk-v.bf16", hash);
    Buffer dq(nq * 4), dk(nk * 2), dv(nk * 2), dck(nc * 2), dcv(nc * 2), dout(nq * 4), bad(4),
        state(sizeof(PrefillState));
    Stream stream;
    for (auto entry : {std::pair{dq.pointer, static_cast<const void *>(q.data())},
                       {dk.pointer, k.data()},
                       {dv.pointer, v.data()},
                       {dck.pointer, ck.data()},
                       {dcv.pointer, cv.data()}}) {
      const auto bytes =
          entry.first == dq.pointer
              ? nq * 4
              : (entry.first == dk.pointer || entry.first == dv.pointer ? nk * 2 : nc * 2);
      Check(cudaMemcpy(entry.first, entry.second, bytes, cudaMemcpyHostToDevice));
    }
    PrefillState command{o.base, o.base, o.tokens, o.capacity, o.base, 1, 1, 1};
    Check(cudaMemcpy(state.pointer, &command, sizeof(command), cudaMemcpyHostToDevice));
    Check(cudaMemset(bad.pointer, 0, 4));
    const dim3 grid((o.tokens + 7) / 8, 4);
    auto launch = [&](bool frozen) {
      const auto run = [&]<bool Local, bool Round>() {
        if (frozen)
          frozen_prefill_attention::PipelineBF16<Local, Round><<<grid, 128, 0, stream.value>>>(
              dq.As<float>(), dk.As<__nv_bfloat16>(), dv.As<__nv_bfloat16>(),
              dck.As<__nv_bfloat16>(), dcv.As<__nv_bfloat16>(), dout.As<float>(), o.base, o.tokens,
              cap, bad.As<int>(), state.As<PrefillState>(), o.capacity);
        else
          padded_single_prefill_attention::PipelineBF16<Local, Round>
              <<<grid, 128, 0, stream.value>>>(
                  dq.As<float>(), dk.As<__nv_bfloat16>(), dv.As<__nv_bfloat16>(),
                  dck.As<__nv_bfloat16>(), dcv.As<__nv_bfloat16>(), dout.As<float>(), o.base,
                  o.tokens, cap, bad.As<int>(), state.As<PrefillState>(), o.capacity);
      };
      if (o.local) {
        if (o.round)
          run.template operator()<true, true>();
        else
          run.template operator()<true, false>();
      } else {
        if (o.round)
          run.template operator()<false, true>();
        else
          run.template operator()<false, false>();
      }
      Check(cudaGetLastError());
    };
    launch(true);
    Check(cudaStreamSynchronize(stream.value));
    Check(cudaMemcpy(ref.data(), dout.pointer, nq * 4, cudaMemcpyDeviceToHost));
    Graph graph;
    if (o.graph) {
      Check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
      launch(false);
      Check(cudaStreamEndCapture(stream.value, &graph.graph));
      Check(cudaGraphInstantiate(&graph.exec, graph.graph, 0));
    }
    auto run = [&] {
      if (o.graph)
        Check(cudaGraphLaunch(graph.exec, stream.value));
      else
        launch(false);
    };
    for (int i = 0; i < o.warmup; ++i)
      run();
    Check(cudaStreamSynchronize(stream.value));
    Event start, stop;
    std::vector<float> ms;
    for (int i = 0; i < o.repeats; ++i) {
      Check(cudaEventRecord(start.value, stream.value));
      run();
      Check(cudaEventRecord(stop.value, stream.value));
      Check(cudaEventSynchronize(stop.value));
      float elapsed = 0;
      Check(cudaEventElapsedTime(&elapsed, start.value, stop.value));
      ms.push_back(elapsed);
    }
    Check(cudaMemcpy(actual.data(), dout.pointer, nq * 4, cudaMemcpyDeviceToHost));
    int failed = 0;
    Check(cudaMemcpy(&failed, bad.pointer, 4, cudaMemcpyDeviceToHost));
    if (failed || std::memcmp(ref.data(), actual.data(), nq * 4))
      throw std::runtime_error("frozen operator mismatch/nonfinite result");
    const double oracle = Oracle(o, cap, q, k, v, ck, cv, actual);
    cudaDeviceProp gpu{};
    Check(cudaGetDeviceProperties(&gpu, 0));
    int driver = 0, runtime = 0;
    Check(cudaDriverGetVersion(&driver));
    Check(cudaRuntimeGetVersion(&runtime));
    cudaFuncAttributes attributes{};
    int active_blocks = 0;
    const auto resources = [&]<bool Local, bool Round>() {
      Check(cudaFuncGetAttributes(&attributes,
                                  padded_single_prefill_attention::PipelineBF16<Local, Round>));
      Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks, padded_single_prefill_attention::PipelineBF16<Local, Round>, 128, 0));
    };
    if (o.local) {
      if (o.round)
        resources.template operator()<true, true>();
      else
        resources.template operator()<true, false>();
    } else {
      if (o.round)
        resources.template operator()<false, true>();
      else
        resources.template operator()<false, false>();
    }
    std::cout << "{\"fixture_sha256\":" << mellum16::json::Quote(hash.HexDigest())
              << ",\"build_source_sha256\":"
              << mellum16::json::Quote(mellum16::kPadraBuildSourceSha256)
              << ",\"benchmark_source_sha256\":"
              << mellum16::json::Quote(PADRA_ATTENTION_BENCH_SHA256) << ",\"nvcc\":\""
              << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__ << '.' << __CUDACC_VER_BUILD__
              << "\""
              << ",\"gpu\":" << mellum16::json::Quote(gpu.name)
              << ",\"sm_count\":" << gpu.multiProcessorCount << ",\"cuda_runtime\":" << runtime
              << ",\"registers_per_thread\":" << attributes.numRegs
              << ",\"static_shared_bytes\":" << attributes.sharedSizeBytes
              << ",\"thread_local_bytes\":" << attributes.localSizeBytes
              << ",\"active_blocks_per_sm\":" << active_blocks << ",\"cuda_driver\":" << driver
              << ",\"base\":" << o.base << ",\"tokens\":" << o.tokens
              << ",\"capacity\":" << o.capacity << ",\"local\":" << (o.local ? "true" : "false")
              << ",\"graph\":" << (o.graph ? "true" : "false")
              << ",\"round_output\":" << (o.round ? "true" : "false")
              << ",\"operand\":\"bf16\",\"accumulator\":\"fp32\",\"probability\":\"bf16\",\"q_"
                 "scaling\":\"post_qk_fp32\",\"tile\":[8,32,128],\"splits\":1,\"workspace_bytes\":"
              << required << ",\"frozen_bitwise_equal\":true,\"fp64_max_abs_error\":" << oracle
              << ",\"milliseconds\":[";
    for (std::size_t i = 0; i < ms.size(); ++i)
      std::cout << (i ? "," : "") << ms[i];
    std::cout << "]}\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
