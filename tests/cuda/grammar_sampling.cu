// No model required: exercise the production sampler and greedy reduction.
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "runtime/sampling.h"
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace mellum16 {
namespace {
void Check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
struct Control {
  int token, bad;
};
#include "cuda/selection.cuh"
#include "cuda/sampling.cuh"
struct Buffer {
  void *p = nullptr;
  explicit Buffer(std::size_t bytes) {
    Check(cudaMallocManaged(&p, bytes));
  }
  ~Buffer() {
    cudaFree(p);
  }
  template <class T> T *As() {
    return static_cast<T *>(p);
  }
};
void Require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Test() {
  constexpr int n = 32;
  Buffer source(n * 4), adjusted(n * 4), sorted(n * 4), ids(n * 4), sorted_ids(n * 4),
      counts(n * 4), cumulative(n * 8), rng(sizeof(gpu_sampling::State)), control(sizeof(Control)),
      bad(4), mask(n), candidates(sizeof(SelectionCandidate));
  const auto bytes = gpu_sampling::WorkspaceBytes(n);
  Buffer workspace(bytes);
  for (int i = 0; i < n; ++i) {
    source.As<float>()[i] = static_cast<float>(i);
    counts.As<int>()[i] = 0;
    mask.As<std::uint8_t>()[i] = 0;
  }
  source.As<float>()[0] = 1000;
  mask.As<std::uint8_t>()[3] = 1;
  mask.As<std::uint8_t>()[7] = 1;
  const std::vector<float> original(source.As<float>(), source.As<float>() + n);
  *bad.As<int>() = 0;
  *rng.As<gpu_sampling::State>() = {42, 0};
  SamplingOptions options;
  options.temperature = .6f;
  options.top_k = 1;
  const auto sample = [&] {
    gpu_sampling::Launch(source.As<float>(), adjusted.As<float>(), sorted.As<float>(),
                         ids.As<int>(), sorted_ids.As<int>(), cumulative.As<double>(),
                         rng.As<gpu_sampling::State>(), control.As<Control>(), bad.As<int>(),
                         workspace.p, bytes, nullptr, options, counts.As<int>(), n,
                         mask.As<std::uint8_t>());
    Check(cudaDeviceSynchronize());
    Require(!control.As<Control>()->bad, "sampling fault");
  };
  sample();
  Require(control.As<Control>()->token == 7, "mask applied after top-k");
  options.top_k = 0;
  options.top_p = .95f;
  for (int i = 0; i < 64; ++i) {
    sample();
    Require(mask.As<std::uint8_t>()[control.As<Control>()->token], "sampled forbidden token");
  }
  Require(rng.As<gpu_sampling::State>()->step == 65, "grammar changed RNG step count");
  SelectTiles<<<1, 256>>>(source.As<float>(), candidates.As<SelectionCandidate>(), n, bad.As<int>(),
                          mask.As<std::uint8_t>());
  Select<<<1, 256>>>(candidates.As<SelectionCandidate>(), control.As<Control>(), 1, bad.As<int>());
  Check(cudaDeviceSynchronize());
  Require(control.As<Control>()->token == 7 && !control.As<Control>()->bad, "greedy mask failed");
  cudaStream_t stream;
  Check(cudaStreamCreate(&stream));
  cudaGraph_t graph;
  cudaGraphExec_t executable;
  Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  gpu_sampling::Launch(source.As<float>(), adjusted.As<float>(), sorted.As<float>(), ids.As<int>(),
                       sorted_ids.As<int>(), cumulative.As<double>(), rng.As<gpu_sampling::State>(),
                       control.As<Control>(), bad.As<int>(), workspace.p, bytes, stream, options,
                       counts.As<int>(), n, mask.As<std::uint8_t>());
  Check(cudaStreamEndCapture(stream, &graph));
  Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  mask.As<std::uint8_t>()[7] = 0;
  Check(cudaGraphLaunch(executable, stream));
  Check(cudaStreamSynchronize(stream));
  Require(control.As<Control>()->token == 3, "captured sampler ignored new mask");
  Require(std::equal(original.begin(), original.end(), source.As<float>()),
          "source logits mutated");
  Check(cudaGraphExecDestroy(executable));
  Check(cudaGraphDestroy(graph));
  Check(cudaStreamDestroy(stream));
}
} // namespace
} // namespace mellum16
int main() {
  try {
    mellum16::Test();
    std::cout << "grammar GPU sampler checks passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
