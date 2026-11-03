#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <vector>
#include "runtime/sampling.h"
using mellum16::SamplingOptions;
struct Control {
  int token, bad;
};
void Check(cudaError_t e) {
  if (e != cudaSuccess) {
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    std::exit(1);
  }
}
#include "cuda/sampling.cuh"

template <class T> T *alloc(size_t n) {
  T *p;
  Check(cudaMalloc(&p, n * sizeof(T)));
  return p;
}
struct Sampler {
  int n;
  float *logits, *adjusted, *sorted;
  int *ids, *sortedIds, *counts, *bad;
  double *cdf;
  gpu_sampling::State *state;
  Control *chosen;
  void *temp;
  size_t bytes;
  Sampler(int n) : n(n) {
    logits = alloc<float>(n);
    adjusted = alloc<float>(n);
    sorted = alloc<float>(n);
    ids = alloc<int>(n);
    sortedIds = alloc<int>(n);
    counts = alloc<int>(n);
    bad = alloc<int>(1);
    cdf = alloc<double>(n);
    state = alloc<gpu_sampling::State>(1);
    chosen = alloc<Control>(1);
    bytes = gpu_sampling::WorkspaceBytes(n);
    Check(cudaMalloc(&temp, bytes));
  }
  void init(const std::vector<float> &v, const std::vector<int> &c, uint64_t step = 0) {
    Check(cudaMemcpy(logits, v.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(counts, c.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    Check(cudaMemset(bad, 0, sizeof(int)));
    gpu_sampling::State s{123456789, step};
    Check(cudaMemcpy(state, &s, sizeof s, cudaMemcpyHostToDevice));
  }
  void launch(SamplingOptions o, bool full, cudaStream_t stream = nullptr) {
    if (full)
      gpu_sampling::Launch<false>(logits, adjusted, sorted, ids, sortedIds, cdf, state, chosen, bad,
                                  temp, bytes, stream, o, counts, n);
    else
      gpu_sampling::Launch(logits, adjusted, sorted, ids, sortedIds, cdf, state, chosen, bad, temp,
                           bytes, stream, o, counts, n);
  }
};
int main() {
  const int n = 98304;
  Sampler a(n), b(n);
  std::vector<float> v(n);
  std::vector<int> c(n);
  for (int i = 0; i < n; ++i) {
    v[i] = float((i * 7919) % 257 - 128) / 17;
    c[i] = i % 11 == 0 ? i % 5 + 1 : 0;
  }
  for (int scenario = 0; scenario < 7; ++scenario)
    for (int k : {0, 1, 2, 16, 40, 256, 1024, 1025, 98304}) {
      auto input = v;
      if (scenario == 1)
        for (int i = 0; i < n; ++i)
          input[i] = 0;
      if (scenario == 2)
        for (int i = 0; i < n; ++i)
          input[i] = i % 2 ? -0.0f : 0.0f;
      if (scenario == 6)
        for (int i = 0; i < n; ++i)
          input[i] = float((uint32_t(i) * 2654435761U) >> 8) / 1048576.f - 8.f;
      if (scenario == 3)
        input[17] = NAN;
      if (scenario == 4)
        input[18] = INFINITY;
      SamplingOptions o;
      o.temperature = .73f;
      o.top_k = k;
      o.top_p = scenario == 5 ? .82f : 1.f;
      if (scenario == 5) {
        o.frequency_penalty = .25f;
        o.presence_penalty = -.3f;
        o.repetition_penalty = 1.1f;
      }
      a.init(input, c);
      b.init(input, c);
      for (int step = 0; step < 20; ++step) {
        a.launch(o, false);
        b.launch(o, true);
        Control x, y;
        gpu_sampling::State sx, sy;
        Check(cudaMemcpy(&x, a.chosen, sizeof x, cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(&y, b.chosen, sizeof y, cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(&sx, a.state, sizeof sx, cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(&sy, b.state, sizeof sy, cudaMemcpyDeviceToHost));
        const int eligible = k ? k : n;
        std::vector<double> ca(eligible), cb(eligible);
        Check(cudaMemcpy(ca.data(), a.cdf, eligible * sizeof(double), cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(cb.data(), b.cdf, eligible * sizeof(double), cudaMemcpyDeviceToHost));
        if (x.token != y.token || x.bad != y.bad || sx.step != sy.step ||
            sx.step != static_cast<uint64_t>(scenario == 3 || scenario == 4 ? 0 : step + 1) ||
            x.bad != (scenario == 3 || scenario == 4) ||
            ((scenario == 1 || scenario == 2) && k == 1 && x.token != 0) ||
            (k > 0 && k <= 1024 && ca != cb)) {
          fprintf(stderr, "mismatch scenario=%d k=%d step=%d: %d/%d %d/%d steps %llu/%llu\n",
                  scenario, k, step, x.token, y.token, x.bad, y.bad, (unsigned long long)sx.step,
                  (unsigned long long)sy.step);
          return 1;
        }
      }
    }
  // Invalid options must still set the same error and leave RNG progression unchanged.
  for (int invalid = 0; invalid < 3; ++invalid) {
    SamplingOptions o;
    o.temperature = invalid == 2 ? 0.f : .8f;
    o.top_k = invalid == 0 ? -1 : invalid == 1 ? n + 1 : 40;
    a.init(v, c);
    b.init(v, c);
    a.launch(o, false);
    b.launch(o, true);
    Control x, y;
    gpu_sampling::State sx, sy;
    Check(cudaMemcpy(&x, a.chosen, sizeof x, cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(&y, b.chosen, sizeof y, cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(&sx, a.state, sizeof sx, cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(&sy, b.state, sizeof sy, cudaMemcpyDeviceToHost));
    if (!x.bad || x.bad != y.bad || sx.step != sy.step || sx.step != 0)
      return 1;
  }
  puts("differential sampling cases passed");
  for (int k : {1, 40, 256, 1024}) {
    SamplingOptions o;
    o.temperature = .8f;
    o.top_k = k;
    a.init(v, c);
    b.init(v, c);
    for (int j = 0; j < 20; ++j) {
      a.launch(o, false);
      b.launch(o, true);
    }
    Check(cudaDeviceSynchronize());
    for (bool full : {false, true}) {
      auto &s = full ? b : a;
      cudaEvent_t start, end;
      Check(cudaEventCreate(&start));
      Check(cudaEventCreate(&end));
      cudaGraph_t graph;
      cudaGraphExec_t exec;
      cudaStream_t stream;
      Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      s.launch(o, full, stream);
      Check(cudaStreamEndCapture(stream, &graph));
      Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
      Check(cudaEventRecord(start));
      for (int j = 0; j < 200; ++j)
        Check(cudaGraphLaunch(exec, nullptr));
      Check(cudaEventRecord(end));
      Check(cudaEventSynchronize(end));
      float ms;
      Check(cudaEventElapsedTime(&ms, start, end));
      printf("k=%d %s %.3f us\n", k, full ? "full scan" : "short scan", ms * 5);
      Check(cudaEventDestroy(start));
      Check(cudaEventDestroy(end));
      Check(cudaGraphExecDestroy(exec));
      Check(cudaGraphDestroy(graph));
      Check(cudaStreamDestroy(stream));
    }
  }
}
