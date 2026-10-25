// Full BF16/F32 Mellum draft. Included after the common single-row operators.
// This block owns its KV; only the original token embedding and LM head alias
// the target. Host scheduling supplies the previous post-final-norm hidden row.
namespace mtp {
inline constexpr int kHidden = 2304, kWindow = 1024, kVocabulary = 98304;
inline constexpr std::size_t kWorkspaceBytes =
    (4608 + 2304 * 4 + 4096 * 2 + 512 * 3 + 64 + 8 + 8 * 896 * 2 + 8 * 2304 + kVocabulary +
     32 * kWindow) *
    sizeof(float);
inline constexpr std::size_t kDraftResidentBytes =
    internal::kMtpWeightsBytes + 192 * sizeof(DeviceMatrix) +
    2 * kWindow * 512 * sizeof(__nv_bfloat16) + kWorkspaceBytes + 8 * sizeof(int);
static_assert(kDraftResidentBytes == kMtpDraftResidentBytes);
using Capture = std::function<void(const char *, const float *, int)>;
__device__ bool Position(const DecodeState *state, int offset, int *bad, int &position) {
  if (!state)
    return true;
  if (!DecodeReadyAtLimit(state, state->capacity) || offset < 0 || offset >= state->remaining ||
      offset >= state->capacity - state->position) {
    if (!threadIdx.x)
      atomicExch(bad, 1);
    return false;
  }
  position = state->position + offset;
  return true;
}
__global__ void EmbedControlledDraft(DeviceMatrix embedding, const DecodeState *state,
                                     const int *inputs, int offset, float *out, int *bad) {
  int position = 0;
  if (!Position(state, offset, bad, position))
    return;
  const int token = offset ? inputs[offset] : state->token;
  if (token < 0 || token >= kVocabulary) {
    if (!threadIdx.x)
      atomicExch(bad, 1);
    return;
  }
  for (int i = threadIdx.x; i < kHidden; i += 128)
    out[i] = BF(embedding.data, token * kHidden + i);
}
__global__ void SaveProposal(const Control *chosen, const DecodeState *state, int *inputs,
                             int offset, int *bad) {
  if (threadIdx.x)
    return;
  if (*bad || chosen->bad || chosen->token < 0 || chosen->token >= kVocabulary) {
    atomicExch(bad, 1);
    return;
  }
  if (offset == 0)
    inputs[0] = state->token;
  inputs[offset + 1] = chosen->token;
}
__global__ void NormF32(const float *x, float *out, const float *weights, int dim) {
  __shared__ float scratch[8], inverse;
  const int h = blockIdx.x;
  float sum = 0, error = 0;
  for (int i = threadIdx.x; i < dim; i += 128)
    Accumulate(__fmul_rn(x[h * dim + i], x[h * dim + i]), sum, error);
  sum = AccurateReduce(sum, error, scratch);
  if (threadIdx.x == 0)
    inverse = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, float(dim)), 1e-6f)));
  __syncthreads();
  for (int i = threadIdx.x; i < dim; i += 128)
    out[h * dim + i] = B(__fmul_rn(B(__fmul_rn(x[h * dim + i], inverse)), weights[i]));
}
__global__ void RouterProjection(const float *weights, const float *x, float *out, int *bad) {
  const int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x & 31;
  float sum = 0, error = 0;
  for (int c = lane; c < kHidden; c += 32)
    Accumulate(__fmul_rn(weights[row * kHidden + c], x[c]), sum, error);
  sum = WarpAccurateReduce(sum, error);
  if (lane == 0) {
    out[row] = B(sum);
    Finite(out[row], bad);
  }
}
__global__ void ApplyRope(float *x, const float *table, int position,
                          const DecodeState *state = nullptr, int offset = 0, int *bad = nullptr) {
  if (!Position(state, offset, bad, position))
    return;
  const int h = blockIdx.x, i = threadIdx.x;
  __shared__ float old[128];
  old[i] = x[h * 128 + i];
  __syncthreads();
  const float c = table[position * 128 + i % 64], s = table[position * 128 + 64 + i % 64];
  const float rotated = i < 64 ? -old[i + 64] : old[i - 64];
  x[h * 128 + i] = B(__fadd_rn(B(__fmul_rn(old[i], c)), B(__fmul_rn(rotated, s))));
}
__global__ void AttentionProbabilities(const float *q, const __nv_bfloat16 *keys,
                                       float *probabilities, int position,
                                       const DecodeState *state = nullptr, int offset = 0,
                                       int *bad = nullptr) {
  if (!Position(state, offset, bad, position))
    return;
  __shared__ float scores[kWindow], scratch[16], maximum, total;
  const int h = blockIdx.x, kv = h / 8, first = max(0, position - kWindow + 1),
            count = position - first + 1, lane = threadIdx.x & 7, group = threadIdx.x / 8;
  const unsigned mask = 0xffU << (threadIdx.x & 24);
  float query[16];
#pragma unroll
  for (int d = 0; d < 16; ++d)
    query[d] = q[h * 128 + lane + d * 8];
  float mx = -INFINITY;
  for (int j = group; j < count; j += 32) {
    float sum = 0, error = 0;
    const int offset = ((first + j) % kWindow) * 512 + kv * 128 + lane;
#pragma unroll
    for (int d = 0; d < 16; ++d)
      Accumulate(__fmul_rn(query[d], __bfloat162float(keys[offset + d * 8])), sum, error);
#pragma unroll
    for (int delta = 4; delta; delta /= 2) {
      const float other = __shfl_down_sync(mask, sum, delta, 8),
                  low = __shfl_down_sync(mask, error, delta, 8);
      if (lane + delta < 8) {
        Accumulate(other, sum, error);
        Accumulate(low, sum, error);
      }
    }
    if (lane == 0) {
      scores[j] = B(B(__fadd_rn(sum, error)) * 0.08838834764831845f);
      mx = fmaxf(mx, scores[j]);
    }
  }
  mx = AttentionMaxReduce8(mx, scratch);
  if (!threadIdx.x)
    maximum = mx;
  __syncthreads();
  float sum = 0, error = 0;
  if (threadIdx.x < 128)
    for (int j = threadIdx.x; j < count; j += 128) {
      scores[j] = ExpRN(__fsub_rn(scores[j], maximum));
      Accumulate(scores[j], sum, error);
    }
  sum = AttentionAccurateReduce8(sum, error, scratch);
  if (!threadIdx.x)
    total = sum;
  __syncthreads();
  for (int j = threadIdx.x; j < count; j += 256)
    probabilities[h * kWindow + j] = B(__fdiv_rn(scores[j], total));
}
__global__ void AttentionValues(const float *probabilities, const __nv_bfloat16 *values, float *out,
                                int position, int *bad, const DecodeState *state = nullptr,
                                int offset = 0) {
  if (!Position(state, offset, bad, position))
    return;
  const int h = blockIdx.y, feature = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x & 31,
            first = max(0, position - kWindow + 1), count = position - first + 1;
  float sum = 0, error = 0;
  for (int j = lane; j < count; j += 32)
    Accumulate(__fmul_rn(probabilities[h * kWindow + j],
                         __bfloat162float(
                             values[((first + j) % kWindow) * 512 + (h / 8) * 128 + feature])),
               sum, error);
  sum = WarpAccurateReduce(sum, error);
  if (lane == 0) {
    out[h * 128 + feature] = B(sum);
    Finite(out[h * 128 + feature], bad);
  }
}
__global__ void StoreDraft(const float *k, const float *v, __nv_bfloat16 *keys,
                           __nv_bfloat16 *values, const DecodeState *state, int offset, int *bad) {
  int position = 0;
  if (!Position(state, offset, bad, position))
    return;
  const int i = blockIdx.x * 128 + threadIdx.x;
  keys[(position % kWindow) * 512 + i] = __float2bfloat16_rn(k[i]);
  values[(position % kWindow) * 512 + i] = __float2bfloat16_rn(v[i]);
}

#include "mtp_prime.cuh"

struct Draft {
  Buffer weights, descriptors{192 * sizeof(DeviceMatrix)};
  Buffer kv{2 * kWindow * 512 * sizeof(__nv_bfloat16)};
  Buffer workspace{kWorkspaceBytes};
  Buffer ids{8 * sizeof(int)};
  DeviceMatrix embedding, head;
  float *simtScratch = nullptr;
  unsigned char *headInput; // Shared scratch, no duplicate head weights.
  const internal::MtpModel &model;
  const float *rope;
  const int capacity;
  float *concat, *x, *normalized, *q, *k, *v, *kn, *attended, *out, *routing, *routingWeights,
      *gate, *up, *down, *hidden, *logits, *probabilities;
  __nv_bfloat16 *keys, *values;

  Draft(const internal::MtpModel &source, DeviceMatrix sharedEmbedding, DeviceMatrix sharedHead,
        const float *localRope, int ropeCapacity, cudaStream_t stream,
        unsigned char *sharedHeadInput = nullptr, float *sharedSimtScratch = nullptr)
      : weights(source.weights.size()), embedding(sharedEmbedding), head(sharedHead),
        simtScratch(sharedSimtScratch), headInput(sharedHeadInput), model(source), rope(localRope), capacity(ropeCapacity) {
    if (embedding.format != 0 || (head.format != 0 && head.format != 7) ||
        (head.format == 7 && !headInput) || embedding.rows != kVocabulary ||
        head.rows != kVocabulary || embedding.cols != kHidden || head.cols != kHidden ||
        !embedding.data || !head.data || !rope || capacity < 1)
      throw std::runtime_error("MTP shared target/position binding mismatch");
    Check(cudaMemcpyAsync(weights.p, source.weights.data(), weights.bytes, cudaMemcpyHostToDevice,
                          stream));
    Check(cudaMemsetAsync(kv.p, 0, kv.bytes, stream));
    auto *cursor = workspace.As<float>();
    const auto take = [&](int n) {
      auto *p = cursor;
      cursor += n;
      return p;
    };
    concat = take(4608);
    x = take(2304);
    normalized = take(2304);
    q = take(4096);
    k = take(512);
    v = take(512);
    kn = take(512);
    attended = take(4096);
    out = take(2304);
    routing = take(64);
    routingWeights = take(8);
    gate = take(8 * 896);
    up = take(8 * 896);
    down = take(8 * 2304);
    hidden = take(2304);
    logits = take(kVocabulary);
    probabilities = take(32 * kWindow);
    if (std::size_t(cursor - workspace.As<float>()) * sizeof(float) != workspace.bytes)
      throw std::runtime_error("MTP workspace accounting mismatch");
    keys = kv.As<__nv_bfloat16>();
    values = keys + kWindow * 512;
    std::array<DeviceMatrix, 192> experts;
    for (int e = 0; e < 64; ++e)
      for (int p = 0; p < 3; ++p) {
        auto m = Matrix(p == 0 ? "ffn_gate_exps" : p == 1 ? "ffn_up_exps" : "ffn_down_exps");
        if (m.format == 3) {
          m.data += std::size_t(e) * m.rows * m.cols / 2;
          m.suh += std::size_t(e) * m.cols;
          m.svh += std::size_t(e) * m.rows;
        } else
          m.data += std::size_t(e) * m.rows * m.cols * (m.format == 1 ? 1 : 2);
        if (m.format == 1)
          m.scales += std::size_t(e) * m.rows * 2;
        experts[e * 3 + p] = m;
      }
    Check(cudaMemcpyAsync(descriptors.p, experts.data(), descriptors.bytes, cudaMemcpyHostToDevice,
                          stream));
    // Source host descriptor storage must outlive its asynchronous upload.
    Check(cudaStreamSynchronize(stream));
  }
  DeviceMatrix Matrix(const char *name) {
    const auto &t = model.tensors.at(std::string("blk.28.") + name + ".weight");
    if (t.f32 || t.shape.size() < 2)
      throw std::runtime_error("MTP BF16 matrix binding mismatch");
    const auto rank = t.shape.size();
    DeviceMatrix m;
    m.data = weights.As<unsigned char>() + t.offset;
    m.rows = int(t.shape[rank - 2]);
    m.cols = int(t.shape[rank - 1]);
    if (t.exl3) {
      if (rank != 3 || t.shape[0] != 64 || t.f32 || t.fp8)
        throw std::runtime_error("MTP EXL3 expert binding mismatch");
      m.format = 3;
      m.suh = reinterpret_cast<const std::uint16_t *>(weights.As<unsigned char>() + t.suh_offset);
      m.svh = reinterpret_cast<const std::uint16_t *>(weights.As<unsigned char>() + t.svh_offset);
    }
    if (t.fp8) {
      if (rank != 3 || t.shape[0] != 64 || t.f32 || t.scale_offset < t.offset)
        throw std::runtime_error("MTP FP8 expert binding mismatch");
      m.format = 1;
      m.scales = weights.As<unsigned char>() + t.scale_offset;
    }
    return m;
  }
  const float *F32(const char *name) {
    const auto &t = model.tensors.at(std::string("blk.28.") + name + ".weight");
    if (!t.f32)
      throw std::runtime_error("MTP F32 binding mismatch");
    return reinterpret_cast<const float *>(weights.As<unsigned char>() + t.offset);
  }
  std::size_t ResidentBytes() const {
    return weights.bytes + descriptors.bytes + kv.bytes + workspace.bytes + ids.bytes;
  }
  void Project(const char *name, const float *input, float *output, int *bad, cudaStream_t s) {
    const auto m = Matrix(name);
    GemvWarp<<<(m.rows + 3) / 4, 128, 0, s>>>(m, input, output, bad);
  }
  void Normalize(const char *name, const float *input, float *output, int heads, int dim,
                 cudaStream_t s) {
    NormF32<<<heads, 128, 0, s>>>(input, output, F32(name), dim);
  }
  static void Record(const Capture &capture, const char *name, const float *p, int n) {
    if (capture)
      capture(name, p, n);
  }
  // Prime and Forward have identical input-to-KV arithmetic. Priming omits Q,
  // attention and MoE because those results do not affect this row's own K/V.
  void Encode(int token, int position, const float *previousHidden, int *bad, cudaStream_t s,
              const Capture &capture = {}, const DecodeState *state = nullptr,
              const int *inputs = nullptr, int offset = 0) {
    if ((!state && (token < 0 || token >= kVocabulary || position < 0 || position >= capacity)) ||
        !previousHidden || !bad || (state && !inputs))
      throw std::runtime_error("MTP input outside admitted bounds");
    if (state)
      EmbedControlledDraft<<<1, 128, 0, s>>>(embedding, state, inputs, offset, out, bad);
    else
      Embed<<<1, 128, 0, s>>>(embedding, token, out);
    Record(capture, "embedding", out, 2304);
    Normalize("nextn.enorm", out, concat, 1, 2304, s);
    Normalize("nextn.hnorm", previousHidden, concat + 2304, 1, 2304, s);
    Record(capture, "concat", concat, 4608);
    Project("nextn.eh_proj", concat, x, bad, s);
    Record(capture, "eh_proj", x, 2304);
    Normalize("attn_norm", x, normalized, 1, 2304, s);
    Record(capture, "attn_norm", normalized, 2304);
    Project("attn_k", normalized, k, bad, s);
    Project("attn_v", normalized, v, bad, s);
    Record(capture, "k_projected", k, 512);
    Record(capture, "v_projected", v, 512);
    Normalize("attn_k_norm", k, kn, 4, 128, s);
    ApplyRope<<<4, 128, 0, s>>>(kn, rope, position, state, offset, bad);
    Record(capture, "k_rope", kn, 512);
    if (state)
      StoreDraft<<<4, 128, 0, s>>>(kn, v, keys, values, state, offset, bad);
    else
      Store<<<4, 128, 0, s>>>(kn, v, keys, values, position, kWindow);
  }
  // Caller supplies dead prefill arenas. Input target hidden rows and the final
  // target residual remain live and must not alias these scratch pointers.
  void EncodeRows(const int *tokens, int first, int rows, const float *previousHidden,
                  float *concatRows, float *xRows, float *normalRows, float *kRows, float *vRows,
                  float *knRows, int *bad, cudaStream_t s) {
    if (!tokens || !previousHidden || !concatRows || !xRows || !normalRows || !kRows || !vRows ||
        !knRows || !bad || first < 0 || rows < 1 || rows > kWindow || first >= capacity ||
        rows > capacity - first)
      throw std::runtime_error("MTP batched prime outside admitted bounds");
    Embed<<<rows, 128, 0, s>>>(embedding, first, xRows, tokens);
    prime::NormRows<<<dim3(1, rows), 128, 0, s>>>(xRows, concatRows, F32("nextn.enorm"), 2304, 2304,
                                                  4608, rows, bad);
    prime::NormRows<<<dim3(1, rows), 128, 0, s>>>(previousHidden, concatRows + 2304,
                                                  F32("nextn.hnorm"), 2304, 2304, 4608, rows, bad);
    prime::Projection<<<dim3(576, (rows + 3) / 4), 128, 0, s>>>(Matrix("nextn.eh_proj"), concatRows,
                                                                xRows, rows, bad);
    prime::NormRows<<<dim3(1, rows), 128, 0, s>>>(xRows, normalRows, F32("attn_norm"), 2304, 2304,
                                                  2304, rows, bad);
    prime::Projection<<<dim3(128, (rows + 3) / 4), 128, 0, s>>>(Matrix("attn_k"), normalRows, kRows,
                                                                rows, bad);
    prime::Projection<<<dim3(128, (rows + 3) / 4), 128, 0, s>>>(Matrix("attn_v"), normalRows, vRows,
                                                                rows, bad);
    prime::NormRows<<<dim3(4, rows), 128, 0, s>>>(kRows, knRows, F32("attn_k_norm"), 128, 512, 512,
                                                  rows, bad);
    prime::RopeStore<<<dim3(4, rows), 128, 0, s>>>(knRows, vRows, rope, keys, values, first, rows,
                                                   capacity, bad);
  }
  void Forward(int token, int position, const float *previousHidden, int *bad, cudaStream_t s,
               const Capture &capture = {}, const DecodeState *state = nullptr,
               const int *inputs = nullptr, int offset = 0) {
    Encode(token, position, previousHidden, bad, s, capture, state, inputs, offset);
    Project("attn_q", normalized, q, bad, s);
    Normalize("attn_q_norm", q, q, 32, 128, s);
    ApplyRope<<<32, 128, 0, s>>>(q, rope, position, state, offset, bad);
    Record(capture, "q_rope", q, 4096);
    AttentionProbabilities<<<32, 256, 0, s>>>(q, keys, probabilities, position, state, offset, bad);
    AttentionValues<<<dim3(32, 32), 128, 0, s>>>(probabilities, values, attended, position, bad,
                                                 state, offset);
    Record(capture, "attended", attended, 4096);
    Project("attn_output", attended, out, bad, s);
    Add<<<18, 128, 0, s>>>(x, out, 2304, bad);
    Record(capture, "attention_residual", x, 2304);
    Normalize("ffn_norm", x, normalized, 1, 2304, s);
    Record(capture, "ffn_norm", normalized, 2304);
    RouterProjection<<<16, 128, 0, s>>>(F32("ffn_gate_inp"), normalized, routing, bad);
    Record(capture, "router", routing, 64);
    Route<<<1, 1, 0, s>>>(routing, ids.As<int>(), routingWeights, bad);
    Record(capture, "routing_weights", routingWeights, 8);
    const bool exl3 = Matrix("ffn_gate_exps").format == 3;
#if defined(PADRA_MTP_SIMT)
    if(!exl3 || !simtScratch)throw std::runtime_error("SIMT MTP requires EXL3 draft experts and scratch");
    exl3_selected::Launch(descriptors.As<DeviceMatrix>(),0,0,ids.As<int>(),normalized,gate,simtScratch,8,bad,s,false,4);
    exl3_selected::Launch(descriptors.As<DeviceMatrix>(),0,1,ids.As<int>(),normalized,up,simtScratch,8,bad,s,false,4);
#else
    if (exl3)
      exl3_mma_dual::Projection<4><<<dim3(7, 8, 2), 256, 0, s>>>(
          descriptors.As<DeviceMatrix>(), 0, 0, ids.As<int>(), normalized, gate, up, 8, bad);
    else {
      ExpertWarp<<<dim3(224, 8), 128, 0, s>>>(descriptors.As<DeviceMatrix>(), 0, 0, ids.As<int>(),
                                              normalized, gate, bad);
      ExpertWarp<<<dim3(224, 8), 128, 0, s>>>(descriptors.As<DeviceMatrix>(), 0, 1, ids.As<int>(),
                                              normalized, up, bad);
    }
#endif
    Record(capture, "gate", gate, 8 * 896);
    Record(capture, "up", up, 8 * 896);
    Activate<<<56, 128, 0, s>>>(gate, up);
    Record(capture, "activated", gate, 8 * 896);
#if defined(PADRA_MTP_SIMT)
    exl3_selected::Launch(descriptors.As<DeviceMatrix>(),0,2,ids.As<int>(),gate,down,simtScratch,8,bad,s,false,4);
#else
    if (exl3)
      exl3_mma_dual::Projection<4><<<dim3(18, 8), 256, 0, s>>>(
          descriptors.As<DeviceMatrix>(), 0, 2, ids.As<int>(), gate, down, nullptr, 8, bad);
    else
      ExpertWarp<<<dim3(576, 8), 128, 0, s>>>(descriptors.As<DeviceMatrix>(), 0, 2, ids.As<int>(),
                                              gate, down, bad);
#endif
    Record(capture, "down", down, 8 * 2304);
    Combine<<<18, 128, 0, s>>>(x, down, routingWeights, bad);
    Record(capture, "residual", x, 2304);
    Normalize("nextn.shared_head_norm", x, hidden, 1, 2304, s);
    Record(capture, "hidden", hidden, 2304);
    if (head.format == 7)
      head_q8::Launch(1, head, hidden, headInput, logits, bad, s);
    else
      GemvWarp<<<(kVocabulary + 3) / 4, 128, 0, s>>>(head, hidden, logits, bad);
    Record(capture, "logits", logits, kVocabulary);
  }
};

// Encoded rows are copied verbatim, including Q8 scales. The draft retains its
// BF16 ring. The fixed backup stride also preserves the BF16 paired-attention ABI.
struct CacheBinding {
  void *key, *value;
  int capacity;
  void *scales = nullptr;
  int row_bytes = 1024, scale_bytes = 0;
};
static_assert(sizeof(CacheBinding) == 40);
inline constexpr int kMaxRows = 5, kCaches = 29;
inline constexpr std::size_t kVerifyResidentBytes =
    kMaxRows * (2304 * 2 + kVocabulary) * sizeof(float) +
    kMaxRows * (sizeof(DecodeState) + sizeof(int)) + kCaches * sizeof(CacheBinding) +
    kCaches * kMaxRows * 1024 * sizeof(__nv_bfloat16) +
    kMaxRows * (2304 + 8 * 896 * 2 + 8 * 2304) * sizeof(float) +
    kMaxRows * 8 * (sizeof(int) + sizeof(float)) +
    64 * (kMaxRows * sizeof(int) + sizeof(unsigned)) + kMaxRows * (4096 + 512 * 2) * sizeof(float) +
    kMaxRows * (4096 + sizeof(float)) + 65 * sizeof(int);
static_assert(kVerifyResidentBytes == kMtpVerifyResidentBytes);
struct Verification {
  Buffer x{kMaxRows * 2304 * sizeof(float)}, hidden{kMaxRows * 2304 * sizeof(float)},
      logits{kMaxRows * kVocabulary * sizeof(float)}, states{kMaxRows * sizeof(DecodeState)},
      inputs{kMaxRows * sizeof(int)}, bindings{kCaches * sizeof(CacheBinding)},
      backup{kCaches * kMaxRows * 1024 * sizeof(__nv_bfloat16)};
  Buffer normalized{kMaxRows * 2304 * sizeof(float)}, gate{kMaxRows * 8 * 896 * sizeof(float)},
      up{kMaxRows * 8 * 896 * sizeof(float)}, down{kMaxRows * 8 * 2304 * sizeof(float)},
      ids{kMaxRows * 8 * sizeof(int)}, routing{kMaxRows * 8 * sizeof(float)},
      slots{64 * kMaxRows * sizeof(int)}, masks{64 * sizeof(unsigned)};
  Buffer activeExperts{64 * sizeof(int)}, activeCount{sizeof(int)};
  Buffer qkv{kMaxRows * (4096 + 512 * 2) * sizeof(float)}, packed{kMaxRows * 4096},
      scales{kMaxRows * sizeof(float)};
  std::size_t ResidentBytes() const {
    return x.bytes + hidden.bytes + logits.bytes + states.bytes + inputs.bytes + bindings.bytes +
           backup.bytes + normalized.bytes + gate.bytes + up.bytes + down.bytes + ids.bytes +
           routing.bytes + slots.bytes + masks.bytes + activeExperts.bytes + activeCount.bytes +
           qkv.bytes + packed.bytes + scales.bytes;
  }
};
__global__ void BuildStates(const DecodeState *current, const int *inputs, DecodeState *rows,
                            int count, int *bad, int maximum) {
  const int i = threadIdx.x;
  if (i >= count)
    return;
  if (!DecodeReadyAtLimit(current, maximum) || count < 1 || count > kMaxRows ||
      count > current->remaining || inputs[i] < 0 || inputs[i] >= kVocabulary || inputs[i] == 28 ||
      (i == 0 && inputs[i] != current->token)) {
    atomicExch(bad, 1);
    return;
  }
  const int p = current->position + i;
  rows[i] = {inputs[i], p, current->capacity, current->remaining - i, p, min(p, kWindow), 0, 0};
}
// Save before any tentative writes. Each row has independent, contiguous K/V
// storage, including rows straddling the local ring boundary.
template <bool Restore>
__global__ void CopyCacheRows(const CacheBinding *bindings, const DecodeState *current,
                              __nv_bfloat16 *backup, int rows, int retained, int *bad) {
  const int row = blockIdx.x, layer = blockIdx.y;
  if (rows < 1 || rows > kMaxRows || layer >= kCaches || retained < 0 || retained > rows ||
      current->position < 0 || current->capacity < rows ||
      current->position > current->capacity - rows) {
    if (!threadIdx.x)
      atomicExch(bad, 1);
    return;
  }
  if (row >= rows || (Restore && row < retained))
    return;
  const auto cache = bindings[layer];
  const bool shape = (cache.row_bytes == 1024 && cache.scale_bytes == 0) ||
                     (cache.row_bytes == 544 && cache.scale_bytes == 0) ||
                     (cache.row_bytes == 512 && cache.scale_bytes == 32);
  if (!shape || !cache.key || !cache.value || bool(cache.scales) != bool(cache.scale_bytes) ||
      cache.capacity < 1 || cache.capacity > current->capacity ||
      (reinterpret_cast<std::uintptr_t>(cache.key) & 3) ||
      (reinterpret_cast<std::uintptr_t>(cache.value) & 3) ||
      (reinterpret_cast<std::uintptr_t>(cache.scales) & 3)) {
    if (!threadIdx.x)
      atomicExch(bad, 1);
    return;
  }
  const int position = current->position + row, slot = position % cache.capacity;
  auto *saved = reinterpret_cast<unsigned *>(backup) + (layer * kMaxRows + row) * 512;
  const int words = cache.row_bytes / 4, scale_words = cache.scale_bytes / 4;
  auto *key = static_cast<unsigned *>(cache.key) + slot * words;
  auto *value = static_cast<unsigned *>(cache.value) + slot * words;
  // Unwritten global/suffix rows are outside valid history. Only a wrapped
  // ring overwrites committed data; never read uninitialized suffix storage.
  const bool valid = position >= cache.capacity;
  for (int d = threadIdx.x; d < words; d += blockDim.x) {
    if constexpr (Restore) {
      key[d] = saved[d];
      value[d] = saved[words + d];
    } else {
      saved[d] = valid ? key[d] : 0;
      saved[words + d] = valid ? value[d] : 0;
    }
  }
  if (cache.scales) {
    auto *scales = static_cast<unsigned *>(cache.scales) + slot * scale_words;
    for (int d = threadIdx.x; d < scale_words; d += blockDim.x) {
      if constexpr (Restore)
        scales[d] = saved[2 * words + d];
      else
        saved[2 * words + d] = valid ? scales[d] : 0;
    }
  }
}
__global__ void CommitState(DecodeState *current, const Control *chosen, int count, int *bad,
                            int maximum) {
  if (threadIdx.x)
    return;
  if (*bad || chosen->bad || !DecodeReadyAtLimit(current, maximum) || count < 1 ||
      count > current->remaining || chosen->token < 0 || chosen->token >= kVocabulary) {
    atomicExch(bad, 1);
    return;
  }
  current->position += count;
  current->remaining -= count;
  current->global_valid = current->position;
  current->local_valid = min(current->position, kWindow);
  current->token = chosen->token;
  current->eos = chosen->token == 28;
  current->stopped = current->eos || current->remaining == 0;
}
template <int Rows>
__global__ void HeadRows(DeviceMatrix m, const float *input, float *output, int *bad) {
  const int lane = threadIdx.x & 31, r = blockIdx.x * 4 + threadIdx.x / 32;
  float sums[Rows] = {}, errors[Rows] = {};
  for (int c = lane; c < 2304; c += 32) {
    const float weight = BF(m.data, r * 2304 + c);
#pragma unroll
    for (int t = 0; t < Rows; ++t)
      Accumulate(__fmul_rn(weight, input[t * 2304 + c]), sums[t], errors[t]);
  }
#pragma unroll
  for (int t = 0; t < Rows; ++t) {
    const float value = B(WarpAccurateReduce(sums[t], errors[t]));
    if (lane == 0) {
      output[t * kVocabulary + r] = value;
      Finite(value, bad);
    }
  }
}
inline void LaunchHeadRows(int rows, DeviceMatrix head, const float *hidden, float *logits,
                           int *bad, cudaStream_t stream) {
  switch (rows) {
  case 2:
    HeadRows<2><<<24576, 128, 0, stream>>>(head, hidden, logits, bad);
    break;
  case 3:
    HeadRows<3><<<24576, 128, 0, stream>>>(head, hidden, logits, bad);
    break;
  case 4:
    HeadRows<4><<<24576, 128, 0, stream>>>(head, hidden, logits, bad);
    break;
  case 5:
    HeadRows<5><<<24576, 128, 0, stream>>>(head, hidden, logits, bad);
    break;
  default:
    throw std::runtime_error("MTP verifier row count unsupported");
  }
}
} // namespace mtp
