# SGLang decode findings — 2026-10-13

There are useful remaining experiments. The new evidence points primarily to
attention, activation conversion and normalization, rather than replacing the
EXL3 expert kernels. No new Padra production kernel is promoted by this report.

A controlled vLLM backend change is particularly informative: the same native
Mellum implementation and INT4 checkpoint reaches **284.65 instead of 250.62
decode token/s** when selecting FlashInfer instead of FlashAttention. The fresh
SGLang result is **288.72 token/s**. Thus much of the previously observed
SGLang-versus-vLLM difference is explained by the attention backend selection.
This does not predict the same percentage gain in Padra: its attention is already
substantially faster than the measured vLLM FlashAttention path.

## Versions and comparison conditions

Live PyPI metadata and GitHub's latest-release API both identify
[SGLang 0.5.21](https://github.com/sgl-project/sglang/releases/tag/v0.5.21) as the
latest stable release at the time of this investigation. The existing installation
already has that version, revision `e00930c5489053f26d86b179cee0d087f846acbb`.
This is a stable-release comparison, not a test of unreleased main/nightlies.

- SGLang: native Mellum, Torch 2.13.0, Transformers 5.12.1, FlashInfer 0.6.18.
- vLLM: native Mellum, 0.31.0, Torch 2.13.0, Transformers 5.17.0,
  FlashInfer 0.7.0.post1. The FlashInfer variant resolves to XQA for decode on SM120;
  it is not precisely the same FlashInfer kernel/version as SGLang.
- Padra: selected `856a04f` composition, ASYNC64=ON, P2 plus compensated D2 norm.
- RTX 5080 Laptop, one request, capacity 8192, prefill chunk/budget 2048,
  BF16 KV, greedy decoding, no prefix hits, offload or speculative decoding.
- Same saved P2000/G500 and P4000/G1000 input IDs. SGLang and vLLM use the same
  community INT4/G32 W4A16 checkpoint. Padra retains EXL3 K4 experts and FP8
  attention weights. Comparisons against Padra therefore mix engine and quantization.

GPU jobs were serial and bounded to 20 GiB host memory, zero swap and explicit
120–420 second timeouts. No clock/power settings changed. Existing installations
and checkpoints were not edited. Diagnostic wrappers live outside the repository.

## Fresh sustained measurements

Median token/s, one complete same-shape warmup and three retained runs per case:

| Engine / attention | P2 prefill | P2 decode | P4 prefill | P4 decode |
| --- | ---: | ---: | ---: | ---: |
| Padra EXL3 + FP8, current selected composition | 11,024.70 | 256.81 | 10,331.15 | 252.84 |
| SGLang INT4, FlashInfer | 13,578.50 | 293.38 | 12,703.00 | 288.72 |
| vLLM native INT4, FlashAttention | 14,043.12 | 254.70 | 13,672.32 | 250.62 |
| vLLM native INT4, FlashInfer/XQA | 14,301.90 | 289.41 | 13,973.91 | 284.65 |

The Padra throughput row is the fresh control from the immediately preceding
same-day EXL3 comparison; the three other rows were run in this investigation.
Padra's operator profiles below were also captured afresh from the selected core.
These are serial engine runs, not randomized paired acceptance blocks. They
support investigation of the large differences, not precise claims about small
percentage gains. Phase timers exclude startup/compilation/tokenization and are
engine/runtime timers, not a common HTTP endpoint.

The vLLM change raises P4 decode by **13.58%** and brings it close to SGLang.
Checkpoint, quantization, request sizes and sampling remain fixed. A new
FlashAttention control after the FlashInfer run reproduces the earlier ~251
result. Neither variant slows prefill in these measurements; this is not a
statistical non-regression qualification.

### SGLang timing correction

SGLang's API `first_token_latency` includes delayed first output delivery. Using
`request_finished_ts - (request_received_ts + first_token_latency)` incorrectly
removes roughly 0.17 seconds of actual decode on these P4 requests and reports
about 304 token/s. The retained correction is:

```
prefill_seconds = prefill_finished_time - forward_entry_time
decode_seconds = request_finished_ts - prefill_finished_time
decode_tps = (completion_tokens - 1) / decode_seconds
```

This uses the GPU-ready scheduler prefill boundary and includes final IPC
transport. It reproduces the earlier corrected ~290 figure. Initial raw rows,
the corrected rows and the original harness are retained separately. All output
counts and zero cache-hit/retraction conditions were checked. SGLang P4 retained
runs produce identical IDs; its P2 retained runs differ, as previously observed.
No claim of deterministic SGLang output or full numerical qualification follows.

## Actual decode profiles

The old SGLang Nsight startup failure is bypassed by starting PyTorch/CUPTI
profiling inside the already initialized worker. Both engines reach actual
positions 4000–4007 and 4990–4997 through genuine P4000/G1000 generation. Padra
uses a small CUPTI host probe linked to the unchanged selected core archive.
Only compact aggregate metrics are persisted, not full tensor/trace dumps.

Representative microseconds per token over positions **4990–4997**:

| Work across all 28 layers | Padra | SGLang INT4 | vLLM INT4 / FlashAttention |
| --- | ---: | ---: | ---: |
| Expert projection kernels | 1,198 | 1,138 | 1,139 |
| QKV and O projections | 815 | 614 | 603 |
| Attention core plus merge / Padra FP8 output encoding | 577 | 369 | 1,015 |
| Two main norms per layer / corresponding fused norms | 368 | 95 | 83 |
| BF16 head plus router GEMVs | 665 | 658 | 652 |
| Router selection and explicit expert alignment | 109 | 93 | 269 |

This is a selected-family table, not an exhaustive sum. Expert transforms,
activation functions, QK norm/RoPE, KV writes and runtime work are separate.
The norm rows do different arithmetic and Padra input norms also encode FP8;
the projection rows operate on different quantizations. Do not interpret those
rows as drop-in speedups. Different profiler front ends, clocks and profile
perturbation also prevent adding the differences as a promised end-to-end gain.
The earlier 4000 window gives the same qualitative ordering. Padra's profiled
outputs match the uninstrumented current control; CUPTI reports zero dropped or
invalid records. Completed traces have eight heads and 224 layer invocations.

### What SGLang does differently

1. **Attention selection and bounded local KV input.** SGLang uses FlashInfer's
   Tensor Core paged attention and merge. Its decode planner supplies a suffix
   limited to the actual sliding window, plus a separate full-attention wrapper.
   The measured vLLM FlashAttention path spends about 0.82 ms/token on the 21
   local layers alone. FlashInfer/XQA in vLLM recovers most of its sustained gap.
   Padra already has local-window specialization and grouped-query Tensor Core
   attention; those basic ideas are already implemented.
2. **Single-token MoE preparation.** SGLang's `M == 1` path uses a single-warp
   alignment/sort for the distinct selected experts. Its alignment costs about
   31 us/token across 28 layers, versus about 189 us for vLLM's more general
   small-batch alignment. Padra already consumes its selected expert IDs directly;
   it does not have the vLLM alignment cost to remove.
3. **Cheaper fused normalization.** FlashInfer uses vectorized residual/RMSNorm,
   an ordinary FP32 reduction and approximate reciprocal square root. Its residual
   sum remains FP32 for the norm calculation, with BF16 output storage. Padra
   preserves additional BF16 rounding boundaries, compensated reductions and
   exact divide/sqrt, plus FP8 activation coding at input norm. Copying the
   SGLang arithmetic changes the numerical experiment, not just scheduling.
4. **W4A16 projections.** SGLang keeps BF16 activations at QKV/O projections.
   Padra's weights are FP8 and decode activations are also quantized to FP8.
   INT4 projection speed cannot simply be imported while preserving our weights,
   but avoiding unnecessary activation conversion is a testable design idea.
5. **No special cheap output head.** Both engines still pay roughly 0.57 ms/token
   for the large BF16 head in this trace. The combined head/router GEMV times are
   very close. SGLang does not demonstrate an overlooked head compression trick.

Relevant installed upstream sources are pinned in `SOURCE_SEAL.json`. Public
entry points: [Mellum](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/models/mellum.py),
[FlashInfer backend](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/layers/attention/flashinfer_backend.py),
[Marlin MoE](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py).

## Recommended next experiments for Padra

Keep the accepted parent and existing quality gates. These are hypotheses, not
accepted changes or a new promise to reach 290 token/s. A small repeatable gain
still qualifies if quality, prefill, complexity and memory remain acceptable.

1. **BF16-valued attention output into O projection, decode only.** Keep EXL3,
   stored FP8 weights/row scales and BF16 KV unchanged. Prototype O's existing
   SIMT dot using the already available BF16-rounded attention values instead of
   encoding and decoding another FP8 activation vector. This permits a per-head
   merge without the global 4096-value amax/encoding stage. Compare *merge plus
   projection together*, not just merge. Current merge/encode costs ~208 us/token;
   some merge work remains and projection input traffic increases, so this is a
   cost budget, not an achievable saving. Test the exact represented-weight FP64
   reference and declared activation-precision change on captured layers 0/14/27,
   both positions, then development NLL/logits before full acceptance testing.
   This is different from the rejected D3 local 16-to-32 split experiment.
2. **FlashInfer-style attention as an operator reference.** Benchmark the saved
   BF16 Q/K/V cases against native attention including layout conversion and
   output merge. Keep exact sliding-window membership and RoPE. Separate local
   and global layers. Only pursue a production kernel port or new split policy
   if the inclusive result leaves a material gain after the required O-input
   conversion. Avoid changing the entire KV allocator merely to obtain an
   attention kernel benchmark. Padra already has the important GQA/TC idea.
3. **Narrow normalization experiments.** First isolate reciprocal-square-root
   evaluation while retaining compensated statistics and current BF16 boundaries;
   then consider a vectorized ordinary-reduction/fewer-rounding variant as a
   separate numerical candidate. Existing D2 register-row inputnorm was slower,
   and ordinary FP32 postnorm did not establish development NLL non-inferiority
   (upper confidence bound +0.001262 versus +0.001). Those results remain valid.
   Do not silently retry the same candidate or waive the failed gate because
   another engine uses it. FlashInfer's 95 us is an arithmetic/layout reference,
   not a guarantee for Padra's stricter contract.
4. **Attention projection tiling after the inclusive conversion experiment.**
   Compare a better FP8 weight decode/reuse schedule or a Tensor Core path with
   the selected QKV/O SIMT kernels. Preserve weight bytes/scales and measure the
   full projection including activation preparation. Use a new concrete design;
   do not repeat the previously rejected exact FP8-MMA variant without resolving
   its failure. The observed ~200 us projection difference includes INT4's lower
   weight bandwidth and cannot all be credited to scheduling.

Expert kernels and head compression are lower priority from this comparison:
EXL3 projection time is already close to Marlin's; previous D1 split-K, P1
reconstructed-GEMM and H1 head experiments were rejected with evidence. The new
measurements do not invalidate those decisions.

## Prefill

The native INT4 advantage remains about 13.7–14.0K versus Padra's 10.3K at P4.
It does not establish an EXL3 ceiling or show that a Marlin kernel can consume
EXL3 codebooks. Padra already groups experts and uses the K64 Tensor Core path;
S0 measured high Tensor Core activity, and P1's predecoded GEMM upper bound was
slower even before reconstruction. With decode now the priority, a broad prefill
rewrite is less justified than the targeted experiments above. The EXL3 vLLM
plugin's very slow prefill is a separate issue documented in
[VLLM_EXL3_COMPARISON.md](VLLM_EXL3_COMPARISON.md).

## Quality and reproducibility

Both vLLM variants passed the existing four short boundary smoke cases and
48 full-vocabulary teacher sanity comparisons. FlashInfer matches source top-1
on 47/48, mean source KL 0.014096, extra NLL +0.027673; the fresh FlashAttention
control has KL 0.013683 and extra NLL +0.022096. These are small sanity
screens, not numerical non-inferiority or task-quality acceptance. The current
SGLang installation is unchanged; no new SGLang full quality qualification was
performed. Earlier source/NLL/task failures remain open.

Evidence workspace:
`/home/sebastian/Development/padra-engine-comparison/2026-10-13-sglang-decode`

- `LATEST_VERSION.json`, `SOURCE_SEAL.json`: live release check and source hashes.
- `baseline/corrected-runs.jsonl`: corrected SGLang sustained measurements.
- `vllm-control/`, `vllm-flashinfer/`: fixed-weight backend comparison, configs,
  actual dispatch logs, quality metrics and all retained runs.
- `sglang-clean-decode{4000,4990}.json`, `vllm-decode{4000,4990}.json`,
  `padra-decode{4000,4990}.json`: actual-position compact profiles.
- `RESULT.json`, `analyze.py`: reproducible medians and family aggregation.
- `bench.py`, `profile_hook.py`, `profile_common.py`, `vllm_profile.py`,
  `vllm_control.py`, `vllm_flashinfer.py`, `padra-profile.cpp`, `profile.h`:
  diagnostic harnesses; `jobs/*` contain commands, logs and memory receipts.

The first SGLang aggregate accidentally included GPU annotation spans as kernels;
only `sglang-clean-*` excludes `is_user_annotation` and is used above. The first
vLLM window ended too early and included only seven head launches; preserved
`.attempt1` files are superseded by the complete eight-head captures. A first
Padra probe run omitted its CUPTI activation environment flag and produced no
profile; `padra-profile-active` is the valid capture. None of these diagnostic
attempts is counted as a sustained throughput sample.

The subsequent [attention experiments](PERFORMANCE_ATTENTION.md) implement and
measure these first two ideas. The exact merge variant is retained with about
1.2–1.4% more decode throughput. Direct BF16 O remains unqualified; the same-input
FlashInfer comparison confirms remaining global-attention headroom.
