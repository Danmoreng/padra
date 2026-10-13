# Pascal compressed weight layout and codec performance

This extends the [FP32 implementation results](pascal-sm61-fp32-performance-results.md)
from baseline `13f75b2` on the GTX 1070 Max-Q, SM61, CUDA 12.4. The default
implementation improves Prefill by approximately 13% and Decode by 7%, with
bit-identical logits on the frozen quality corpus. An optional DP4A Prefill
profile provides a larger speedup with additional arithmetic quantization.
llama.cpp remains faster; the optional profile is not the default.

The native checkpoint, quantized weight payload, sidecars, KV format and memory
admission policy are unchanged. The default implementation also preserves all
activation formats. The optional DP4A profile
changes arithmetic, not the compressed checkpoint or persistent weight layout. The new kernel profile is
`pascal-fp32-v3-lane-window`; the arithmetic profile remains
`mellum16-pascal-fp32-v2`.

## Changes in the default implementation

Each EXL3 lane reconstructs eight weights from its compressed word and the
preceding lane's word. The old implementation repeatedly fetched arbitrary
words for each 16-bit codec state. The new implementation exchanges the
preceding word once, then extracts overlapping pairs locally. It preserves
the codec, Hadamard transforms, BF16 boundaries and accumulation order.
Selected Decode projections, sparse M8 Prefill and dense M32 Prefill use this
reuse. No expanded weights persist in GPU memory.

The dense M32/N128/K16 kernel now assigns four rows and four columns to each
thread, instead of two rows and eight columns. Every weight is reused for four
accumulators. Every output still visits K in the same order. The production
kernel uses 72 registers, 14,656 shared bytes and no local memory or spills.
Sparse experts retain M8. The existing N64/K32 experiment retains its original
two-row mapping.

## Repeated server measurements

Three warm runs per implementation and input length, alternating old/new server
order and prompt order. Each request has identical frozen input tokens, exactly
1,000 output tokens, 999 Decode forwards, context 8192 and zero cached prompt
tokens. No speculative decoding. Only one GPU workload runs at a time.
All paired generated responses are identical.

| Input | Phase | Baseline median Tok/s | New median Tok/s | New min to max |
| --- | --- | ---: | ---: | ---: |
| 2000 | Prefill | 282.86 | 319.53 | 319.12 to 320.21 |
| 2000 | Decode | 39.92 | 42.59 | 42.53 to 42.68 |
| 4000 | Prefill | 286.13 | 322.41 | 322.08 to 322.61 |
| 4000 | Decode | 38.66 | 41.28 | 41.16 to 41.30 |

The earlier five-run llama.cpp reference is 881.15/56.98 Tok/s at 2000 and
895.34/55.65 at 4000, for Prefill/Decode respectively. Those are the previous
measurements, not a new llama.cpp run in this experiment. Its Q4_K_M weights and
Q8 KV differ from Padra's EXL3 K4 and BF16 KV. The new Padra implementation is
still about 2.8 times slower in Prefill and 1.34 times slower in Decode.

## Lossless rearrangement of quantized weights

[The layout benchmark](../tools/benchmark_pascal_layout.cu) tests permutations
of the existing compressed words. It never dequantizes at load and never
changes the represented weights or the persistent byte count.

| Variant | Arrangement and loads | Median speedup over the new native-layout kernel |
| --- | --- | --- |
| 0 | Original payload, preload four separate K16 words per lane | 1.001 to 1.024 |
| 1 | `[N16][K64][lane][four K16 words]`, one 128-bit load per lane | 1.025 to 1.032 |
| 2 | `[N128][K64][N16 within N128][lane][four K16 words]` | 0.975 to 0.993 |

These are synthetic full-shape gate/up/down Decode operators with eight
distinct experts, five alternating timing pairs and bitwise output comparison.
Clock changes affect absolute times, so compare paired ratios rather than
absolute times across variants. The layouts are not enabled in the server.
The best variant's roughly 3% operator gain does not yet justify changing every
consumer of the payload: Decode, sparse and dense Prefill, reference kernels and
descriptor validation must all agree on the layout.

The original accesses already fetch adjacent compressed words across a warp.
Repacking primarily changes the number and scheduling of load instructions;
it does not remove the codec reconstruction or FP32 dot products. This explains
why a lossless permutation alone is not a demonstrated route to llama.cpp parity.

## DP4A with EXL3 storage

DP4A computes four 8-bit integer products and adds them into a 32-bit integer
accumulator. GP104 supports this instruction. See
[NVIDIA's explanation](https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/).
The pinned llama.cpp implementation uses it for Q4_K times Q8_1 dot products:
[vecdotq.cuh](https://github.com/ggml-org/llama.cpp/blob/77757a34672a888845a09ab45728b8d34aa475ff/ggml/src/ggml-cuda/vecdotq.cuh#L508).
Padra's exact EXL3 decoder uses DP4A to sum hash bytes, but its matrix
multiplications still use FP32 FMA.

The EXL3 bit stream is not an affine array of INT4 numbers. Enumerating all
65,536 codec states produces 913 distinct weight values. A common INT8 scale
cannot represent all of them exactly. An exact common fixed-point representation
needs a quantum of 2^-17 and reaches integer magnitude 452608; multiple byte
components and compatible activation arithmetic would be needed for an exact
integer decomposition. Such a decomposition has no measured speed advantage.

[The DP4A experiment](../tools/benchmark_pascal_dp4a.cu) retains the original
compressed GPU payload. Inside the dense Prefill kernel, a lookup maps decoded
weights to signed INT8 with scale `3.453125 / 127`. A preceding kernel quantizes
transformed activations per H128 block into reused scratch. M32/N128/K64 tiles
use DP4A, accumulate/scaled-combine into FP32, and retain the output Hadamard and
BF16 boundary. This is additional arithmetic quantization, not a lossless layout
change. The synthetic test's 5% operator NRMSE guard is not a model-quality gate.

The final optional `mellum16-pascal-dp4a-prefill-v2` profile additionally:

- Uses DP4A for groups/tails of at least 16 rows. Smaller tails use exact M8;
  padding a one-row remainder to another full M32 tile is avoided.
- Fuses the output H128 into the dense kernel. The first version used 93
  registers and regressed; limiting occupancy to permit three 256-thread CTAs
  yields 80 registers, 14,784 shared bytes and no spills. Full dense tiles match
  the separate-Hadamard DP4A implementation bit for bit in the synthetic sweep.
- At context capacity 8192, captures Prefill attention graphs bounded to
  1024, 4096 and 8192 positions. Runtime selects the smallest graph covering the
  complete current chunk. Scratch is shared and allocation-free during requests.
  Other capacities retain the existing graph to respect the memory reserve.

The first DP4A profile measured 477.93/485.76 Prefill Tok/s at 2000/4000 input,
with Decode 42.58/41.22. The final optional profile has three warm runs per
length, the same frozen requests, 1000 output tokens, 999 Decode forwards, 8192
context and zero prompt-cache hits. These runs are compared with the separately
recorded FP32 and earlier llama.cpp runs, not a new interleaved llama.cpp run.

| Input | Phase | Final DP4A median Tok/s | Min to max | Earlier llama.cpp median |
| --- | --- | ---: | ---: | ---: |
| 2000 | Prefill | **582.50** | 578.38 to **586.01** | 881.15 |
| 2000 | Decode | **42.60** | 42.54 to **42.73** | 56.98 |
| 4000 | Prefill | **580.93** | 578.89 to **582.01** | 895.34 |
| 4000 | Decode | **41.26** | 41.17 to **41.34** | 55.65 |

Relative to this turn's original FP32 baseline, Prefill improves 2.03–2.06x and
Decode about 1.07x. Relative to the first DP4A profile, the final Prefill improves
approximately 20–22%. Decode arithmetic remains unchanged in the optional profile;
its improvement over the original baseline comes from the exact codec work.
The earlier 62.84 Tok/s short 128-token Decode result is not directly comparable
to these long-context 1000-output-token runs.

Server metrics confirm one model load, one weight upload and seven startup graph
captures, with no additional captures during these requests. Post-graph available
memory is 825,098,240 bytes, above the unchanged 700 MiB reserve. The native
checkpoint was not modified.

A separate EXL3 Decode DP4A prototype improved synthetic gate/up operators only
about 5% and down about 9%, with approximately 1.2–1.3% operator NRMSE. It is not
integrated: this is not evidence of a twofold Decode speedup. The compressed
codec reconstruction remains even when integer dot products become cheaper.

## Final optional-profile quality

The final binary replays 2,304 teacher-forced continuation tokens across 18 cases
from 16 source documents. The checkpoint and all input/target IDs are unchanged.
Saved original-reference scores are reused with verified binary/corpus hashes;
`dp4a-v2-quality/reference-provenance.json` records their origin.

Against the original reference arithmetic, mean NLL increases by **0.003851
Nat/token**, with a document-bootstrap 95% interval of **[-0.000801, 0.009627]**.
The upper limit is below the existing 0.01 gate. Against the current fast FP32
profile, the additional NLL increase is **0.001878**, interval
**[-0.005494, 0.008463]**.

Full-vocabulary distribution comparison against that current fast FP32 profile:

| Metric | Value |
| --- | ---: |
| Mean KL(FP32 || DP4A), nats | **0.007982** |
| Mean KL 95% interval | 0.003869 to 0.013173 |
| KL median / p95 / p99 | 0.0000397 / 0.04003 / 0.10894 |
| Maximum individual-token KL | 1.38622 |
| Mean reverse KL | 0.008281 |
| Mean Jensen–Shannon divergence | 0.001925 |
| Top-1 agreement | 98.13% |

This measures distribution preservation on the frozen corpus, not unrestricted
coding performance or equivalence to unquantized model weights. The 24 executable
code-generation tasks were **not rerun for this optional profile**. The profile
remains explicitly selectable and disabled by default.

## Why the remaining llama.cpp gap is not just a kernel comparison

The actual reference GGUF uses Q6_K for the output matrix, Q4_K for 98 attention
projection tensors and Q6_K for 14. The output matrix is 185,794,560 bytes versus
452,984,832 bytes for native BF16; attention projection payloads total
338,688,000 bytes versus 594,542,592 bytes for native FP8. Together this is roughly
523 MB more native weight payload per Decode step, before sidecars/cache effects.
These are tensor inventory counts, not measured DRAM traffic.

For the selected expert payloads the direction reverses: the GGUF contains
Q4_K gate/up and a mixture of Q8_0/Q5_0 down tensors. Reading eight of 64 experts
in all 28 layers corresponds to 924,844,032 bytes, versus 693,633,024 native EXL3
payload bytes. EXL3 is smaller but has a more expensive reconstruction operation.
The reference also uses Q8 KV while this implementation retains BF16 KV.

The measured remaining targets are therefore:

1. Reduce Prefill Q/K/V/O conversion and SGEMM cost; these still take about
   92 ms in the profiled 160-row last chunk, even with DP4A experts.
2. Improve EXL3 Decode reconstruction/occupancy. Its three projections still
   total approximately 9.9 ms per token in eager operator timing.
3. Evaluate a separately identified lower-precision output/attention projection
   profile, with full-vocabulary KL and task evaluation. Changing these formats
   is distinct from losslessly rearranging the EXL3 payload.
4. Improve attention beyond removing unused context work. A Flash Attention-style
   implementation and Q8 KV need their own numerical validation; neither is
   implemented or claimed by this change.

No demonstrated result yet establishes llama.cpp parity.

## Verification and reproduction

All four SM61 CTest groups pass. The added dense-tile regression requires
bit-identical R4 and R2 outputs for real expert shapes and sparse/skewed groups.
The complete 18-case frozen corpus produces 2,304 logit vectors bit-identical
to the previous fast implementation. Consequently its previously measured NLL
values on that corpus are unchanged. This is a corpus result, not a population
quality guarantee. SM86 and SM120a cross-build and pass their host profile tests;
GPU execution is verified on SM61. The optional build passes five CTest groups,
including poisoned-output, zero-activation, sparse/dense boundary and remainder
cases for DP4A, and poisoned future-KV checks for bounded attention.

Evidence directory:
`/srv/agent/workspaces/padra-pascal-native-20261013`.
It contains archived baseline/candidate servers, binary hashes, frozen request
hashes, `run_paired.py`, `paired-summary.json`, individual HTTP results,
`quality-comparison.json`, full captured logits, operator sweeps and compiler
resource usage. Baseline binaries were archived before editing.

Build the standalone compressed-layout and DP4A experiments from the repository:

```sh
nvcc -O3 -std=c++20 -arch=sm_61 -ccbin=g++-13 \
  -Iinclude -Isrc -I/tmp/padra-sm61-build/generated \
  tools/benchmark_pascal_layout.cu -o /tmp/padra-layout-bench
/tmp/padra-layout-bench
nvcc -O3 -std=c++20 -arch=sm_61 -ccbin=g++-13 \
  -Iinclude -Isrc -I/tmp/padra-sm61-build/generated \
  tools/benchmark_pascal_dp4a.cu -lcublas -o /tmp/padra-dp4a-bench
/tmp/padra-dp4a-bench
```

Run GPU measurements sequentially. The default build and CTest commands remain
those in the [validation instructions](pascal-sm61-validation.md).

### Optional build and distribution comparison

Configure the ordinary SM61 build with `-DPADRA_PASCAL_DP4A_PREFILL=ON` to select
this arithmetic profile. The option defaults to OFF and configuration rejects
non-SM61/non-SIMT targets. `simt_reference_math` retains reference arithmetic.
The CLI, native checkpoint and tokenizer are unchanged. Use a separate build
directory so the exact FP32 binary remains available for comparison.

```sh
cmake -S . -B /tmp/padra-sm61-dp4a-qualified-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPADRA_CUDA_ARCHITECTURE=61 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-13 -DPADRA_PASCAL_DP4A_PREFILL=ON \
  -DPADRA_BUILD_TESTS=ON -DPADRA_BUILD_DIAGNOSTICS=ON \
  -DPADRA_CARGO_EXECUTABLE=/tmp/padra-rust/bin/cargo \
  -DPADRA_CARGO_HOME=/tmp/padra-cargo-cache
cmake --build /tmp/padra-sm61-dp4a-qualified-build -j 6
ctest --test-dir /tmp/padra-sm61-dp4a-qualified-build --output-on-failure
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tools/compare_pascal_logits.py \
  --corpus tests/data/pascal_math_corpus.json \
  --reference /srv/agent/workspaces/padra-pascal-fast-20261013/quality-final/candidate \
  --candidate /srv/agent/workspaces/padra-pascal-native-20261013/dp4a-v2-quality/candidate \
  --output /srv/agent/workspaces/padra-pascal-native-20261013/dp4a-v2-kl.json
```

KL uses all 98,304 vocabulary entries at temperature 1,
identical teacher-forced contexts and natural logarithms. Confidence intervals
resample whole source documents, not individual correlated tokens.

### Sanitizer scope

The exact Prefill suite and the final DP4A boundary subset pass memcheck with
zero errors. The changed exact EXL3 M32 kernels pass
focused racecheck. The final DP4A kernels also pass namespace-filtered racecheck
across the boundary subset with zero hazards. A broad racecheck reports shared-memory hazards inside
cuBLAS `sgemm_32x32x32_NT_vec`; the same report is reproduced with the archived
baseline binary. This is not a clean broad racecheck and is not established to
be a false positive. The focused custom-kernel result must not be generalized
to the entire CUDA library path. Logs are retained in the evidence directory.
