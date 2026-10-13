# Pascal FP32 matrix and Attention profile

Measured engine implementation: `1e11421`, `codex/pascal-sm61-port`, GTX 1070
Max-Q / SM61, CUDA 12.4 and GCC 13. This follows the
[exact implementation](pascal-sm61-performance-results.md) and the
[performance plan](pascal-sm61-performance-plan.md).

The native EXL3 K4 checkpoint and tokenizer are unchanged. SM61 now selects
`mellum16-pascal-fp32-v2`. Other SIMT targets retain their earlier arithmetic;
the SM120 backend retains its Tensor Core dispatch. No INT8 approximation of
EXL3 weights or activations, KV quantization, or new checkpoint was introduced.

## Implementation

- Q/K/V/O Prefill projections decode FP8 weights once into reused FP32 scratch
  and use cuBLAS SGEMM. The physical batch remains 256; inactive input rows are
  zero and only active output rows are scaled and rounded to BF16. Decode retains
  the selected FP32 vector projection, where full-matrix decoding is slower.
- EXL3 Prefill uses M32/N128/K16 tiles with independent FP32 accumulators and
  shared weight/activation reuse. Experts with fewer than 32 active rows retain
  M8/N128. Both dispatches share the original stable grouping and exact codec,
  input H128, output H128, sidecars and ordered combine. Sparse experts avoid the
  padding regression measured with unconditional M32 tiles.
- Prefill Attention gathers one KV head into reused FP32 scratch, batches its
  eight query heads through SGEMM QK and PV, and keeps the BF16 score, probability
  and output rounding boundaries. Old ring KV and current-chunk KV remain
  separate until Commit. At global capacity 16384 it processes 64 queries at a
  time to limit score workspace to 32 MiB. Up to 8192, it uses 256 queries; above
  16384, it uses the bounded FP32 split-kernel implementation instead of SGEMM.
- Decode Attention shares key tiles among four query heads, uses FP32 FMA and
  `expf`, and splits PV over four CTAs per head for contexts above 1024. Shorter
  contexts retain fused Softmax/PV. The Softmax denominator retains compensated
  summation. A broadcast needs a barrier after every warp has read shared
  reduction scratch; the final implementation includes that barrier.

All owned buffers are allocated before graph capture. The cuBLAS handle uses a
fixed 16 MiB workspace. Five production graphs are captured once: short/middle/
long Decode, Head and Prefill. The original 700 MiB reserve and 42 MiB local KV
snapshot remain enforced. The real HTTP instances pass post-capture admission
with 850,264,064 free bytes at 8192 and 753,795,072 at 16384.

The large physical batch and full-capacity Attention GEMMs can increase latency
for short prompts and small cache append requests. The long-profile improvement
below is not a claim that every request size is faster.

## Quality preservation

The offline `reference_math` option reproduces all 83 original real activation
and logit captures bit for bit. Per-request `teacher_sequences` let the offline
qualification executable reuse one loaded model across the corpus; these options
are not available through HTTP.

The frozen [native-token corpus](../tests/data/pascal_math_corpus.json) contains
18 cases from 16 source documents, with contexts 128/1024/2000/4000 and two 8000
cases. It scores 2304 next tokens on identical teacher-forced inputs. Sources
include German/English engineering prose, repository code, TypeScript and SQL;
long cases repeat their source text. Corpus SHA-256:
`f9e36b7a3b32da431f330bf7ecd48e6f126430023db1b677bf8c6bbf73fee452`.

Against the previous quantized Padra arithmetic:

- Mean NLL increase: **0.001973 Nat/token**.
- Paired bootstrap 95% interval: **[-0.003759, 0.008062]**.
- Predeclared upper limit: **0.01 Nat/token**. The preservation gate passes.

The bootstrap resamples whole source documents, grouping repeated-context cases
from the same source; 10,000 samples, seed 20261013. This measures preservation
on a fixed engineering corpus, not equivalence to original unquantized weights
or quality on a broad external benchmark.

The frozen [executable task set](../tests/data/pascal_code_tasks.json) contains
eight Python, eight TypeScript and eight SQLite tasks. Both old and new servers
pass **24/24**, with no task regressions. Programs are checked on fixed edge
cases; SQLite queries execute against a fixed in-memory fixture. This small
paired set does not establish a one-percentage-point population-loss bound;
that broader P7 qualification remains open.

## Correctness and build verification

- All four SM61 CTest groups pass, including full-size expert projections,
  sparse/skewed groups, poisoned future/tail data, graph replay after active-row
  changes, local ring wrap, and Attention near 8192/16384.
- Matrix tests retain a normalized-RMSE limit of 0.0005. Attention accepts either
  normalized RMSE <= 0.002 or absolute RMSE <= 0.0001 and maximum absolute error
  <= 0.0005 for the fixed fixture values in [-1,1]. BF16 probability rounding can
  amplify small `expf` changes in nearly cancelling PV sums; relative error alone
  is not a suitable bound there. Real-model acceptance uses the separate NLL gate.
- Compute Sanitizer memcheck: zero errors. Focused racecheck on the new Pascal
  and dense expert kernels: zero errors or warnings. The focused suite includes
  dense counts 33/65, sparse tails, captured SGEMM, causal staged KV and long/ring
  Attention. An earlier broad run containing repeated timing loops was stopped
  and is not counted as a pass.
- Real HTTP checks pass at capacities 8192 and 16384: Greedy and seeded sampling,
  streaming, disconnect cancellation/recovery, cached/fresh agreement, local ring
  wrap, and restoration of 1100 cached tokens. Model loads/uploads stay at one;
  graph captures stay at five in the new profile.
- SM86 and SM120a cross-build and pass their host GPU-profile tests. The
  CUDA-disabled host build also passes. GPU execution was tested only on SM61.
- The dense EXL3 tile uses 72 registers and 14,656 shared bytes without spills;
  the split PV kernel uses 32 registers and 4096 shared bytes without spills.

## Repeated long-profile comparison

Five warm repetitions per engine and profile, alternating engine order. Context
8192, unchanged saved native input IDs, Greedy, zero cached prompt tokens, no
speculation and one request/model on the GPU at a time. Each run generates
**1000 output tokens**; Decode counts 999 forward steps. Prefill includes first
selection. Loading, tokenization and queueing are outside phase timing. HTTP
totals and first-token headers are retained in the raw records. No profiling,
compilation or other GPU tests ran during these measurements.

Padra uses EXL3 K4 / FP8 attention weights and activations / BF16 KV;
llama.cpp uses the earlier Q4_K_M / Q8_0 KV setup and a different source
revision. This remains a practical setup comparison, not quality equivalence
between quantizations.

| Input / output | Phase | Previous Padra Tok/s | New Padra seconds [min, max] | New Padra Tok/s | llama.cpp seconds [min, max] | llama.cpp Tok/s | Paired speedup 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2000 / 1000 | prefill | 139.14 | 7.073 [7.045, 7.088] | 282.77 | 2.270 [2.258, 2.277] | 881.15 | [0.320, 0.321] |
| 2000 / 1000 | decode | 34.96 | 25.075 [24.996, 25.160] | 39.84 | 17.531 [17.510, 17.558] | 56.98 | [0.697, 0.701] |
| 4000 / 1000 | prefill | 123.01 | 13.972 [13.969, 13.998] | 286.29 | 4.468 [4.456, 4.486] | 895.34 | [0.319, 0.321] |
| 4000 / 1000 | decode | 32.56 | 25.867 [25.824, 25.932] | 38.62 | 17.951 [17.936, 17.968] | 55.65 | [0.692, 0.695] |

Speedup is `llama_seconds / padra_seconds`; values below one favor llama.cpp.
Intervals resample the five per-repetition ratios 10,000 times, seed 20261013.
The earlier Padra values are the five-repeat exact-profile medians, not a new
interleaved A/B measurement of the two Padra profiles.

- 2000 prefill: **2.03x** versus the previous exact profile.
- 2000 decode: **1.14x** versus the previous exact profile.
- 4000 prefill: **2.33x** versus the previous exact profile.
- 4000 decode: **1.19x** versus the previous exact profile.

**The llama.cpp targets remain unmet in all four phases**, against both the
historical plan limits and the contemporary comparison. llama.cpp remains about
3.1x faster in Prefill and 1.4x in Decode. The plan has not been fully accepted.

GPU clocks/power settings were not changed. Active telemetry (>=50% utilization,
>=6000 MiB resident) reports:

- padra: median SM/memory 1544/3802 MHz; temperature 53-67 C; median power 80.5 W; sampled peak VRAM 7297 MiB.
- llama: median SM/memory 1544/3802 MHz; temperature 63-67 C; median power 82.9 W; sampled peak VRAM 7845 MiB.

Raw telemetry retains throttle flags. These are sampled peaks, not an exact
allocator high-water mark; the final metadata contains binary/request hashes,
actual launch commands and order for all five pairs.

## Remaining operator budget

A separate final eager CUDA-event run uses the saved 4000-token input and two
outputs. It is diagnostic and is not included in benchmark throughput. The one
Decode forward at position 4000 attributes about 11.63 ms to EXL3 Gate/Up/Down,
5.67 ms to QKV/O, 3.31 ms to Attention and 2.63 ms to Head. Other layer groups
contribute about 2.70 ms. The two Head calls are counted separately from the
single Decode forward.

Attention is substantially cheaper than the earlier roughly 8.25 ms at 4k;
the unchanged Decode matrix kernels and BF16 Head now account for most of the
remaining budget. The last 160-row Prefill chunk attributes about 455 ms to
expert projections, 152 ms to Attention and 92 ms to QKV/O. These tail-chunk
figures cannot be treated as an entire 4000-token Prefill budget.

Pascal DP4A still accelerates the exact EXL3 hash only. Integer matrix products
would require a separately defined activation/decoded-weight approximation and
new quality qualification; this implementation does not silently reinterpret
EXL3 K4 payloads as linear Q4 weights. Original-model quality, larger task sets,
full CPU/GPU timeline accounting and llama.cpp performance parity remain open.

## Reproduction and raw artifacts

All raw results and exact commands are under
`/srv/agent/workspaces/padra-pascal-fast-20261013/`:
`quality-final/`, `functional/`, `final-paired/`, `final-telemetry.csv`, final
build/test/sanitizer logs and the earlier rejected candidate measurements.
Only final qualified serving binaries count in the repeated comparison.

```sh
python3 tools/qualify_pascal_math.py \
  --corpus tests/data/pascal_math_corpus.json \
  --reference-binary /path/to/reference-profile-qualify \
  --candidate-binary /path/to/padra-simt-qualify \
  --artifact /path/to/native \
  --tokenizer-sha256 58548a346eb073e5132bf7d8ad17dc6971bca36ade378ca4d2bfbc49bf60da2a \
  --output /path/to/quality
```

Both executables must support `teacher_sequences` and `reference_math`. The
reference is independently checked against the original captures; NumPy is
required for scoring. The HTTP task runner is `tools/evaluate_code_quality.py`
and requires Node with TypeScript stripping. Frozen corpus/task files avoid
re-tokenizing or regenerating the benchmark inputs.
