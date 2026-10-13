# Padra performance plan for Sol

Date: 2026-10-13. Starting source: `d57873b267781fceb3f7b89b220407c0f64b6f39`.

## 1. Assignment and changed numerical policy

Implement and evaluate this plan sequentially when the owner asks you to execute
it. Finish one experiment with an explicit decision before starting the next.
This document is an implementation handoff, not evidence that any new candidate
has already passed.

The owner explicitly no longer requires bit-identical inference. Different FP32
reduction orders, Tensor Core accumulation schedules, GEMM tilings, attention
splits and normalizations may be explored. Generated tokens and intermediate
values may differ. **Preserve the EXL3 K4 checkpoint and demonstrate quality
non-inferiority; do not substitute bit equality for quality evaluation.**

This supersedes historical instructions that reject a candidate solely because
its logits, routes or token IDs differ from the current implementation. It does
not erase historical quality failures or authorize a new checkpoint, weight
requantization, lower-precision KV, fewer experts, approximate attention,
context truncation, CPU offload, or speculative decoding in this campaign.

Keep original FP16 EXL3 codebook values and sidecars exact when reconstructing
weights. Keep BF16 KV, top-8 routing over 64 experts, and original model semantics.
Start numerical experiments with FP32 accumulation. Do not enable global
`--use_fast_math`, FP16 accumulation, or change every rounding boundary at once.
Localized changes to reduction order, reciprocal/square-root evaluation and
intermediate rounding must be declared and independently evaluated.

Proceed through routine reads, builds, isolated experiments, rejected candidates,
and qualified implementation without asking for confirmation at each stage.
Only ask for genuinely missing information or a change outside this scope.
Do not create another chat, delegate, publish, push, or change product defaults
merely because this plan exists. Keep execution sequential and resumable.

Primary workload: one request, P4000/G1000. Secondary: P2000/G500. Historical
milestones are 12,000 prefill tokens/s and 280 decode tokens/s, measured together
on the same selected build. They are targets, not promised outcomes or permission
to sacrifice quality. Useful smaller verified gains are worth retaining.

## 2. Establish the actual baseline

The selected performance control is **async64 ON**, although the public CMake
default is OFF. An ordinary default build silently selects a different control.
Use `-DPADRA_EXL3_PREFILL_ASYNC64=ON` and verify actual runtime dispatch.

Historical paired medians for this control:

| Workload | Prefill tokens/s | Decode tokens/s |
| --- | ---: | ---: |
| P2000/G500 | 10,851.114 | 255.175 |
| P4000/G1000 | 10,164.761 | 251.341 |

Use a fresh paired control for new acceptance measurements; historical numbers
are context. Pin source, build flags, compiler, binary, model, tokenizer, inputs,
KV format, capacity, chunk, sampler and timings. Inspect user changes first and
preserve them. If HEAD has moved, record the difference and establish a new
control deliberately rather than overwriting the checkout.

Hardware previously measured: RTX 5080 Laptop, 60 SMs, SM120a, approximately
16 GiB VRAM; CUDA 13.4/GCC 15. Verify current hardware/toolchain. Do not tune
clocks or power limits. Default performance configuration for this campaign:
BF16 KV, capacity 8192, chunk 2048, resident weights, batch size one, no prefix
reuse, greedy generation. Preserve separately the server's larger context and
8/12 GiB admission contracts.

The older P4000 comparison measured vLLM at approximately 13,761/251 and SGLang
at 12,818/290 prefill/decode tokens/s using INT4/G32 W4A16. vLLM's measured
advantage is prefill, not decode. Those checkpoints and numerical profiles differ
from Padra. The SGLang profiler attempt failed at startup: its operator-level
advantage has **not** been established.

Current short P4000/G16 trace, not sustained G1000 throughput:

| Component | Share of phase GPU kernel time |
| --- | ---: |
| Prefill EXL3 projections alone | 55.0% |
| Prefill all experts/transforms/combine | 65.6% |
| Prefill attention including projections | 26.5% |
| Decode EXL3 projections alone | 31.0% |
| Decode QKV and O projections | 20.7% |
| Decode attention splits and merge/quantization | about 14.1% |
| Decode BF16 head | 13.8% |
| Decode norm/residual | 10.3% |

Decode host/device gaps were about 3.7% of wall time. CUDA Graphs already exist.
The prefill 55% projection fraction implies roughly 1.39x projection speedup
to reach 12K tokens/s if everything else stays constant. This is an Amdahl
estimate, not a measured prediction. Do not prioritize tiny launch savings over
the dominant work.

## 3. Experiment workspace and execution discipline

Keep this plan and retained production source in Padra. Put experimental source
copies, harnesses, reports and captures in a new sibling directory, for example
`/home/sebastian/Development/padra-performance/2026-10-13-sol-numerical`.
Never overwrite prior experiment directories. Neighbor engines and historical
Mellum16 source/evidence are read-only references.

Create these small campaign files:

- `ENTRY.json`: immutable initial control, scope, hashes and protocols.
- `QUALITY_PROTOCOL.json`: datasets, teacher identity, tolerances, statistics.
- `PERFORMANCE_PROTOCOL.json`: cases, timing definitions, run order and gates.
- `PROGRESS.md`: current stage, decision, unresolved issue and exact next command.
- `RESULTS.json`: all candidates, including failures, with evidence paths.
- Per candidate: `HYPOTHESIS.md`, source patch, commands, resource receipts,
  oracle results, timings, quality results and `DECISION.json`.

Maintain both the original campaign control and the current accepted composition.
Compare each candidate to its immediate parent and the final composition to the
original control, so several small quality or performance regressions cannot
accumulate unnoticed.

Use bounded foreground/joined jobs: 20 GiB host RAM, zero swap, explicit timeout,
normally two build workers. Reuse/adapt the existing bounded runner; verify that
limits are enforced before a large job. No detached children or GPU lockfiles.
Never repeat the previous unbounded full-model racecheck that caused OOM.
Use small targeted memcheck/synccheck/racecheck cases for changed CUDA code.

Existing raw diagnostic limits are 16 MiB per file and 128 MiB aggregate under
`padra-performance`. Audit remaining capacity first. Stream full-vocabulary
metrics and retain per-case metrics, hashes and replay recipes rather than
duplicating all tensors. Reuse verified immutable teacher outputs. Do not delete
old evidence or silently expand limits. Counters may use enough targeted replay
passes to answer the question; there is no obsolete three-pass limit.

## 4. Quality protocol: replace equality gates with measured non-inferiority

### 4.1 Three distinct references

1. **Operator oracle:** independent decoding/indexing and FP64 represented dot
   products or attention references on bounded cases. Establish correctness of
   the intended new arithmetic, not merely similarity to another buggy kernel.
2. **Frozen Padra control:** measure incremental numerical and functional changes.
3. **Original source model teacher:** measure whether changes move output
   distributions toward or away from the source model. Use the pinned source
   revision and teacher arithmetic. Never substitute vLLM's INT4 output for this.

Verify existing teacher output hashes and provenance before reuse. The historical
complete-prefill and forced-one-decode protocols are different experiments;
their mean KL values must not be compared as if they were the same protocol.
Existing source/raw-logit/NLL/SQL/task failures remain separately reported.

### 4.2 Freeze evaluation data before candidate selection

Use the existing 48 source-reference positions and 12 executable coding tasks
as regression coverage. They are too small and too familiar to be the only gate.

Add 64 independent code contexts, balanced over the eight existing language
categories where practical, with 32 development and 32 untouched holdout
contexts. Include short, 2K, 4K and 8K prefixes; real code and multi-turn/tool
contexts; avoid only repeating a short snippet. Freeze provenance, IDs, split,
target continuations and hashes. Never use these examples for weight calibration.

For each context evaluate four declared full-vocabulary checkpoints, covering
complete prefill, forced decode and chunk transitions across the suite. Also
teacher-force at least 128 genuine continuation tokens per context for NLL.
Average within each context before aggregate statistics. Include router output
and representative layer activations from early/middle/late layers on a small
subset. Capture actual native states, not one early activation relabelled as a
later-layer state.

For full-model functional qualification add 24 executable coding/SQL tasks,
half development and half holdout, with fixed prompts, tests and token budgets.
Use one greedy attempt per task and identical extraction/scoring rules. Generated
code must run through the existing isolated, bounded task runner without secrets
or unrestricted network access. Count truncation, malformed tools and execution
failures; do not repair candidate output manually.

Prepare a pinned original-source teacher path that fits the existing memory
limits, using the existing layerwise/reference tooling where necessary. Missing
teacher coverage blocks promotion, not unrelated operator work. Do not silently
fall back to a different quantized teacher. Avoid building a new evaluation
framework if the existing scripts can be adapted.

### 4.3 Metrics and decision rules

Compute FP64-normalized log-probabilities over **all 98,304 vocabulary entries**,
source-to-candidate KL, target NLL, top-1 agreement with source, probability
margins, and per-context/per-language distributions. Never infer quality from
matching only the largest logits or a speed prompt's generated IDs.

The following are new conservative engineering non-inferiority margins for this
campaign, not empirically proven perceptual thresholds or an owner request to
accept lower quality. Freeze them before candidate results. Report both the
observed delta and uncertainty; do not call an increase an improvement.

| Metric, candidate minus control | Final numerical gate |
| --- | --- |
| Mean target NLL, nats/token | One-sided 95% upper bound <= +0.001 |
| Mean source-to-output KL, nats | One-sided 95% upper bound <= max(0.0005, 10% of matching control KL) |
| Language/context-length NLL slices | No slice mean increase > 0.003 nats/token |
| Functional tests | Preserve every previously passing legacy task; no lower pass count on either new development or holdout suite |
| Runtime correctness | No new nonfinite values, memory errors, deadlocks, invalid routing, tool/schema failures or state corruption |

Use paired resampling by independent context, not by correlated tokens, with a
fixed seed and 10,000 bootstrap replicates. Inspect outliers and high-margin
source top-1 losses. Top-1 agreement and raw-logit cosine/L2 are diagnostics,
not universal equality or absolute-logit gates. A large regression hidden by
an aggregate average remains a reason to reject a candidate.

Apply aggregate numerical gates separately to the complete-prefill and decode
protocols. Apply them to the final composition versus the original control too.
Development gates permit progression; untouched holdout is for the selected
final composition. If holdout fails, do not tune against it. Return to the last
qualified composition; a new attempt needs a fresh holdout.

If an interval is inconclusive, use at most one additional independent corpus
batch of the same size with the same frozen construction rules, then score all
data together. Do not keep sampling until a pass appears. If the evidence still
cannot establish non-inferiority, mark the candidate inconclusive and leave it
unselected. Do not loosen a margin after seeing results.

Different generated IDs, routes near a tie, or floating-point bits are allowed.
Correct expert-count/tie rules, model masks, RoPE, positions and KV chronology
are not optional. Direct execution and CUDA Graph replay of a given candidate
must implement the same arithmetic; prefer deterministic reductions and do not
introduce unordered float atomics just to gain speed.

## 5. Performance protocol and decisions

Distinguish operator, complete MoE/layer, full-model GPU timing and user-visible
latency. Include every new reconstruct, transform, gather, scatter, reduction,
scratch initialization and dispatch cost. A GEMM-only win is not a prefill win.

Use one warmup per configuration. For local screens retain three paired runs;
for final model comparisons retain six pairs, alternating A/B and B/A order.
Declare fixed inner repetitions in advance. Keep all valid runs. Record GPU
temperature, clocks and power without changing them. Exclusions are permitted
only for independently observed infrastructure failures under predeclared rules;
retain their logs and do not discard a valid slow run.

Time prefill through the first ready generated token. Time decode over G-1
additional tokens and document exclusions consistently. No initialization,
weight upload or graph instantiation in steady-state timing. Verify this in the
trace rather than assuming it. Record TTFT and full request latency separately.

Numerical candidates can follow different generation trajectories. Therefore use:

- Fixed-token, teacher-forced performance replay with each implementation's own
  native activations/routes, to compare identical token histories.
- Free-running greedy P2000/G500 and P4000/G1000 with actual generated counts,
  EOS policy and output-quality results. Do not force route IDs across engines
  or compare different generated lengths as equivalent workloads.

Keep legacy length-controlled throughput fixtures, and add four frozen natural
code prompts for confirmation. Report phase distributions as well as medians.
Do not invent a performance result when early EOS makes a requested shape absent.

Local progression should reflect model impact. Estimate speedup as
`1 / ((1 - f) + f / s)` using measured affected fraction `f` and inclusive local
speedup `s`. Advance simple candidates with a plausible positive end-to-end gain,
including gains around 1% or smaller; weigh measurement cost and complexity rather
than applying a hard minimum gain. Do not enforce an
arbitrary 5% gain on every small sub-operator or reject a model win because one
small layer is slightly slower. Inspect worst cases explicitly.

Final retention requires all quality/correctness gates plus:

- A reproducible positive median target-phase throughput gain on P4000/G1000,
  with the paired 95% confidence interval for the speedup above 1.0. A real 1%
  improvement is sufficient; there is no arbitrary 2% minimum.
- No reproducible slowdown in the other phase or at P2000/G500; natural-prompt
  confirmation must not show a consistent regression concealed by the main fixture.
  Report uncertainty explicitly rather than promising mathematically zero slowdown.
- Keep implementation complexity proportionate to measured benefit. A small,
  simple, qualified gain should be retained. Kernel-only or projected improvements
  remain hypotheses until measured in the model.

These replace the old experiment program's local percentage/range rules for new
candidates only. Do not relabel old rejected results as passes. Final timings
must be from an independently frozen finalist, not the best tuning run.

## 6. Ordered work program

Run S0, P1, P2, D1, D2, D3, then conditional H1 and composition qualification.
If a stage passes local screening, integrate it in an experimental branch/source
copy and run development quality plus model performance before accepting it as
the next parent. Re-profile after a meaningful gain; later stages may be skipped
with a written cost-based reason if their bottleneck has disappeared.

For each family predeclare at most three implementation geometries. Do not run
the Cartesian product of every tile, stage count and split. Select using cost,
resource and counter evidence. A correctness repair can repeat validation;
another performance candidate needs a changed, explicit hypothesis.

### S0 — Freeze controls, evaluation and representative profiles

1. Verify current source, artifact and async64 dispatch; inventory old failures.
2. Freeze the quality/performance protocols and prepare the new development data,
   sealed holdout and bounded teacher path.
3. Reproduce the selected control with one fresh paired baseline program.
4. Profile a P4000 request with short capture windows near decode positions 4000
   and 4990, after actually reaching those positions. Preserve actual activations,
   KV and routing; do not synthesize a long context by changing a position integer.
5. Collect focused NCU evidence for expert gate/up and down, QKV/O and norm;
   separate DRAM/L2 traffic, instruction issue, tensor activity, achieved
   occupancy, register/shared use and stalls. Explain counter units correctly.
6. Attempt one bounded repair of the SGLang profiling setup if useful. Its failure
   must not block Padra work. Pin any compared backend rather than assuming Marlin
   or FlashInfer from the engine name.

Deliver: frozen baseline, expanded quality protocol, updated component cost
table, and a concrete initial geometry for P1 and D1. This is preparation, not
an invitation to spend the campaign rebuilding all historical benchmarks.

### P1 — Reconstruct hot experts once for prefill

Hypothesis: repeated procedural EXL3 decoding across M32 token tiles costs more
than exact FP16 reconstruction plus a more efficient grouped GEMM at high expert
occupancy. Average assignments at T2048 are 256/expert, or about eight M32 tiles.
Routing skew and real padding must be measured, not assumed uniform.

Start with an isolated complete-MoE comparison using real routes/inputs from
layers 0, 14 and 27. Test T128, T512, T2000/2048 and the real 1952-token tail.

1. Implement exact K4-to-FP16 reconstruction in the rotated basis. Exhaustively
   validate the 65,536 codebook states and production address/permutation cases.
   Keep sidecars and input/output H128 explicit; do not fold them into weights.
2. Establish a diagnostic GEMM ceiling on predecoded weights, clearly excluding
   reconstruction. If even this ceiling cannot beat control meaningfully, stop
   this family before building a complicated scheduler.
3. Implement the inclusive path: GPU route grouping, reconstruct, grouped/batched
   FP16 GEMM with FP32 accumulation/output, original H128 and BF16 boundaries,
   SiLU and ordered top-8 combine. Avoid a hidden FP16 GEMM-output rounding before
   H128. A custom epilogue is acceptable; unchanged dot summation order is not
   required. Preserve independent gate/up sidecars.
4. Use a preallocated reusable scratch arena and device-side descriptors. No
   per-expert host readback, allocation or graph rebuild per token/chunk. Start
   with one projection resident at a time and small expert groups.
5. Select a deterministic occupancy crossover from the development measurements.
   Try only a small declared set, e.g. 64/128/256 assigned rows. Below crossover
   use the current path. Measure threshold boundaries and imbalanced routing.

Full reconstruction of one projection across all 64 experts costs 252 MiB;
all three projections of a layer cost 756 MiB. Two chunks across 28 layers write
about 44.4 GB of FP16 weights before subsequent reads. Account for this traffic,
not just available VRAM. Grouped GEMM, activation slabs and metadata add storage.
The full model must remain compressed/resident; never materialize all layers.

Upstream reference:
https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/modules/moe_batch_recon.py

Inspect and pin the actual source used. Its folded mode changes weight rounding;
its Python allocations/host metadata flow are not Padra's graph-safe design.
Follow attribution/license requirements for reused code.

Reject if reconstruction/traffic consumes the apparent GEMM gain, representative
small/tail cases cannot be handled without regressions, or quality fails.
Do not interpret an upper-bound predecoded benchmark as a production speedup.

### P2 — Async64 operand ownership and decode reuse

Hypothesis: the successful K64 transport leaves enough generic Pair/shuffle and
operand-assembly overhead that transposed ownership can reduce instructions.

Start from the currently selected prefill path. Combine K64 asynchronous transport
with weights as MMA A and activations as B, using the specialized four-fragment
lane codec where the exact mapping permits it. Preserve H128 semantics and masks.
Prove reconstructed operand bits; qualify accumulated outputs numerically.

Inspect SASS counts/dependencies and compiled registers/shared/spills against
control. Keep the first candidate M32/N128/eight warps. Consider M64 only if a new
shared-memory lifetime/layout addresses the earlier 43 KiB/two-CTA failure;
merely increasing M repeats a rejected experiment. Do not assume SM90/SM100
warp-group instructions exist on SM120a; use the actual supported instructions.

If P1 was retained, evaluate P2 on the cold-expert fallback and residual hot path
only where its actual model fraction justifies the work. If it saves no material
inclusive time, stop rather than chasing instruction counts alone.

### D1 — Increase parallelism of EXL3 decode projections

Hypothesis: gate/up's 112 CTAs and down's 144 CTAs underfill the 60-SM GPU or leave
long dependent K loops poorly hidden. Theoretical resource occupancy alone does
not prove this; require S0's achieved issue/occupancy and memory evidence.

Prototype deterministic split-K on the existing N128 output ownership. First
candidate: two K partitions; consider four only if measurements justify it.
Gate/up and down may need different choices. Split only at legal K16 boundaries;
the 896-wide down input is not divisible by every proposed larger transport tile.

Keep native weight decoding and exact input transformation. Write FP32 partials
to preallocated scratch, then reduce in a fixed order and apply output H128,
sidecars and BF16 rounding. Do not round each partial to BF16 or apply separately
rounded output transforms. Account for repeated input transforms, scratch
traffic, extra launches and the merge. Avoid inter-CTA spin barriers and unordered
atomic accumulation.

Use real early/middle/late layer inputs near both 4K and 5K, plus a full 28-layer
weight sweep. Warm repeats of one operator are secondary diagnostics. If bandwidth
is already saturated or merge cost erases the gain, reject split-K and move on.

### D2 — Faster FP32 norm/reduction arithmetic

Hypothesis: compensated reductions and scalar dependency chains cost material
time on the roughly 10% norm/residual share. Relaxing bit equality makes a simpler
FP32 tree and localized reciprocal-square-root evaluation worth testing.

Start with `ResidualPostNormWideExact` only. Retain residual addition and declared
BF16 boundaries; replace compensated sum-of-squares with a stable FP32 parallel
tree, then separately evaluate `rsqrtf` versus current sqrt/div. Test zero, tiny,
large finite and representative rows, including overflow handling.

After a passing first variant, apply the same design to input norm/FP8 encoding
as a second candidate. Preserve the declared E4M3 rounding/saturation algorithm;
new amax/scale values resulting from new normalized inputs are expected and must
be evaluated through the model. Do not accidentally bypass quantization boundary
handling or nonfinite detection.

Changed norms can alter router choices; evaluate actual routing and downstream
quality. Do not force baseline experts to manufacture agreement. Keep QK/RoPE
normalization unchanged initially so the error source is identifiable.

This family has a limited ceiling: 30% faster norm/residual alone suggests about
2.4% faster decode. Use that to bound implementation effort.

### D3 — Attention specialization; conditional projection experiment

First focus on local attention, whose window saturates at 1024 tokens and whose
existing 16 splits process only two K32 tiles each. Compare a narrowly chosen
alternative such as 32 single-tile splits (128 CTAs instead of 64), including the
larger merge. The different online-softmax/merge order is permitted, but must
pass represented attention and model quality checks. Derive small-context active
split handling and poisoned/unused scratch behavior explicitly.

Treat global attention separately at 4K/5K and larger supported contexts. Select
splits from actual work/occupancy, not context capacity alone. Preserve causal
and local masks, absolute ring chronology, RoPE and BF16 KV. Profile shared-memory
lifetimes and register pressure before redesigning the pipeline. Changing shared
memory alone cannot improve occupancy when the launch grid is the real limit.

Only if QKV/O remain a substantial measured opportunity, consider one numerical
FP8 Tensor Core projection candidate. The old `attention_fp8_mma` branch failed
source/SQL/top-1 gates. Read its report first. It may be used as a diagnostic
reference under the new protocol, but is not a newly discovered or preapproved
optimization. Prefer a changed accumulation/layout design justified by its old
error and performance evidence. Keep stored weights and input FP8 encoding fixed
to isolate accumulation changes. Reject on current quality gates.

Do not simultaneously replace the entire attention stack and all projections.
An external FlashInfer/FlashAttention comparison is optional diagnostic evidence,
not permission to add a large runtime dependency without assessing integration.

### H1 — Conditional lossless head compression

Do this only after the larger candidates, if the residual head share and memory
bandwidth evidence still justify it. Current head is 98,304 x 2,304 BF16, about
453 MB, with 5,871 distinct raw values. A 13-bit dictionary saves about 18.75%
of weight bytes. Ideal proportional savings imply only about 2.7% overall decode
throughput gain at a 13.8% head share, before unpack/lookup overhead.

Start with fixed-width 13-bit indices and an appropriate cached dictionary,
preserving every BF16 weight bit. Prove full reconstruction and address safety.
Do not use divergent constant-memory lookups without testing serialization.
Benchmark inclusive cold/full-model head behavior, not just a small cached slice.

Account for load-time conversion, temporary duplicate storage and final steady
state. Do not permanently keep both GPU representations. No lossy head
quantization is included in this plan. Stop if codec overhead eliminates the
byte-saving benefit; do not start an open-ended entropy-code project.

## 7. Prior experiments: avoid rediscovering failures

Read the actual receipts before a related candidate. Failures motivate a changed
hypothesis, not a ban on every related algorithm.

| Previous attempt | Evidence / consequence |
| --- | --- |
| Async64 transport | Retained; selected ON baseline, default still OFF |
| Ldmatrix-only prefill | About 0.6% inclusive gain, insufficient |
| Old M64 prefill | Higher resource use; slower across tested fixtures |
| Older prefill transpose/lane specialization | Little meaningful prefill gain; K64 combination needs new evidence |
| Earlier/deeper EXL3 register loads | Slower; unchanged retries are low priority |
| Shared decoded QKV input | About 6.08 to 11.21 us; substantially slower |
| Separate decode input H128 | About 37.09 to 43.91 us/FFN; extra kernels/traffic lost |
| Cooperative full-MoE fusion | About 9% slower |
| Combine plus next norm fusion | Slower in measured cases |
| BF16 head layout/prefetch | Approximately 0.4% local, no meaningful model gain |
| Naive BF16 Tensor Core head | Approximately 3x slower |
| Generic codebook LUT | Approximately 22–26x slower in tested decode cases |
| FP8 attention projection MMA | Historical quality failure, not automatically cured by removing equality |
| Smaller KV formats | Different quality/memory question; outside this campaign |

Do not prioritize standalone SiLU removal: its current decode share is below 1%.
Do not repeat generic paired-QKV, graph installation, or residual/RMS fusion as
if missing; inspect active dispatch and existing implementations first.

## 8. Integration and final qualification

For every locally promising candidate:

1. Independent operator/oracle checks and small targeted sanitizer checks.
2. Experimental model integration, truthful kernel-profile metadata and complete
   memory accounting. Use a narrow opt-in selector until qualified.
3. Development numerical quality and executable task checks.
4. Paired model performance against its parent and original control as needed.
5. Keep or reject explicitly; preserve artifacts and the rejected patch outside
   the public source tree. Do not mark integration alone as acceptance.

For the selected composition, run fresh performance confirmation and untouched
holdout quality. Test direct/graph execution, repeated graph replay, prompt/chunk
boundaries (including 1/3/4/127/128 and 2047/2048/2049 tokens as applicable), local
window wrap, EOS, cancellation, request reset and session/KV reuse. Preserve the
server's output and error contracts despite changed numerical results.

Verify actual memory admission for chunks 256/512/2048, budget ceilings 8/12 GiB
and actual free memory, and context capacities through the supported 131072.
New scratch must not cause a silent context/precision change. A deterministic
fallback to the established kernel when new scratch does not fit is acceptable
if explicit in planning/metadata and separately checked. A capped 16 GiB card
does not qualify performance on physical 8/12 GiB hardware. Long-context memory
admission is not long-context quality qualification; report which actually ran.

Retain only qualified production code, small relevant tests and concise English
documentation in Padra. Keep research harnesses and large evidence outside it.
Do not change `PADRA_EXL3_PREFILL_ASYNC64` or experimental runtime defaults as a
side effect. Do not commit/push or claim broad source/product-quality acceptance
unless separately requested and supported.

Final report: exact selected source/build, original and final P2K/P4K phase
throughput and variability, quality deltas/intervals, task outcomes, memory/context
coverage, rejected experiments, remaining uncertainty and next best hypothesis.
If none passes, say so and leave the selected control intact. Targets may remain
unmet; completing a finite experiment program is not achieving the speed targets.

## 9. Local evidence and code map

These absolute paths are machine-local research references, not shipped assets.
Verify existence/hashes; never copy weights or large captures into the repo.

- Active repository: `/home/sebastian/Development/padra`.
- Local handoff: `ASTRA_PERFORMANCE_HANDOFF.local.md` in that repository (ignored).
- Qualified async64 build and receipts:
  `/home/sebastian/Development/padra-performance/2026-10-13-public-async64-integration`.
  Core: `build-ON/engine/libpadra_core.a`; probe: `build-ON/padra-runtime-probe`.
- Current profile and recent rejected candidates:
  `/home/sebastian/Development/padra-performance/2026-10-13-next-program`.
  Start with `P0/RESULT.md`, `P0/profile-summary.json`, `CONTROL_SOURCE_PINS.json`.
- Prior isolated P1/D1/H1 results:
  `/home/sebastian/Development/padra-performance/2026-10-13-followup-summary.md`.
- Engine comparison, prompts, teacher metadata, quality scripts:
  `/home/sebastian/Development/padra-engine-comparison/2026-10-12`.
  Start with `summary.json`, `quality_summary.json`, `quality_teacher_protocol.json`,
  `control_identity.json`, `quality_collect.py`, `evaluate_quality.py` and
  `padra_quality.py`. Inspect hard-coded old binaries/options before reuse.
- Original-source reference outputs:
  `/home/sebastian/Development/mellum16/build/quality-v2/original`.
- Older experiment reports:
  `/home/sebastian/Development/mellum16/docs/EXL3_K4_OVERNIGHT_PROGRESS.md` and
  `/home/sebastian/Development/mellum16/build/exl3-overnight`.
- Existing bounded runner helpers:
  `/home/sebastian/Development/mellum16/tools/diagnostics/bounded_gpu_job.py`,
  `bounded_scope_payload.py`, `diagnostic_storage.py` in the same directory.
  Example orchestration: `run_job.py` in the recent `next-program` directory.
- Native artifact:
  `/home/sebastian/.cache/huggingface/derived/mellum16/92ddae9fc7665e9f801d141d2e5a6b2caf2460c4/mellum16-exl3-k4-v1/native`.
  Weights SHA256: `e614375f2eb26c17321365d4bd8ee7aa4b716f6badee171bb2040b85b754804c`.
  Manifest SHA256: `0f30d4ead6ae3d0683ec1eb9ad7af1adfebe343d804e3313e8c01d6c31f76f0b`.
  Tokenizer SHA256: `58548a346eb073e5132bf7d8ad17dc6971bca36ade378ca4d2bfbc49bf60da2a`.

Follow actual dispatch in `src/cuda/inference.cu`, not header names alone:

| Area | Active source / useful reference |
| --- | --- |
| Prefill EXL3 | `expert_exl3_prefill_transport_async64.cuh`, `expert_exl3_prefill_packed_activations.cuh` |
| Codec/fragment mapping | `expert_exl3_dot_codec.cuh`, `expert_exl3_lane_codec.cuh` |
| Decode EXL3 | `expert_exl3_decode_dual.cuh` |
| Decode QKV / O | `fp8_projection_fast.cuh`, `fp8_projection_pair_exact.cuh` |
| Norms | `residual_norm_wide_exact.cuh`, `norm_fp8_wide_exact.cuh`, `rms_wide_exact.cuh` |
| Decode attention | `attention_gqa8_fragment_validation.cuh`, `attention_merge_quantize_wide_exact.cuh` |
| Prefill attention | `attention_prefill_padded_single.cuh` |
| Head | `GemvWarp` in `inference.cu`; alternative head headers are not necessarily selected |
| Memory/contracts | `src/runtime/server_memory.h`, `server_profile.h`, `context_capacity.h`, `src/cuda/context_admission.cuh` |

## 10. Resume checklist

- [x] S0: control, quality/performance protocols and representative profiles frozen.
- [x] P1: exact reconstruction plus inclusive grouped GEMM decided.
- [x] P2: retained after full-model small-gain qualification; see execution record.
- [x] D1: evidence-backed split-K decode decided.
- [x] D2: compensated postnorm retained with the qualified short-K64 prefill repair; see follow-up decision below.
- [x] D3: local32 attention rejected; conditional projection retry skipped with evidence.
- [x] H1: lossless13-bit dictionary implemented, exact, slower; rejected.
- [x] Selected P2 + D2/short-K64 composition passes quality and speed confirmation, including a fresh short-prefix holdout.
- [x] Memory, graph, server-state and context compatibility checked and scoped.
- [x] Qualified composition retained; final report distinguishes gains from unmet targets.

At every interruption update `PROGRESS.md` with completed evidence and the next
single action. Resume there instead of restarting the campaign or rerunning
rejected variants without a new hypothesis.

## Execution record — Sol

S0 preparation completed in `/home/sebastian/Development/padra-performance/2026-10-13-sol-numerical`. See `S0/REPORT.md`, `S0/DECISION.json`, `S0/SEAL.json`, and `PROGRESS.md` there. Expanded quality evaluation and candidate qualification remain pending; no numerical candidate, product default, commit or push is accepted. Fresh async64 ON control P2K medians:10,819.239/255.594prefill/decode tok/s; P4K:10,204.001/252.486. Next stage:P1 exact reconstruction and grouped-GEMM ceiling, initial group8.

P1 early ceiling stop: all15tested all-active group8 cuBLAS cases are slower than async64 even before reconstruction/transforms. Exact codebook/full-weight/FP64/graph and targeted safety checks pass. Inclusive scheduler/model integration unrun by the plan’s stop rule. See `P1-reconstruct/REPORT.md` and `DECISION.json`; next:P2.

Astra takeover: P2 completed, deferred for insufficient standalone model benefit (see campaign `P2-transpose64/DECISION.json`); D1 split-K2 implementation started. No retained improvement yet.

2026-10-13 user amendment: Real gains around 1% are worth retaining when simple and without regressions. The former 2% final / 1% projected minimums are superseded. P2 is reopened for model integration and paired qualification; earlier deferral remains historical evidence, not an acceptance decision.

D1 split-K2 implemented and rejected after measured inclusive slowdowns in all six local cases and 28-layer weight sweep (see campaign `D1-splitk/REPORT.md`). Rejection is due to regression, independent of the small-gain amendment.

P2 qualification clarification (before expanded results): if every full-vocabulary head in the complete declared comparison is byte-identical to control, candidate-minus-control NLL and source KL are exactly zero for any common teacher. This proves non-inferiority without recomputing the absolute source baseline. It does not qualify existing source failures. Any changed head requires the original numerical teacher protocol; task/state gates remain. See campaign `P2-model/EXACT_NONINFERIORITY_RULE.json`.

P2 retained after user amendment: P4 prefill10229.304 ->10309.004 tok/s (+0.779% ratio of medians; +1.130% median paired gain, paired95% CI +0.096%..+1.526%). P2K +0.193% inconclusive; decode within observed measurement variation, natural-prompt decode point estimates -0.025%..-0.051% explicitly retained. All8320 expanded full-vocabulary heads and96legacy heads exact, new task passes22/24 each and12legacy sequences unchanged. Eager/graph boundaries, cancellation/reset/append/checkpoint and capacity admission through131072 pass with identical memory. Only short content was executed at the largest capacity profiles. Qualified P2 CUDA patch and README retained in repo; no default/commit/push change. See campaign `P2-model/REPORT.md` and `SELECTED_PARENT.json`. Overall program D2/D3/H1 and final composition remain open; next:D2 from P2 parent. Future numerical candidates must not tune on P2 holdout results.

D2 continuation: ordinary FP32 postnorm is numerically inconclusive (development NLL mean +0.000133, upper95 +0.001262 exceeds +0.001 margin). Compensated256-lane register-row follow-up matches all4128 development heads and12development task sequences. Initial paired P4 decode+0.237% (95% +0.067%..+0.530%); P2K+0.659%. Inputnorm follow-up rejected for0.768–0.776 local speed ratios. Small negative natural-prefill measurements remain disclosed; a minimal integration cleanup is frozen for final qualification in `D2-clean-model`. No D2 public adoption yet.

D3 local32: exact native control replay and FP64/graph/mask/ring/sanitizers pass, but inclusive split+merge+FP8 takes18.7–19.1us versus12.9–13.2us on4native states. Reject. Global split behavior unchanged. Conditional FP8 projection retry skipped after reviewing prior source-KL/top1 failure; no new layout/accumulation hypothesis justified another numerical branch.

H1 dictionary13: all226492416BF16 words reconstruct exactly, two full output vectors and sampled FP64 dots match; targeted sanitizers/graph checks pass. Storage452984832→368061922bytes, but fullhead0.529ms→1.099ms (cached global table) or2.562ms (shared table). Both rejected without model integration.

Validation correction: the old P2 boundary script changed `controlled_chunks` inside one persistent runtime; only the first request selects the mode. Its result alone did not prove a mode comparison. `D2-comp-model/boundaries.py` closes the gap with separate processes:42full heads per arm/mode match over14prefix lengths andtwo decode steps. This compares prefill modes; decode still uses the selected full-step graph. New operator direct/graph checks separately pass.

Final holdout refreshed before evaluation:32independent new code files and12new executable tasks, all gold/mutant controls validated. Source/build and final test hashes are sealed in `D2-clean-model/FINAL_SEAL.json`. Do not tune against this holdout.

## Earlier decision — 2026-10-13 (superseded by D2 repair below)

The finite plan is decided. **P2 remains the selected production implementation.**
No additional D2/D3/H1 code was promoted. Small gains remain eligible; D2 was
withheld because the requested no-regression condition was not met.

| New experiment | Result | Decision |
| --- | --- | --- |
| Ordinary FP32 postnorm | NLL upper confidence bound +0.001262 exceeds +0.001 margin | Inconclusive |
| Compensated parallel postnorm | Decode +0.22% at P4, +0.58% at P2K; repeated natural-prompt prefill -0.2% to -0.9% | Keep prototype; do not select |
| Parallel input norm/FP8 | About 29–30% longer locally | Reject |
| Local attention with 32 splits | About 45% longer including merge | Reject |
| Lossless 13-bit head | 453 to 368 MB, but 0.529 to 1.099 ms with cached dictionary | Reject |

The D2 final candidate passed all 8,320 development/fresh-holdout full-vocabulary
comparisons, 96 historical head checks, scoped sanitizers, memory/admission and
session-state checks. Development and new holdout tasks remained 11/12 each;
all twelve historical generations were unchanged. The numerical policy was not
reverted to mandatory bit equality. Its simpler FP32 predecessor was evaluated
numerically and remains inconclusive.

A G1 phase-focused A/A-controlled check showed smaller prefill shifts, but did
not reproduce the natural suite's long-generation request history. The final
natural suite repeated the negative prefill result, so that focused check cannot
establish absence of regression. The underlying cause remains unresolved.

The retained P2 binary was separately rechecked on the fresh holdout: 4,160 full
heads and twelve task sequences match the original control; holdout passes 11/12.
Fresh six-block measurements give P4 paired prefill +1.087% (95% interval
+0.228% to +1.394%), with decode within measurement variation. This confirms the
previously retained small prefill gain; it is not an additional D2 gain.

Detailed report and resumable evidence:
`/home/sebastian/Development/padra-performance/2026-10-13-sol-numerical/FINAL_REPORT.md`.
The 12,000 prefill / 280 decode targets remain unmet. The next useful diagnostic
is a full-natural-request identical-binary A/A comparison and bounded host/GPU
timing to explain the short-prefill issue before more numerical tuning. All
new holdout data is now consumed and must not guide future numerical tuning.

## D2 repair follow-up — 2026-10-13 (current selection)

The user requested diagnosis and repair of the prefill regression so the small
D2 decode gain could be retained. **Select P2 + compensated D2 postnorm + short
K64 dispatch**, under the existing ASYNC64=ON option. Default OFF is unchanged.
The preceding decision remains as experiment history and is superseded here.

Identical-binary controls reproduced the old short-prefill regression. The
prefill kernel instructions, counts and resources were unchanged; telemetry
implicates power/clock/request-history coupling, without isolating the entire
cause. Blocking CPU wait did not repair speed. The successful repair removes the
unmeasured T<128 packed-K32 fallback: existing K64 is locally faster at all six
short sizes tested across three native layers. T<4 keeps its original SIMT path.

Final six-block balanced comparisons versus P2 show short prefill +10.6–11.5%,
P4000/G1000 decode +0.557% (95% +0.334..+0.665%), and P2000/G500 decode +0.616%
(+0.435..+0.734%). Long prefill is unresolved around zero; all timing samples,
including negative blocks, remain in the reported intervals. There is no repeated
other-phase regression. Small gains are accepted without an arbitrary threshold.

Qualification includes 4,128 full heads from 32 fresh disjoint short contexts,
8,320 replayed expanded heads, 96 legacy heads, unchanged task sequences,
independent FP64/operator/sanitizer checks, separate-process execution-mode and
chunk boundaries, state transitions and scoped memory admission. OFF metadata,
32 generated tokens and four full heads also match. The frozen tested CUDA bytes
are retained in the repo; no extra buffers, launches, commit or push are added.
All fresh holdouts are now consumed. See [PERFORMANCE_D2.md](PERFORMANCE_D2.md)
for measurements, uncertainty, validation limits and exact reproduction paths.
Targets of 12,000 prefill / 280 decode token/s remain unmet.

## Follow-up evidence: SGLang decode comparison (2026-10-13)

The [decode investigation](SGLANG_DECODE_FINDINGS.md) now includes working
SGLang/native-vLLM profiles at actual positions 4000 and 4990. SGLang's current
stable release reproduces 288.72 decode token/s. Changing only vLLM's attention
backend from FlashAttention to FlashInfer raises its native INT4 result from
250.62 to 284.65 token/s. Padra's selected EXL3 control remains 252.84 token/s.

The report prioritizes a decode-only BF16-valued input to the O projection with
unchanged FP8 weight bytes, an inclusive FlashInfer attention operator comparison,
and narrowly scoped normalization experiments. These are new hypotheses, not
accepted implementations. Existing rejected experiments and quality gates remain
in force; the older failed SGLang profile no longer limits operator investigation.

## Follow-up result: attention merge (2026-10-13)

The [attention experiments](PERFORMANCE_ATTENTION.md) retain a per-head BF16 merge
followed by the original exact FP8 row encoding and O projection, under the
existing ASYNC64=ON option. Full-vocabulary regression, graph/direct, state,
OFF-build and targeted operator/sanitizer checks pass. Six paired final blocks
show +1.370% P4 decode (95% +0.950..+1.519%) and +1.236% P2 decode. Prefill medians
are -0.136%/-0.273%; short prompts show -0.122..-0.688% prefill. These fit the
existing 1% median tolerance, but strictly zero slowdown is not proven. The
identical-control diagnostic and unchanged existing kernel SASS do not erase
negative candidate samples. Default OFF remains unchanged; no commit/push here.

The direct BF16-input O prototype improves exploratory P4 decode by 3.09% but
fails development numerical non-inferiority; it is not promoted. The inclusive
FlashInfer test shows substantial global-attention headroom (about 15 versus
24–26 microseconds including Q conversion and the same O projection), but no
native backend integration or full-model quality claim. Normalization is deferred.

Current selected source/build and reproducible receipts:
`/home/sebastian/Development/padra-performance/2026-10-13-attention-bf16/SELECTED_PARENT.json`.
The old campaign remains an immutable parent. Future numerical changes require
fresh holdout/source-quality qualification; this exact decomposition preserves
checked parent outputs and does not resolve historical absolute quality failures.

## Follow-up result: native global attention (2026-10-13)

Two [native FlashInfer global-attention prototypes](GLOBAL_ATTENTION_EXPERIMENTS.md)
are implemented and screened in the sibling evidence workspace. G1 uses BF16
partials; G2 keeps internal partial accumulators in FP32, with BF16 Q/K/V/output.
Both retain the parent's FP8 O path and replace only the seven global decode
layers. Full-model P4 decode is 258.32 versus 256.25 token/s for G1, and 258.24
versus 256.37 for G2. Corresponding prefill is 10,443.31 versus 10,482.26 and
10,426.91 versus 10,453.74 token/s. These are exploratory paired screens.

Both fail development NLL non-inferiority (95% upper +0.003778/+0.003526 versus
the +0.001 gate, plus language-slice failures). Do not promote them. Small speed
gains remain acceptable; the blocker here is quality qualification. The newly
frozen holdout is unscored, source-KL/tasks are unrun, and the retained parent
remains the exact merge: 10,346.96 prefill / 256.05 decode at P4000/G1000 and
10,820.03 / 259.61 at P2000/G500 in its final six-block series.

Next prioritize Q/K-normalization + RoPE + BF16 KV-store fusion with preserved
arithmetic, then incremental global-kernel scheduling changes, then normalization
data movement. The linked report records concrete scope and validation steps.
Evidence: `/home/sebastian/Development/padra-performance/2026-10-13-global-attention`.
No new production CUDA change, default, commit or push in this follow-up.

## Follow-up result: Q/K, RoPE and KV-store fusion (2026-10-13)

The preceding source and comparison reports were committed and pushed as
`c1c337f` before this follow-up, as requested. **Retain the exact Q/K-normalization,
RoPE and BF16 KV-store fusion** under the existing ASYNC64=ON option. Default
OFF remains unchanged. The new implementation removes 28 kernel launches per
decode token without new buffers or altered arithmetic. All 238 pre-existing
kernel SASS bodies are unchanged; one fused kernel is added.

Six paired blocks versus the committed parent give P4000/G1000 decode
256.32 to 257.86 token/s, paired +0.562% (95% +0.454..+0.686%), and P2000/G500
259.95 to 261.18, paired +0.475% (+0.323..+0.581%). Candidate prefill is
10,403.48 and 10,864.71 token/s respectively. Paired prefill changes are -0.132%
and +0.025%, with intervals including zero; short-prompt prefill is
-0.457..+0.395%. Retention uses the existing 1% median other-phase tolerance,
not a claim of strictly zero slowdown. All valid samples remain included.

The second requested experiment caches native global Q operands in registers
across KV tiles, preserving MMA/softmax/partition arithmetic. Operator time
improves, and full-model outputs are exact. An independently qualified combined
build is then compared against fusion alone: P4 decode 257.66 to 257.79 token/s,
paired +0.055% (95% -0.075..+0.191%); P2 decode 261.33 to 261.12, paired -0.032%
(-0.161..+0.032%). **Do not promote query reuse:** the extra complete-model gain
is unproven. No arbitrary minimum speed threshold is applied.

Each prototype and the combined build reproduce 4,128 complete heads, 42 boundary
heads in each direct/graph mode and ten state cases. Fusion and combined OFF
checks, independent operator checks and scoped sanitizers pass. Historical
absolute-quality issues remain; no numerical holdout was consumed.

See [PERFORMANCE_QK_KV.md](PERFORMANCE_QK_KV.md) for scope, all absolute rates,
uncertainty and reproduction. Selected source/binary pins:
`/home/sebastian/Development/padra-performance/2026-10-13-qk-kv-global-reuse/SELECTED_PARENT.json`.
Both requested experiments are completed; further tile/partition and normalization
work remains unqualified. The 12K prefill / 280 decode targets are still unmet.
