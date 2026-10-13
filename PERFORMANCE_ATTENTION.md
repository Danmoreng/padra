# Attention output and merge experiments

Follow-up: [native global attention experiments](GLOBAL_ATTENTION_EXPERIMENTS.md)
now include two complete-engine FlashInfer prototypes, absolute rates and quality
results. Neither replaces the exact merge retained below.

Date: 2026-10-13. Parent: `856a04f`, selected ASYNC64=ON (P2 + compensated D2).
Hardware: RTX 5080 Laptop, SM120a, CUDA 13.4/GCC 15. Same EXL3 K4 artifact,
BF16 KV, top-8 routing, chunk 2048 and capacity 8192 for sustained timing.

## Precision and scope

The owner permits FP32 accumulation and softmax statistics, with BF16/FP8
activation storage. EXL3 K4 expert weights remain unchanged. Q/K/V/O weights
are separately stored as FP8 in this artifact; W8A8 describes those attention
projections, not the EXL3 experts. Converting a quantized value to a wider
arithmetic type does not recover information lost during quantization.

These experiments use actual packed BF16 storage for the new attention-output
intermediate. They do not migrate every existing float allocation in the engine;
some older buffers still store BF16-rounded values in float containers.

## A1: BF16 attention output directly into O

Implemented a per-head BF16 merge and a paired-row BF16-input/FP8-weight O
projection. FP8 weight bytes/scales, attention split arithmetic, BF16 KV and
FP32 accumulation are unchanged. The activation FP8 round trip is removed.

Six native layer/position fixtures show 1.27–1.35x inclusive operator throughput
(attention + merge + O). The attention outputs remain exact; relative O error
against independent FP64 represented dot products decreases from 0.0063–0.0458
to 0.00051–0.00166. Better operator accuracy does not establish model quality.

Two exploratory AB/BA blocks, one warmup and two retained requests per shape:

| Workload | Control prefill | A1 prefill | Control decode | A1 decode |
| --- | ---: | ---: | ---: | ---: |
| P2000/G500 | 10,919.17 | 10,739.54 | 257.57 | 264.84 |
| P4000/G1000 | 10,458.89 | 10,285.46 | 253.03 | 260.84 |

Units: token/s. P4 decode improves 3.09%; prefill decreases 1.66% in this screen.
No acceptance claim follows from these short timing runs.

The 32-context development screen changes 4,096 decode heads; the 32 prefill
heads remain exact. Mean target-NLL change is -0.0002717, but the context-bootstrap
one-sided 95% upper bound is +0.0025458 (gate +0.001). Go averages +0.0097942 and
Java +0.0030775 (per-language gate +0.003). The mean does not show aggregate
harm, but non-inferiority is unproven and the declared language gates fail.
**Do not promote A1.** Fresh holdout/source-KL/task qualification was not run;
the existing development result is not a substitute. The prototype is preserved.

## A2: inclusive FlashInfer comparison

FlashInfer 0.6.18, tensor-core paged decode, BF16 Q/K/V/output, page size one;
32 Q heads, four KV heads, dimension 128. Compare the same native Q/K/V snapshots
and the same A1 O projection. FlashInfer timing includes the Q storage conversion
from float to BF16; both variants include attention, merge and O. CUDA Graphs
contain 128 replays, with five alternating timing rounds per native fixture.

| Native fixture | Padra BF16 path, µs | FlashInfer path, µs |
| --- | ---: | ---: |
| Layer 0, position 4000 | 13.25 | 12.78 |
| Layer 0, position 4990 | 13.23 | 12.77 |
| Layer 14, position 4000 | 13.25 | 12.76 |
| Layer 14, position 4990 | 13.26 | 12.74 |
| Layer 27, position 4000 | 23.54 | 14.95 |
| Layer 27, position 4990 | 26.32 | 15.20 |

Global attention has substantial remaining headroom. This is a fixed-position
operator comparison: planning and index preparation happen outside replay,
and there are no KV copies per replay. It is not a drop-in engine benchmark or
an accepted model-quality result. FlashInfer's relative attention error against
the independent FP64 oracle is about 0.0018–0.0022, versus Padra's 0.0015–0.0017;
its arithmetic differs. A native backend integration must account for changing
positions, local-ring chronology, graph-safe metadata updates, workspace
admission, cancellation and quality before claiming an engine speedup.

The first capture used a stale CUDA stream and yielded an empty native graph;
its timing is explicitly invalid. A second attempt incorrectly captured the
default stream and failed. Only attempt 3 queries the current stream inside
both closures, passes graph/direct output checks and is reported above.

## B: exact FP8 output with a parallel merge

Separate the expensive merge from row quantization: 32 CTAs each merge one
head into packed BF16, then one 512-thread CTA computes the original row scale
and exact FP8 encoding. The original paired FP8 O projection consumes the same
bytes and scale. The reduction and ascending split FMA order are preserved;
finite amax reduction is order-independent. Stream/graph dependencies order the
two kernels, without a cross-CTA barrier or completion counter.

Six native inclusive operators save approximately 9–14% time. All 14 local-ring,
tile-boundary and native-position cases preserve the parent attention values,
FP8 bytes/scales and O output; independent FP64, graph parity and buffer-canary
checks pass. Memcheck reports zero errors; targeted Synccheck/Racecheck of the
new merge, quantizer and A1 projection report zero errors/hazards. The unchanged
split kernels are not newly race-qualified by the filtered run.

The final source primes new kernels before graph capture and limits this path
to the existing ASYNC64=ON selected EXL3 configuration. Other paths retain their
original dispatch. It reuses the existing attention arena, adds no resident
buffer and adds one kernel per layer. Compiled merge: 40 registers; quantizer:
32 registers; both have zero local memory/stack spills.

Final qualification has reproduced 4,128 full heads from 32 contexts, all 42
boundary heads in separate direct/graph processes, ten checkpoint/cancel/append/
reset requests, and the OFF build's metadata, 32 generated IDs and four full
heads. Checked admission/resident/KV inventories match. This preserves the
parent's checked quality; historical absolute source/task failures remain open.
Fresh numerical holdout qualification is still required for A1 or a different
attention arithmetic. Existing context data are regression coverage, not fresh
holdouts.

**Retain B under the existing ASYNC64=ON option.** Public default OFF is unchanged.
Six alternating AB/BA blocks, alternating P2/P4 shape order, one warmup and one
retained request per shape/arm/block, give:

| Workload/phase | Control token/s | Candidate token/s | Paired median gain | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| P2000/G500 decode | 256.51 | 259.61 | +1.236% | +0.953..+1.339% |
| P4000/G1000 decode | 252.60 | 256.05 | +1.370% | +0.950..+1.519% |
| P2000/G500 prefill | 10,850.88 | 10,820.03 | -0.273% | -1.467..+0.637% |
| P4000/G1000 prefill | 10,370.35 | 10,346.96 | -0.136% | -0.642..+0.677% |

Intervals bootstrap six paired block ratios (10,000 replicates, seed 20261013).
The paired median is not necessarily the ratio of marginal medians. All valid
samples remain included. Decode is G-1 actual steps divided by decode time;
model load/upload and graph preparation are excluded. All generated sequences
match, and runtime load/upload counters stay at one. There is no new full-model
CUPTI trace; graph execution and the allocation-free replay construction were
checked through the native harness and source, not a new profiler capture.

Four short natural prompts at G128, two AB/BA blocks with one warmup and two
retained requests per prompt/process, give +0.643..+0.731% decode and
-0.122..-0.688% prefill. Generated IDs remain exact. No new task grading is
claimed. A subsequent identical-control A/A diagnostic shows prefill medians
from -0.350% to +3.615%, including the first Python arm's large positive drift;
no valid run is excluded. This demonstrates measurement variability, not a
correction that removes the candidate's negative observations.

All 235 pre-existing kernel SASS bodies match after anonymous-symbol name
normalization; only the three new kernels are added. Prefill kernel code and
dispatch are unchanged. The small phase-level prefill differences are unresolved;
**strictly zero slowdown is not proven**, and the P2 interval includes losses
larger than 1%. Retention uses the existing 1% median other-phase tolerance,
positive paired decode intervals, unchanged arithmetic and modest implementation
size. There is no arbitrary minimum gain threshold. The wider targets of 12K
prefill / 280 decode remain unmet.

The tested final source files are copied unchanged into the repository. No
commit or push was performed. The next selected baseline is recorded in the
new evidence root's `SELECTED_PARENT.json`; the preceding campaign is preserved.

## Reproduction

Evidence root:
`/home/sebastian/Development/padra-performance/2026-10-13-attention-bf16`.

- `ENTRY.json`: pinned parent and owner-approved precision policy.
- `operator/`: three-way native operator harness and experimental kernels.
- `model-A/`: BF16 O prototype, development metrics and exploratory timing.
- `FLASHINFER_OPERATOR.json`, `flashinfer_screen.py`: valid same-input comparison.
- `model-B/`: first exact-merge prototype. Later exploratory timing blocks overlap
  a CPU build; all samples are retained and are not the final acceptance series.
- `model-final/`: exact-merge integration, ON/OFF builds, qualification and final
  performance protocols/results. Final timing runs have no concurrent builds.
- `jobs/`: commands, timeouts, bounded 20 GiB/zero-swap memory receipts and logs.

Failed compiles, an operator harness fixture-index typo, a sanitizer filter
syntax error, and OFF-harness differences between logged hashes and raw heads
were repaired without discarding valid benchmark observations. The original
failed attempts remain available. No clock/power settings were changed.
