# Native global attention experiments

Date: 2026-10-13. RTX 5080 Laptop, SM120a, CUDA 13.4/GCC 15.
Parent: the selected ASYNC64=ON exact per-head merge from
[PERFORMANCE_ATTENTION.md](PERFORMANCE_ATTENTION.md).

## Decision and current absolute throughput

Keep the existing selected implementation. Two native FlashInfer-based global
attention prototypes improve exploratory decode throughput, but both fail the
predeclared development quality gates. Their small speed gain is **not** the
reason for rejection. No new production CUDA changes, dependency, default,
commit or push are introduced by this investigation. The prototypes, frozen
sources, binaries and measurements remain in the evidence directory below.

The retained implementation's six-block final measurement remains:

| Workload | Prefill token/s | Decode token/s |
| --- | ---: | ---: |
| P2000/G500 | 10,820.03 | 259.61 |
| P4000/G1000 | 10,346.96 | 256.05 |

P is input length and G is generated length. Decode throughput counts G-1
actual decode steps; model load/upload and graph preparation are excluded.
These are the selected optimized build's rates, not the public default OFF.
The previous report retains uncertainty and small negative prefill observations.
New controls below fluctuate somewhat around these rates; compare each candidate
with its own contemporaneous control, not across separate campaigns.

## Native implementation

The pinned local FlashInfer 0.6.18 headers supply its tensor-core attention device
implementation. A separate CUDA translation unit wraps it for the seven global
layers, with 64-key tiles and at most 16 partitions. Every graph replay reads the
live position from Padra's device state. Global KV is already contiguous, so the
wrapper needs no paged indices, Python, host planning or KV copies during replay.
The 21 local layers and the prefill dispatch retain their parent implementation.

The complete path packs the existing BF16-valued Q into BF16 storage, runs split
attention, merges the partial results into BF16, then runs the parent's exact FP8
row quantization and original paired FP8 O projection. EXL3 K4 expert weights,
FP8 Q/K/V/O weights, BF16 KV and top-8 routing are unchanged. Both variants use
FP32 sums and softmax statistics, as authorized by the owner.

- G1 (`model`): BF16 partial outputs between split attention and merge.
- G2 (`model-f32`): FP32 temporary partial accumulators, removing G1's additional
  intermediate rounding. Q/K/V and final attention output remain BF16. This is
  internal accumulation storage, not FP32 model weights or a new activation mode.

The wrappers reuse the existing scratch arena (141,312 bytes for G1, 272,384 for
G2), with a capacity guard of at least 512. Required/resident/KV memory inventories
in the measured full-model runs match their controls. Kernel entry points are
primed before capture. These are experimental integrations, not fully qualified
production backends or a claim about all FlashInfer configurations.

## Inclusive operator results

These timings include Q packing, attention, merge, FP8 encoding and the same O
projection. Three alternating rounds, 256 graph replays per measurement, using
actual layer-27 Q/K/V snapshots. Units are microseconds; smaller is better.

| Experiment | Position | Parent | Candidate |
| --- | ---: | ---: | ---: |
| G1 | 4000 | 28.602 | 21.189 |
| G1 | 4990 | 31.475 | 22.250 |
| G2 | 4000 | 26.202 | 19.472 |
| G2 | 4990 | 28.776 | 20.543 |

The separate G1/G2 sessions have different absolute baselines. Both show a useful
paired operator improvement. Only seven of 28 layers use global attention, and
attention is only one part of each layer; this cannot imply a similar percentage
gain for the complete model. Local operator outputs remain unchanged.

At position 4000, attention relative L2 error against an independent FP64 oracle
is 0.001719 for the parent, 0.001905 for G1 and 0.001724 for G2. At 4990 it is
0.001626, 0.001894 and 0.001618 respectively. G2 removes most of G1's extra local
error, but this does not establish model-level quality.

## Complete-model speed screen

Each variant has two balanced AB/BA blocks, one same-shape warmup plus two retained
requests per shape/process. All P2 requests precede P4 within each process.
Chunk 2048, capacity 8192; all requests reach G500/G1000, with one model load and
weight upload, no cached prompt, no cancellation. No CPU builds overlap timing.
All valid samples are retained. This is screening, not the final six-block
acceptance protocol or a confidence-bound claim.

| Variant | Workload | Control prefill | Candidate prefill | Control decode | Candidate decode |
| --- | --- | ---: | ---: | ---: | ---: |
| G1 | P2000/G500 | 10,869.38 | 10,893.39 | 260.63 | 261.92 |
| G1 | P4000/G1000 | 10,482.26 | 10,443.31 | 256.25 | 258.32 |
| G2 | P2000/G500 | 10,907.88 | 10,859.12 | 260.10 | 261.90 |
| G2 | P4000/G1000 | 10,453.74 | 10,426.91 | 256.37 | 258.24 |

Units: token/s. Ratios of marginal medians give P4 decode +0.81% for G1 and
+0.73% for G2. P4 prefill is -0.37%/-0.26%; P2 prefill is +0.22%/-0.45%.
Unchanged prefill dispatch does not erase the observed timing differences.

## Quality and safety

Each variant evaluates the same 32 independent development contexts in eight
languages, with 128 genuine next-token targets per context. NLL uses FP64
log-sum-exp over the full 98,304-entry vocabulary. Bootstrap units are contexts,
10,000 replicates, seed 20261013. The cached P2 reference used for scoring was
previously proved identical to all 4,128 heads of the current retained parent.
Both prototypes preserve the 32 prefill heads and change all 4,096 decode heads.

| Metric | G1 | G2 | Required maximum |
| --- | ---: | ---: | ---: |
| Mean target-NLL increase | +0.001486 | +0.001128 | — |
| One-sided 95% upper bound | +0.003778 | +0.003526 | +0.001 |
| Worst language mean increase | +0.006875 (JavaScript) | +0.006713 (Python) | +0.003 |

G1 also exceeds the slice gate for Bash, Java and Go. G2 also exceeds it for SQL
(+0.003722). These results do not prove a broad practical coding regression, but
they do not satisfy the agreed non-inferiority requirements. Better FP64 operator
accuracy or a similar mean is insufficient to override the declared gates.
G2 shows that BF16 partial rounding is not the sole reason for the quality gap;
partitioning, softmax/reduction order and other kernel arithmetic still differ.

Both native operator variants pass 24 actual/boundary cases, independent FP64
checks, direct/graph output parity, canaries and unused-storage poisoning. Their
full operator memchecks report zero errors. G1's split-kernel Synccheck/Racecheck
also pass; its new merge is covered by memcheck, not those filtered checks.
G2's targeted split and merge Synccheck/Racecheck also report zero errors/hazards;
their receipts are recorded in the evidence summary. No full-model racecheck or checkpoint/cancellation qualification
is claimed for these rejected numerical prototypes.

No new source-model KL, executable task or final holdout qualification was run
after the development failures. The separately frozen 32-context holdout remains
unscored: three Java, five Python, four contexts in every other language, because
only three unused eligible Java files were available. Its manifest explicitly
records that limitation. It must not be treated as consumed validation or as a
replacement for source-quality testing. Historical absolute-quality issues remain.

## Next experiments, in order

1. Fuse the BF16 KV write into the four K-head CTAs of `QKNormRopeControlled`.
   Preserve normalization, RoPE and BF16 rounding order; store K/V directly in the
   existing local/global slots. This removes 28 separate `StoreControlled` launches
   per decode token. The prior position-4990 trace spends 41.38 microseconds/token
   in those launches: about 1% of token latency is an upper bound, not a promised
   gain. Check local wraparound, global boundaries, graph/direct and state behavior,
   register pressure and whole-model speed before retaining it. Keep diagnostic
   capture and alternate cache-format dispatches intact.
2. Transfer the useful FlashInfer scheduling ideas incrementally to Padra's own
   global kernel: tile layout/loading and then partition geometry, one change at
   a time. Preserve the parent reduction/rounding where possible. Bit identity is
   not required, but numerical changes still require development, fresh holdout,
   source-KL and task qualification. These two prototypes do not exhaust the design
   space. Further tuning of their graph launch overhead alone cannot fix quality.
3. Revisit normalization after these steps. The older actual-position profile
   attributes about 368 microseconds/token to the input/post-attention normalization
   families together. That is their total cost, not removable overhead. Start with
   data movement and fusion while preserving compensated arithmetic; relaxing
   reductions needs numerical qualification and should remain a separate experiment.

For any survivor, freeze the full composition, run the relevant functional and
quality checks, then balanced six-block full-model measurements including short
prompts. Accept reproducible small gains under the existing other-phase tolerance;
do not substitute operator speed for complete-model evidence.

## Reproduction

Evidence root:
`/home/sebastian/Development/padra-performance/2026-10-13-global-attention`.

- `ENTRY.json`, `FLASHINFER_SOURCE_PINS.json`: parent, artifact, protocol and
  SHA256-pinned local upstream headers (their license notices are preserved).
- `operator/`, `operator-f32/`: native wrappers, oracle/boundary/timing harnesses.
- `model/`, `model-f32/`: full source snapshots, build/harnesses, `DECISION.json`
  with source and binary hashes, `development/RESULT.json`, `performance-screen/`.
- `holdout/MANIFEST.json`, `holdout/contexts.json`: sealed and unscored data.
- `RESULTS.json`, `jobs/`: summary, exact commands, original failed attempts,
  all measurements, sanitizer logs, bounded memory/time receipts.

GPU work is serial, with 20 GiB host RAM, zero swap, two build workers and bounded
timeouts. No clocks/power settings were changed. Prior experiment directories
and the retained parent's `SELECTED_PARENT.json` remain unchanged.
