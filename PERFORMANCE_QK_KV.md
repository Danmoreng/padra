# Q/K, RoPE and KV fusion; native global query reuse

Date: 2026-10-13. RTX 5080 Laptop, SM120a, CUDA 13.4/GCC 15.
Parent `c1c337f` was committed and pushed to `origin/main` before this work,
as requested.
Same EXL3 K4 artifact, FP8 Q/K/V/O weights, BF16 KV and top-8 routing.
Existing ASYNC64=ON selected build; public default OFF is unchanged.

## Decision

**Retain the exact Q/K-normalization + RoPE + BF16 KV-store fusion.** It gives
reproducible decode gains while preserving the checked full-model outputs.
**Do not add query-register reuse to the retained fusion.** It is implemented and
qualified for exactness, but its incremental complete-model result does not
establish the required gain. The isolated operator and short screen are
insufficient. This is not an arbitrary minimum-speed rejection.

The new selected source is copied unchanged from `fusion/src` in the evidence
root below. `SELECTED_PARENT.json` records its source, binary and core-library
hashes. Existing numerical holdout data remains unconsumed. Historical absolute
source/task quality issues are unchanged; these exact rewrites do not resolve them.

## Point 1: fuse the BF16 KV store

`QKNormRopeStoreControlled` preserves the original compensated RMS reduction,
BF16 rounding boundaries and RoPE operation order. The four K-head CTAs also
write their K/V cache rows. Heads have disjoint addresses, and normal stream/graph
dependencies order all writes before attention. There is no cross-CTA barrier,
completion counter or extra allocation. Original Q/K outputs are still written.
The selected BF16 device-decode path removes one kernel launch per layer,
or 28 per decode token. Diagnostic captures and alternate cache/configuration
paths retain their original dispatch.

The operator harness covers 24 local/global boundary cases through position
16,383, synthetic input scales 1e-4/1/1e4, every untouched cache row, allocation
canaries and direct/graph parity. Parent Q/K and BF16 KV outputs are exact, and
an independent FP64 RMS oracle with declared rounding stages agrees. Operator
medians at positions 4000/4990 improve from about **4.70 to 3.79 microseconds**.
Both arms include the same two device-to-device input resets per replay; those
are harness overhead, not new production work. Registers remain 26, with zero
stack/local spill bytes. All 238 pre-existing kernel SASS bodies are unchanged;
one new fused kernel is added. Prefill code and dispatch are unchanged.

Six final paired blocks versus the committed parent:

| Workload/phase | Control token/s | Candidate token/s | Paired gain | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| P2000/G500 prefill | 10,895.79 | 10,864.71 | +0.025% | -0.503..+0.487% |
| P2000/G500 decode | 259.95 | 261.18 | +0.475% | +0.323..+0.581% |
| P4000/G1000 prefill | 10,428.31 | 10,403.48 | -0.132% | -0.622..+0.089% |
| P4000/G1000 decode | 256.32 | 257.86 | +0.562% | +0.454..+0.686% |

Short natural prompts at G128 give +0.655..+0.798% decode and
-0.457..+0.395% prefill. The small negative prefill observations remain
in the report; strictly zero slowdown is not proven. Retention uses the existing
1% median other-phase tolerance and positive paired decode intervals.

## Point 2: reuse native global Q operands

After inspecting FlashInfer's Q/KV load and iteration schedule, this experiment
moves Padra's 32 packed Q operand registers outside the global KV-tile loop.
This is a native load/reuse adaptation, not a verbatim FlashInfer kernel port;
no new dependency is added. The 32-key tiles, partition count, MMA sequence,
softmax/reduction order, BF16 probability rounding, merge and FP8 O stay intact.
Local attention uses the original specialization. The default-false template
specializations preserve their original SASS after symbol-name normalization.

On actual layer-27 inputs, inclusive attention/merge/FP8/O operator medians are
**26.215 to 25.641 microseconds** at position 4000 and **28.895 to 28.229** at 4990.
All 24 native/boundary cases preserve attention, FP8 bytes/scales and O output
exactly, including independent FP64 checks, poisoned unused KV/scratch and
allocation canaries. Registers rise from 83 to 115; shared storage remains
42,592 bytes and there are no spills. Seven of 28 layers use this global path,
so the implied whole-token headroom is small.

The first two-block standalone screen versus the committed parent gives P4
prefill 10,326.81 to 10,338.49 and decode 255.62 to 255.81 token/s; P2 gives
10,766.08 to 10,744.11 and 259.24 to 259.56. No selection was made from this
screen. The final combined build was independently qualified and measured
against the already qualified **fusion-only** build, isolating the additional
effect of query reuse:

| Workload/phase | Control token/s | Candidate token/s | Paired gain | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| P2000/G500 prefill | 10,937.33 | 10,859.65 | -0.162% | -0.936..+0.336% |
| P2000/G500 decode | 261.33 | 261.12 | -0.032% | -0.161..+0.032% |
| P4000/G1000 prefill | 10,354.10 | 10,355.63 | -0.145% | -0.613..+0.515% |
| P4000/G1000 decode | 257.66 | 257.79 | +0.055% | -0.075..+0.191% |

The same incremental short-prompt comparison gives -0.128..+0.039%
decode and -0.416..-0.028% prefill. These are separate measurement
sessions; do not compare absolute rates across tables or add the marginal
median ratios as if they were one end-to-end measurement.

## Qualification and measurement limits

Fusion-only, query-reuse-only and the combined build each reproduce all 4,128
complete vocabulary heads from 32 frozen contexts, 42 boundary heads in separate
direct/graph processes and ten checkpoint/cancellation/append/reset cases.
Fusion and combined OFF builds additionally reproduce 32 generated IDs, four
complete heads and metadata. Resident, required and KV inventories match.
Full operator Memcheck and targeted new-kernel Synccheck/Racecheck pass for both
changes. There is no full-model racecheck, new full-context qualification or new
CUPTI trace. Kernel-launch removal follows the checked dispatch; it is not a
claim from a new trace. Existing operator sanitizer results apply to the same
unchanged kernel source in the combined build.

Each final long series uses six alternating AB/BA blocks and alternates P2/P4
shape order, with one warmup and one retained request per shape/arm/block.
Decode counts G-1 actual steps; load/upload and graph preparation are excluded.
Chunk 2048, capacity 8192. All requested output lengths are reached, generated
IDs match, no prompt-cache hits occur, and load/upload counts remain one.
Short prompts use two AB/BA blocks, one warmup and two retained requests per
prompt. Confidence intervals bootstrap six paired block ratios with 10,000
replicates, seed 20261013. Paired gains need not equal ratios of marginal medians.
All valid samples, including negative ones, are retained. No build overlaps
performance measurement. No clocks or power settings are changed.

## Reproduction

Evidence root: `/home/sebastian/Development/padra-performance/2026-10-13-qk-kv-global-reuse`.

- `ENTRY.json`: committed parent, scope, artifact and acceptance requirements.
- `operator-fusion/`, `operator-qregs/`: independent checks, timing and resources.
- `fusion/`, `qregs/`, `combined/`: frozen source, build/quality harnesses,
  `SOURCE_BINARY_PINS.json`, complete-head checks and functional results.
- `fusion/performance/`, `combined/performance/`: frozen protocols, all samples,
  telemetry and final paired results; corresponding `short-prompts/` directories.
- `fusion/DECISION.json`, `combined/DECISION.json`, `SELECTED_PARENT.json`:
  explicit decisions and the new baseline. Earlier campaign evidence is unchanged.
- `jobs/`: exact commands, timeouts, 20 GiB host/zero-swap receipts and logs.

GPU jobs run serially with bounded timeouts; builds use two workers. The accepted
implementation and evidence address the requested first two steps. Normalization
arithmetic changes and further global tile/partition variants remain separate,
unqualified work; no new numerical-quality claim is implied.
