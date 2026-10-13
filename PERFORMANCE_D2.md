# D2 decode gain with repaired short prefill

Decision, 2026-10-13: retain the compensated D2 postnorm together with short-batch
K64 dispatch in the existing `PADRA_EXL3_PREFILL_ASYNC64=ON` profile. The default
remains OFF. No new runtime switch, resident buffer or kernel launch is required.
The ordinary FP32 norm, fused prefill norms and blocking-wait experiments are not
included. The 12,000 prefill / 280 decode token/s targets remain unmet.

## Why decode affected measured prefill

The original regression was real in the tested request sequence. Six balanced
blocks with two identical P2 control binaries reproduced short-prefill losses of
0.22–0.82% after long generations. Three of four prefill confidence intervals
excluded zero. The gain was not rejected merely for being small.

D2 changes only the decode residual/postnorm kernel. Disassembly of all 235
kernels found identical instructions in the other 234. CUPTI recorded the same
791 kernel invocations, 26 kernel types and resource counts for the measured
short prefills. The old packed EXL3 projection consumed about 60% of GPU time.
Thus the slowdown was not extra prefill launches or a changed prefill kernel.

Unprofiled NVML snapshots after long decode showed D2 SM clocks 8–45 MHz below
matched P2 snapshots, unchanged memory clocks, and an active software power cap.
Power/clock coupling and request history are the leading explanation. These are
cached clock samples, not a continuous in-kernel trace: the exact causal share
is unresolved, and code-placement/cache effects were not independently excluded.
A blocking-wait experiment reduced prefill CPU time by about 90% without repairing
prefill speed, so CPU spinning alone does not explain the loss.

## Implemented repair

The optional K64 path previously sent every batch below 128 tokens to the older
packed K32 implementation. That crossover had not been measured. Reusing the
existing P2 weights-as-A K64 kernel at 4/8/16/32/64/127 tokens gave 1.172–1.287x
inclusive local MoE speedups in layers 0, 14 and 27. Removing this fallback saves
actual prefill work. Batches below four tokens retain their SIMT dispatch.

The decode norm retains the D2 compensated FP32 reduction and explicit BF16
boundaries. It keeps nine residual values per thread in registers and uses all
256 threads for the reduction, replacing the 2,304-value shared row. Its compiled
resources fall from 40 registers / 9,252 shared bytes to 34 registers / 68 shared
bytes. The OFF build keeps the original reduction and profile metadata.

The final binary has the same 235 kernel instruction bodies as the earlier D2
binary after normalizing compiler-generated namespace hashes. The prefill repair
changes host dispatch; no numerical retuning or clock/power adjustment is added.

## Full-model measurements

RTX 5080 Laptop, CUDA 13.4/GCC 15, Release SM120a, EXL3 K4, BF16 KV, chunk 2048,
capacity 8192. Six balanced orders: ABC, CBA, BCA, ACB, CAB, BAC. A is the selected
P2 parent, B this composition, C the original campaign control. Each process runs
four natural warmups followed by four measured natural prompts (G768), then
warmup/measured P2000/G500 and P4000/G1000 pairs. One model load/upload per process,
no cached prompt tokens, all requested generations complete, identical generated
IDs. All valid runs are retained, including the first block's slower long prefill.

Entries are median paired throughput changes versus P2, with 95% paired bootstrap
intervals in brackets. Phase rates exclude model loading and setup.

| Workload | Prefill change | Decode change |
| --- | ---: | ---: |
| Python, 42 prompt tokens / G768 | +10.64% [+10.38, +11.10] | +0.568% [+0.347, +0.666] |
| Rust, 42 / G768 | +11.54% [+10.15, +11.66] | +0.570% [+0.311, +0.652] |
| C++, 44 / G768 | +10.64% [+10.15, +11.14] | +0.612% [+0.388, +0.643] |
| SQL, 58 / G768 | +10.69% [+10.21, +11.06] | +0.574% [+0.415, +0.649] |
| P2000/G500 | +0.069% [-1.207, +0.219] | +0.616% [+0.435, +0.734] |
| P4000/G1000 | +0.083% [-0.861, +1.031] | +0.557% [+0.334, +0.665] |

There is no resolved new long-prefill gain or reproducible long-prefill regression;
these intervals do not prove exactly zero slowdown. Decode improves in every
measured block/workload. No minimum percentage threshold is used for acceptance.

At P4000/G1000, median phase rates are 9,981.67 → 10,005.35 prefill and
250.825 → 252.124 decode token/s. At P2000/G500 they are 10,442.21 → 10,465.07 and
253.860 → 255.361. Ratios of these marginal medians differ from paired medians.
Against the original campaign control, paired P4 prefill/decode changes are
+1.109%/+0.635%; P2K changes are +0.948%/+0.620%.

The combined measured prefill-plus-decode generation span improves by 0.51% at
P4 and 0.57% at P2K (paired speed ratios); natural cases improve by 0.63–0.68%.
This is not HTTP latency and excludes setup, loading and tokenization. Large
short-prefill percentage gains save only about two milliseconds per request.

## Validation and limits

- 18 native short-MoE cases match combined output bits and direct/graph replay.
  An independent FP64 projection oracle covers 45 represented rows at three
  layers/sizes, with exact input boundaries and maximum relative L2 0.000274.
  Targeted memcheck, synccheck and racecheck pass; the unchanged D2 norm retains
  its existing native-state, FP64, finite/extreme-input and sanitizer checks.
- 32 fresh disjoint source files across eight languages, prefixes 16/42/64/127,
  were frozen before evaluation. All 4,128 forced-continuation full-vocabulary
  hashes match P2. Another 8,320 development/previous-holdout heads and 96 legacy
  heads match their recorded controls: 12,544 observations in total. Candidate
  activations/routes are its own; only continuation IDs are teacher-forced.
- All 24 development/holdout task token sequences match control, preserving 22/24
  passes; 12 legacy sequences preserve 10/12. Prior functional grading is reused
  for identical token sequences, not represented as new test execution.
- Separate processes for both prefill execution modes cover 14 prefix lengths
  (1 through 4000, including dispatch/chunk boundaries), with three heads each.
  Decode still uses the selected full-step graph. Cancellation, reset, append
  and checkpoint state comparisons pass.
- Admission, required/resident/KV bytes and short-content outputs match at five
  capacities through 131072, chunks 256/512/2048 and explicit 8/12 GiB ceilings.
  The 8 GiB ceiling rejects identically. This is not 128K-content quality testing
  or qualification on physical 8/12 GiB GPUs.
- A rebuilt OFF profile matches the original metadata, 32 generated tokens and
  four full heads at prompt lengths 33/128/2000/4000.

Bit identity is an observation on these checks, not a newly imposed requirement
or proof for every possible input. Identical checked logits give zero relative
NLL/KL change on those observations; this does not newly establish absolute quality
against the original unquantized model. Fresh holdout is now consumed and must not
be used to tune future numerical candidates. Other GPUs and full-context throughput
remain unqualified. No commit, push or product-default change was made.

## Reproduction and retained evidence

Campaign root:
`/home/sebastian/Development/padra-performance/2026-10-13-sol-numerical`.
All build/GPU jobs use its `run_job.py`: 20 GiB host RAM, no swap, explicit timeout,
normally two build workers. Raw captures remain under the existing 128 MiB budget;
full-vocabulary checks retain hashes instead of new tensor dumps.

- `D2-prefill-diagnosis/REPORT.md`: identical-binary controls, disassembly, CUPTI,
  telemetry, and unsuccessful fusion/wait repairs.
- `D2-short-k64/SUMMARY.json`, `ORACLES.json`: local timing and independent oracle.
- `D2-short-final/performance/{PROTOCOL,ROWS,RESULT,SPAN_RESULT}.json`: frozen
  binaries/source hashes, all final timings, intervals and combined generation span.
- `D2-short-final/fresh-short/`: dataset manifest, pre-evaluation seal and results.
- `D2-short-final/{replay-quality,boundaries,qualification,off-check}/`: checks.
- `D2-short-final/src/`: exact tested source; `build/padra-runtime-probe` is the
  ordinary timing binary, SHA256
  `3ab1059156282cc2852175c5a0bbe54c940ff4f26d1508d6ec3eaecb5044777b`.
- `SELECTED_PARENT.json`: selected source/binary/core pins and evidence links.

The three public CUDA files were copied byte-for-byte from the frozen tested
source and verified against the pre-performance hashes. Build with the documented
repository procedure and `-DPADRA_EXL3_PREFILL_ASYNC64=ON` to enable this composition.
