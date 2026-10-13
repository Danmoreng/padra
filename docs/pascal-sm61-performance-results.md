# Pascal SM61 performance implementation, 2026-10-13

The exact Pascal implementation improves both long profiles substantially, but
**does not meet the llama.cpp targets** in the [performance plan](pascal-sm61-performance-plan.md).
Measured engine code: `640c662`, branch `codex/pascal-sm61-port`. The checkpoint, tokenizer,
EXL3 K4 payload and BF16 KV format were not regenerated or changed. The source
identity and original timings remain in [baseline validation](pascal-sm61-validation.md).

## Repeated long-profile comparison

GTX 1070 Max-Q, SM61, CUDA 12.4 / GCC 13, driver 580.178.04, 8192 context,
Padra physical batch 256. Five warm repetitions per engine and profile; engine
order alternates by repetition. Only one model uses the GPU at a time, with no
concurrent builds or GPU tests. Both engines receive the original frozen input
IDs (independently tokenized and checked against llama.cpp), use greedy selection,
and generate exactly 1000 tokens without prompt-cache reuse or speculation.
Prefill includes first selection; Decode counts the remaining 999 forward steps.
Model loading, tokenization and queueing are outside the phase timings.

Padra uses EXL3 K4 / FP8 activations / BF16 KV; llama.cpp uses Q4_K_M / Q8_0 KV
and a different source revision. This is the practical setup comparison requested
in the plan, not a comparison at identical quantization or a quality-equivalence test.

| Input / output | Phase | Padra median s [min, max] | Padra Tok/s | llama.cpp median s [min, max] | llama.cpp Tok/s | Paired speedup 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2000 / 1000 | prefill | 14.374 [14.284, 14.418] | 139.14 | 2.270 [2.264, 2.284] | 880.91 | [0.157, 0.158] |
| 2000 / 1000 | decode | 28.579 [28.525, 28.604] | 34.96 | 17.521 [17.485, 17.526] | 57.02 | [0.613, 0.614] |
| 4000 / 1000 | prefill | 32.518 [32.440, 32.612] | 123.01 | 4.474 [4.465, 4.492] | 893.96 | [0.137, 0.138] |
| 4000 / 1000 | decode | 30.686 [30.665, 30.708] | 32.56 | 17.924 [17.911, 17.940] | 55.73 | [0.584, 0.585] |

Speedup is `llama_seconds / padra_seconds`; values below one favor llama.cpp.
Intervals use paired bootstrap resampling of the five per-repetition ratios
(10,000 samples, fixed seed 20261013). All four contemporary and historical
plan targets remain unmet. Five pairs are a small sample; the remaining gaps
are much larger than the observed spread.

Compared with the original single-run Padra baseline:

- 2000 prefill: 7.75× faster.
- 2000 decode: 2.15× faster.
- 4000 prefill: 7.10× faster.
- 4000 decode: 2.09× faster.

HTTP totals and first-token headers are retained in each raw response record.
One runner restart failed to bind a reused loopback port before any measurement.
Completed warm pairs were retained; subsequent launches used unique ports.
No time from the failed startup was included. Exact commands, port assignments,
binary/request hashes and the engine order are saved in `final-paired/metadata.json`.

## Implemented packages and current acceptance

| Package | Implementation and evidence | Remaining work |
| --- | --- | --- |
| P0 | Offline real-tensor captures, teacher replay, eager operator events for Prefill/Decode/Head at contexts 128/1024/2000/4000/8000; original unprofiled long controls retained. | A full CPU/GPU timeline and repeatable original long-context operator budget are not complete. |
| P1 | Exact FP32/unsigned-DP4A codec and lookup implementation. All 65,536 states match the original half-FMA codec; production SASS has DP4A and no HFMA2 in these projection kernels. | Speed targets require additional groups to improve. |
| P2 | Real layer-batch 256 Prefill, device active-row state, staged current-chunk KV, commit after causal reads, and splitting at snapshot anchors. Three stable production graphs captured once. | Full long-context model-quality qualification remains open. |
| P3 | Exact FP8 M8/N32 reuse with the original 32-lane partial sums and reduction tree; direct/lookup FP8 conversion checked for all 256 encodings. Separate cuBLAS comparison includes decode/scaling and bounded scratch. | SGEMM changes some BF16 outputs and is not a qualified serving profile. |
| P4 | Stable grouping by expert, M8/N128 weight reuse, original H128/sidecars and ordered Top-8 combine. Tests include empty groups, skew, tails and real Gate/Up/Down dimensions. | Full real-layer routing-distribution measurements and target Prefill throughput remain open. |
| P5 | Exact direct global KV addresses and local 1024-ring masks; Decode prefetch8 retains compensated accumulation order. Split/GQA and batch-prefetch candidates measured, numerically checked and excluded from defaults. | Attention remains a large long-context cost. |
| P6 | Exact lookup codecs, warp H128, faster QKV/O conversion and isolated larger-tile comparisons. | Decode parity remains unmet; no faster Head/sampling profile has been introduced. |
| P7 | Mechanical changes pass bitwise activation/logit and functional regression checks; no alternative-math serving profile is enabled. | Frozen broad corpus, trusted-source NLL, executable task quality and qualification of any changed-math profile remain open. |

Commits separate diagnostics (`a0fd2b8`), codec (`a20cfd1`), batched scheduling
(`cab8bd5`), FP8 reuse (`5c4ac37`), expert grouping (`4ce59ae`), attention
(`09fb277`) and lookup/FP8 tile selection (`640c662`).

## Variant selection

The default expert tile remains M8. At M256 with uniform synthetic routing,
M8 Gate/Up/Down measured about 8.18/8.18/8.36 ms, versus 10.52/10.53/10.54 ms
for the M16/N64 candidate. Skewed routing also favored M8. The M16 implementation
is retained as an explicitly tested experiment, and is not used by the server.

At context 8000, isolated original/global attention measured about 1.56 ms,
direct addressing about 1.27 ms, and direct addressing plus Decode prefetch8
about 1.20 ms. For eight Prefill queries, original/global attention measured
about 18.1 ms versus 5.62 ms with direct addresses; local attention was about
2.05 ms versus 0.83 ms. Batch prefetch8 and split attention were slower and are
not defaults. These are operator measurements, not whole-engine speedup factors.

SGEMM is a separate optional benchmark (`PADRA_BUILD_SGEMM_BENCHMARK=ON`) and
does not add a cuBLAS dependency or workspace to the server. Its speed includes
FP8 decoding and output scaling. Some outputs differ because of accumulation
order; it cannot count toward the target until P7 qualifies a real model profile.

## Numerical, functional and memory checks

- All four SM61 CTest groups pass: GPU profiles, scalar/SIMT kernels, operators and Prefill.
- Final teacher-forced replay matches all 83 original real activation/logit files bit for bit.
- All ten final long Greedy responses match the original saved response messages, with 1000 output tokens each.
- Real prompt lengths 1/3/4/15/16/31/32/63/64/127/128/255/256/257 match original final logits exactly.
- Synthetic attention checks cover poisoned future/tail data, 1024-ring wrap, contexts near 8192 and staged-versus-sequential results.
- Final compute-sanitizer memcheck reports zero errors; focused racecheck reports zero hazards. An earlier broader racecheck hit a 180-second limit and was not counted as passed.
- Real HTTP checks cover greedy code, seeded sampling, streaming, disconnect cancellation/recovery, cached/fresh equivalence and restoration of 1100 cached tokens after ring wrap.
- The same bounded HTTP suite passes with capacity 16384, without filling a 16384-token prompt.
- Model loads/uploads remain one and graph captures remain three in each default server instance.
- SM86 and SM120a (CUDA 13.2, existing async64 backend) cross-build; their host profile tests and the CUDA-disabled host CTest pass. New-GPU kernels/performance were not executed here.

The server keeps the 700 MiB reserve and 42 MiB local snapshot. Batch 256 adds
about 79 MB of owned workspace; there are no per-request GPU allocations or
weight transfers. At capacity 16384, the HTTP instance reports 7,526,936,431
owned bytes and 852,361,216 bytes free after graph capture, above the reserve.
Per-repetition 8192 admission/resident figures are preserved in the final metrics.
Sampled peak VRAM in the final comparison is 7167 MiB for Padra and 7845 MiB
for llama.cpp. Active memory clocks remain 3802 MHz; median active SM clock is
1544 MHz for both engines. Active temperatures range from 56–66 °C for Padra
and 63–66 °C for llama.cpp. Telemetry preserves power and throttle flags (0 or
0x4 in active samples); neither clocks nor power settings were changed.

## Remaining performance budget

At context 4000, final event measurements attribute roughly 12 ms to the three
expert groups, 8 ms to Attention, 6 ms to QKV/O, plus the Head and other work.
The Decode target is about 17.9 ms per forward step. Removing only one of these
groups cannot provide a practical complete solution. Prefill similarly needs
a further large matrix-throughput improvement; reduced host launches alone
cannot close the measured gap. Event timings are diagnostic eager measurements
and are not used as the final throughput result.

The plan is not fully accepted: parity and the listed P0/P4/P7 work remain open.
In particular, no precision change has been promoted to conceal the remaining
performance gap.

## Raw results and reproduction

All configurations, captures, logs, variants, SASS/resource reports and telemetry
are under `/srv/agent/workspaces/padra-pascal-opt-20261013/`. Final records are in
`final-paired/`, with `summary.json` containing every phase sample and interval.
`selected-*` contains final exact-reference captures, events and sanitizer logs.
`expert-tile-comparison.txt`, `attention-batch-comparison.txt` and the separate
`sgemm*-comparison.txt` files retain candidate measurements. The server/request
commands and hashes are in the final metadata; requests come from the unchanged
`/srv/agent/workspaces/padra-bench-20261013/` directory.

```sh
python3 tools/compare_benchmark_runs.py \
  /srv/agent/workspaces/padra-pascal-opt-20261013/final-paired \
  --output /tmp/padra-paired-summary.json
```

For offline diagnosis, build with `PADRA_BUILD_DIAGNOSTICS=ON` and run
`padra-simt-qualify ARTIFACT TOKENIZER_SHA256 CONFIG_JSON OUTPUT_DIRECTORY`.
The JSON specifies `capacity`, `chunk`, `requests` (native token `ids` and
`generate`), `diagnostics`, selected `layer`/`positions`, and optionally
`teacher_tokens`, `reference_codec`, `reference_prefill`, `eager` or
`events_only`. `split_attention=true` explicitly selects the rejected exact
split candidate for research comparison; HTTP requests cannot enable it.

The later experimental update `92f6dff` pads shared BF16 rows to 136 and removes
KV modulo from the split prototype; it does not enable the prototype in the
server. It passes all 83 original captures, CTest and focused memcheck/racecheck.
At context 8000 its global Decode time improves to about 1.28 ms, still above
the chosen 1.20 ms default; eight-query Prefill is about 7.43 ms versus 5.61 ms.
These later candidate measurements are in `attention-padded-direct-comparison.txt`.
The five paired serving measurements above identify the unchanged chosen
production path at `640c662`, independently of this experimental update.
