# vLLM EXL3 comparison — 2026-10-13

vLLM is installed and successfully runs Mellum through its native model class
with Padra's exact EXL3 expert tensors and FP8 attention weights. On this machine,
the tested EXL3 plugin is substantially slower than Padra, especially at prefill.
The earlier vLLM INT4 result was not an EXL3 comparison. A fresh native INT4 run
also shows that the old Transformers model implementation was not responsible
for a large performance penalty on these workloads.

Follow-up: the [SGLang decode investigation](SGLANG_DECODE_FINDINGS.md) finds
that selecting FlashInfer raises native INT4 vLLM from about 251 to 285 decode
token/s. The INT4 results below describe its earlier FlashAttention configuration,
not the fastest vLLM configuration tested. The EXL3-plugin rows remain unchanged.

## What the previous benchmark actually used

The 2026-10-12 comparison used exact saved prompt token IDs, one sequence,
8192 context capacity, a 2048-token prefill budget, greedy decoding and no prefix
cache hits. It measured runtime/engine interfaces, not a common HTTP endpoint.

- vLLM 0.31.0: explicitly `model_impl=transformers`, `TransformersMoEForCausalLM`,
  Marlin experts, FlashAttention 2 and BF16 KV. The weights were the community
  INT4/G32 W4A16 checkpoint, not EXL3. Historical P4 prefill/decode: 13,761/251 tok/s.
- SGLang 0.5.21: native `MellumForCausalLM`, WNA16 Marlin MoE, FlashInfer, BF16 KV,
  and the same community INT4 checkpoint. Historical P4: 12,818/291 tok/s.
- SGLang's old 2K timing series was incomplete after a restart; it does not support
  a steady-state ranking. The old vLLM 2K warmup was in a different process after
  a harness failure. The new series below uses one warmup and three retained runs
  in one persistent process per engine. SGLang has not been rerun here.

## Working EXL3 setup

The [yeasah EXL3 plugin](https://github.com/yeasah/vllm-exl3-plugin) registers an
out-of-tree vLLM quantization backend using exllamav3 kernels. Its MoE path is
implemented through three `exl3_mgemm` calls per layer. The project's exllamav3
fork disables a hand-written SM90+ barrier by default to avoid documented MoE
hangs. This fork was built locally; `EXL3_SM90_BARRIER` was not enabled.

Pins used here:

| Component | Identity |
| --- | --- |
| vLLM | Official installed 0.31.0, `db9527a46873454610df6dbedf79a36d6bf1a7f6` |
| Model implementation | Native `MellumForCausalLM`, explicitly `model_impl=vllm` |
| EXL3 plugin | `1a73cefe66bae9cb1f00372cf9e3e559912817f1` |
| Plugin exllamav3 fork | `2ce30866316606f40a18bce2937db07f2e7fe14d`, package 1.4.3 |
| Torch / Transformers | 2.13.0+cu130 / 5.17.0 |
| Extension build | CUDA 13.4, SM120, two compilation workers |
| GPU / driver | RTX 5080 Laptop, 16,303 MiB, driver 615.71.09 |
| Padra | Commit `856a04f`, qualified ASYNC64=ON P2+D2 composition |

The plugin documentation describes a patched vLLM fork, but our native Mellum
case works with the existing official 0.31.0 installation without patching vLLM.
This does not imply that its other supported architectures need no patches.
Mellum's native implementation is also visible in the
[official v0.31.0 source](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/models/mellum.py).

The new environment references the previous vLLM environment's packages read-only
through a `.pth` file. New packages live in the separate comparison environment.
The original installations, checkpoints and Padra implementation were not modified.
All build/GPU jobs used a 20 GiB host-memory cap, zero swap and explicit timeouts.

## Making the weights match

Our original exported EXL3 checkpoint has original BF16 nonexpert weights, whereas
Padra's native artifact has FP8 attention. Loading that export directly would only
match the experts. Instead, `export_matched.py` created a separate HF-compatible
view of the native artifact:

- All 5,376 experts retain their original trellis and FP16 sidecars. Every trellis,
  `suh` and `svh` byte was compared with the corresponding native artifact region;
  expert safetensors are symlinked, not requantized.
- All 112 attention projection matrices retain the native E4M3 bytes. BF16 row
  scales are widened exactly to FP32 for vLLM's serialized scale convention.
- Embeddings, head, routers and norms retain the native BF16 values.
- A small comparison-only `exl3_padra` configuration selects the existing EXL3
  method for experts and vLLM's existing `CompressedTensorsW8A8Fp8` method for
  attention. The latter actually selects `CutlassFP8ScaledMMLinearKernel`.
  No new projection kernel was written.

Thus the represented quantized weights match, including attention. Activation
rounding, Hadamard execution, routing arithmetic, reductions and attention kernels
remain engine-specific. This is not a bit-identical arithmetic comparison.
Both engines use BF16 KV, no CPU weight offload and no speculative decoding.
vLLM reserves a 1 GiB KV pool; Padra uses its own exact inventory. The admitted
request capacity is 8192 in both cases, but allocator footprints are not identical.

## Fresh measurements

Each row uses one complete warmup and three retained runs per workload, with the
same saved P2000/G500 and P4000/G1000 token IDs. All requested output counts were
reached, prefix hits were zero and repeated output IDs were stable within each
engine. First generated tokens agree across engines; full free generations need
not match. First-token selection belongs to prefill; decode counts 499/999 steps.
Loading, compilation, warmup and tokenization are excluded from phase rates.

| Engine / weights | P2K prefill | P2K decode | P4K prefill | P4K decode |
| --- | ---: | ---: | ---: | ---: |
| Padra, current EXL3 K4 + FP8 | 11,024.70 | 256.81 | 10,331.15 | 252.84 |
| vLLM native + EXL3 plugin, matched EXL3 K4 + FP8 | 605.93 | 178.74 | 599.45 | 176.73 |
| vLLM native, community INT4 G32 | 14,059.15 | 254.67 | 13,671.91 | 250.68 |

All values are median token/s. At P4, retained EXL3-vLLM ranges were
599.36–600.44 prefill and 176.73–176.77 decode; Padra ranges were
10,301.34–10,384.06 and 251.98–253.31. Padra is 17.23x faster at prefill and
1.431x at decode than this EXL3-vLLM configuration. Against native INT4, Padra
still trails at prefill. The small native-versus-historical-Transformers INT4
rate differences are not a statistical equivalence result.

These are sequential same-device runs, not randomized paired blocks. No clocks
or power limits were changed. vLLM measures scheduled-to-first-token and
first-to-last-token engine timestamps; Padra uses its native GPU-ready runtime
phase timers. They are not identical HTTP latency measurements. No best-of
selection or extra retained-run sweep was performed.

## Why this plugin's prefill is slow

A separate P2000/G1 diagnostic, after an unprofiled warmup, recorded aggregate CUDA
kernel durations without writing a raw trace. Its 84 `exl3_mgemm` launches took
3,245.79 ms out of 3,292.68 ms total kernel time: **98.58%**. FP8 CUTLASS projection
kernels took about 17.13 ms. The slowdown is in the expert kernels, not weight
loading, the FP8 adapter or only Python scheduling.

The adapter repeats the input for each selected expert and passes tensors shaped
`[tokens * top_k, 1, features]` to `exl3_mgemm`. It does not group prompt rows by
expert into a batched prefill GEMM as Padra does. The source and profile together
identify the inefficient expert path; this is not a claim that every EXL3
implementation or a future vLLM plugin must have this speed. No new performance
optimization was attempted as part of this benchmark.

## Output checks and limitations

The matched EXL3 run passed four eight-token smoke requests at prefix lengths
32/1023/1024/1025, followed by full 98,304-token-vocabulary comparisons on 48 fixed
historical teacher prefixes. Source and Padra top-1 tokens agree on 48/48 positions.
Mean source KL is 0.006916; mean KL from the historical Padra output distribution
is 0.001563. Mean source extra NLL is +0.013956 nats. Full logprob hashes and
per-case metrics are retained, without dumping new raw tensors.

The native INT4 check agrees with source top-1 on 47/48 positions, with mean source
KL 0.013683 and extra NLL +0.022099 nats. These small historical screens establish
basic functioning and expose numerical differences; they are not broad task,
long-context or full-model-quality qualification. No new executable coding-task
suite was run. Earlier Padra quality failures remain open.

The initial EXL3 launch successfully loaded, compiled and captured graphs, but our
custom-function inspection RPC was rejected by vLLM's serialization policy before
smoke/timing requests. The harness was changed to the built-in named
`get_model_inspection` RPC. No unsafe serialization setting was enabled. The first
startup took about two minutes; subsequent runs reused exllamav3's autotune cache.
Failed-job logs remain alongside the successful run. No engine processes were
left resident after the comparison.

## Local reproduction and evidence

Workspace:
`/home/sebastian/Development/padra-engine-comparison/2026-10-13-exl3`.

- `ENTRY.json`, `SOURCE_SEAL.json`, `FINAL_SOURCE_PINS.json`: scope and source pins.
- `MATCHED_EXPORT.json`: exact region comparison and matched checkpoint path.
- `export_matched.py`, `matched-plugin/`: format adapter and quantization composition.
- `bench.py`, `quality.py`: native vLLM benchmark and compact quality checks.
- `matched-attempt2/`: successful EXL3 dispatch inspection, config, quality and runs.
- `native-awq/`: native INT4 control with the same harness.
- `padra/runs.jsonl`, `RESULT.json`: fresh Padra controls and comparison summaries.
- `profile_prefill.py`, `PREFILL_PROFILE.json`: aggregate expert bottleneck evidence.
- `jobs/`: exact commands, logs, exit codes and memory receipts, including failures.

To repeat the EXL3 run, choose fresh job/output names and use the existing bounded
runner and isolated environment:

```sh
cd /home/sebastian/Development/padra-engine-comparison/2026-10-13-exl3
python3 run_job.py matched-repeat 600 env \
  COMPARISON_CHECKPOINT=/home/sebastian/.cache/huggingface/derived/mellum16/92ddae9fc7665e9f801d141d2e5a6b2caf2460c4/mellum16-exl3-k4-v1/vllm-exl3-fp8-v1 \
  VLLM_DISABLE_COMPILE_CACHE=1 MAX_JOBS=2 TORCHINDUCTOR_COMPILE_THREADS=2 \
  CUDA_HOME=/opt/cuda env/bin/python bench.py matched-repeat
```

This runs native vLLM offline inference, not an HTTP server. Do not overlap GPU
runs. The benchmark and quality screen are complete; starting a serving process
or optimizing the plugin is separate work.
