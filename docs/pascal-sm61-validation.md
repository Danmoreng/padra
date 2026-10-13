# Pascal SM61 checkpoint validation, 2026-10-13

The first SIMT port runs the original native EXL3 K4 checkpoint on a GTX 1070
Max-Q, but is substantially slower than the existing llama.cpp Q4_K_M setup.
This is a functional baseline; Pascal performance and model quality are not
fully qualified. No checkpoint was regenerated or modified.

Hardware: GTX 1070 Max-Q, 8192 MiB physical VRAM, driver 580.178.04,
i7-8750H, 32 GiB RAM. Padra runtime commit: `b3ca187`.
Pascal build: Release, CUDA 12.4, GCC 13 CUDA host compiler, SM61 SIMT,
native text enabled. SM86 and SM120a cross-builds pass; SM120a uses its original
optimized backend, including a build with async64 enabled. Those builds were
not executed on newer GPUs here.

Checkpoint source revision: `92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`.
Native weights: 7,092,933,120 bytes,
SHA256 `e614375f2eb26c17321365d4bd8ee7aa4b716f6badee171bb2040b85b754804c`.
Tokenizer SHA256:
`58548a346eb073e5132bf7d8ad17dc6971bca36ade378ca4d2bfbc49bf60da2a`.

The real-checkpoint checks pass for repeated greedy Python generation, seeded
sampling with penalties, streaming parity, disconnect cancellation followed by
a successful request, and local KV snapshot restoration. A wrapped snapshot
restores 1100 cached tokens and produces the same answer as a fresh request.
The local attention ring has 1024 tokens. Throughout these tests the model is
loaded/uploaded once and only two GPU graphs are captured. Synthetic CPU-reference
kernel checks pass, with zero compute-sanitizer memcheck errors and racecheck
hazards. Full source/NLL/task quality gates and other-GPU inference remain open.

At 8192 context, chunk256, budget8, and BF16 KV, nvidia-smi reports 7079 MiB used
and 1029 MiB free; this remains stable through both long benchmark requests.
At 16384 context the model loads, the bounded short smoke suite passes, and
7199 MiB is used with 909 MiB free. This does not establish quality or speed
with an actually filled 16384-token prompt.

## Long-sequence performance

Each profile is one measured request after a 32-output-token warm-up. Both
servers use 8192 context, greedy sampling, no prompt-cache reuse and no
speculation. Servers run sequentially. Prompts request a long Python module;
both engines actually produce all 1000 tokens and stop at the length limit.
Native-rendered token IDs are supplied to llama.cpp and independently checked
against its tokenizer: both 2000- and 4000-token sequences match exactly.
Prefill includes first-token selection. Decode rate counts the subsequent
999 forward steps, consistently with this llama.cpp server's timing definition.
Model load, HTTP queuing and tokenization are excluded from phase times.

| Engine | Input / output | Prefill s | Prefill tokens/s | Decode s | Decode tokens/s | HTTP total s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Padra EXL3 K4 SIMT | 2000 / 1000 | 111.410 | 17.95 | 61.458 | 16.26 | 172.872 |
| llama.cpp Q4_K_M | 2000 / 1000 | 2.245 | 890.99 | 17.473 | 57.17 | 19.722 |
| Padra EXL3 K4 SIMT | 4000 / 1000 | 230.814 | 17.33 | 64.229 | 15.55 | 295.050 |
| llama.cpp Q4_K_M | 4000 / 1000 | 4.476 | 893.64 | 17.906 | 55.79 | 22.403 |

llama.cpp is about 50–52 times faster for prefill and 3.5–3.6 times faster for
decode in these configurations. Yesterday's saved short Q4_K_M benchmark
reported 764.10 tokens/s for pp512 and 62.84 tokens/s for tg128 (three repeats);
its short chat check reported 61.30 decode tokens/s.

The comparison includes different weight formats and math/cache profiles:
Padra uses EXL3 K4, FP8 activations and BF16 KV; llama.cpp uses Q4_K_M and Q8_0 KV.
The GGUF source revision is `20b439426f1e997ba575eaccaf4856ac0f76f5d5`.
Thus these are practical setup comparisons, not an isolated engine comparison
at identical quantization or a quality-equivalence result.
llama.cpp build `77757a346` uses batch512/ubatch256, six threads, flash attention,
GPU layers99, CPU MoE0 and fit off. All transformer layers/experts are on GPU;
its normal input embedding is on CPU. VRAM during inference is about 7845 MiB.
Telemetry records stable Padra SM clocks at 1544 MHz and llama.cpp clocks around
1518 MHz. These single-request measurements provide no confidence interval.

The SIMT implementation launches its one-token forward graph for each prompt
token. Chunk256 bounds progress/cancellation, but does not batch matrix work
across tokens. A batched Pascal prefill is therefore the first clear performance
opportunity. Decode needs kernel profiling before selecting optimizations;
these end-to-end timings do not identify an individual kernel bottleneck.

## Repeat and inspect

Raw requests, responses, headers, timing summaries, GPU telemetry, metadata and
server logs are saved at `/srv/agent/workspaces/padra-bench-20261013/`.
The previous llama.cpp results are at
`/srv/agent/workspaces/inference-bench-20261012/results.txt` and
`mellum-Q4_K_M-bench.json`. No system service was started or changed; the temporary
test servers and telemetry process have been stopped.

Start the desired server with the configuration above, warm it up, then run:

```sh
python3 tools/benchmark_http.py --engine padra \
  --base-url http://127.0.0.1:18081 \
  --request /srv/agent/workspaces/padra-bench-20261013/prompt-2000.json \
  --output /tmp/padra-2000-repeat.json --prompt-tokens 2000

python3 tools/benchmark_http.py --engine llama \
  --base-url http://127.0.0.1:18082 \
  --request /srv/agent/workspaces/padra-bench-20261013/llama-request-4000.json \
  --output /tmp/llama-4000-repeat.json --prompt-tokens 4000
```

Corresponding request files exist for both lengths. The script saves raw results
and fails if the prompt/output counts differ or any prompt tokens were reused.
`tests/checkpoint_smoke.py --long` repeats bounded functional checks against a
running Padra server without downloading, converting or loading a checkpoint.
