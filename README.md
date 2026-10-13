<p align="center">
  <img src="assets/padra_logo.svg" alt="Padra logo" width="160">
</p>

# Padra

A standalone C++20/CUDA inference server for
**JetBrains/Mellum2.1-12B-A2.5B-Thinking**, using the native EXL3K4 checkpoint.
Padra is inspired by Padparadscha; its name is independent of GPU memory size.

The server includes native tokenization, the original model chat/thinking template,
GPU sampling with frequency/presence/repetition penalties, Chat Completions,
Responses, streaming, function calls and bounded session/KV reuse. Weights remain
GPU-resident. SM120 defaults to 131,072 tokens with BF16 KV; the SIMT backend
defaults to 8,192 tokens. 8/12 GiB memory ceilings are explicit profiles.
Alternative-bpw checkpoints and complete-model Pascal qualification remain pending.

This repository contains serving source, build files and required licenses only.
Historical tests, reviews, benchmarks, captures and conversion campaigns remain in
the separate research project; checkpoint-free portability tests live in `tests/`.
A prepared native artifact is required; weights and
conversion outputs are not distributed here. Internal namespaces/checkpoint IDs
retain their legacy names for compatibility.

## Build

Linux requirements: CMake 3.28+, Ninja, a CUDA-compatible C++20 host compiler,
Rust 1.89+ and Cargo. Pascal needs CUDA Toolkit 12.x (tested with 12.4 / GCC 13);
CUDA 13 removed compilation for targets older than SM75. SM120 needs CUDA 12.8+
with SM120 support. The original measured toolchain is CUDA 13.4 / GCC 15 / SM120a.
Dependencies are prepared automatically: the SM120 build downloads pinned CUTLASS
headers, and the build runs Cargo for the locked native tokenizer. The first
build needs internet access; no separate CUTLASS build, Git submodule setup or
Python preparation script is required.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15
cmake --build build -j2
```

Use compiler paths appropriate to your installed CUDA toolchain. CUDA and native
text are on by default. `PADRA_ENABLE_CUDA=OFF` and
`PADRA_ENABLE_NATIVE_TEXT=OFF` permit a host compile check; such a build cannot run
native model inference.

`PADRA_CUDA_ARCHITECTURE=auto` (default) queries CUDA device 0 with a host-only
probe, honoring `CUDA_VISIBLE_DEVICES` without requiring `nvidia-smi`.
Set one target explicitly for cross-compilation or builds without an installed GPU:

```sh
# GTX 1070 / Pascal, with the local CUDA 12.4 host compiler:
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPADRA_CUDA_ARCHITECTURE=61 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13
cmake --build build -j2

# RTX 5080 / Blackwell (requires an appropriate CUDA toolkit):
# add -DPADRA_CUDA_ARCHITECTURE=120a to the configure command
```

`PADRA_CUDA_BACKEND=auto` selects the existing optimized Tensor Core backend
for SM120/SM120a, and a portable SIMT backend for other SM61+ targets. A numeric
`120` selection uses `120a` for this backend, as its FP4/MMA instructions require
the accelerated target. Explicit
`simt` enables comparison on newer GPUs; `sm120` requires a 120/120a target.
The compiler must support the chosen target. Other architectures currently use
the SIMT backend; additional architecture-specific Tensor Core backends are future
work. A single GPU target is built per binary.

The SIMT backend keeps the original K4 payload, FP8 attention weights/activations
and BF16 KV storage. It uses selected EXL3 FP32 FMA kernels, compensated scalar
attention and reusable single-token CUDA graphs for both prefill and decode.
It does not download CUTLASS or compile SM120/FP4/MMA/`cp.async` kernels.
Prefill chunk size controls progress/cancellation commit boundaries; its workspace
is independent of that size. Session append and local KV snapshot/restore use
the existing worker protocol. Reduction order differs from the Tensor Core path;
complete-model logits, quality and speed need the original checkpoint for validation.
Q8 KV and the SM120 research capture/teacher-forcing options are currently unsupported
in SIMT and fail explicitly.

Checkpoint-free checks (the GPU test requires a device compatible with the target):

```sh
cmake -S . -B build -DPADRA_BUILD_TESTS=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

For an offline build, set `-DPADRA_CUTLASS_ROOT=/path/to/cutlass` to the pinned
source tree or an existing prepared header cache, and use
`-DPADRA_CARGO_OFFLINE=ON` with an already populated Cargo cache. Local CUTLASS
headers receive the same hash checks as downloaded headers. See
[Dependencies](DEPENDENCIES.md) for versions, licenses, local HTTP adaptations
and cache/toolchain options.

On the SM120 backend, optional `-DPADRA_EXL3_PREFILL_ASYNC64=ON` enables
experimental K64 weight/input
transport for EXL3 K4 prefill batches of at least four tokens; smaller batches
retain the existing SIMT path. Initial paired 2K/4K measurements with BF16 KV and
chunk 2048 showed about 3.5%/5.2% faster prefill. The subsequent P2 weights-as-A
mapping added about 0.8–1.0% at P4000/G1000 in separate six-block comparisons.

The option now also includes the D2 compensated register-row decode postnorm.
Extending K64 to short batches repairs the earlier natural-prompt prefill loss:
against P2, final paired measurements show 10.6–11.5% faster short prefill and
0.56%/0.62% faster decode at P4000/G1000 and P2000/G500. Long prefill has no
resolved change. The checked full-vocabulary outputs and session states match;
no extra buffers or launches are added. The option remains off by default.
These RTX 5080 Laptop results do not qualify full-context performance or resolve
the open quality gates below. See [the D2 report](PERFORMANCE_D2.md) for evidence,
uncertainty and validation scope, and [the performance plan](SOL_PERFORMANCE_PLAN.md)
for the retained experiment history.

## Run

```sh
build/padra-server \
  "$PADRA_ARTIFACT" "$PADRA_TOKENIZER_JSON" "$PADRA_TOKENIZER_SHA256" 8080
```

Arguments:

```text
ARTIFACT TOKENIZER_JSON TOKENIZER_SHA256 PORT
[GENERATION_TIMEOUT_MS] [CONTEXT_CAPACITY] [PREFILL_CHUNK] [DEVICE_BUDGET_GIB]
```

Defaults: 600000 ms and actual-free-memory admission. SM120: 131072 tokens,
chunk 2048. SIMT: 8192 tokens, chunk 256. Explicit arguments override build defaults.
Explicit chunks 512/256 and ceilings 8/12 GiB are supported. Admission includes a
700 MiB reserve and 42 MiB session snapshot. Unsupported or oversized configurations
fail explicitly; no silent context/precision/chunk/CPU-offload fallback.

Use an admitted mellum16-exl3-k4-v1/native artifact and the matching original
source tokenizer. Source revision: 92ddae9fc7665e9f801d141d2e5a6b2caf2460c4.
Tokenizer SHA256: 58548a346eb073e5132bf7d8ad17dc6971bca36ade378ca4d2bfbc49bf60da2a.

The listener binds 127.0.0.1. OpenAI clients use http://127.0.0.1:8080/v1 and model
ID mellum21-exl3-k4-research. Text/function subsets of /v1/chat/completions and
/v1/responses are implemented; forced/strict-schema tools, media, background mode
and public authentication/TLS deployment are unsupported. Responses supports stored
previous_response_id and manual item replay. Retention is bounded and process-local;
the server executes no generated functions. Reasoning is returned separately.

## Status

Experimental: source/NLL/SQL/task and full product-quality gates remain open.
The research version passed bounded native Python/TypeScript server workflows;
long/load/expiry/concurrency/deployment qualification remains unfinished. Source
extraction/build checks are not new model-inference or performance qualification.
SM61 builds and synthetic kernel tests run on a GTX 1070 Max-Q (8 GiB).
With the original EXL3 K4 checkpoint, bounded SM61 checks cover greedy and seeded
sampling, streaming, cancellation/recovery, and local KV snapshot restore after
the 1024-token ring wraps. Context allocation and short requests pass at 8192
and 16384 tokens. These checks do not qualify long-context quality or other GPUs.
Run `python3 tests/checkpoint_smoke.py --base-url http://127.0.0.1:8080 --long`
against an already running server to repeat the bounded checks.
See [Pascal validation and benchmarks](docs/pascal-sm61-validation.md) for the
real-checkpoint results, 2000/1000 and 4000/1000 timings, and llama.cpp comparison.

Padra is licensed under [Apache 2.0](LICENSE). See [NOTICE](NOTICE) and
licenses/ for third-party attribution.
