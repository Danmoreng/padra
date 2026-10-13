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
GPU-resident. Default context is 131,072 tokens with BF16 KV. 8/12 GiB memory ceilings
are explicit profiles; alternative-bpw checkpoints and physical smaller GPUs remain
future qualification targets.

This repository contains serving source, build files and required licenses only.
Historical tests, reviews, benchmarks, captures and conversion campaigns remain in
the separate research project. A prepared native artifact is required; weights and
conversion outputs are not distributed here. Internal namespaces/checkpoint IDs
retain their legacy names for compatibility.

## Build

Linux requirements: CMake 3.28+, Ninja, C++20 compiler, CUDA 12.8+ with SM120 support,
Rust 1.89+ and Cargo. The measured toolchain is CUDA 13.4 / GCC 15 / SM120a.
Dependencies are prepared automatically: CMake downloads the pinned CUTLASS
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

For an offline build, set `-DPADRA_CUTLASS_ROOT=/path/to/cutlass` to the pinned
source tree or an existing prepared header cache, and use
`-DPADRA_CARGO_OFFLINE=ON` with an already populated Cargo cache. Local CUTLASS
headers receive the same hash checks as downloaded headers. See
[Dependencies](DEPENDENCIES.md) for versions, licenses, local HTTP adaptations
and cache/toolchain options.

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

Defaults: 600000 ms, 131072 tokens, chunk 2048, actual-free-memory admission.
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
Other GPU architectures and physical 8/12 GB cards have not been qualified.

Padra is licensed under [Apache 2.0](LICENSE). See [NOTICE](NOTICE) and
licenses/ for third-party attribution.
