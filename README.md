<p align="center">
  <img src="assets/padra_logo.svg" alt="Padra logo" width="160">
</p>

# Padra

A standalone C++20/CUDA inference server for
**JetBrains/Mellum2.1-12B-A2.5B-Thinking**, using admitted native EXL3 K3, K4 or K6 checkpoints.
Padra is inspired by Padparadscha; its name is independent of GPU memory size.

The server includes native tokenization, the original model chat/thinking template,
GPU sampling with frequency/presence/repetition penalties, Chat Completions,
Responses, streaming, function calls and bounded session/KV reuse. Weights remain
GPU-resident. Default context is 131,072 tokens with BF16 KV. Explicit
[16/12/8 GB profiles](docs/gpu-profiles.md) provide 131K context and MTP with
K6/BF16-KV, K6/Q8_0-KV and K3/Q8_0-KV respectively. Physical 8/12 GB GPUs are
not qualified; their memory ceilings are checked on the tested 16 GB GPU.

Optional [Mellum MTP](docs/mtp.md) supports BF16, FP8 or EXL3 K4 draft experts
and fixed D1–D4 speculation with K3/K4/K6 targets and BF16, Q8 or Q8_0 KV.
The GPU profiles select D1 and EXL3 draft experts. The direct server defaults to
ordinary decoding; MTP requires a separately prepared native draft artifact.

K3, K4 and K6 use FP16 MMA expert kernels with rate-specific packed-weight decoders
and BF16 activation boundaries. All retain FP8 attention projections and BF16
routers and embeddings. New profiles use a stored [shared Q8 head](docs/mtp.md)
for ordinary and MTP execution, without runtime head conversion or duplicate
GPU weights. Legacy BF16-head checkpoints remain loadable as a reference.
K6 requires more VRAM for its larger
expert weights.

This repository contains serving source, build files, server regression tests and
required licenses. A prepared native artifact is required; weights and conversion
outputs are not distributed here. The runtime uses the `mellum16` namespace and
the pinned native K3/K4/K6 artifact identities.

## Pascal / portable SIMT backend

The separate SIMT backend supports SM61 and newer, including the GTX 1070,
with CUDA Toolkit 12.x for Pascal. It uses the pinned EXL3 K4 artifact,
BF16 KV and an 8192-token default context with 256-token chunks. MTP,
Q8/Q8_0 KV and the stored shared Q8 head require additional SIMT porting;
the SM120 GPU profiles above do not enable these features on Pascal.

```sh
cmake -S . -B build/sm61 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPADRA_CUDA_ARCHITECTURE=61 -DPADRA_CUDA_BACKEND=simt
cmake --build build/sm61
```

Architecture auto-detection selects the SIMT backend for non-SM120 devices.
An explicit target is required when cross-compiling. CUDA 13 cannot compile
Pascal; use a CUDA 12.x compiler and toolkit.

## Build

Linux requirements: CMake 3.28+, Ninja, C++20 compiler, CUDA 12.8+ with SM120 support,
Rust 1.89+ and Cargo. The tested toolchain is CUDA 13.4 / GCC 15 / SM120a.
Dependencies are prepared automatically: CMake downloads the pinned CUTLASS
headers, and the build runs Cargo for the locked native tokenizer. The first
build needs internet access; no separate CUTLASS build, Git submodule setup or
Python preparation script is required.

```sh
cmake --preset release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15
cmake --build --preset release
```

Use compiler paths appropriate to your installed CUDA toolchain. CUDA and native
text are on by default. `PADRA_ENABLE_CUDA=OFF` and
`PADRA_ENABLE_NATIVE_TEXT=OFF` permit a host compile check; such a build cannot run
native model inference.

Server regressions are registered with CTest and run without weights or CUDA:
`ctest --preset release`. See [tests](tests/README.md) for
host/sanitizer commands and coverage. All owned C++ targets use strict warnings;
[host CI](.github/workflows/host.yml) covers Debug/Release and native text builds.

For an offline build, set `-DPADRA_CUTLASS_ROOT=/path/to/cutlass` to the pinned
source tree or an existing prepared header cache, and use
`-DPADRA_CARGO_OFFLINE=ON` with an already populated Cargo cache. Local CUTLASS
headers receive the same hash checks as downloaded headers. See
[Dependencies](DEPENDENCIES.md) for versions, licenses, local HTTP adaptations
and cache/toolchain options.

`PADRA_KERNEL_PROFILE` selects the CUDA kernel set:

- `baseline-v1` is the default for builds without a preset.
- `exact-fusions-v1` is selected by the `release` preset for K3/K4/K6 ordinary
  and MTP execution. K4 uses K64 EXL3 prefill transport; BF16 KV uses fused
  normalization/RoPE/cache storage and per-head attention merge. Quantized KV
  uses its format-specific attention kernels.

The executable is `build/release/padra-server`. The `release-k4` preset remains
available with its existing `build/release-k4` output directory.

Global attention, native Q8 chunk transport and EXL3 prefill transport have
independent build selections; see [prefill profiles](docs/attention-benchmark.md).

The compatible `PADRA_EXL3_PREFILL_ASYNC64=ON` alias selects `exact-fusions-v1`
when `PADRA_KERNEL_PROFILE` is empty. Unsupported profiles and `baseline-v1`
combined with that alias set to `ON` fail configuration. `/v1/models` exposes the
resolved build profile.

## Run

```sh
build/release/padra-server \
  "$PADRA_ARTIFACT" "$PADRA_TOKENIZER_JSON" "$PADRA_TOKENIZER_SHA256" 8080
```

Arguments:

```text
ARTIFACT TOKENIZER_JSON TOKENIZER_SHA256 PORT
[GENERATION_TIMEOUT_MS] [CONTEXT_CAPACITY] [PREFILL_CHUNK] [DEVICE_BUDGET_GIB] [KV_FORMAT]
[--mtp-artifact DIRECTORY] [--mtp-depth 1..4] [--head-format bf16|q8] [--config JSON]
```

Defaults: 600000 ms, 131072 tokens, chunk 2048, actual-free-memory admission,
and the head format stored in the checkpoint.
Explicit chunks 512/256 and ceilings 8/12 GiB are supported. Admission includes a
700 MiB reserve and a target session snapshot in the selected KV format.
BF16, Q8 and Q8_0 support prefix reuse. MTP also reserves its draft state and
snapshot; see the [MTP contract](docs/mtp.md). `KV_FORMAT` defaults
to `bf16`; opt-in `q8` and `q8_0` reduce cache memory with lossy KV storage. See the
[KV cache contract](docs/kv-cache.md). Unsupported or oversized configurations
fail explicitly; no silent context/precision/chunk/CPU-offload fallback.

Use an admitted native K3, K4 or K6 artifact and the matching original
source tokenizer. Source revision: 92ddae9fc7665e9f801d141d2e5a6b2caf2460c4.
Tokenizer SHA256: 58548a346eb073e5132bf7d8ad17dc6971bca36ade378ca4d2bfbc49bf60da2a.

The [model repository](https://huggingface.co/danmoreng/Mellum2.1-12B-A2.5B-Thinking-Padra-EXL3)
provides K3/K4/K6 targets with stored shared Q8 heads and the EXL3 MTP draft.
See [model setup](docs/models.md) for downloads,
verification, launch commands and memory requirements.

Padra exposes an OpenAI-compatible API server on `127.0.0.1`, including Chat
Completions and Responses. See the [API guide](docs/api.md) for client setup and
usage, and [API support](docs/api-support.md) for supported features and limitations.

## Status

Experimental. Full model/task quality, full-context performance, sustained load
and deployment behavior are not qualified. Host tests cover bounded server/API
behavior; native CUDA/Rust HTTP smoke tests cover serving with the real artifact.
Other GPU architectures and physical 8/12 GB cards are not qualified.

Padra is licensed under [Apache 2.0](LICENSE). See [NOTICE](NOTICE) and
licenses/ for third-party attribution.
