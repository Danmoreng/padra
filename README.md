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
are explicit profiles. Other checkpoint precisions and physical 8/12 GB GPUs are
not qualified.

This repository contains serving source, build files, server regression tests and
required licenses. A prepared native artifact is required; weights and conversion
outputs are not distributed here. The runtime uses the `mellum16` namespace and
`mellum16-exl3-k4-v1` artifact identifier.

## Build

Linux requirements: CMake 3.28+, Ninja, C++20 compiler, CUDA 12.8+ with SM120 support,
Rust 1.89+ and Cargo. The tested toolchain is CUDA 13.4 / GCC 15 / SM120a.
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

Server regressions are registered with CTest and run without weights or CUDA:
`ctest --test-dir build --output-on-failure`. See [tests](tests/README.md) for
host/sanitizer commands and coverage. All owned C++ targets use strict warnings;
[host CI](.github/workflows/host.yml) covers Debug/Release and native text builds.

For an offline build, set `-DPADRA_CUTLASS_ROOT=/path/to/cutlass` to the pinned
source tree or an existing prepared header cache, and use
`-DPADRA_CARGO_OFFLINE=ON` with an already populated Cargo cache. Local CUTLASS
headers receive the same hash checks as downloaded headers. See
[Dependencies](DEPENDENCIES.md) for versions, licenses, local HTTP adaptations
and cache/toolchain options.

`PADRA_KERNEL_PROFILE` selects the CUDA kernel set:

- `baseline-v1` is the default profile.
- `exact-fusions-v1` is experimental. It includes K64 weight/input transport for
  EXL3 K4 prefill, compensated register-row decode postnorm, per-head attention
  merge, and fused Q/K normalization, RoPE and BF16 KV storage.

The compatible `PADRA_EXL3_PREFILL_ASYNC64=ON` alias selects `exact-fusions-v1`
when `PADRA_KERNEL_PROFILE` is empty. Unsupported profiles and `baseline-v1`
combined with that alias set to `ON` fail configuration. `/v1/models` exposes the
resolved build profile.

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

Experimental. Full model/task quality, full-context performance, sustained load
and deployment behavior are not qualified. Host tests cover bounded server/API
behavior; native CUDA/Rust HTTP smoke tests cover serving with the real artifact.
Other GPU architectures and physical 8/12 GB cards are not qualified.

Padra is licensed under [Apache 2.0](LICENSE). See [NOTICE](NOTICE) and
licenses/ for third-party attribution.

## Request lifecycle and observability

Queue admission starts an independent five-second waiting deadline. Expiry,
disconnect and session deletion remove queued work and release its byte/token
reservation and session lease without waiting for the active GPU job. Generation
has its own deadline, starting when execution begins; the CLI default is 600
seconds. There is no separately configured total-request deadline.

Normal stop/EOS, length and tool completion are generation outcomes. Errors and
cancellations have a separate state. Streaming keeps 64 ordinary frames and a
separate three-frame completion lane for finish reason, optional usage and
`[DONE]`. A full delta/tool buffer produces an explicit error instead of reporting
success with a truncated protocol. Idle streams emit an SSE comment heartbeat
every 15 seconds; deployment-specific proxy behavior still needs testing.

Both APIs return `503/queue_deadline` or `503/server_shutdown` with `Retry-After: 1`,
and `504/generation_deadline`. SSE/Responses failures use the same domain codes.
Response records, bytes and new physical roots are admitted together. Eviction
must actually free all references to a root before that root counts as an available
session slot. Manual sessions, active leases, pending and pinned records are
protected. Failed admission releases new roots and pending reservations. Cache
eviction is final, including when a later step fails. Record and physical-root
evictions have separate counters; TTL/deletion-only cleanup is excluded. See
[the lock contract](docs/server-locks.md).
At most four generation requests prepare/tokenize concurrently, with an early
queue/readiness check followed by final byte/token admission.

HTTP shutdown removes readiness, rejects new work and fails queued requests.
Active generation gets a five-second grace period before cancellation; backend
cancellation takes effect at its next cooperative boundary. The listener stays
open during this grace period so admitted responses can finish writing, and stops
when their transports finish or the grace deadline expires. Socket timeouts and
cooperative backend boundaries still govern connection/thread teardown; this is
not a hard process-exit deadline. Repeated HTTP stop calls are safe. Direct coordinator
shutdown cancels immediately. The GPU continues to execute one request at a time.

`/v1/models` reports validated loaded weight/manifest hashes, tokenizer hash,
build source fingerprint, and runtime math/kernel profiles. Missing artifact
identities are `null`. The build fingerprint identifies configure-time source
content; compiler/toolchain and build options are separate qualification inputs.
`/metrics` exposes fixed-bucket latency histograms, output-token totals, and prefill
chunk/eager counts. Queue time excludes preparation; output availability measures
internal text/tool readiness before transport. Job lifetime ends at coordinator
completion, so these metrics are not client-observed TTFT/end-to-end latency.
Transport disconnect/backpressure counters include failures after generation ends.
`mellum_first_visible_write_seconds` measures handler entry to the first successful
body write containing text or tool identity/arguments; protocol-only and reasoning
events do not count. `mellum_response_write_seconds` measures handler entry through
transport completion or failure. Active transports, written responses, write failures
and successfully written body bytes are counted separately. These are server-side
socket-write observations, not confirmation of client receipt; request upload,
network transit and client buffering are excluded. Early admission errors and stored
GET responses are outside these generation-transport counters.

Tool identity is emitted after a complete offered function name is recognized,
then argument fragments stream during generation. Arguments retain their original
JSON bytes; the completed call still undergoes full JSON validation. Clients must
wait for successful completion before using the arguments. If arguments precede
the name, their bounded prefix waits for identification. A length-truncated call
remains partial and does not commit session history. If stored, its incomplete
envelope remains retrievable, but `previous_response_id` referencing it returns
HTTP 400 with Padra's own `response_not_continuable` code. Retry the original
request, or continue from the last valid predecessor, with enough output budget.
Unfinished argument JSON is never repaired or added to valid continuation history.
Text-only length endings remain continuable. Python SDK compatibility has
an optional pinned loopback suite; proxy/load qualification and authenticated
external deployment require separate qualification.

Responses output items keep their first-observed order across streaming deltas,
done events, terminal output and stored GET. There is at most one reasoning item,
one message item and eight tool items. Text/reasoning emitted again extends its
existing item, including text on both sides of a tool call. Nonstreaming requests
use the same logical ordering; items are not regrouped by type at completion.
