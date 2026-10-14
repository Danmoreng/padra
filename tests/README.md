# Server regression tests

The suite compiles the production coordinator, HTTP handlers, stores, parsers,
SSE encoders and C++ UTF-8 decoder. A controlled `RuntimeBackend` and test-only
vocabulary/renderer replace model inference and native tokenization. No weights,
CUDA, Rust or Python are needed for these tests. They exercise actual loopback
HTTP connections in addition to deterministic blocked-backend lifecycle cases.

```sh
cmake -S . -B build/host -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DPADRA_ENABLE_CUDA=OFF -DPADRA_ENABLE_NATIVE_TEXT=OFF
cmake --build build/host -j2
ctest --test-dir build/host --output-on-failure
```

For ASan/UBSan, add `-DPADRA_ENABLE_SANITIZERS=ON` to a separate host build and run
CTest with `ASAN_OPTIONS=detect_leaks=1:abort_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1`. Assertions remain enabled in Release tests.
The host CI matrix also builds the real native Rust tokenizer when enabled;
server regression requests still use the explicitly synthetic fixture.
The transport/drain subset is independently repeatable with
`build/host/padra-server-tests --transport`.
Disconnect cases are registered independently as `server-disconnect-queue` and
`server-disconnect-active`. Repeat each without masking failures:

```sh
ctest --test-dir build/host -R 'server-disconnect-' --repeat until-fail:100 \
  --output-on-failure
```

These use backend/queue barriers and a client socket close, then wait for terminal
outcome counts, empty queue budgets, transport completion and released root/lease.
The shared header-only `padra_test_support` target supplies the controlled backend,
loopback fixture and SSE reader; no test includes another test's `.cpp`/`main`.

Covered regressions include:

- Full SSE ring followed by stop, EOS, length, optional usage, or a tool call;
  ordered completion frames and explicit errors on delta/tool backpressure.
- Queue expiry while another job is blocked, released bytes/tokens/session leases,
  and queued disconnect, shutdown and session deletion.
- Graceful completion or bounded cancellation during drain, including actual
  HTTP Chat/Responses streaming and nonstreaming completion, repeated stop calls,
  and a blocked backend that exceeds the grace deadline.
- Successful body-write bytes, first visible writes, full transport accounting
  and abandoned transports (separate from generation completion).
- Forty independent stored Responses over HTTP, shared/pinned roots, lineage,
  deletion, expiry, concurrent reservations and rollback on failed generation.
- Consistent Chat/Responses HTTP queue errors, Retry-After, both SSE protocols,
  real client disconnects, artifact identity and bounded latency histograms.
- Incremental tool bytes, split markers/fields, Unicode, reordered fields, duplicate
  and unoffered names, early delivery while the backend is paused, and both HTTP
  protocols before generation completes.
- Duplicate JSON keys, Unicode surrogates, split UTF-8/stop/thinking and owner/lease
  isolation.
- Named kernel profiles, legacy alias compatibility, unsupported profiles and
  conflicting selection.

The base host suite does not qualify CUDA arithmetic, real model quality, native
renderer semantics, heartbeat/proxy behavior or sustained multi-client performance.
Those require separate runs with the actual artifact. The optional Python SDK
suite below covers its explicitly supported loopback API subset.
The 2026-10-13 Pro review's reproductions originally asserted the broken behavior;
these tests require corrected behavior.

## Native artifact smoke test

An optional Python standard-library runner starts the real CUDA/Rust server,
checks loaded artifact hashes, session follow-ups, both streaming APIs, forty
independent stored Responses and deletion with retained children, then stops and
joins the server. Use an unused loopback port and a qualified artifact:

```sh
python3 tests/native_http_smoke.py \
  --server build/padra-server --artifact /path/to/native \
  --tokenizer /path/to/tokenizer.json --tokenizer-sha256 EXPECTED_SHA256 \
  --output-dir build/native-http-smoke
```

Run under an appropriate foreground memory/time limit on the target GPU. This is
a serving integration check, not numerical quality or performance qualification.

## Python SDK compatibility

The optional suite pins the real Python SDK and starts a synthetic serving fixture.
It makes only loopback requests. No API key, model artifact or CUDA is required.
The fixture replaces inference/tokenization; handlers and streaming are production
code. It covers both APIs, tool/Unicode assembly, the Responses stream helper,
usage, forty retained Responses, queue errors and SDK stream cancellation.

```sh
python3 -m venv build/sdk-venv
build/sdk-venv/bin/pip install -r tests/sdk-requirements.txt
cmake -S . -B build/sdk -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DPADRA_ENABLE_CUDA=OFF -DPADRA_ENABLE_NATIVE_TEXT=OFF \
  -DPADRA_ENABLE_SDK_TESTS=ON \
  -DPython3_EXECUTABLE="$PWD/build/sdk-venv/bin/python"
cmake --build build/sdk -j2
ctest --test-dir build/sdk --output-on-failure
```

The event contracts follow the official [function-calling streaming guide](https://developers.openai.com/api/docs/guides/function-calling).
