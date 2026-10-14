# Client contract qualification

The public support matrix is [API support](../docs/api-support.md). The versioned
normative request matrix is `contracts/api-fields.json`, version 1. Its 247 cases
exercise the production HTTP adapters, accepting supported values and rejecting
unsupported fields, ranges, content forms, strict/forced tools and excess tool
limits. Error status/code and request identity are checked. Unsupported endpoints
are separately rejected. The synthetic backend records all sampling options at
runtime preparation; both API adapters must forward temperature, top-p, top-k,
seed and all three penalties unchanged. This does not qualify sampling numerics.

`api-wire-golden` compares 32 protocol transcripts captured before the typed-event
refactor. Only generated identities and timestamps are normalized. Separate C++
output-order and partial-tool tests compare reconstructed live bytes, terminal
objects and retained GET, including incomplete output and continuation policy.
SDK acceptance alone is not used as the wire-contract oracle.

## SDK suite

The supported baseline is `openai==3.29.0` from `sdk-requirements.txt`. Run with
`-DPADRA_ENABLE_SDK_TESTS=ON -DPython3_EXECUTABLE=/path/to/venv/bin/python` and CTest,
or invoke `sdk_tests.py --fixture /path/to/padra-sdk-fixture --report report.json`.
All requests target synthetic loopback inference with dummy namespace credentials.
OpenAI cloud/API keys and model weights are not used.

Twelve named groups cover Chat and Responses JSON/SSE, SDK accumulation helpers,
usage, mixed tool/text output, exact raw arguments, early tool deltas, UTF-8/stop,
sampling extensions, retained continuation/GET/delete, partial-tool policy, queue
503/Retry-After/request IDs, disconnect, invalid tools, paused readers, rejected
options and endpoints. Client retries are zero and timeout is three seconds.
Paused-reader cleanup is observed independently through server metrics, within a
six-second test deadline; no further SDK read is needed to release the server.
The dedicated paused-reader HTTP transport also uses zero retries and a 1024-byte
requested socket receive buffer, preventing TCP autotuning from invalidating the
load condition.

Early-tool delivery uses a synthetic-backend barrier. With `--control-stdin`, the
fixture pauses after an incomplete argument prefix. The SDK must receive argument
bytes while generation remains active and completion counters are unchanged, then
send `release-tool` over the test-only stdin control channel. This is independent
of CPU/network timing. Cleanup closes stdin before terminating the fixture, which
releases blocked generation. Normal fixture mode has no stdin control thread.

The suite writes actual SDK/Python version, engine source fingerprint, resolved
server settings, explicit client settings, passed group names, success and fixture
exit code to `sdk-compatibility.json` in CTest's build directory. Failed runs retain
the current group name. The pinned CI job uploads this and CTest output.

## Update canary

`.github/workflows/sdk-canary.yml` is separate from pinned host qualification. It
runs weekly and on manual dispatch, resolves the then-current pip SDK, builds the
same synthetic fixture and runs SDK plus exact wire tests. It uploads the actual
package resolution, report and test output, including on failure. A canary failure
requires compatibility investigation; it neither upgrades the supported pin nor
silently changes release requirements.

Local PR09 verification on 2026-10-14 tested baseline SDK 3.29.0 and independently
resolved SDK 3.31.0; both passed all twelve groups with fixture exit zero. Host
ASan/UBSan and Clang TSan cover the field matrix as separate C++ tests. Native
artifact smoke is a separate runner, `native_http_smoke.py`, including explicit
JSON configuration/admin checks, real tools, stored roots, lineage and both API
streams. It does not replace full model/task or sustained-load qualification.

The complete local SDK configuration passed 21/21 CTest tests. GCC 16 Release,
Clang TSan and GCC ASan/UBSan host configurations passed 19/19 each; the completed
247-case matrix was additionally rerun under both sanitizers. The final CUDA
server build and real-artifact smoke also passed with the explicit configuration
and admin checks. GitHub workflow artifacts record remote compiler/Rust/SDK runs.
