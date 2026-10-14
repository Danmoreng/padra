# Follow-up implementation evidence (2026-10-14)

Scope: finish the existing transport/drain patch, then deterministic disconnect
tests, stable output ordering, partial-tool continuation policy, joint root/record
admission, and consistent admission errors/counters. Each is a separate change.

## 1. Transport and drain

The HTTP listener previously stopped before active generation/transport could use
the grace period. Shutdown now waits for admitted generation and transport within
one shared deadline, then explicitly cancels any remaining generation. Repeated
stop calls are serialized because the pinned HTTP library cannot safely close its
listener twice. Generation completion and successful transport are separate
counters; protocol/reasoning events do not start the first-visible-write timer.
Metrics describe successful server socket writes, not client receipt.

Executed:

```sh
cmake --build build/review-sdk -j4
ctest --test-dir build/review-sdk --output-on-failure
cmake --build build/review-release15 -j4
ctest --test-dir build/review-release15 --output-on-failure
cmake --build build/review-sanitizers -j4
ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build/review-sanitizers --output-on-failure
```

Results: Debug/SDK 3/3, GCC 15 Release 2/2, ASan/UBSan 2/2. The `--transport`
subset passed 100 consecutive executions in each build, stopping on the first
failure (300 total; logs in each build's `transport-repeat.log`). Covers both
APIs, streaming/nonstreaming, visible/empty output, repeated/concurrent stop,
deadline cancellation, idempotent accounting and abandoned transport cleanup.
The Python SDK used the pinned version from `sdk-requirements.txt`.

`cmake --build build/review-cuda -j2` and the bounded real-model
`tests/native_http_smoke.py` run passed (CUDA/Rust, exact-fusions-v1). Results and
server log are under `build/review-cuda/native-http-smoke/`. This is a serving
smoke, not full numerical/task quality or sustained load qualification. No CUDA
arithmetic or numerical defaults changed. Baseline-profile GPU qualification,
slow-reader saturation and full model-quality measurements were not run here.

A separate Release attempt with the machine's new GCC 16 default failed under
`-Werror=maybe-uninitialized` in the existing JSON/`std::variant` literal path.
No warning was disabled; the Release checks above use the available GCC 15
toolchain. This change does not claim GCC 16 Release support.

## 2. Deterministic disconnect regression (PR01)

The old test treated `!active` as completion even if the request was still queued.
Queue and active HTTP disconnects now have separate tests for both APIs, plus an
explicit-session Chat case. The predecessor/backend is held at a real barrier;
the client closes its socket only after queue admission or backend entry is
observed. Queue cancellation completes before releasing the predecessor. Tests
require terminal outcome counts, empty token/byte reservations, no active
transport, one failed transport, temporary root cleanup and an acquirable explicit
session lease. Backend/HTTP/SSE helpers live in `server_test_support.h` exposed by
the header-only CMake `padra_test_support` target.

Executed the same build/CTest commands above: Debug/SDK 6/6; GCC 15 Release 5/5;
ASan/UBSan 5/5. In all three builds:

```sh
ctest --test-dir BUILD -R 'server-disconnect-' --repeat until-fail:100 \
  --output-on-failure
```

Both named cases passed 100 consecutive runs per build (600 executions, three
HTTP/session scenarios each). Logs: `BUILD/disconnect-repeat.log`. No sanitizer
diagnostic or assertion failure. No production behavior changed in this step;
CUDA/Rust/model quality and sustained GPU load were not rerun for this test-only
change. The existing native smoke evidence belongs to step 1.

## 3. Canonical output ordering (PR02)

`server-output-order` first failed with `live/final output index mismatch` on
Tool→Text. The worker now records a bounded logical order (one reasoning item,
one message, up to eight calls) when output first appears. Both streaming and
nonstreaming observe this order; serialization/retention use it instead of
regrouping by type. Repeated text extends the same message item. HTTP item IDs
remain adapter-owned. The stream still delivers tool bytes before generation
finishes; no completion buffering or larger resource limit was introduced.

The HTTP matrix covers Tool→Text, Text→Tool, Reasoning→Tool→Text, multiple calls,
text on both sides of a call, empty/text-only output, each with EOS/length. It
checks all live/done indices, reconstructed text/argument bytes, exact terminal
envelope vs GET, and stream/nonstream semantic equality. The pinned SDK stream
accumulator also checks mixed output and retrieval (excluding SDK-added null
parsed helper fields; wire equality is checked without normalization).

Executed the same build/CTest commands: Debug/SDK 7/7, GCC 15 Release 6/6,
ASan/UBSan 6/6, CUDA/Rust build successful. No sanitizer diagnostic. The SDK
comparison initially exposed its extra null parsed fields, which are accounted
for explicitly. Native model smoke was not rerun in this step; full numerical,
model-quality and sustained-load qualification remain outside these host changes.

## 4. Partial-tool continuation policy (PR03)

The new budget-truncation test first failed because the follow-up returned a
generic history parse error. Retention now distinguishes the display envelope
from continuability: incomplete tools remain retrievable but are marked internally
non-continuable; their incomplete arguments never enter valid history. Resolving
their previous ID returns HTTP 400/`response_not_continuable` before prompt/history
construction. This is a Padra-specific contract. Text-only length endings remain
continuable. Explicit sessions retain their last valid state on a partial tool.

The shared backend now actually honors output budgets. New HTTP tests supply more
tokens than the budget and check exact output-token accounting at truncated names,
strings, escapes, Unicode bytes, and a complete call followed by an incomplete
second call. All cover store on/off, stream/nonstream, GET, follow-up and explicit
sessions; text-only truncation is a positive continuation control. The pinned SDK
also checks incomplete tool GET and the precise follow-up error.

Executed Debug/SDK CTest (8 cases; the added SDK check passed after correcting a
test-file indentation error), GCC 15 Release 7/7 and ASan/UBSan 7/7, using the build
commands above. No sanitizer diagnostic. CUDA/Rust/native smoke and full model
quality were not rerun for this host/API change.

## 5. Joint response/root admission (PR04)

The new HTTP test first reproduced rejection with one manual session plus 31
independent stored responses. `Reserve` now plans record slots, response bytes,
new root slots and initial session bytes under the two store locks. It counts a
physical root only when all its eligible references are retired; manual sessions,
leases, pending records and pins are protected. New map nodes are preallocated;
the eviction commit and new root/reservation publish together. RAII releases both
new resources after a later failure. Expired/deleted cleanup and committed cache
evictions remain final. Root callbacks/destructors are deferred until store locks
are released, and core deletion precedes optional notifications. The lock contract
is in `docs/server-locks.md`. Physical-root and record eviction counters differ.

Tests cover the mixed HTTP case, the existing 40-response case, shared roots,
all-pinned/all-manual/active/pending protection, session byte pressure, TTL,
abandonment after reserve/root creation and Submit failure, reentrant/throwing
discard notifications, and concurrent manual/response admission/deletion.
The two-resource transaction removes the former externally visible gap between
record reservation and new-root creation; rollback is tested at its returned
boundary, not via an arbitrary callback inside the locks.

Executed Debug/SDK 9/9, GCC 15 Release 8/8, ASan/UBSan 8/8 with the commands above.
The admission suite additionally passed 100 consecutive ASan/UBSan runs before
the final notification-throw guard; the final suite includes that new case too.
CUDA/Rust build and native HTTP smoke passed. No numerical arithmetic/defaults
changed; full model quality and sustained multi-client load were not qualified.
