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
