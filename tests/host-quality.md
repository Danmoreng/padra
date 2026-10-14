# Host quality checks

Host tests use the production server, parsers and UTF-8 decoder with a synthetic
inference backend. They do not qualify CUDA arithmetic or model quality. The
native-text CI variants additionally build the locked Rust dependencies and run
six Rust tests, including real BPE create/encode/decode through the C ABI using a
small generated test vocabulary of the required size. They require no weights.

## CI coverage

`.github/workflows/host.yml` runs GCC 13 and Clang 18 in Debug and Release, with
native text enabled and disabled. Debug variants also run ASan/UBSan. Separate
jobs run TSan, pinned Python SDK compatibility, clang-format 22.1.0, and a Clang
libFuzzer target under ASan/UBSan. Missing tools fail configuration; sanitizer
failures are not retried or suppressed. TSan uses compatible ASLR entropy on the
disposable Ubuntu runner and cannot be combined with ASan or CUDA in one build.

The formatter checks tracked project-owned C++/CUDA files in `src`, `include` and
`tests`. Include order is preserved because CUDA headers have ordering contracts.
Run `python3 tools/check_format.py --clang-format /path/to/clang-format`.

## Parser properties and fuzzing

`parser-properties` runs the versioned corpus `tests/corpus/parsers/v1`, 512 seeded
arbitrary-byte inputs and long-input boundary probes. It checks JSON semantic
roundtrips, stop/thinking/tool results under whole, byte, seven-byte and seeded
irregular chunks, raw tool-argument preservation, and lossy UTF-8 decoding under
byte and combined multibyte tokens. Malformed tools must reject consistently;
bytes delivered before detecting an error can depend on the failing chunk.

Retained parser data is checked after every push: stop carry ≤255 bytes,
thinking carry ≤7 bytes, UTF-8 carry ≤3 bytes and tool framing/raw data ≤8×64 KiB
plus 16 marker bytes. The caller's collected output is separate and grows with
input. These are retention bounds, not a claim about zero allocations or total
process RSS. JSON fuzzing uses explicit depth, value-count and string-byte limits.

Build a separate Clang host configuration with
`-DPADRA_ENABLE_CUDA=OFF -DPADRA_ENABLE_NATIVE_TEXT=OFF
-DPADRA_ENABLE_SANITIZERS=ON -DPADRA_ENABLE_FUZZING=ON` and build target
`padra-parser-fuzz`. Copy the corpus into the build directory and run
`padra-parser-fuzz -seed=1346454610 -max_len=4096 -runs=10000 build/corpus`.
Only the build copy is mutated. CI preserves crash/leak/timeout reproducers on
failure. Larger or longer fuzz campaigns can use the same target; the harness
rejects inputs above 128 KiB.

## PR10a verification, 2026-10-14

- GCC 16 Debug + OpenAI Python SDK 3.29.0: 11/11 CTest tests.
- GCC 16 Release: 10/10; literal parsing constructs the known variant alternative
  directly, without disabling `-Werror=maybe-uninitialized`.
- GCC 15 Release: 10/10.
- Clang 23.1.1 Release: 10/10. Native-disabled builds explicitly allow the unused
  native tokenizer handle; other warning errors remain enabled.
- GCC 15 TSan, separate Debug build: 10/10, no suppression list.
- GCC 16 ASan/UBSan: 10/10.
- Clang 23.1.1 libFuzzer + ASan/UBSan: 10,000 seeded runs, no findings.
- Locked Rust native text: six tests pass; this is separate from synthetic CTest.
- CUDA 13.4 / GCC 15 native-text build passed the formatting baseline. No CUDA
  arithmetic or execution defaults are changed by these host checks.

Local commands use `cmake --build build/<configuration> -j4` followed by
`ctest --test-dir build/<configuration> --output-on-failure`. Configurations are
`review-sdk`, `review-release`, `review-release15`, `review-clang`,
`review-tsan`, `review-sanitizers` and `review-fuzz`. TSan uses
`TSAN_OPTIONS=halt_on_error=1`; ASan/UBSan use
`ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 UBSAN_OPTIONS=halt_on_error=1`.

PR10a GitHub Actions run [38569963687](https://github.com/Danmoreng/padra/actions/runs/38569963687)
completed successfully for all 16 jobs, including every GCC/Clang/native-text
variant, TSan, ASan/UBSan, format, fuzz and pinned SDK.
