# Dependencies

CMake owns the server build, including fetching CUTLASS headers and invoking
Cargo. No dependency checkout beside this repository is required. CMake, the
C++/CUDA compilers, Ninja and Rust/Cargo are installed build tools; Padra does not
install or replace system toolchains. Dependencies are not downloaded at runtime.

| Dependency | Pinned version/source | Integration | License |
| --- | --- | --- | --- |
| NVIDIA CUTLASS | `db1c288993354c88e551c40c19a8fb93a774a241` | CMake FetchContent, header-only | BSD-3-Clause |
| Hugging Face Tokenizers | `0.22.1` | Cargo, statically linked through the local C ABI | Apache-2.0 |
| serde_json | `1.0.145` | Cargo, `preserve_order` and `float_roundtrip` | MIT OR Apache-2.0 |
| cpp-httplib | `0.40.0` with local transport adaptations | Vendored sources, built by CMake | MIT |
| OpenSSL Crypto | Installed system version, optional | CMake `find_package`; existing native SHA-256 when absent | See installed distribution |

## CUTLASS

`cmake/PadraCutlass.cmake` downloads the fixed upstream commit archive into the
build tree using FetchContent. The archive SHA-256 is
`e84e148a587cbe62ab29118389fd842fa77af5c58c1f6a940d6297669c8c3a6f`.
All 892 header/license files are also checked against
`cmake/cutlass-source-lock.json`, preserving the original verified file set.
The archive was checked against that complete manifest. CUTLASS examples, tests
and tools are not configured or built; its templates compile with Padra's CUDA
sources through the `Padra::CUTLASS` interface target.

For local or offline use, pass `-DPADRA_CUTLASS_ROOT=/path/to/cutlass`. This can
be a source tree at the pinned revision or the legacy prepared header cache.
Both receive the same per-file checks. Existing prepared caches retain their
source-lock identity check. `tools/prepare_cutlass.py` remains an optional way
to produce such a cache, not a normal build prerequisite.

CMake's `FETCHCONTENT_BASE_DIR` and
`FETCHCONTENT_SOURCE_DIR_PADRA_CUTLASS_SOURCE` overrides are also supported.
For offline use, provide an existing source tree before configuring; merely
enabling `FETCHCONTENT_FULLY_DISCONNECTED` does not populate an empty cache.

Upstream: <https://github.com/NVIDIA/cutlass>. License:
`licenses/CUTLASS-BSD-3-Clause.txt` and the downloaded `LICENSE.txt`.

## Native text

`src/text/native/src/lib.rs` is Padra's local C ABI adapter to the upstream Rust
tokenizer. `renderer.rs` translates the pinned Mellum chat template; it is local
code, not an upstream tokenizer feature. The inference engine and incremental
stream decoder remain C++/CUDA.

`src/text/native/Cargo.toml` specifies exact direct versions.
`src/text/native/Cargo.lock` fixes transitive versions and registry checksums.
`cmake/PadraNativeText.cmake` invokes `cargo build --locked --release`; Cargo
downloads and builds the pinned packages, then CMake links the static library.
Rust 1.89 or newer is required. No Python process or Cargo invocation is needed
to run the built server.

Optional CMake settings:

- `PADRA_CARGO_EXECUTABLE`: Cargo executable, otherwise discovered on `PATH`.
- `PADRA_CARGO_HOME`: private Cargo dependency cache; default is Cargo's normal cache.
- `PADRA_RUSTUP_HOME`: private Rustup installation; default is Rustup's normal location.
- `PADRA_CARGO_JOBS`: Cargo build parallelism, default `2`.
- `PADRA_CARGO_OFFLINE=ON`: prohibit Cargo network access; missing cached packages
  produce an error instead of changing dependency versions.

Upstream: <https://github.com/huggingface/tokenizers> and
<https://github.com/serde-rs/json>. Dependency license texts remain in Cargo's
source cache; individual transitive dependency licenses also apply.

## HTTP transport

`third_party/cpp-httplib` contains the split header/implementation for version
0.40.0, originally copied from the Gem16 server reference at commit
`c0a807afcbcd2528f304bada513afd11d1e9c979`. Padra retains the local changes for
bodyless requests, persistent keep-alive read-ahead, pipelining, connection-close
handling and unsupported transfer-encoding rejection. Fetching an unchanged
upstream version would lose these changes.

`third_party/cpp-httplib/SOURCE_LOCK.json` records the upstream version, donor,
adaptations and exact hashes of the vendored header, implementation and license.
`cmake/PadraHttp.cmake` verifies these files before adding the existing
`padra_http` target. Intentional transport updates must update that manifest.

Upstream: <https://github.com/yhirose/cpp-httplib>. License:
`third_party/cpp-httplib/LICENSE`. Other adapted kernel/runtime source attribution
is retained in `NOTICE` and `licenses/`; this dependency cleanup does not replace
those adapted sources.
