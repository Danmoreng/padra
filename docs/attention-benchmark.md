# Isolated prefill attention benchmark

Configure a CUDA build with `-DPADRA_ENABLE_ATTENTION_BENCH=ON`, then build the
`padra-attention-bench` target. It requires no model weights or tokenizer and
compares the active padded BF16 attention kernel with a frozen K32 control from
engine commit `61851c0` on identical post-RoPE Q and old/current-chunk K/V.

```sh
python tools/attention_bench.py --binary BUILD/padra-attention-bench \
  --output RESULT.json -- --base 65536 --tokens 256 --capacity 131072 --graph
```

Use an exclusive GPU lease. The wrapper records local commands, worktree state,
binary hash and before/after GPU telemetry. `--provenance FILE.json` attaches
bounded capture metadata and its hash (for example model/artifact/tokenizer
identities, input-token hash, KV format and rotations). Keep machine-specific reports and
model-derived fixtures outside the public checkout.

Output includes fixture/engine/benchmark source hashes, GPU/SM count, CUDA
compiler/runtime/driver, register/shared/local memory and occupancy limits,
shape, graph mode, precision, explicit device-buffer
bytes and each warmed CUDA-event duration. These durations cover only attention,
excluding host setup, packing, rotations, projections and transfers; they are not
application latency or time to first token. Defaults: three warmups, seven runs.
Allocation admission preserves 700 MiB of currently available device memory.

Options: `--local`, `--round-output`, `--poison-invisible`, `--warmup N`,
`--repeats N`. Local attention uses a 1024-position ring and separate current
chunk. `--graph` captures the controlled kernel with a device `PrefillState`.
Output rounding selects BF16, or the FP32 intermediate used before native Q8's
inverse V rotation. QK/PV accumulation is FP32, probabilities BF16, and Q scaling
follows the QK product. Input fixtures must already have any required rotations.

`--dump-fixture DIRECTORY` exports native little-endian `q.f32`, `k.bf16`,
`v.bf16`, `chunk-k.bf16` and `chunk-v.bf16`. `--fixture DIRECTORY` loads that layout
and checks byte counts against explicit CLI shape. Q shape is `[tokens,32,128]`,
cache `[cap,4,128]`, chunks `[tokens,4,128]`; cap is global capacity or
`min(1024,capacity)` locally. Capture provenance/shape metadata must accompany
model fixtures. The fixture hash includes shape, output rounding and all bytes.

Runs reject nonfinite outputs and require complete bitwise agreement with the
frozen control. Small cases (at most 16 queries, 2048 total positions) additionally
use an independent scalar FP64 softmax oracle with BF16 operands. Its tolerance
is `0.002 + 0.02*abs(reference)` because production rounds intermediate
unnormalized probabilities to BF16. This catches indexing/mask errors separately
from frozen bitwise checks; it does not qualify downstream model quality.

## Exact global dataflow selection

`PADRA_PREFILL_PROFILE=baseline-v1` is the default. The opt-in
`global-dataflow-v1` build uses direct global-prefix addressing and a warp-uniform
unit-alpha fast path. Local ring addressing, softmax denominator updates,
probability rounding, output rounding and decode kernels retain their contracts.
No extra device workspace is allocated. Build/profile identity and kernel labels
identify the selected path; captured graph identity includes source and profile.

The benchmark's `--variant` selects `baseline`, `address`, `alpha`, or
`address-alpha` independently of the server build selection. Every variant is
checked against the frozen control. Warmed timing alternates control/candidate
order and reports both raw arrays (`control_milliseconds`, `milliseconds`).
Compare representative shapes and downstream model results before changing a
runtime default; an isolated-kernel result is not an application speed claim.

## Tile and partition controls

The operator harness additionally supports `ldmatrix`, `split2`, `split4` and
`matrix-adaptive`. Split variants change the partition maximum and BF16
probability rounding. They require finite
outputs within `0.002 + 0.02*abs(frozen)` and the small-shape FP64 oracle bound.
ULP and RMS differences are reported as diagnostics. Exact address, unit-alpha
and K32 `ldmatrix` controls require bitwise equality. A separate fragment test
compares every QK/PV fragment bit before any timing. These operator bounds do not
replace downstream logit, routing or task-quality qualification.

`--capture-base 0 --base 129024 --graph` replays the same captured graph at both
prefixes, with independently computed controls. The capacity and active query
count stay fixed. This exercises device-driven prefix and partition selection.

The server build profile `global-matrix-v1` retains K32 arithmetic with exact
fragment loads. `global-matrix-split-v1` also selects four partitions for global
chunks of at most 512 queries once the device prefix reaches 8192. Earlier
prefixes and larger chunks retain unsplit K32 arithmetic. Local attention and
native Q8 decode retain their existing kernels. This split profile has a distinct
arithmetic identity; it is an explicit selection, not the default.

Split workspace is admitted and allocated before execution: at most
`min(chunk,512) * 32 * 4 * 130 * sizeof(float)`, or 32.5 MiB. Each partition stores
128 unnormalized FP32 values plus one complete maximum and denominator per
query/head. Empty partitions use `(-infinity,0,zero)`. The final stable merge
applies output rounding once. Prefix selection reads `PrefillState` on replay;
no capture-time prefix length determines the partitioning. Source, build profile
and arithmetic identity distinguish the graph cache entries.

## Native Q8 fixtures and packed current chunk

`--q8` generates native block32 Q8 rows with FP16 scales and derives their BF16
materialization using the production rounding boundary. The deterministic
fixture is already in the rotated K128/V64 storage domain; the harness introduces
no query quantization. Fixture export includes the native rows and their BF16
images. Timed attention consumes the BF16 materialization, as the server does.
The reported workspace includes the independent frozen-control buffers.

`PADRA_Q8_PREFILL_PROFILE=packed-chunk-v1` dequantizes the current Q8 chunk directly
into its packed BF16 workspace. It preserves the existing rotations, BF16
rounding and materialized-prefix contract. `materialized-v1` retains the explicit
BF16-to-float-to-BF16 chunk control. The selection is independent of attention
partitioning and decode kernels, requires no additional device memory, and is
included in build and graph identity.
