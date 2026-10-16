# Isolated prefill attention benchmark

Configure a CUDA build with `-DPADRA_ENABLE_ATTENTION_BENCH=ON`, then build the
`padra-attention-bench` target. It requires no model weights or tokenizer and
compares the active padded BF16 attention kernel with a frozen K32 control from
engine commit `61851c0` on identical post-RoPE Q and old/current-chunk K/V.

Fresh server builds select `PADRA_PREFILL_PROFILE=global-matrix-v1`,
`PADRA_Q8_PREFILL_PROFILE=packed-chunk-v1` and
`PADRA_EXL3_PREFILL_TRANSPORT=async64-v1`. These preserve the existing arithmetic
and leave `PADRA_KERNEL_PROFILE=baseline-v1` as the decode profile. Existing CMake
caches retain their configured selections.

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

`PADRA_PREFILL_PROFILE=baseline-v1` selects the frozen arithmetic/dataflow control.
The opt-in `global-dataflow-v1` build uses direct global-prefix addressing and a
warp-uniform unit-alpha fast path. Local ring addressing, softmax denominator updates,
probability rounding, output rounding and decode kernels retain their contracts.
No extra device workspace is allocated. Build/profile identity and kernel labels
identify the selected path; captured graph identity includes source and profile.

The benchmark's `--variant` selects `baseline`, `address`, `alpha`, or
`address-alpha` independently of the server build selection. Every variant is
checked against the frozen control. Warmed timing alternates control/candidate
order and reports both raw arrays (`control_milliseconds`, `milliseconds`).
Compare representative shapes and downstream model results before changing a
runtime default; an isolated-kernel result is not an application speed claim.

## Exact matrix loads

The operator harness additionally supports `--variant ldmatrix`. It retains K32
arithmetic and requires bitwise equality with the frozen control. A separate
fragment test compares every QK/PV fragment bit before timing.

`--capture-base 0 --base 129024 --graph` replays the same captured graph at both
prefixes, with independently computed controls. Capacity and query count stay
fixed; prefix selection reads `PrefillState` at replay time.

The server profile `global-matrix-v1` uses exact matrix fragment loads. Local
attention and native Q8 decode retain their existing kernels. No additional
persistent workspace is allocated.

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

## Independent EXL3 prefill transport

`PADRA_EXL3_PREFILL_TRANSPORT=async64-v1` selects the double-buffered K64 transport
for EXL3 K4 prefill at four or more tokens. It preserves sequential K16 MMA order
and BF16 boundaries. K3/K6 and smaller requests retain their existing codec
paths. Gate and up inputs retain their separate matrix-specific transforms.
`baseline-v1` retains the packed-activation transport control.
This selection changes no decode fusion flags and allocates no additional
persistent workspace. `PADRA_KERNEL_PROFILE` continues to select decode fusions
independently. Build and graph identity include the prefill transport selection.

## Timing boundaries

`InferenceResult.prompt_forward_stream_seconds` measures CUDA-stream elapsed time
from the generation boundary through final prompt logits, before token selection.
It is separate from wall-clock `prefill_seconds`, which includes selection and
control transfer, and from external time to first token. `decode_seconds` covers
the subsequent N-1 token steps. These ranges are alternatives, not additive GPU
and CPU costs. Preparation/graph construction is reported separately.
