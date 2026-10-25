# Multi-token prediction

Padra provides opt-in Mellum2.1 Thinking MTP for admitted EXL3 K3/K4/K6 targets
with BF16, Q8 or Q8_0 target KV. The default standalone draft retains its original BF16 matrices and F32 norms
and router. Separate opt-in FP8 and EXL3 K4 expert profiles reduce persistent memory.
The [131K GPU profiles](gpu-profiles.md) select EXL3 draft experts and one stored
Q8 head. The engine and general full-context/task quality remain experimental.

Without `--mtp-artifact`, execution is ordinary decoding. With an artifact, the
default is D1; `--mtp-depth 1`, `2`, `3` or `4` fixes the number of proposals.
The one trained Mellum head is applied recursively for deeper drafts. Depth is
fixed at startup. Higher depth does not guarantee higher throughput.

## Prepare the artifact

Use the original official BF16 GGUF from
[JetBrains](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF/tree/4159d684ffdb09ba89dc2a58feb3e75c54eef036):

```text
mtp-Mellum2.1-12B-A2.5B-Thinking-BF16.gguf
SHA-256 9c7b2c37dcdcb28140118debc81175cfc22bb00a7aa51bcaf0be1768bcfd139b
```

Import once, offline, with Python 3 and an admitted native target. Stored-Q8
target heads additionally require NumPy for the offline shared-weight check:

```sh
python3 tools/import_mtp.py \
  --source /path/to/mtp-Mellum2.1-12B-A2.5B-Thinking-BF16.gguf \
  --target "$PADRA_ARTIFACT" --output /path/to/native-mtp
```

The output directory must not exist. The importer checks the complete source
and target payload identities, architecture, tokenizer and shared tensor bytes.
It writes a native artifact; recurring inference does not load or convert GGUF.
The native draft payload is 857,061,376 bytes with SHA-256
`b03d5abcf0adf5f5089694668f723fb262f78dc104aeb14dcd00be7db40b8389`.
The runtime requires bounded regular manifest/payload files and validates their
identities, tensor inventory, types, shapes, offsets and payload checksum.

## Run

Use the common release build for ordinary and MTP execution:

```sh
cmake --preset release
cmake --build --preset release
```

Pass any local CUDA compiler and offline dependency options when configuring,
as described in the main [build instructions](../README.md#build).

```sh
build/release/padra-server \
  "$PADRA_ARTIFACT" "$PADRA_TOKENIZER_JSON" "$PADRA_TOKENIZER_SHA256" \
  8080 600000 8192 2048 0 bf16 \
  --mtp-artifact /path/to/native-mtp --mtp-depth 1
```

The flags follow the positional server arguments and may accompany `--config`.
`--mtp-depth` requires an artifact. Invalid depths and incompatible target/KV
profiles fail startup explicitly.

`--head-format bf16|q8` selects the output projection for both the target and
draft. Without an override, the server uses the format stored in the checkpoint.
The new GPU profiles use a prequantized Q8 head. Loading it performs validation
and one upload, with no head conversion or repack. Target and draft share the
same GPU matrix. Requesting BF16 from a stored-Q8 checkpoint fails explicitly.

Legacy BF16-head checkpoints remain loadable for reference. An explicit
`--head-format q8` converts their validated host head before upload; the BF16
GPU head is omitted. For new distributions, convert once offline as described
in [model packages](models.md). The recommended profiles use the stored format.

Q8 uses signed INT8 blocks of 32 weights with BF16 scales. Per-row inputs use
the same block codec, followed by integer dot products, FP32 accumulation and
BF16 logits. Admission includes the smaller weight arena and 12,240-byte shared
activation workspace, saving 212,324,400 resident bytes overall.

Q8 changes target logits and can change generated tokens or sampled outputs.
It is an explicit numerical profile, independent of the draft expert format.
The kernel/math profile reports the head format and converted-head checksum.
MTP verifies against the ordinary target using that same selected head format.
The [shared-head quality report](performance/shared-head-q8-2026-10-25.txt)
records paired teacher-forced and executable-code diagnostics. These are limited
local tests, not a general coding-quality certification.

`--mtp-expert-dispatch full|compact` selects verifier expert dispatch and also
requires an artifact. The default is `compact`. This path
builds an ascending active-expert list on the GPU and launches a fixed upper
bound of eight experts per verifier row. It preserves expert slots, projection
arithmetic and weighted-combine order. Both modes preallocate their dispatch
storage; no per-group allocation or graph recapture is needed.

`--mtp-draft-experts bf16|fp8|exl3` selects the separately pinned draft artifact
profile and requires an artifact. The default is `bf16`. The FP8 profile is an
optional memory profile, not the selected performance profile. Only its expert
gate/up/down weights use E4M3FN codes and BF16 row scales; activations and
reconstructed weights retain the BF16 arithmetic boundary. Its norms, router
and dense projections remain unchanged. The shared target head follows the
independent `--head-format` selection.

Create that separate artifact offline using Python 3 with NumPy installed:

```sh
python3 tools/quantize_mtp_experts.py \
  --source-native /path/to/native-mtp --output /path/to/native-mtp-fp8
```

Then select both `--mtp-artifact /path/to/native-mtp-fp8` and
`--mtp-draft-experts fp8`. Each profile rejects incompatible payloads.
The FP8 payload is 461,223,936 bytes with SHA-256
`2b6b785cea1296bdf5a45abb89d7306208288d761951c16a6d7ace0ab1252d30`.
The replacement arena saves 377.5 MiB; it does not add a private output head.
This does not increase the engine's 131,072-token admission ceiling.
D1 memory/throughput qualification uses the common 32K-capacity fixtures and two
held-out code prompts. D2–D4 have state, sampling and API checks; their FP8
throughput and full-context task quality are not qualified.

The EXL3 profile requires a separately prepared, calibrated native artifact:

```text
Profile: mellum21-mtp-exl3-k4-experts-bf16-f32-v1
Payload bytes: 263747584
SHA-256: 985312eda174b32499468fb60291584e63da3b930674ccb0c9a59a700d8e2464
```

Select `--mtp-artifact /path/to/native-mtp-exl3 --mtp-draft-experts exl3`.
Only the 64 draft experts' gate, up and down matrices use EXL3 K4 with H128
transforms and FP16 sidecars. All 13 nonexpert tensors are copied unchanged
from the pinned BF16/F32 artifact. The runtime consumes packed weights directly
through the existing EXL3 MMA kernels; it does not allocate a BF16 expert shadow.
The replacement saves 593,313,792 bytes (565.8 MiB) of resident memory. Weights
remain on the GPU during generation; the benefit concerns GPU weight reads and
resident size, rather than per-token host/device weight copies.

The artifact uses routed draft activations from a separate calibration corpus,
with its identity checked at load. Preparation provenance and the measured
qualification scope are recorded in [the performance report](performance/mtp-2026-10-25.txt).
The selected opt-in throughput configuration is D1 with EXL3 draft experts and
the shared Q8 head:

```sh
build/release/padra-server \
  "$PADRA_ARTIFACT" "$PADRA_TOKENIZER_JSON" "$PADRA_TOKENIZER_SHA256" \
  8080 600000 131072 2048 0 bf16 \
  --mtp-artifact "$PADRA_MTP_EXL3_ARTIFACT" --mtp-depth 1 \
  --mtp-draft-experts exl3 --head-format q8
```

EXL3 draft experts also work with the BF16 head. Their proposals can change, but
the target still chooses every emitted token. D0–D4 state and API checks cover
both head formats; D1 throughput covers natural 4K/16K prompts and two additional
code prompts. Higher-depth throughput and general task quality are not qualified.
The shared Q8 head's target-quality caveat above remains applicable.

Embedding and output projection share the validated target allocation. The
draft owns an independent 1024-token BF16 KV ring. Admission includes all draft
weights, KV, hidden rows, verifier workspaces and rollback buffers. BF16 session
checkpoints also retain the draft ring and the preceding target hidden row.
Graph allocations receive the existing post-capture reserve check. Admission
never changes context, precision, chunk size or offload policy to make MTP fit.

## Execution contract

The draft consumes the current input token and the preceding target's
post-final-norm hidden state. Prompt catch-up initializes its own KV from those
aligned inputs. Every committed prefill chunk retains a complete visible draft
window, including at cancellation and append boundaries.
Catch-up batches at least four visible rows through the dead target-prefill
scratch buffers, preserving each row's compensated projection and KV rounding.
Shorter tails and decode-time repairs retain the single-row path. No additional
resident workspace is needed for batched catch-up.

A group verifies up to D+1 causal target input rows. Target normalization,
projection, attention and expert arithmetic retain the ordinary per-row
contract. Packed Q/K/V projections and shared expert tiles reuse
weights across verifier rows. Residual/post-normalization, router projection and
routing process the verifier rows in batched launches, writing directly into
existing verifier buffers. With BF16 target KV, two-row verification batches attention split, merge
and encoding. Q8 and Q8_0 use the ordinary per-row attention path. Adjacent rows share a KV/MMA tile when their causal partitions
match; other positions retain the ordinary independent partitions. Overwritten
local history is read from the pre-group snapshot, so a later row's ring writes
cannot affect an earlier row's attention. Rejected rows are restored before the next group;
retained draft KV is repaired with the actual target hidden states. Q8 rollback
copies encoded K/V rows and their scales byte-for-byte; Q8_0 inline scales are
included in each saved row. The draft ring remains BF16 for all target formats.

Only target-selected outputs reach token callbacks and the HTTP stream. A draft
token is accepted when it equals the ordinary target-selected token at that
position. Target RNG steps and penalty history advance with emitted outputs.
This is deterministic target-sample coupling, not probability-ratio sampling.
The target's original sampling, EOS, stop, output-budget and cancellation
behavior applies. Ordinary target steps handle short budget tails and a draft
EOS boundary. Reset, append and checkpoint restore retain their normal session
contracts.

Chat Completions and Responses use the existing API schemas. `/v1/models`
reports `mtp_depth` (zero for ordinary) and `mtp_resident_bytes`. Admin metrics
include depth, resident draft bytes, verification groups, draft calls, proposed
and accepted tokens, and ordinary tails. Decode timing includes drafting,
verification, selection and state repair. Prefill and graph setup are separate.
Speculative bursts are not a measurement of each individual token's latency.
