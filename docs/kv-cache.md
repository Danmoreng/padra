# KV cache storage

The native controlled-chunk server supports BF16 KV and explicitly selected `q8` or
`q8_0` KV at capacities 8192, 16384, 32768, 65536 and 131072. Context capacity includes
both input and generated tokens. The default storage format is BF16.

Select a quantized cache with the final positional server argument:

```sh
build/padra-server "$PADRA_ARTIFACT" "$PADRA_TOKENIZER_JSON" \
  "$PADRA_TOKENIZER_SHA256" 8080 600000 131072 256 8 q8_0
```

This requests a 131072-token cache, prefill chunk 256 and an 8 GiB device-memory
ceiling. Admission depends on the loaded model and actual available device memory;
the command does not imply that every native checkpoint fits. Physical 8/12 GiB
GPUs are not qualified. Oversized configurations fail without changing the selected
context or storage precision. The runtime includes its 700 MiB admission reserve
and rechecks available memory after graph creation.

`q8` uses `q8_token_head128_f32scale_v1`: signed codes in [-127, 127]
with independent FP32 K and V scales for each token and 128-element KV head.
Keys are encoded after BF16 Q/K normalization and RoPE; values come from their
BF16 boundary. Attention reconstructs consumed tiles as BF16 and uses BF16
matrix multiplication with FP32 accumulation and softmax statistics.

`q8_0` uses `q8_0_token512_block32_f16scale_rot_k128_v64_v2`: each logical block of 32 signed
codes has one FP16 scale. Each K or V token row stores16 FP16 scales followed
by512 signed codes, with32-byte-aligned head payloads. Four query heads share
asynchronously staged K/V tiles during decode. Normalized Hadamard rotations cover 128 key elements
and 64 value elements before encoding. Decode rotates and quantizes queries to
Q8 with FP32 scales, computes integer QK dot products using DP4A, accumulates
softmax and weighted values in FP32, then applies the inverse value rotation
before the BF16 output boundary. Prefill uses the BF16 attention kernels with
rotated queries and decoded K/V. Its FP32 result is inverse-rotated before BF16
rounding. This is a native cache layout inspired by llama.cpp Q8_0; it does not
provide GGUF loading or cache interchange. Both cache formats leave model
weights unchanged and are distinct from FP8 storage.

With four KV heads, per-token cache payloads including scales are:

| Storage | Bytes per token per layer | Payload at capacity 131072 |
| --- | ---: | ---: |
| BF16 | 2048 | 1923088384 |
| `q8` | 1056 | 991592448 |
| `q8_0` | 1088 | 1021640704 |

Seven global layers retain the selected capacity; 21 local layers retain a
1024-token ring. Model weights, RoPE tables, temporary buffers, graph resources
and the admission reserve are additional. Neither format retains a BF16 cache
copy for every layer. `q8_0` allocates one BF16 prefix workspace of
`capacity * 2048` bytes, reused across all layers during prefill (256 MiB at
capacity 131072), plus a 4608-byte query workspace. Current chunks use the same
codec as older entries; local rings commit after attention reads finish.

Both quantized formats are lossy and can change logits and generated tokens.
The strict BF16 logit-equivalence gate is not satisfied; the small diagnostic
quality sample does not qualify either format as a quality-equivalent replacement. Successful allocation and
execution through 131072 do not establish general long-context model quality or
equivalence to BF16. BF16 remains the default quality reference. Inspect
`GET /v1/models` for the active `kv_format` and `execution_profile`.

The runtime supports reset, cancellation, forward prefix append and local prefix
checkpoint reuse with all three cache formats. The server retains one snapshot
of the 21 local rings in their original format: 42 MiB for BF16, 21.65625 MiB for
`q8`, or 22.3125 MiB for `q8_0`. Quantized snapshots copy codes and scale bits
without decoding or requantizing them. Global rows before the retained anchor
remain in place. With MTP, the snapshot additionally retains the draft's BF16
ring and preceding target hidden state (2 MiB plus 9216 bytes).

Reuse applies to the matching resident Chat session or Responses continuation.
The runtime validates the canonical token prefix, runtime binding, generation,
capacity and cache format before restoring. A different GPU session, changed
prefix, reset or stale snapshot causes a full history replay through the server;
this is not a multi-session GPU cache. Usage reports the actually reused prefix
in `cached_tokens`. Restore preserves the original chunk boundaries. An anchor
within three tokens of a chunk end is moved back to leave four rows, preserving
the full-chunk expert arithmetic instead of switching to a narrow tail path.
Q8 trace/snapshot diagnostics and the legacy non-controlled
runtime are unsupported.

MTP supports both quantized target caches. Its rollback saves and restores the
encoded rows and their scales without dequantizing history. The draft retains
its separate 1024-token BF16 ring. See the [GPU profiles](gpu-profiles.md) for
the selected 131K/MTP combinations and measured memory requirements.
