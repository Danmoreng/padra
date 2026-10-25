# MTP porting and reproduction

MTP has separate SM120 and SM61 CUDA implementations. The SM61 backend uses
SIMT FP32 expert projections, a shared stored Q8 head and the pinned EXL3 K4
draft. Target artifacts may be K3 or K4. BF16 and native rotated Q8_0 KV are
supported; the different legacy `q8` head-scale cache layout is rejected.
CUDA 12.x is required to build for Pascal. The SM120 8 GB profile is not a
Pascal performance qualification.

## SM61 reproduction

Build with `PADRA_CUDA_ARCHITECTURE=61`, `PADRA_CUDA_BACKEND=simt`,
and native text enabled. `PADRA_PASCAL_DP4A_PREFILL=OFF` selects the default FP32
expert arithmetic. Setting it to `ON` selects approximate INT8 DP4A dense
prefill for K3 and K4; sparse prefill and ordinary/MTP Decode retain the exact
codec and FP32 arithmetic. Q8_0 runtime identities explicitly identify this
compute profile. Use the current published K3 directory (stored
Q8 head) and `MTP-EXL3-K4/native`; no BF16 output head or duplicate draft head
is uploaded. For an 8 GiB GPU:

```sh
build/sm61/padra-server /models/K3/native /models/K3/tokenizer.json \
  58548a346eb073e5132bf7d8ad17dc6971bca36ade378ca4d2bfbc49bf60da2a \
  8080 600000 8192 256 8 q8_0 \
  --head-format q8 --mtp-artifact /models/MTP-EXL3-K4/native \
  --mtp-draft-experts exl3 --mtp-depth 1
```

The verifier batches 2–5 target rows, shares FP8 weight reads and grouped EXL3
weights, and uses the same per-row attention and Q8 head arithmetic as ordinary
SM61 decoding. Tentative KV rows are backed up in their stored format and
restored after rejection. Retained draft rows are repaired using actual target
hidden states. RNG and penalty counts advance only for emitted target samples.
BF16 and Q8_0 session checkpoints also include draft KV and the preceding target
hidden row, retaining complete 256-token prefill chunks. Q8_0 selects the target
cache; the draft retains its native 1024-row BF16 ring (2 MiB). This does not
allocate a BF16 output head. SIMT uses a fixed verifier dispatch; SM120 research
flags do not select alternative SIMT kernels.

Enable `PADRA_BUILD_DIAGNOSTICS=ON` for `padra-simt-qualify`, or
`PADRA_ENABLE_MTP_PROBES=ON` for the artifact-backed `padra-mtp-probe` on either
backend. CTest's `simt-k3-experts`, `simt-verifier` and `simt-mtp-operators`
exercise the independent expert/projection references, encoded-cache attention
oracle and byte-level ring rollback. `padra-mtp-operators` below remains an
SM120-specific operator probe. GPU performance depends on speculation depth and
acceptance; enabled MTP is not a speed guarantee.

## Relevant execution paths

- `src/cuda/inference.cu`: ordinary and verifier orchestration, graph capture,
  target sampling and commits. Start from the selected ordinary target as the
  numerical reference before enabling speculative verification.
- `src/cuda/mtp.cuh`, `mtp_prime.cuh`, `mtp_rows.cuh`: draft state, prefill catch-up,
  encoded KV backup/rollback and batched row operations.
- `src/cuda/mtp_experts.cuh`: K3/K4/K6 verifier expert tile reuse, compact dispatch
  and per-row FP16 MMA with BF16 boundaries. Draft experts use EXL3 K4.
- `src/cuda/mtp_attention.cuh`, `mtp_fp8.cuh`: BF16 paired attention and packed
  FP8 projection paths. Q8/Q8_0 verification currently uses ordinary per-row
  attention and restores encoded cache bytes including scales.
- `src/cuda/head_q8.cuh`: one shared signed INT8-block32 head, BF16 block scales,
  integer dot products, FP32 accumulation and BF16 logit output.
- `src/runtime/mtp_schedule.h`, `mtp_memory.h`, `server_memory.h`: transaction
  schedule and complete resident-memory admission.
- `src/model/compiled.cpp`, `mtp.cpp`, `src/runtime/native_artifacts.h`: native
  tensor layouts and pinned identities. No duplicate BF16 GPU heads/experts.

Preserve ordinary/MTP token, final-logit, hidden-state, RNG/penalty, EOS,
callback/cancellation, append and ring-wrap contracts independently from speed.
A different numerical backend needs its own ordinary reference and explicit
qualification; do not label approximate changes bit-exact.

## Optional native diagnostics

On a supported CUDA device, configure the ordinary release build with
`-DPADRA_ENABLE_MTP_PROBES=ON`. This adds two diagnostics, not automatic CTest
jobs. Supply real admitted artifacts and an explicit process time/memory bound.

```sh
build/release/padra-mtp-operators /path/to/K3/native 3
build/release/padra-mtp-operators --checkpoint
compute-sanitizer --tool memcheck build/release/padra-mtp-operators /path/to/K3/native 3
build/release/padra-mtp-probe /path/to/K3/native /path/to/MTP-EXL3-K4/native \
  /path/to/requests.json /path/to/results.jsonl
```

The operator probe checks encoded BF16/Q8/Q8_0 cache restore against a byte
oracle and K3/K4/K6 verifier expert outputs against ordinary expert projection.
Use `synccheck` and `racecheck` separately as appropriate. The native probe runs
D0 through D4, or one depth supplied as a final argument. A request file is an
array, for example:

```json
[{
  "input": "/path/to/input.json",
  "expert_rate": 3,
  "kv_format": "q8_0",
  "capacity": 131072,
  "prefill_chunk": 256,
  "device_budget_gib": 8,
  "q8_head": true,
  "exl3_experts": true,
  "generate": 128
}]
```

The input file contains `{"input_ids": [1, 2, 3]}` with your actual tokenized
prompt. Startup options come from the first request. Each result records tokens,
final logit/hidden hashes, callbacks, committed/pending state, MTP acceptance,
memory admission and timings. Natural EOS can end generation before the budget.
Prefix snapshots are enabled for every cache format; `checkpoint_enabled: false`
on the first request disables them for a control run. A positive `checkpoint_anchor`
captures the canonical prefix; a subsequent `restore_checkpoint: true` restores
the retained snapshot, optionally extending it with a new `checkpoint_anchor`.
The requested anchor is an upper bound: the runtime may retain up to three fewer
tokens to avoid changing expert arithmetic at a partial chunk boundary.
The output includes `cached_tokens`, snapshot bytes, capture/restore counts and
`prepare_s`, which includes restore copies. The `--checkpoint` operator probe
checks encoded snapshots against a chronological byte oracle and exercises graph
replay and invalid-command guards without loading model weights.
Use identical prompts and target precision for ordinary/MTP comparisons; join
all GPU jobs, retain warmups separately, and record graph/driver memory too.
The probe is a diagnostic executable, not a hardened input-facing service.

The current measured scope and identities are recorded in
[the profile report](../performance/gpu-profiles-2026-10-25.txt) and
[the MTP optimization report](../performance/mtp-2026-10-25.txt).
The [prefix reuse report](../performance/prefix-reuse-2026-10-25.txt) records
the Q8 checkpoint tests, full-context memory and replay/reuse comparison.

For a fresh-replay comparison at chunk and ring boundaries, run
`tests/cuda/prefix_reuse.py --probe /path/to/padra-mtp-probe --artifact /path/to/target
--draft /path/to/draft --input /path/to/input.json --output-dir /new/result/directory
--expert-rate 3 --kv-format q8_0`. It tests Ordinary and MTP D1/D4, checks full
outputs and state, and preserves all requests and results. Supply at least
`3 * chunk + 2` input tokens; `--chunk` defaults to 256. Use `--kv-format q8` or
`bf16` for the other formats. This is an explicit GPU diagnostic, not a host CTest.

For SM61, add `--whole-chunks --chunk 256` and supply at least 1538 input tokens.
SIMT retains complete expert chunks, so the requested anchor rounds down to a
multiple of 256. This mode also checks that shorter prefixes produce no snapshot
and tests restore across the 1024-token local ring. Q8_0 snapshots preserve codes
and scales; MTP snapshots also preserve the draft ring and target hidden state.
