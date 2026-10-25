# MTP porting and reproduction

The production CUDA implementation requires SM120/SM120a. The 8 GB profile is a
memory/precision configuration, not Pascal support. A GTX 1070 needs a separate
SM61 implementation; changing only `PADRA_CUDA_ARCHITECTURE` is insufficient.
The current CMake guard intentionally refuses unsupported architectures.

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
Use identical prompts and target precision for ordinary/MTP comparisons; join
all GPU jobs, retain warmups separately, and record graph/driver memory too.
The probe is a diagnostic executable, not a hardened input-facing service.

The current measured scope and identities are recorded in
[the profile report](../performance/gpu-profiles-2026-10-25.txt) and
[the MTP optimization report](../performance/mtp-2026-10-25.txt).
