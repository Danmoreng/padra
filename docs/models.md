# Model packages

Padra loads pinned native EXL3 K3, K4 and K6 checkpoints derived from
JetBrains/Mellum2.1-12B-A2.5B-Thinking, revision
`92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`. Use the matching source tokenizer.
The recommended [131K/MTP GPU profiles](gpu-profiles.md) require a stored Q8
output head shared by target and draft. Each profile uses the same EXL3 K4 MTP
expert artifact; target and draft do not hold separate copies of shared weights.

| Native checkpoint | Target expert rate | Weight payload | API model ID |
| --- | --- | ---: | --- |
| K3 with stored Q8 head | 3 bits/weight | 5.493 GB | `mellum21-exl3-k3-research` |
| K4 with stored Q8 head | 4 bits/weight | 6.881 GB | `mellum21-exl3-k4-research` |
| K6 with stored Q8 head | 6 bits/weight | 9.655 GB | `mellum21-exl3-k6-research` |

Sizes are decimal GB and exclude the separate 0.264 GB draft payload, KV caches
and runtime memory. EXL3 expert sidecars (`suh`/`svh`) are IEEE FP16 input/output scaling vectors,
not head weights. They occupy 32.8125 MiB per target and remain unchanged. Attention projections retain
FP8; routers, norms and embeddings retain BF16. The output head uses signed
symmetric INT8 blocks of 32 values with one BF16 scale per block: 8.5 effective
bits/weight. It occupies 229.5 MiB and saves 202.5 MiB against BF16. The loader
uploads the stored representation directly, without runtime quantization or a
BF16 shadow. See [head quality and validation](performance/shared-head-q8-2026-10-25.txt).

BF16-head checkpoints are deprecated as the recommended profile format. They
remain loadable as compatibility and quality references. Their default head
execution stays BF16; explicit `--head-format q8` converts their head in host
memory before upload. New stored-Q8 checkpoints select Q8 automatically and
reject `--head-format bf16`. Neither path stores both heads on the GPU.

## Artifact availability and conversion

The public [Padra EXL3 repository](https://huggingface.co/danmoreng/Mellum2.1-12B-A2.5B-Thinking-Padra-EXL3)
provides stored-Q8 K3, K4 and K6 target packages and the EXL3 K4 MTP draft.
The complete release is pinned at revision
`c9029d1ace9e891ab41bee380f05f0a61c99bb79`. The latest native files in all three
target directories use the admitted Q8-head identities. Legacy BF16-head
checkpoints remain available through historical Hub revisions.

Download the K3 target and matching MTP draft:

```sh
export PADRA_MODEL_ROOT="$HOME/.cache/padra/mellum21-exl3"
hf download danmoreng/Mellum2.1-12B-A2.5B-Thinking-Padra-EXL3 \
  --revision c9029d1ace9e891ab41bee380f05f0a61c99bb79 \
  --include "K3/*" "MTP-EXL3-K4/*" "LICENSE" "NOTICE" \
  --local-dir "$PADRA_MODEL_ROOT"
(cd "$PADRA_MODEL_ROOT/K3" && sha256sum -c SHA256SUMS)
(cd "$PADRA_MODEL_ROOT/MTP-EXL3-K4" && sha256sum -c SHA256SUMS)
```

Replace `K3/*` with `K4/*` or `K6/*` to download another target; the draft is shared.
The paths for `tools/run_profile.py --profile 8gb` are
`K3/native`, `MTP-EXL3-K4/native` and `K3/tokenizer.json` under this directory.
The current CUDA engine requires SM120/SM120a; the 8 GB memory profile does not
provide GTX 1070/Pascal support. See [porting diagnostics](development/mtp-porting.md).

Each target package also provides `launch.py --verify-only` and
`launch.py --server /path/to/padra-server`; the launcher enables MTP and uses the
sibling `MTP-EXL3-K4/native` directory. K6 accepts `--profile 12gb` for Q8_0 KV.
K4 uses its BF16-KV configuration with a 12 GiB ceiling. `--dry-run` prints the
command; `--ordinary` disables MTP. The source is `tools/launch_package.py`.

## Optional conversion from a legacy reference

The published packages already contain the Q8 head. To reproduce a K6 conversion,
download the old BF16-head package into a separate directory:

```sh
export PADRA_LEGACY_ROOT="$HOME/.cache/padra/mellum21-legacy"
hf download danmoreng/Mellum2.1-12B-A2.5B-Thinking-Padra-EXL3 \
  --revision 9b9444c6decd04b9e798a7ef68b86f95f08fbaf2 \
  --include "K6/*" --local-dir "$PADRA_LEGACY_ROOT"
python3 tools/quantize_shared_head.py \
  "$PADRA_LEGACY_ROOT/K6/native" /path/to/fresh/K6-q8head/native
```

The converter uses Python 3 and NumPy and runs entirely on CPU.
The destination must not exist. Conversion verifies the complete source identity,
changes only `lm_head.weight`, copies all other tensors unchanged, writes fresh
native metadata and verifies the saved weight hash. It accepts only the three
pinned BF16-head parent identities. The output is deterministic; the engine
admits the exact resulting manifest and payload hashes listed in
`src/runtime/native_artifacts.h`. Downloaded manifests and weights must be
regular files, not symlinks.

Build the [common engine](../README.md#build), prepare the [MTP draft](mtp.md),
and launch through `tools/run_profile.py` as described in
[GPU profiles](gpu-profiles.md). Use the tokenizer SHA256
`58548a346eb073e5132bf7d8ad17dc6971bca36ade378ca4d2bfbc49bf60da2a`.
`GET /v1/models` reports the active model ID, which requests must use; see the
[API guide](api.md).

## Memory and compatibility

Weights remain GPU-resident. Admission includes target and draft weights, both
KV caches, working buffers and a 700 MiB reserve, followed by a check after
CUDA graph creation. Configurations that exceed available memory fail explicitly
without CPU offload or a precision fallback. See measured profile budgets in
the [profile report](performance/gpu-profiles-2026-10-25.txt).

The tested hardware is an SM120 RTX 5080 Laptop GPU with 16 GiB VRAM. Physical
8/12 GiB GPUs and other architectures are not qualified. Loading and executing
131,072 tokens does not establish model quality at that context length. The
native layout is Padra-specific; compatibility with standard ExLlamaV3 or other
EXL3 runtimes is not established.
