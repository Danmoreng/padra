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
and runtime memory. EXL3 expert sidecars are FP16. Attention projections retain
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

The existing public [Padra EXL3 repository](https://huggingface.co/danmoreng/Mellum2.1-12B-A2.5B-Thinking-Padra-EXL3)
provides the legacy K4/K6 packages with BF16 heads. The stored-Q8 artifacts and
K3 profile are prepared locally; their publication is not established by these
instructions. A legacy package's bundled launcher is independent of the
131K/MTP profiles supplied here.

To obtain a legacy K6 package with the Hugging Face CLI:

```sh
export PADRA_MODEL_ROOT="$HOME/.cache/padra/mellum21-exl3"
hf download danmoreng/Mellum2.1-12B-A2.5B-Thinking-Padra-EXL3 \
  --include "K6/*" --local-dir "$PADRA_MODEL_ROOT"
```

Convert its admitted native directory once on CPU (Python 3 and NumPy required):

```sh
python3 tools/quantize_shared_head.py \
  "$PADRA_MODEL_ROOT/K6/native" "$PADRA_MODEL_ROOT/K6-q8head/native"
```

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
