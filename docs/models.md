# Model packages

Padra supports the native K4 and K6 packages of
[Mellum2.1-12B-A2.5B-Thinking Padra EXL3](https://huggingface.co/danmoreng/Mellum2.1-12B-A2.5B-Thinking-Padra-EXL3).
Both derive from JetBrains/Mellum2.1-12B-A2.5B-Thinking, revision
`92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`, and use the matching source tokenizer.

| Package | Expert weights | Weight payload | API model ID |
| --- | --- | --- | --- |
| [K4](https://huggingface.co/danmoreng/Mellum2.1-12B-A2.5B-Thinking-Padra-EXL3/tree/main/K4) | EXL3 4-bit with FP16 sidecars | 7.09 GB | `mellum21-exl3-k4-research` |
| [K6](https://huggingface.co/danmoreng/Mellum2.1-12B-A2.5B-Thinking-Padra-EXL3/tree/main/K6) | EXL3 6-bit with FP16 sidecars | 9.87 GB | `mellum21-exl3-k6-research` |

Sizes are decimal GB. Both packages retain FP8 attention projections and BF16
routers, norms, embeddings and output weights. The native manifest and weight
identities are checked before inference; the loaded package selects its K4 or K6
decoder automatically.

The loader also admits the pinned native K3 identity in
`src/runtime/native_artifacts.h`, using its own packed expert decoder and API
model ID `mellum21-exl3-k3-research`. The download instructions below describe
the K4 and K6 packages linked above.

## Download and launch

With the Hugging Face `hf` CLI installed, download the K6 package into a directory
outside the engine checkout:

```sh
export PADRA_MODEL_ROOT="$HOME/.cache/padra/mellum21-exl3"
hf download danmoreng/Mellum2.1-12B-A2.5B-Thinking-Padra-EXL3 \
  --include "K6/*" --local-dir "$PADRA_MODEL_ROOT"
python3 "$PADRA_MODEL_ROOT/K6/launch.py" --verify-only
python3 "$PADRA_MODEL_ROOT/K6/launch.py" --server ./build/padra-server --port 8080
```

Build the engine using the [main build instructions](../README.md#build) before
launching. The launcher requires Python 3.11+, verifies package hashes, and selects
an 8,192-token context, chunk 2048 and BF16 KV. Replace `K6` with `K4` in these
commands to use K4. Download into regular local files: the native loader rejects
symlinked manifests or weights.

`GET /v1/models` reports the active API model ID. Chat Completions and Responses
requests must use that ID; see the [API guide](api.md).

For direct engine arguments, pass the package's `native/` directory, its
`tokenizer.json`, and tokenizer SHA256
`58548a346eb073e5132bf7d8ad17dc6971bca36ade378ca4d2bfbc49bf60da2a`.
The supplied layout is Padra-native; loading it in standard ExLlamaV3 or other
EXL3 runtimes is not established.

## Memory and compatibility

Weights remain fully GPU-resident. The weight payload excludes KV cache, runtime
buffers, graphs and the 700 MiB admission reserve. K6 cannot fit an 8 GiB device
budget. Available memory and the selected context determine admission; an
oversized configuration fails explicitly without CPU offload or a precision
fallback.

The tested hardware is an SM120 RTX 5080 Laptop GPU with 16 GiB VRAM. Physical
8/12 GiB GPUs and other architectures are not qualified. The launcher uses 8,192
tokens; the direct engine's default is 131,072, subject to memory admission.
Loading or running a context does not establish model quality at that length.
