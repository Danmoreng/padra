# GPU profiles

The profiles in `profiles/` select 131,072 tokens of total context and MTP D1.
They use one stored Q8-block32 output head shared by target and draft, with no
BF16 head shadow. The draft's 64 experts use EXL3 K4 (4 bits/weight;
4.0248 including FP16 sidecars). Its other 13 tensors remain unchanged, and its
1024-token KV ring remains BF16. Target attention projections, routers, norms
and embeddings retain their existing precision.

| GPU class | Target experts | Target KV | Prefill chunk | Admission ceiling |
| --- | --- | --- | ---: | --- |
| 16 GB | EXL3 K6 | BF16 | 2048 | actual device memory |
| 12 GB | EXL3 K6 | Q8_0 | 256 | 12 GiB |
| 8 GB | EXL3 K3 | Q8_0 | 256 | 8 GiB |

The runtime admits the complete model, both KV caches, working buffers and a
700 MiB reserve, then checks available memory again after graph creation.
The reserve is headroom, not an additional allocated buffer. BF16 target KV
also supports a local prefix checkpoint. Q8 sessions replay history instead.
No profile silently reduces context or precision or offloads weights to the CPU.

These are experimental SM120/SM120a profiles. Measurements use an RTX 5080
Laptop GPU with 16 GB VRAM. The 8/12 GiB limits are explicit admission ceilings
on that GPU; they do not establish performance or driver overhead on physical
8/12 GB cards. See the measured memory, throughput and validation scope in
[the profile report](performance/gpu-profiles-2026-10-25.txt).
Successful execution at 131K does not establish retrieval or coding quality at
that context length.

## Prepare and run

Build the common engine once:

```sh
cmake --preset release
cmake --build --preset release
```

Pass compiler paths and offline dependency options as described in the main
[build instructions](../README.md#build). The same binary serves all profiles.

Supply the matching stored-Q8 native target, the pinned EXL3 MTP draft and the
original tokenizer. The launcher validates their identities before starting;
the native loader additionally hashes both weight payloads before upload.

```sh
python3 tools/run_profile.py --profile 16gb \
  --artifact /path/to/K6-q8head/native \
  --mtp-artifact /path/to/mtp-exl3/native \
  --tokenizer /path/to/tokenizer.json --port 8080
```

Use `12gb` with K6 or `8gb` with K3. `--dry-run` validates the inputs and prints
the exact server command. `--server` selects another compiled binary, and
`--config` passes a server configuration file. The context, cache, chunk,
shared-head format and MTP depth come from the selected profile.

For the weight identities and offline head conversion, see
[model packages](models.md). Draft preparation and its pinned EXL3 payload are
described in [MTP](mtp.md). Other MTP depths remain available through the direct
server CLI; greater depth is not guaranteed to be faster.
