#!/usr/bin/env python3
"""Create the separate pinned Mellum draft FP8-expert W8A16 artifact.

Only expert gate/up/down weights change. Scale is ceil_BF16(maxabs/448),
zero rows use1. Codes use E4M3FN nearest-even after FP32 division, finite
saturation at448, canonical positive zero. Decoding rounds code*scale toBF16.
The original BF16 native artifact is immutable and independently hash-checked.
"""
import argparse
import hashlib
import json
import math
import mmap
from pathlib import Path
import numpy as np
from import_mtp import (PROFILE as PARENT_PROFILE, MODEL_REVISION, SOURCE_REVISION,
                        SOURCE_SHA256, SOURCE_BYTES, TOKENIZER_SHA256, inventory)

PROFILE = "mellum21-mtp-fp8-experts-bf16-f32-v1"
CONVERSION = "e4m3fn-rne-f32div-ceilbf16-maxabs448-zero1-positivezero-bf16decode-v1"
PARENT_SHA256 = "b03d5abcf0adf5f5089694668f723fb262f78dc104aeb14dcd00be7db40b8389"
PARENT_BYTES = 857061376
EXPERTS = {f"blk.28.ffn_{part}_exps.weight" for part in ["gate", "up", "down"]}
ALIASES = {"output.weight": "lm_head.weight", "token_embd.weight": "model.embed_tokens.weight",
           "output_norm.weight": "model.norm.weight"}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def quantize(source, output):
    manifest_path = source / "manifest.json"
    weights_path = source / "weights.bin"
    for p in [manifest_path, weights_path]:
        if p.is_symlink() or not p.is_file():
            raise ValueError("source must contain regular files")
    if manifest_path.stat().st_size > 65536:
        raise ValueError("manifest too large")
    parent = json.loads(manifest_path.read_text())
    expected = dict(profile=PARENT_PROFILE, schema_version=1, model_revision=MODEL_REVISION,
                    source_revision=SOURCE_REVISION, source_sha256=SOURCE_SHA256,
                    source_bytes=SOURCE_BYTES, tokenizer_sha256=TOKENIZER_SHA256,
                    weights_file="weights.bin", weights_bytes=PARENT_BYTES,
                    weights_sha256=PARENT_SHA256, shared_target=ALIASES)
    if any(parent.get(k) != v for k, v in expected.items()):
        raise ValueError("parent identity mismatch")
    if weights_path.stat().st_size != PARENT_BYTES or digest(weights_path) != PARENT_SHA256:
        raise ValueError("parent payload mismatch")
    shapes = {k: v for k, v in inventory().items() if k not in ALIASES}
    if set(parent["tensors"]) != set(shapes):
        raise ValueError("parent inventory mismatch")
    cursor = 0
    for name in sorted(shapes):
        dims, kind = shapes[name]
        t = parent["tensors"][name]
        size = math.prod(dims) * (4 if kind == 0 else 2)
        if (t != dict(shape=list(reversed(dims)), dtype="F32" if kind == 0 else "BF16",
                      offset=(cursor + 255) // 256 * 256, bytes=size)):
            raise ValueError("parent descriptor mismatch: " + name)
        cursor = t["offset"] + size
    if cursor != PARENT_BYTES or output.exists():
        raise ValueError("parent bounds or existing output")
    # Positive finite E4M3FN codes0..126; code127 isNaN and never emitted.
    code = np.arange(127, dtype=np.int32)
    exponent, mantissa = code >> 3, code & 7
    lut = np.where(exponent == 0, mantissa / 512., (1. + mantissa / 8.) * np.exp2(exponent - 7)).astype(np.float32)
    output.mkdir(parents=True, exist_ok=False)
    result = {k: v for k, v in parent.items() if k != "tensors"}
    result.update(profile=PROFILE, conversion=CONVERSION, parent_weights_sha256=PARENT_SHA256, tensors={})
    audit = {}
    with weights_path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as raw, (output / "weights.bin").open("xb") as dest:
        for name in sorted(shapes):
            t = parent["tensors"][name]
            offset = (dest.tell() + 255) // 256 * 256
            dest.write(bytes(offset - dest.tell()))
            view = memoryview(raw)[t["offset"]:t["offset"] + t["bytes"]]
            original_sha = hashlib.sha256(view).hexdigest()
            if name not in EXPERTS:
                dest.write(view)
                result["tensors"][name] = {**t, "offset": offset}
                audit[name] = dict(source_sha256=original_sha, unchanged=True)
                del view
                continue
            experts, rows, cols = t["shape"]
            all_scales = np.empty((experts, rows), dtype="<u2")
            codes_hash = hashlib.sha256()
            histogram = np.zeros(256, dtype=np.int64)
            squared_error = 0.
            squared_original = 0.
            for e in range(experts):
                start = e * rows * cols * 2
                bits = np.frombuffer(view[start:start + rows * cols * 2], dtype="<u2").reshape(rows, cols)
                values = (bits.astype(np.uint32) << 16).view(np.float32)
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite parent weights")
                maxima = np.max(np.abs(values), axis=1)
                exact_scale = (maxima / np.float32(448)).astype(np.float32)
                sbits = exact_scale.view(np.uint32)
                ceil_bits = ((sbits >> 16) + ((sbits & 65535) != 0)).astype(np.uint16)
                ceil_bits[maxima == 0] = 0x3f80
                scales = (ceil_bits.astype(np.uint32) << 16).view(np.float32)
                if not np.isfinite(scales).all() or not (scales > 0).all():
                    raise ValueError("invalid stored scale")
                normalized = np.abs(values / scales[:, None]).astype(np.float32)
                if np.max(normalized) > 448:
                    raise ValueError("scale rounding overflow")
                high = np.searchsorted(lut, normalized, side="left").clip(0, 126)
                low = np.maximum(0, high - 1)
                dl = normalized.astype(np.float64) - lut[low]
                dh = lut[high].astype(np.float64) - normalized
                pick_high = (dh < dl) | ((dh == dl) & ((high & 1) == 0))
                chosen = np.where(pick_high, high, low).astype(np.uint8)
                chosen |= np.where(np.signbit(values) & (chosen != 0), 128, 0).astype(np.uint8)
                encoded = chosen.tobytes()
                dest.write(encoded)
                codes_hash.update(encoded)
                histogram += np.bincount(chosen.ravel(), minlength=256)
                all_scales[e] = ceil_bits
                decoded = lut[chosen & 127] * scales[:, None]
                decoded = np.where(chosen & 128, -decoded, decoded).astype(np.float32)
                dbits = decoded.view(np.uint32)
                rounded = ((dbits + np.uint32(0x7fff) + ((dbits >> 16) & 1)) >> 16).astype(np.uint16)
                decoded = (rounded.astype(np.uint32) << 16).view(np.float32)
                delta = decoded.astype(np.float64) - values
                squared_error += float(np.sum(delta * delta))
                squared_original += float(np.sum(values.astype(np.float64) ** 2))
                del bits, values
            scale_offset = dest.tell()
            scale_bytes = all_scales.tobytes()
            dest.write(scale_bytes)
            tensor = dict(shape=t["shape"], dtype="FP8_E4M3FN", offset=offset,
                          bytes=dest.tell() - offset, scale_dtype="BF16", scale_offset=scale_offset,
                          scale_bytes=len(scale_bytes), source_sha256=original_sha,
                          codes_sha256=codes_hash.hexdigest(), scales_sha256=hashlib.sha256(scale_bytes).hexdigest())
            result["tensors"][name] = tensor
            audit[name] = {**tensor, "relative_rmse": math.sqrt(squared_error / squared_original),
                           "code_histogram": histogram.tolist()}
            del view
        result["weights_bytes"] = dest.tell()
    result["weights_sha256"] = digest(output / "weights.bin")
    (output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output.parent / "CONVERSION_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(dict(artifact=str(output), weights_bytes=result["weights_bytes"], weights_sha256=result["weights_sha256"], saving_bytes=PARENT_BYTES-result["weights_bytes"])))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-native", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    quantize(a.source_native, a.output)
