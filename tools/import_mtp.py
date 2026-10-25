#!/usr/bin/env python3
"""Import the pinned official Mellum2.1 BF16 MTP GGUF into a native artifact.

Large matrices remain BF16; upstream F32 norms/router remain F32. Target
embedding/output weights are verified against the target (using the identical
offline codec for a stored Q8 head) and physically shared by
the runtime. Conversion is offline and never runs in the generation loop.
"""
import argparse
import hashlib
import json
import math
import mmap
from pathlib import Path
import struct

SOURCE_SHA256 = "9c7b2c37dcdcb28140118debc81175cfc22bb00a7aa51bcaf0be1768bcfd139b"
SOURCE_BYTES = 1766609280
SOURCE_REVISION = "4159d684ffdb09ba89dc2a58feb3e75c54eef036"
MODEL_REVISION = "92ddae9fc7665e9f801d141d2e5a6b2caf2460c4"
TOKENIZER_SHA256 = "58548a346eb073e5132bf7d8ad17dc6971bca36ade378ca4d2bfbc49bf60da2a"
PROFILE = "mellum21-mtp-bf16-f32-v1"


def inventory():
    result = {
        "output.weight": ([2304, 98304], 30),
        "token_embd.weight": ([2304, 98304], 30),
        "output_norm.weight": ([2304], 0),
        "blk.28.nextn.eh_proj.weight": ([4608, 2304], 30),
        "blk.28.ffn_gate_inp.weight": ([2304, 64], 0),
        "blk.28.ffn_down_exps.weight": ([896, 2304, 64], 30),
        "blk.28.ffn_gate_exps.weight": ([2304, 896, 64], 30),
        "blk.28.ffn_up_exps.weight": ([2304, 896, 64], 30),
    }
    for name in ["attn_norm", "ffn_norm", "nextn.enorm", "nextn.hnorm",
                 "nextn.shared_head_norm"]:
        result[f"blk.28.{name}.weight"] = ([2304], 0)
    for name in ["attn_q_norm", "attn_k_norm"]:
        result[f"blk.28.{name}.weight"] = ([128], 0)
    for name, rows, cols in [("attn_q", 4096, 2304), ("attn_k", 512, 2304),
                             ("attn_v", 512, 2304), ("attn_output", 2304, 4096)]:
        result[f"blk.28.{name}.weight"] = ([cols, rows], 30)
    return result


class Header:
    def __init__(self, source):
        self.source = source
        self.at = 0

    def take(self, length):
        if length < 0 or length > 8 * 1024 * 1024 - self.at:
            raise ValueError("GGUF header exceeds bounded extent")
        value = self.source[self.at:self.at + length]
        if len(value) != length:
            raise ValueError("truncated GGUF header")
        self.at += length
        return value

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def string(self):
        return self.take(self.number("Q")).decode("utf-8", errors="strict")

    def value(self, kind, nested=False):
        if kind == 8:
            return self.string()
        if kind == 9:
            element, count = self.number("I"), self.number("Q")
            if nested or count > 400000:
                raise ValueError("unsupported GGUF metadata array")
            return [self.value(element, True) for _ in range(count)]
        types = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
                 7: "?", 10: "Q", 11: "q", 12: "d"}
        if kind not in types:
            raise ValueError("unsupported GGUF metadata type")
        return self.number(types[kind])

    def parse(self):
        if self.take(4) != b"GGUF" or self.number("I") != 3:
            raise ValueError("expected little-endian GGUF v3")
        tensor_count, metadata_count = self.number("Q"), self.number("Q")
        if tensor_count != 19 or metadata_count > 256:
            raise ValueError("unexpected MTP inventory")
        metadata = {}
        for _ in range(metadata_count):
            name = self.string()
            if name in metadata:
                raise ValueError("duplicate metadata key")
            metadata[name] = self.value(self.number("I"))
        expected = inventory()
        tensors = {}
        for _ in range(tensor_count):
            name = self.string()
            rank = self.number("I")
            if name not in expected or name in tensors or rank not in (1, 2, 3):
                raise ValueError("unknown/duplicate tensor or unsupported rank")
            dims = [self.number("Q") for _ in range(rank)]
            kind, offset = self.number("I"), self.number("Q")
            if (dims, kind) != expected[name]:
                raise ValueError(f"wrong shape/type: {name}")
            tensors[name] = dict(dimensions=dims, type=kind, offset=offset,
                                 bytes=math.prod(dims) * (2 if kind == 30 else 4))
        if set(tensors) != set(expected):
            raise ValueError("incomplete tensor inventory")
        alignment = metadata.get("general.alignment", 32)
        if alignment != 32:
            raise ValueError("unsupported GGUF alignment")
        start = (self.at + 31) // 32 * 32
        cursor = 0
        for tensor in sorted(tensors.values(), key=lambda x: x["offset"]):
            if tensor["offset"] != (cursor + 31) // 32 * 32:
                raise ValueError("overlapping/noncanonical tensor range")
            cursor = tensor["offset"] + tensor["bytes"]
            if start + cursor > len(self.source):
                raise ValueError("tensor outside file")
        if start + cursor != len(self.source):
            raise ValueError("unaccounted GGUF payload")
        required = {"general.architecture": "mellum", "mellum.block_count": 29,
                    "mellum.embedding_length": 2304, "mellum.expert_count": 64,
                    "mellum.expert_used_count": 8, "mellum.nextn_predict_layers": 1,
                    "mellum.attention.head_count": 32, "mellum.attention.head_count_kv": 4}
        if any(metadata.get(k) != v for k, v in required.items()):
            raise ValueError("incompatible Mellum MTP architecture")
        return metadata, tensors, start


def digest_file(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def import_model(source_path, target_path, output):
    source_path = source_path.resolve(strict=True)
    if not source_path.is_file() or source_path.stat().st_size != SOURCE_BYTES:
        raise ValueError("wrong BF16 MTP file size")
    if digest_file(source_path) != SOURCE_SHA256:
        raise ValueError("BF16 MTP source SHA-256 mismatch")
    manifest_path = target_path / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("invalid target manifest")
    target = json.loads(manifest_path.read_text())
    target_weights = target_path / "weights.bin"
    if (target.get("model_revision") != MODEL_REVISION or
            target.get("tokenizer_sha256") != TOKENIZER_SHA256 or
            target_weights.is_symlink() or not target_weights.is_file() or
            target_weights.stat().st_size != target.get("weights_bytes") or
            digest_file(target_weights) != target.get("weights_sha256")):
        raise ValueError("invalid target identity/weights")
    if output.exists():
        raise ValueError("output already exists; artifacts are immutable")
    with source_path.open("rb") as sf, target_weights.open("rb") as tf:
        with mmap.mmap(sf.fileno(), 0, access=mmap.ACCESS_READ) as source, mmap.mmap(tf.fileno(), 0, access=mmap.ACCESS_READ) as target_data:
            metadata, tensors, start = Header(source).parse()
            aliases = {"output.weight": "lm_head.weight", "token_embd.weight": "model.embed_tokens.weight",
                       "output_norm.weight": "model.norm.weight"}
            for name, alias in aliases.items():
                actual = target["tensors"][alias]
                tensor = tensors[name]
                raw = source[start + tensor["offset"]:start + tensor["offset"] + tensor["bytes"]]
                if name == "output.weight" and actual["format"] == "Q8_HEAD_BLOCK32":
                    import numpy as np
                    from quantize_shared_head import encode_head, CONVERSION
                    if (actual.get("shape") != [98304, 2304] or actual.get("block_size") != 32 or
                            actual.get("scale_dtype") != "BF16" or target.get("head_conversion") != CONVERSION or
                            set(actual["regions"]) != {"packed", "scales"}):
                        raise ValueError("unsupported stored shared head")
                    if (actual["regions"]["packed"]["bytes"] != 98304 * 2304 or
                            actual["regions"]["scales"]["bytes"] != 98304 * 2304 // 32 * 2):
                        raise ValueError("wrong stored shared-head region size")
                    values = np.frombuffer(raw, dtype="<u2")
                    for at in range(0, values.size, 256 * 2304):
                        packed, scales = encode_head(values[at:at + 256 * 2304])
                        for part, relative, expected in [("packed", at, packed), ("scales", at // 32 * 2, scales)]:
                            region = actual["regions"][part]
                            offset = region["offset"] + relative
                            if (offset < 0 or offset + len(expected) > len(target_data) or
                                    target_data[offset:offset + len(expected)] != expected):
                                raise ValueError("stored Q8 head differs from quantized shared source")
                    del values
                    continue
                if actual["format"] != "BF16":
                    raise ValueError("shared target tensor is not BF16")
                region = actual["regions"]["data"]
                original = target_data[region["offset"]:region["offset"] + region["bytes"]]
                if name == "output_norm.weight":
                    original = b"".join(struct.pack("<I", x[0] << 16) for x in struct.iter_unpack("<H", original))
                if raw != original:
                    raise ValueError(f"shared source/target tensor differs: {name}")
            output.mkdir(parents=True, exist_ok=False)
            result = {"schema_version": 1, "profile": PROFILE, "model_revision": MODEL_REVISION,
                      "tokenizer_sha256": TOKENIZER_SHA256, "source_revision": SOURCE_REVISION,
                      "source_sha256": SOURCE_SHA256, "source_bytes": SOURCE_BYTES,
                      "shared_target": aliases, "weights_file": "weights.bin", "tensors": {}}
            with (output / "weights.bin").open("xb") as dest:
                for name in sorted(set(tensors) - set(aliases)):
                    tensor = tensors[name]
                    offset = (dest.tell() + 255) // 256 * 256
                    dest.write(bytes(offset - dest.tell()))
                    begin = start + tensor["offset"]
                    # Bound memory independently of the largest expert tensor.
                    for at in range(0, tensor["bytes"], 1024 * 1024):
                        dest.write(source[begin + at:begin + min(at + 1024 * 1024, tensor["bytes"])])
                    result["tensors"][name] = {"shape": list(reversed(tensor["dimensions"])),
                                                "dtype": "BF16" if tensor["type"] == 30 else "F32",
                                                "offset": offset, "bytes": tensor["bytes"]}
                result["weights_bytes"] = dest.tell()
            result["weights_sha256"] = digest_file(output / "weights.bin")
            (output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"artifact": str(output), "weights_bytes": result["weights_bytes"],
                              "weights_sha256": result["weights_sha256"], "shared_bytes": sum(r["bytes"] for alias in aliases.values() for r in target["tensors"][alias]["regions"].values()),
                              "draft_attention": "sliding_attention", "draft_window": metadata["mellum.attention.sliding_window"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import_model(args.source, args.target, args.output)
