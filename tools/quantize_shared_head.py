#!/usr/bin/env python3
"""Convert an admitted EXL3 checkpoint to one stored Q8-block32 target/draft head.

Only lm_head.weight changes. Requires NumPy; runs on CPU, never loads GPU weights.
The destination must not exist. A failure leaves .partial files for inspection.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

PARENTS = {
    '0f30d4ead6ae3d0683ec1eb9ad7af1adfebe343d804e3313e8c01d6c31f76f0b': 4,
    '2a19089d1d50eb7f9dfec63ebcd3e066591db34bcada038cf5072a7602056f57': 6,
    '6aa0adfb5ba3b137ac547a15cba5d8c80a0020737c26c4be7dcc695fbf887271': 3,
}
CONVERSION = 'symmetric-int8-block32-bf16scale-rne-v1'


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def encode_head(raw):
    """Same FP32 divide, BF16-scale ties-to-even and integer ties-to-even as C++."""
    values = (raw.astype(np.uint32) << 16).view(np.float32).reshape(-1, 32)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite BF16 head')
    maximum = np.abs(values).max(axis=1)
    exact = np.where(maximum == 0, np.float32(1), maximum / np.float32(127))
    bits = exact.view(np.uint32)
    sbits = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype('<u2')
    scale = (sbits.astype(np.uint32) << 16).view(np.float32)
    if not (np.isfinite(scale).all() and (scale > 0).all()):
        raise ValueError('invalid BF16 head scale')
    packed = np.clip(np.rint(values / scale[:, None]), -127, 127).astype('i1')
    return packed.tobytes(), sbits.tobytes()


def convert(source, destination):
    manifest_path, weights_path = source / 'manifest.json', source / 'weights.bin'
    for path in (manifest_path, weights_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError('source must contain regular, non-symlink files')
    source_bytes = manifest_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    rate = PARENTS.get(source_hash)
    if rate is None:
        raise ValueError('source manifest is not an admitted K3/K4/K6 BF16-head checkpoint')
    original = json.loads(source_bytes)
    if weights_path.stat().st_size != original['weights_bytes'] or sha_file(weights_path) != original['weights_sha256']:
        raise ValueError('source weight identity mismatch')
    if destination.exists():
        raise ValueError('destination already exists; never overwrite a checkpoint')
    destination.mkdir(parents=True)
    matrix = original['tensors']['lm_head.weight']
    if matrix['format'] != 'BF16' or matrix['shape'] != [98304, 2304]:
        raise ValueError('unsupported head shape/format')
    region = matrix['regions']['data']
    raw = np.memmap(weights_path, mode='r', dtype='<u2', offset=region['offset'], shape=(98304, 2304))
    codes = bytearray()
    scales = bytearray()
    for row in range(0, 98304, 256):
        packed, scale = encode_head(raw[row:row + 256])
        codes.extend(packed)
        scales.extend(scale)
    del raw
    head_hash = hashlib.sha256(codes + scales).hexdigest()
    manifest = copy.deepcopy(original)
    manifest['profile'] = f'mellum16-exl3-k{rate}-fp8-q8head-v2'
    manifest['parent_manifest_sha256'] = source_hash
    manifest['parent_weights_sha256'] = original['weights_sha256']
    manifest['head_conversion'] = CONVERSION
    manifest['head_sha256'] = head_hash
    manifest['quality_status'] = 'unqualified'
    replacement = manifest['tensors']['lm_head.weight']
    replacement.update(format='Q8_HEAD_BLOCK32', block_size=32, scale_dtype='BF16')
    replacement['regions'] = {'packed': {'offset': 0, 'bytes': len(codes)},
                              'scales': {'offset': len(codes), 'bytes': len(scales)}}
    digest = hashlib.sha256()
    cursor = 0
    with weights_path.open('rb') as src, (destination / 'weights.bin.partial').open('xb') as dst:
        def write(block):
            dst.write(block)
            digest.update(block)
        for name, tensor in sorted(manifest['tensors'].items()):
            for part, r in sorted(tensor['regions'].items()):
                offset = (cursor + 255) // 256 * 256
                write(bytes(offset - cursor))
                if name == 'lm_head.weight':
                    write(codes if part == 'packed' else scales)
                else:
                    src.seek(original['tensors'][name]['regions'][part]['offset'])
                    remaining = r['bytes']
                    while remaining:
                        block = src.read(min(remaining, 8 << 20))
                        if not block:
                            raise ValueError('source truncated during conversion')
                        write(block)
                        remaining -= len(block)
                r['offset'] = offset
                cursor = offset + r['bytes']
    if cursor != original['weights_bytes'] - 212336640:
        raise ValueError('unexpected repacked weight size')
    manifest['weights_bytes'], manifest['weights_sha256'] = cursor, digest.hexdigest()
    text = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    (destination / 'manifest.json.partial').write_bytes(text)
    # Re-read the persisted payload before making its final manifest visible.
    if sha_file(destination / 'weights.bin.partial') != manifest['weights_sha256']:
        raise ValueError('persisted weight hash mismatch')
    (destination / 'weights.bin.partial').rename(destination / 'weights.bin')
    (destination / 'manifest.json.partial').rename(destination / 'manifest.json')
    return dict(expert_rate=rate, manifest_sha256=hashlib.sha256(text).hexdigest(),
                weights_sha256=manifest['weights_sha256'], weights_bytes=cursor,
                head_sha256=head_hash, parent_manifest_sha256=source_hash,
                parent_weights_sha256=original['weights_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.destination), indent=2))
