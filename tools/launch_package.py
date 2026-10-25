#!/usr/bin/env python3
"""Verify and launch a stored-Q8 Padra target package with its EXL3 MTP draft."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def verify_package(root):
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = root / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError('Nonregular or unsafe package file: ' + name)
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError('Package hash mismatch: ' + name)
    return json.loads((root / 'release.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--server', default=shutil.which('padra-server'))
    parser.add_argument('--profile', choices=['8gb', '12gb', '16gb', 'k4'])
    parser.add_argument('--mtp-artifact', type=Path)
    parser.add_argument('--mtp-depth', type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument('--ordinary', action='store_true')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    try:
        root = args.package.resolve()
        release = verify_package(root)
        if args.verify_only:
            print('Package hashes and regular-file layout verified')
            return
        rate = release['expert_rate']
        profile = args.profile or {3: '8gb', 4: 'k4', 6: '16gb'}[rate]
        configurations = {'8gb': (3, 256, 8, 'q8_0'), '12gb': (6, 256, 12, 'q8_0'),
                          '16gb': (6, 2048, 0, 'bf16'), 'k4': (4, 2048, 12, 'bf16')}
        required_rate, chunk, budget, kv = configurations[profile]
        if rate != required_rate:
            raise ValueError('Profile does not match this target checkpoint')
        if not args.server or not Path(args.server).is_file():
            raise ValueError('Provide --server /path/to/padra-server from the pinned engine')
        if not 1024 <= args.port <= 65535:
            raise ValueError('Port must be 1024..65535')
        server = Path(args.server).resolve()
        cmd = [str(server), str(root / 'native'), str(root / 'tokenizer.json'),
               release['tokenizer_sha256'], str(args.port), '600000', '131072',
               str(chunk), str(budget), kv, '--head-format', 'q8']
        if not args.ordinary:
            draft = (args.mtp_artifact or root.parent / 'MTP-EXL3-K4/native').resolve()
            manifest_path, weights = draft / 'manifest.json', draft / 'weights.bin'
            if any(p.is_symlink() or not p.is_file() for p in [manifest_path, weights]):
                raise ValueError('Provide the regular native EXL3 K4 MTP artifact')
            manifest = json.loads(manifest_path.read_text())
            if (manifest.get('profile') != 'mellum21-mtp-exl3-k4-experts-bf16-f32-v1' or
                    manifest.get('weights_sha256') != '985312eda174b32499468fb60291584e63da3b930674ccb0c9a59a700d8e2464' or
                    weights.stat().st_size != 263747584):
                raise ValueError('Draft identity mismatch')
            # The engine independently verifies the complete draft payload hash.
            cmd += ['--mtp-artifact', str(draft), '--mtp-depth', str(args.mtp_depth),
                    '--mtp-draft-experts', 'exl3', '--mtp-expert-dispatch', 'compact']
        if args.dry_run:
            print(json.dumps({'profile': profile, 'command': cmd}, indent=2))
        else:
            os.execv(cmd[0], cmd)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
