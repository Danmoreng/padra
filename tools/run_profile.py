#!/usr/bin/env python3
"""Launch one checked Padra 131K/MTP profile using explicitly supplied artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected a regular, non-symlink file: {path}')
    return path


def command(args):
    profile = json.loads((ROOT / 'profiles' / (args.profile + '.json')).read_text())
    manifest = regular(args.artifact / 'manifest.json').read_bytes()
    if hashlib.sha256(manifest).hexdigest() != profile['target']['manifest_sha256']:
        raise ValueError('Target checkpoint does not match the selected stored-Q8 profile')
    target = json.loads(manifest)
    if target['weights_bytes'] != regular(args.artifact / 'weights.bin').stat().st_size:
        raise ValueError('Target payload size mismatch')
    draft = json.loads(regular(args.mtp_artifact / 'manifest.json').read_bytes())
    if (draft.get('profile') != profile['draft']['format'] or
            draft.get('weights_sha256') != profile['draft']['weights_sha256'] or
            regular(args.mtp_artifact / 'weights.bin').stat().st_size != profile['draft']['weights_bytes']):
        raise ValueError('Draft checkpoint must be the pinned EXL3 K4 MTP artifact')
    # The native loader independently checks full weight hashes before uploading.
    if hashlib.sha256(regular(args.tokenizer).read_bytes()).hexdigest() != profile['tokenizer_sha256']:
        raise ValueError('Tokenizer identity mismatch')
    server = regular(args.server).resolve()
    cmd = [str(server), str(args.artifact.resolve()), str(args.tokenizer.resolve()),
           profile['tokenizer_sha256'], str(args.port), str(args.timeout_ms),
           str(profile['context_capacity']), str(profile['prefill_chunk']),
           str(profile['device_budget_gib']), profile['kv_format'],
           '--mtp-artifact', str(args.mtp_artifact.resolve()), '--mtp-depth', str(profile['mtp_depth']),
           '--mtp-expert-dispatch', 'compact', '--mtp-draft-experts', 'exl3', '--head-format', 'q8']
    if args.config:
        cmd += ['--config', str(regular(args.config).resolve())]
    return profile, cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['16gb', '12gb', '8gb'], required=True)
    parser.add_argument('--artifact', type=Path, required=True, help='Stored-Q8 target native directory')
    parser.add_argument('--mtp-artifact', type=Path, required=True, help='EXL3 K4 draft native directory')
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--server', type=Path, default=ROOT / 'build/release/padra-server')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--timeout-ms', type=int, default=600000)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and print the exact command')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or not 1 <= args.timeout_ms <= 600000:
        parser.error('Port must be 1024..65535 and timeout 1..600000 ms')
    try:
        profile, cmd = command(args)
        if args.dry_run:
            print(json.dumps({'profile': args.profile, 'qualification': profile['qualification'],
                              'command': cmd}, indent=2))
        else:
            os.execv(cmd[0], cmd)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
