#!/usr/bin/env python3
"""Check only project-owned tracked C++/CUDA sources with the pinned formatter."""
import argparse
from pathlib import Path
import subprocess
import re

VERSION = '22.1.0'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--clang-format', default='clang-format')
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
version = subprocess.check_output([args.clang_format, '--version'], text=True)
if not re.search(r'\bversion ' + re.escape(VERSION) + r'(?:\s|$)', version):
    raise SystemExit(f'Expected clang-format {VERSION}, got {version.strip()}')
files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
files = [f for f in files if f.startswith(('src/', 'include/', 'tests/'))
         and Path(f).suffix in ('.cpp', '.h', '.cu', '.cuh')]
subprocess.run([args.clang_format, '--dry-run', '--Werror', *files], cwd=root, check=True)
print(f'clang-format {VERSION}: {len(files)} owned source files checked')
