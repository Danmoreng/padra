#!/usr/bin/env python3
"""Copy the pinned, clean CUTLASS headers into an owned dependency cache."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
REV='db1c288993354c88e551c40c19a8fb93a774a241'
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,default=Path.home()/'.cache/padra/cutlass'/REV);a=p.parse_args()
source=a.source.resolve();destination=a.destination.resolve()
if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=REV:raise ValueError('CUTLASS revision mismatch')
if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():raise ValueError('CUTLASS source must be clean')
if destination.exists():raise ValueError('dependency cache already exists')
if source==destination or source in destination.parents:raise ValueError('destination must not mutate the source checkout')
files={}
for relative in ['include','tools/util/include','LICENSE.txt']:
    path=source/relative
    for f in sorted(path.rglob('*')) if path.is_dir() else [path]:
        if f.is_symlink():raise ValueError('unexpected source symlink')
        if f.is_file():files[str(f.relative_to(source))]=hashlib.sha256(f.read_bytes()).hexdigest()
destination.mkdir(parents=True)
try:
    for relative in ['include','tools/util/include','LICENSE.txt']:
        path=source/relative;target=destination/relative
        if path.is_dir():shutil.copytree(path,target)
        else:shutil.copy2(path,target)
    for relative,digest in files.items():
        if hashlib.sha256((destination/relative).read_bytes()).hexdigest()!=digest:raise ValueError('copied dependency hash mismatch')
    (destination/'SOURCE_LOCK.json').write_text(json.dumps({'repository':'https://github.com/NVIDIA/cutlass','revision':REV,'sha256':files},indent=2)+'\n')
except BaseException:
    shutil.rmtree(destination);raise
print(json.dumps({'path':str(destination),'revision':REV,'files':len(files),'lock_sha256':hashlib.sha256((destination/'SOURCE_LOCK.json').read_bytes()).hexdigest()}))
