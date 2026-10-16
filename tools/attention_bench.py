"""Record isolated CUDA benchmark results and local provenance as JSON.

Run under the same exclusive GPU lease used by inference benchmarks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent

    def read(command):
        result = subprocess.run(command, capture_output=True, text=True, cwd=root)
        return {"command": command, "status": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}

    telemetry = ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.used,"
                 "temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory",
                 "--format=csv,noheader,nounits"]
    dirty = read(["git", "diff", "HEAD", "--", "src", "include", "tests", "CMakeLists.txt"])
    command = [str(args.binary.resolve()), *(args.arguments[1:] if args.arguments[:1] == ["--"]
                                            else args.arguments)]
    report = {"version": 1, "started_unix": time.time(), "command": command,
              "commit": read(["git", "rev-parse", "HEAD"])["stdout"].strip(),
              "dirty_diff_sha256": hashlib.sha256(dirty["stdout"].encode()).hexdigest(),
              "worktree_status": read(["git", "status", "--porcelain"])["stdout"],
              "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              "telemetry_before": read(telemetry), "model": None, "tokenizer": None,
              "input_token_ids_sha256": None}
    if args.provenance:
        data = args.provenance.read_bytes()
        if len(data) > 65536:
            raise ValueError("fixture provenance exceeds 64 KiB")
        report["capture_provenance"] = json.loads(data)
        report["capture_provenance_sha256"] = hashlib.sha256(data).hexdigest()
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    report.update(finished_unix=time.time(), exit_code=result.returncode,
                  stderr=result.stderr, telemetry_after=read(telemetry))
    if result.returncode == 0:
        report["result"] = json.loads(result.stdout)
    else:
        report["stdout"] = result.stdout
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if result.returncode:
        raise SystemExit(result.returncode)
    print(json.dumps(report["result"]))


if __name__ == "__main__":
    main()
