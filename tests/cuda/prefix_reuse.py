"""Real-model prefix reuse regression; requires admitted artifacts and a GPU."""
import argparse
import hashlib
import json
import pathlib
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=pathlib.Path, required=True)
    parser.add_argument("--artifact", type=pathlib.Path, required=True)
    parser.add_argument("--draft", type=pathlib.Path, required=True)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--expert-rate", type=int, choices=(3, 4, 6), required=True)
    parser.add_argument("--kv-format", choices=("bf16", "q8", "q8_0"), required=True)
    parser.add_argument("--chunk", type=int, choices=(256, 512, 2048), default=256)
    parser.add_argument("--depths", type=int, nargs="+", choices=range(5), default=[0, 1, 4])
    parser.add_argument("--whole-chunks", action="store_true", help="SIMT complete-chunk snapshot policy")
    args = parser.parse_args()
    tokens = json.loads(args.input.read_text())["input_ids"]
    length = (6 if args.whole_chunks else 3) * args.chunk + 2
    if len(tokens) < length:
        parser.error(f"input requires at least {length} token IDs")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    requests, pairs = [], []
    anchors = [1, args.chunk + 1]
    anchors += [boundary - tail for boundary in (args.chunk, 2 * args.chunk) for tail in (4, 3, 2, 1)]
    if args.whole_chunks:
        anchors = [args.chunk, args.chunk + 1, 2 * args.chunk - 1, 2 * args.chunk,
                   4 * args.chunk - 1, 4 * args.chunk, 4 * args.chunk + 1, 5 * args.chunk - 1]
        for anchor in (1, args.chunk - 1):
            requests.append({"input": str(args.input.resolve()), "prefix": args.chunk - 1,
                             "generate": 2, "checkpoint_anchor": anchor})
    for anchor in anchors:
        base = {"input": str(args.input.resolve()), "prefix": length, "generate": 32}
        requests.append({**base, "checkpoint_anchor": anchor})
        restored = len(requests)
        requests.append({**base, "restore_checkpoint": True, "mutate_at": anchor + 5})
        fresh = len(requests)
        requests.append({**base, "mutate_at": anchor + 5})
        pairs.append((restored, fresh, anchor))
    requests[0].update(
        expert_rate=args.expert_rate, kv_format=args.kv_format, prefill_chunk=args.chunk,
        capacity=8192, q8_head=True, exl3_experts=True, checkpoint_enabled=True,
    )
    protocol = output / "requests.json"
    protocol.write_text(json.dumps(requests, indent=2) + "\n")
    fields = ("ids", "head_sha256", "hidden_sha256", "callbacks", "cancelled", "committed", "pending")
    references, differences, rows_by_depth = {}, [], {}
    for depth in sorted(set(args.depths)):
        path = output / f"d{depth}.jsonl"
        command = [str(args.probe.resolve()), str(args.artifact.resolve()), str(args.draft.resolve()),
                   str(protocol), str(path), str(depth)]
        with (output / f"d{depth}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == len(requests)
        rows_by_depth[depth] = len(rows)
        if args.whole_chunks:
            assert all(not r["checkpoint_available"] and r["cached_tokens"] == 0 for r in rows[:2])
        if depth == 0:
            references = {row["request"]: row for row in rows}
        for restored, fresh, anchor in pairs:
            assert 0 < rows[restored]["cached_tokens"] <= anchor
            if args.whole_chunks:
                assert rows[restored]["cached_tokens"] == anchor // args.chunk * args.chunk
            else:
                assert anchor - rows[restored]["cached_tokens"] <= 3
            for field in fields:
                if rows[restored][field] != rows[fresh][field]:
                    differences.append([depth, restored, fresh, field])
        if references:
            for row in rows:
                for field in fields:
                    if row[field] != references[row["request"]][field]:
                        differences.append([depth, row["request"], "ordinary", field])
    report = {
        "exact": not differences, "differences": differences, "rows_by_depth": rows_by_depth,
        "kv_format": args.kv_format, "expert_rate": args.expert_rate, "chunk": args.chunk,
        "whole_chunks": args.whole_chunks,
        "probe_sha256": hashlib.sha256(args.probe.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    }
    (output / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
