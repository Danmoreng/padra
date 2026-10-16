"""Optional real Pi CLI smoke against production HTTP with synthetic inference.

Run with Pi 1.3 installed; no Pi extension, request hook, proxy or cloud key is
used. All model/session configuration lives in a temporary agent directory.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import queue
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--pi", default="pi")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {"pi_version": subprocess.check_output([args.pi, "--version"], text=True).strip(),
              "synthetic": True, "passed": False, "cases": []}
    process = subprocess.Popen([str(args.fixture.resolve()), "0"], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        port = json.loads(process.stdout.readline())["port"]
        with tempfile.TemporaryDirectory(prefix="padra-pi-") as directory:
            root = Path(directory)
            model = "mellum21-exl3-k4-research"
            for api in ("openai-completions", "openai-responses"):
                run_api(root, port, model, api, args, report)
        report["passed"] = True
    finally:
        process.terminate()
        process.communicate(timeout=10)
        report["fixture_exit_code"] = process.returncode
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def run_api(root, port, model, api, args, report):
    (root / "models.json").write_text(json.dumps({"providers": {"padra": {
        "baseUrl": f"http://127.0.0.1:{port}/v1", "api": api,
        "apiKey": "local-fixture", "models": [{"id": model, "reasoning": True,
        "thinkingLevelMap": {"off": "none"},
        "input": ["text"], "contextWindow": 131072, "maxTokens": 1024}]
    }}}))
    (root / "probe.txt").write_text("PADRA_PI_READ_SENTINEL\n")
    for name, prompt, thinking in [
        ("text", "Say hello.", "off"),
        ("reasoning", "reasoning-text", "medium"),
        ("tool-continuation", "pi-read-fixture: read probe.txt using the read tool.", "medium"),
        ("reasoning-tool-continuation", "pi-thinking-read-fixture: read probe.txt.", "medium"),
    ]:
        command = [args.pi, "--offline", "--no-extensions", "--no-mcp", "--no-skills",
                   "--no-prompt-templates", "--no-context-files", "--no-session",
                   "--model", f"padra/{model}", "--thinking", thinking,
                   "--tools", "read", "--mode", "json", "--print", prompt]
        result = subprocess.run(command, cwd=root,
            env=dict(os.environ, PI_CODING_AGENT_DIR=str(root)),
            capture_output=True, text=True, timeout=30)
        events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        messages = [e["message"] for e in events
                    if e.get("type") == "message_end" and "message" in e]
        assistants = [m for m in messages if m.get("role") == "assistant"]
        assert result.returncode == 0, (name, result.stderr, result.stdout)
        assert assistants and all(m.get("stopReason") != "error" for m in assistants), (name, messages)
        blocks = [block for m in assistants for block in m.get("content", [])]
        if name == "reasoning-tool-continuation":
            assert any(b.get("type") == "thinking" and "inspect € probe" in b.get("thinking", "")
                       for b in blocks), blocks
        if name == "reasoning":
            assert any(b.get("type") == "thinking" and "why €" in b.get("thinking", "")
                       for b in blocks), blocks
        if name in ("tool-continuation", "reasoning-tool-continuation"):
            assert any(b.get("type") == "toolCall" and b.get("name") == "read" for b in blocks), blocks
            assert any(m.get("role") == "toolResult" and "PADRA_PI_READ_SENTINEL" in json.dumps(m)
                       and not m.get("isError") for m in messages), messages
            assert len(assistants) == 2, messages
        assert any(b.get("type") == "text" and ("hello" in b.get("text", "") or
                   "answer" in b.get("text", "")) for b in blocks), blocks
        report["cases"].append({"api": api, "name": name, "passed": True,
                                "assistant_turns": len(assistants)})
    rpc_continuation(root, model, api, args, report)


def rpc_continuation(root, model, api, args, report):
    command = [args.pi, "--offline", "--no-extensions", "--no-mcp", "--no-skills",
               "--no-prompt-templates", "--no-context-files", "--no-session",
               "--model", f"padra/{model}", "--thinking", "medium", "--no-tools", "--mode", "rpc"]
    with tempfile.TemporaryFile(mode="w+t") as errors:
        process = subprocess.Popen(command, cwd=root, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=errors, text=True,
            env=dict(os.environ, PI_CODING_AGENT_DIR=str(root)))
        records = queue.Queue()

        def read_records():
            for line in process.stdout:
                records.put(json.loads(line))
            records.put(None)

        reader = threading.Thread(target=read_records, daemon=True)
        reader.start()
        try:
            assistants = []
            for index, prompt in enumerate(("pi-history-fixture", "pi-history-check")):
                process.stdin.write(json.dumps({"type": "prompt", "id": str(index), "message": prompt}) + "\n")
                process.stdin.flush()
                deadline = time.monotonic() + 15
                while True:
                    event = records.get(timeout=max(.01, deadline - time.monotonic()))
                    assert event is not None, "Pi RPC exited before settling"
                    if event.get("type") == "response":
                        assert event.get("success"), event
                    if event.get("type") == "message_end" and event.get("message", {}).get("role") == "assistant":
                        assistants.append(event["message"])
                        assert event["message"].get("stopReason") != "error", event
                    if event.get("type") == "agent_settled":
                        break
            assert len(assistants) == 2, assistants
            assert any(b.get("text") == "history verified" for b in assistants[-1]["content"]), assistants
            report["cases"].append({"api": api, "name": "two-user-turn-reasoning-replay",
                                    "passed": True, "assistant_turns": 2})
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
            reader.join(timeout=1)


if __name__ == "__main__":
    main()
