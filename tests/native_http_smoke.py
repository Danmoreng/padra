import hashlib, json, pathlib, signal, subprocess, time, urllib.request, urllib.error
import argparse

parser = argparse.ArgumentParser(
    description="Run native Padra HTTP integration smoke tests with an admitted EXL3 K4/K6 artifact."
)
parser.add_argument("--server", type=pathlib.Path, required=True)
parser.add_argument("--artifact", type=pathlib.Path, required=True)
parser.add_argument("--tokenizer", type=pathlib.Path, required=True)
parser.add_argument("--tokenizer-sha256", required=True)
parser.add_argument("--port", type=int, default=18463)
parser.add_argument("--server-config", type=pathlib.Path)
parser.add_argument("--capacity", type=int, default=8192)
parser.add_argument("--prefill-chunk", type=int, default=2048)
parser.add_argument("--device-budget-gib", type=int, default=0)
parser.add_argument("--kv-format", choices=("bf16", "q8", "q8_0"), default="bf16")
parser.add_argument("--mtp-artifact", type=pathlib.Path)
parser.add_argument("--mtp-depth", type=int, choices=range(1, 5), default=1)
parser.add_argument("--output-dir", type=pathlib.Path, required=True)
args = parser.parse_args()
artifact = args.artifact.resolve()
tokenizer = args.tokenizer.resolve()
port = args.port
base = f"http://127.0.0.1:{port}"
sha = args.tokenizer_sha256
args.output_dir.mkdir(parents=True, exist_ok=True)
log = args.output_dir / "native-http-smoke-server.log"


def request(path, body=None, method=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        base + path,
        data=data,
        method=method,
        headers={} if data is None else {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode(), response.headers


def fetch(path, body=None, method=None):
    return json.loads(request(path, body, method)[0])


with log.open("w") as output:
    process = subprocess.Popen(
        [
            str(args.server.resolve()),
            str(artifact),
            str(tokenizer),
            sha,
            str(port),
            "30000",
            str(args.capacity),
            str(args.prefill_chunk),
            str(args.device_budget_gib),
            args.kv_format,
        ] + (["--config", str(args.server_config.resolve())] if args.server_config else [])
        + (["--mtp-artifact", str(args.mtp_artifact.resolve()), "--mtp-depth", str(args.mtp_depth),
            "--mtp-draft-experts", "exl3"] if args.mtp_artifact else []),
        stdout=output,
        stderr=output,
    )
    try:
        until = time.monotonic() + 60
        while time.monotonic() < until:
            if process.poll() is not None:
                raise RuntimeError(
                    f"server exited {process.returncode}: {log.read_text()[-2000:]}"
                )
            try:
                ready = fetch("/readyz")
                break
            except (OSError, urllib.error.HTTPError):
                time.sleep(0.1)
        else:
            raise RuntimeError("server readiness timeout")
        identity = fetch("/v1/models")["data"][0]
        model = identity["id"]
        assert model in ("mellum21-exl3-k3-research", "mellum21-exl3-k4-research", "mellum21-exl3-k6-research")
        manifest = json.loads((artifact / "manifest.json").read_text())
        assert identity["checkpoint_sha256"] == manifest["weights_sha256"]
        assert (
            identity["manifest_sha256"]
            == hashlib.sha256((artifact / "manifest.json").read_bytes()).hexdigest()
        )
        assert (
            identity["tokenizer_sha256"] == sha
            and len(identity["build_source_sha256"]) == 64
        )
        assert not identity["test_backend"]
        assert identity["kv_format"] == (
            {"bf16": "bf16", "q8": "q8_token_head128_f32scale_v1", "q8_0": "q8_0_token512_block32_f16scale_rot_k128_v64_v2"}[args.kv_format]
        )
        assert identity["context_capacity"] == args.capacity
        assert identity["prefill_chunk"] == args.prefill_chunk
        assert identity["device_budget_gib"] == args.device_budget_gib
        assert identity["native_kv_reuse"]
        assert identity["checkpoint_snapshot_bytes"] == (
            21 * 1024 * {"bf16": 2048, "q8": 1056, "q8_0": 1088}[args.kv_format]
            + (2 * 1024 * 512 * 2 + 2304 * 4 if args.mtp_artifact else 0)
        )
        assert identity["build_kernel_profile"] in ("baseline-v1", "exact-fusions-v1")
        resolved = identity["server_config"]
        admin_base = f"http://127.0.0.1:{resolved['admin_port']}"
        with urllib.request.urlopen(admin_base + "/config", timeout=3) as response:
            admin_config = json.load(response)
        assert admin_config["server_config"] == resolved
        assert admin_config["build_source_sha256"] == identity["build_source_sha256"]
        for endpoint in ("/healthz", "/readyz", "/metrics"):
            with urllib.request.urlopen(admin_base + endpoint, timeout=3) as response:
                assert response.status == 200
        if args.server_config:
            for section, fields in json.loads(args.server_config.read_text()).items():
                for key, value in fields.items():
                    if key == "admin_port":
                        assert value == 0 or value == resolved["admin_port"]
                    else:
                        assert resolved[section][key] == value, (section, key)
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_completion_tokens": 4,
            "enable_thinking": False,
        }
        chat = fetch("/v1/chat/completions", body)
        assert chat["choices"][0]["finish_reason"] in ("stop", "length")
        wire, _ = request(
            "/v1/chat/completions",
            dict(body, stream=True, stream_options={"include_usage": True}),
        )
        assert (
            "data: [DONE]" in wire
            and '"usage":{' in wire
            and '"finish_reason":"' in wire
        )
        tool_body = dict(
            body,
            messages=[{"role": "user", "content": "Use lookup with the text padra."}],
            max_completion_tokens=256,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                        },
                    },
                }
            ],
        )
        tool_result = fetch("/v1/chat/completions", tool_body)
        tool_wire, _ = request("/v1/chat/completions", dict(tool_body, stream=True))
        tool_calls = {}
        for record in tool_wire.split("\n\n"):
            if not record.startswith("data: ") or record == "data: [DONE]":
                continue
            chunk = json.loads(record[6:])
            assert "error" not in chunk
            for choice in chunk.get("choices", []):
                for delta in choice.get("delta", {}).get("tool_calls", []):
                    call = tool_calls.setdefault(
                        delta["index"], {"name": "", "arguments": "", "identities": 0}
                    )
                    if "id" in delta:
                        call["identities"] += 1
                    call["name"] += delta["function"].get("name", "")
                    call["arguments"] += delta["function"].get("arguments", "")
        expected_tools = tool_result["choices"][0]["message"].get("tool_calls", [])
        assert len(tool_calls) == len(expected_tools)
        for index, call in enumerate(expected_tools):
            assert tool_calls[index]["identities"] == 1
            assert tool_calls[index]["name"] == call["function"]["name"]
            assert tool_calls[index]["arguments"] == call["function"]["arguments"]
        session = fetch("/v1/sessions", {"model": model})["id"]
        turn = fetch("/v1/chat/completions", dict(body, session_id=session))
        assert turn["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        history = fetch("/v1/sessions/" + session)["messages"]
        assert history[-1]["role"] == "assistant"
        history.append({"role": "user", "content": "What did I ask?"})
        follow = fetch(
            "/v1/chat/completions", dict(body, session_id=session, messages=history)
        )
        cached = follow["usage"]["prompt_tokens_details"]["cached_tokens"]
        assert 0 < cached < follow["usage"]["prompt_tokens"]
        # A different request owns the single live GPU cache. The retained
        # session history must still work through a full replay on its next turn.
        history = fetch("/v1/sessions/" + session)["messages"]
        history.append({"role": "user", "content": "Continue."})
        replay = fetch("/v1/chat/completions", dict(body, messages=history))
        miss = fetch("/v1/chat/completions", dict(body, session_id=session, messages=history))
        assert replay["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert miss["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert miss["choices"] == replay["choices"]
        fetch("/v1/sessions/" + session, method="DELETE")
        ids = []
        for i in range(40):
            response = fetch(
                "/v1/responses",
                {
                    "model": model,
                    "input": f"Say {i}.",
                    "max_output_tokens": 1,
                    "enable_thinking": False,
                },
            )
            assert (
                response["status"] in ("completed", "incomplete")
                and response["error"] is None
            )
            ids.append(response["id"])
        try:
            fetch("/v1/responses/" + ids[0])
        except urllib.error.HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError("old root not evicted")
        child = fetch(
            "/v1/responses",
            {
                "model": model,
                "input": "Next.",
                "max_output_tokens": 2,
                "previous_response_id": ids[-1],
                "enable_thinking": False,
            },
        )
        assert 0 < child["usage"]["input_tokens_details"]["cached_tokens"] < child["usage"]["input_tokens"]
        fetch("/v1/responses/" + ids[-1], method="DELETE")
        assert fetch("/v1/responses/" + child["id"])["id"] == child["id"]
        wire, _ = request(
            "/v1/responses",
            {
                "model": model,
                "input": "Hello.",
                "max_output_tokens": 2,
                "stream": True,
                "store": False,
                "enable_thinking": False,
            },
        )
        assert "event: response.created" in wire and (
            "event: response.completed" in wire or "event: response.incomplete" in wire
        )
        metrics, _ = request("/metrics")
        assert "mellum_job_lifetime_seconds_bucket" in metrics
        result = {
            "actual_cuda_model": True,
            "kv_format": identity["kv_format"],
            "capacity": args.capacity,
            "prefill_chunk": args.prefill_chunk,
            "device_budget_gib": args.device_budget_gib,
            "artifact_identity_verified": True,
            "admin_listener_verified": True,
            "server_config": resolved,
            "chat_and_responses_streams": True,
            "session_followup": True,
            "chat_cached_tokens": cached,
            "responses_cached_tokens": child["usage"]["input_tokens_details"]["cached_tokens"],
            "session_cache_miss_replay": True,
            "mtp_depth": args.mtp_depth if args.mtp_artifact else 0,
            "native_tool_request": True,
            "native_tool_calls": len(expected_tools),
            "independent_stored_responses": len(ids),
            "stored_lineage_delete": True,
            "metrics": True,
            "build_source_sha256": identity["build_source_sha256"],
            "build_kernel_profile": identity["build_kernel_profile"],
            "math_profile": identity["math_profile"],
            "kernel_profile": identity["kernel_profile"],
            "scope": "Serving smoke test; not numerical quality, sustained load or speed qualification",
        }
        (args.output_dir / "native-http-smoke-result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(
            "PASS native CUDA HTTP: artifact identity, sessions, 40 stored roots, lineage, both streams and metrics"
        )
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        assert (
            process.returncode == 0
        ), f"server exit: {process.returncode}; {log.read_text()[-2000:]}"
