"""Exercise the production API through the pinned Python SDK and a synthetic backend."""

import argparse
import concurrent.futures
import json
import importlib.metadata
import platform
import socket
import httpx2
import pathlib
import queue
import subprocess
import tempfile
import threading
import time
import urllib.request
from openai import OpenAI, APIStatusError, NotFoundError, DefaultHttpxClient

MODEL = "mellum21-exl3-k4-research"
CHAT_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
    },
}
RESPONSE_TOOL = {
    "type": "function",
    "name": "lookup",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
}
ARGS = {"text": "hello €", "nested": [1, {"ok": True}]}


def chat(client, content, **options):
    return client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=256,
        **options,
    )


def response(client, content, **options):
    return client.responses.create(
        model=MODEL, input=content, max_output_tokens=256, **options
    )


def test_chat(client):
    reply = chat(client, "hello")
    assert reply.choices[0].message.content == "hello"
    assert reply._request_id.startswith("req_")
    for prompt, expected in [
        ("tool-valid", ARGS),
        ("tool-reverse", {"text": "reverse"}),
    ]:
        calls = {}
        chunks = 0
        finish = None
        for event in chat(client, prompt, tools=[CHAT_TOOL], stream=True):
            chunks += 1
            for delta in event.choices[0].delta.tool_calls or []:
                call = calls.setdefault(
                    delta.index,
                    {"id": None, "name": "", "arguments": "", "identities": 0},
                )
                if delta.id:
                    assert call["id"] is None
                    call["id"] = delta.id
                    call["identities"] += 1
                if delta.function:
                    call["name"] += delta.function.name or ""
                    call["arguments"] += delta.function.arguments or ""
            if event.choices[0].finish_reason:
                finish = event.choices[0].finish_reason
        assert (
            calls[0]["identities"] == 1
            and calls[0]["name"] == "lookup"
            and json.loads(calls[0]["arguments"]) == expected
        )
        assert finish == "tool_calls" and chunks > 2
    usage = None
    for event in chat(
        client, "hello", stream=True, stream_options={"include_usage": True}
    ):
        if event.usage:
            usage = event.usage
    assert usage and usage.completion_tokens == 6


def test_responses(client):
    reply = response(client, "hello", store=False)
    assert reply.output_text == "hello"
    for prompt, expected in [
        ("tool-valid", ARGS),
        ("tool-reverse", {"text": "reverse"}),
    ]:
        calls = {}
        deltas = 0
        finished = False
        sequences = []
        for event in response(
            client, prompt, tools=[RESPONSE_TOOL], stream=True, store=False
        ):
            sequences.append(event.sequence_number)
            if (
                event.type == "response.output_item.added"
                and event.item.type == "function_call"
            ):
                assert event.output_index not in calls
                calls[event.output_index] = {
                    "name": event.item.name,
                    "id": event.item.id,
                    "arguments": "",
                }
            elif event.type == "response.function_call_arguments.delta":
                assert calls[event.output_index]["id"] == event.item_id
                calls[event.output_index]["arguments"] += event.delta
                deltas += 1
            elif event.type == "response.function_call_arguments.done":
                assert calls[event.output_index]["arguments"] == event.arguments
            elif event.type == "response.completed":
                finished = True
        assert finished and deltas > 0 and sequences == list(range(len(sequences)))
        assert (
            calls[0]["name"] == "lookup"
            and json.loads(calls[0]["arguments"]) == expected
        )
    # SDK stream helpers also accumulate the protocol into a final response.
    with client.responses.stream(
        model=MODEL,
        input="tool-valid",
        tools=[RESPONSE_TOOL],
        max_output_tokens=256,
        store=False,
    ) as stream:
        final = stream.get_final_response()
    assert (
        final.status == "completed"
        and final.output[0].name == "lookup"
        and json.loads(final.output[0].arguments) == ARGS
    )
    identities = {}
    with client.responses.stream(
        model=MODEL,
        input="tool-mixed",
        tools=[RESPONSE_TOOL],
        max_output_tokens=256,
    ) as stream:
        for event in stream:
            if event.type == "response.output_item.added":
                identities[event.item.id] = event.output_index
            elif hasattr(event, "item_id") and hasattr(event, "output_index"):
                assert identities[event.item_id] == event.output_index
        mixed = stream.get_final_response()
    assert [item.type for item in mixed.output] == ["function_call", "message"]
    assert all(identities[item.id] == i for i, item in enumerate(mixed.output))
    assert mixed.output_text == "after" and json.loads(mixed.output[0].arguments) == ARGS
    # The SDK accumulator adds null `parsed`/`parsed_arguments` helper fields.
    # Exact wire-envelope equality is independently required by the C++ test.
    stored = client.responses.retrieve(mixed.id).model_dump(exclude_none=True)
    assert stored == mixed.model_dump(exclude_none=True)


def test_partial_tools(client):
    for streaming in (False, True):
        reply = client.responses.create(
            model=MODEL,
            input="tool-valid",
            tools=[RESPONSE_TOOL],
            max_output_tokens=60,
            stream=streaming,
        )
        if streaming:
            with reply:
                finals = [event.response for event in reply if event.type == "response.incomplete"]
            assert len(finals) == 1
            reply = finals[0]
        assert reply.status == "incomplete" and reply.output[0].status == "incomplete"
        assert client.responses.retrieve(reply.id).status == "incomplete"
        try:
            response(client, "next", previous_response_id=reply.id)
        except APIStatusError as error:
            assert error.status_code == 400 and error.code == "response_not_continuable"
        else:
            raise AssertionError("incomplete tool response accepted as continuation")


def test_retention(client):
    ids = []
    for _ in range(40):
        ids.append(response(client, "hello").id)
    try:
        client.responses.retrieve(ids[0])
    except NotFoundError:
        pass
    else:
        raise AssertionError("old stored response retained")
    child = response(client, "hello", previous_response_id=ids[-1])
    client.responses.delete(ids[-1])
    assert client.responses.retrieve(child.id).id == child.id
    client.responses.delete(child.id)  # Supported SDK returns None for this operation.
    try:
        client.responses.retrieve(child.id)
    except NotFoundError as error:
        assert error.code == "response_not_found"
    else:
        raise AssertionError("deleted response remained retrievable")


def test_queue(client, base):
    for is_response in (False, True):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            blocker = executor.submit(chat, client, "queue-blocker")
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                with urllib.request.urlopen(base + "/metrics", timeout=2) as wire:
                    metrics = wire.read().decode()
                if "mellum_active_requests 1\n" in metrics:
                    break
                time.sleep(0.002)
            else:
                raise AssertionError("queue blocker never became active")
            try:
                (response if is_response else chat)(client, "hello")
            except APIStatusError as error:
                assert error.status_code == 503 and error.code == "queue_deadline"
                assert error.request_id.startswith("req_")
                assert error.response.headers["Retry-After"] == "1"
                assert (
                    not blocker.done()
                ), "SDK waited for active generation before deadline error"
            else:
                raise AssertionError("queued SDK request did not expire")
            blocker.result(timeout=2)


def metric(base, name):
    with urllib.request.urlopen(base + "/metrics", timeout=2) as wire:
        for line in wire.read().decode().splitlines():
            if line.startswith(name + " "):
                return int(line.split()[1])
    raise AssertionError("missing metric " + name)


def test_disconnect(client, base):
    for is_response in (False, True):
        before = metric(base, "mellum_disconnects_total")
        stream = (response if is_response else chat)(client, "slow-text", stream=True)
        next(iter(stream))
        stream.close()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if (
                metric(base, "mellum_disconnects_total") == before + 1
                and metric(base, "mellum_active_requests") == 0
            ):
                break
            time.sleep(0.005)
        else:
            raise AssertionError("SDK stream close did not cancel backend")


def test_invalid_tool(client):
    failed = False
    for event in response(
        client, "tool-malformed", tools=[RESPONSE_TOOL], stream=True, store=False
    ):
        assert event.type != "response.completed"
        if event.type == "response.failed":
            failed = True
            assert event.response.error.code == "invalid_tool_output"
    assert failed, "malformed tool stream ended without failure"
    try:
        response(client, "tool-malformed", tools=[RESPONSE_TOOL], store=False)
    except APIStatusError as error:
        assert error.status_code == 500 and error.code == "invalid_tool_output"
    else:
        raise AssertionError("malformed tool response succeeded")


def release_tool(process):
    process.stdin.write("release-tool\n")
    process.stdin.flush()


def test_early_tool_delta(client, base, process):
    for is_response in (False, True):
        before = metric(base, "mellum_requests_completed_total")
        stream = (response if is_response else chat)(
            client, "tool-early", tools=[RESPONSE_TOOL if is_response else CHAT_TOOL], stream=True
        )
        released = False
        arguments = ""
        try:
            for event in stream:
                if is_response:
                    delta = event.delta if event.type == "response.function_call_arguments.delta" else ""
                else:
                    delta = "".join(item.function.arguments or "" for item in
                                    (event.choices[0].delta.tool_calls or []) if item.function)
                arguments += delta
                if delta and not released:
                    assert metric(base, "mellum_active_requests") == 1
                    assert metric(base, "mellum_requests_completed_total") == before
                    release_tool(process)
                    released = True
            assert released and json.loads(arguments) == {"text": "early complete"}
        finally:
            if not released:
                release_tool(process)
            stream.close()


def test_utf8_stop(client):
    for is_response in (False, True):
        for streaming in (False, True):
            options = dict(stream=streaming)
            if is_response:
                reply = response(client, "utf8-stop", extra_body={"stop": ["STOP"]}, **options)
                if streaming:
                    events = list(reply)
                    text = "".join(e.delta for e in events if e.type == "response.output_text.delta")
                    final = next(e.response for e in events if e.type == "response.completed")
                    assert client.responses.retrieve(final.id).output_text == text
                else:
                    text = reply.output_text
            else:
                reply = chat(client, "utf8-stop", stop=["STOP"], **options)
                text = ("".join(e.choices[0].delta.content or "" for e in reply)
                        if streaming else reply.choices[0].message.content)
            assert text == "€ hello "


def test_sampling_extensions(client):
    reply = chat(client, "hello", temperature=.75, top_p=.9, seed=7,
                 frequency_penalty=.25, presence_penalty=-.5,
                 extra_body={"top_k": 42, "repetition_penalty": 1.125})
    assert reply.choices[0].message.content == "hello"
    reply = response(client, "hello", temperature=.75, top_p=.9,
                     extra_body={"top_k": 42, "repetition_penalty": 1.125,
                                 "seed": 7, "frequency_penalty": .25, "presence_penalty": -.5})
    assert reply.output_text == "hello" and reply.temperature == .75 and reply.top_p == .9


def test_slow_reader(client, base):
    # A bounded receive window makes nonreading deterministic instead of relying
    # on platform TCP autotuning to fill before generation ends.
    transport = httpx2.HTTPTransport(retries=0, trust_env=False,
        socket_options=[(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)])
    with OpenAI(api_key="sdk-loopback-fixture", base_url=base + "/v1", max_retries=0, timeout=3,
                http_client=DefaultHttpxClient(transport=transport, timeout=3, trust_env=False)) as slow:
        for is_response in (False, True):
            before = metric(base, "mellum_response_write_failures_total")
            if is_response:
                stream = slow.responses.create(model=MODEL, input="slow-reader", stream=True,
                                                 max_output_tokens=5000)
            else:
                stream = slow.chat.completions.create(model=MODEL,
                    messages=[{"role": "user", "content": "slow-reader"}], stream=True,
                    max_completion_tokens=5000)
            try:
                next(iter(stream))
                deadline = time.monotonic() + 6
                while time.monotonic() < deadline:
                    if (metric(base, "mellum_response_write_failures_total") == before + 1
                            and metric(base, "mellum_active_transports") == 0
                            and metric(base, "mellum_active_requests") == 0):
                        break
                    time.sleep(.01)
                else:
                    raise AssertionError("paused SDK consumer retained generation/transport")
            finally:
                stream.close()


def test_unsupported(client):
    for call in [lambda: chat(client, "hello", n=2),
                 lambda: chat(client, "hello", response_format={"type": "json_object"}),
                 lambda: response(client, "hello", stream=True,
                                  extra_body={"stream_options": {"include_usage": True}})]:
        try:
            call()
        except APIStatusError as error:
            assert error.status_code == 400 and error.code == "invalid_request_error"
            assert error.request_id.startswith("req_")
        else:
            raise AssertionError("unsupported SDK option accepted")
    try:
        client.embeddings.create(model=MODEL, input="hello")
    except NotFoundError as error:
        assert error.code == "not_found"
    else:
        raise AssertionError("unsupported SDK endpoint accepted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, default=pathlib.Path("sdk-compatibility.json"))
    args = parser.parse_args()
    report = {"sdk_version": importlib.metadata.version("openai"),
              "python_version": platform.python_version(), "backend": "synthetic",
              "max_retries": 0, "timeout_seconds": 3, "slow_reader_receive_buffer_bytes": 1024, "passed_tests": [], "success": False}
    print(f"OpenAI Python SDK {report['sdk_version']}; synthetic loopback; retries=0, timeout=3s", flush=True)
    with tempfile.TemporaryFile(mode="w+t") as errors:
        process = subprocess.Popen(
            [str(args.fixture.resolve()), "0", "--control-stdin"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
        )
        try:
            lines = queue.Queue()
            threading.Thread(
                target=lambda: lines.put(process.stdout.readline()), daemon=True
            ).start()
            startup = json.loads(lines.get(timeout=10))
            assert startup["synthetic"]
            base = f"http://127.0.0.1:{startup['port']}"
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    urllib.request.urlopen(base + "/healthz", timeout=0.2).close()
                    break
                except OSError:
                    time.sleep(0.01)
            else:
                raise AssertionError("fixture startup timeout")
            with urllib.request.urlopen(base + "/v1/models", timeout=2) as wire:
                identity = json.load(wire)["data"][0]
            assert identity["test_backend"]
            report["build_source_sha256"] = identity["build_source_sha256"]
            report["server_config"] = identity["server_config"]
            # Dummy key names a local owner; base_url always targets loopback.
            with OpenAI(
                api_key="sdk-loopback-fixture",
                base_url=base + "/v1",
                max_retries=0,
                timeout=3,
            ) as client:
                tests = [
                    ("chat", test_chat, (client,)),
                    ("responses", test_responses, (client,)),
                    ("early_tool_delta", test_early_tool_delta, (client, base, process)),
                    ("utf8_stop", test_utf8_stop, (client,)),
                    ("sampling_extensions", test_sampling_extensions, (client,)),
                    ("partial_tools", test_partial_tools, (client,)),
                    ("retention_and_delete", test_retention, (client,)),
                    ("queue_errors", test_queue, (client, base)),
                    ("disconnect", test_disconnect, (client, base)),
                    ("invalid_tool", test_invalid_tool, (client,)),
                    ("slow_reader", test_slow_reader, (client, base)),
                    ("unsupported_options_endpoints", test_unsupported, (client,)),
                ]
                for name, function, arguments in tests:
                    report["running_test"] = name
                    function(*arguments)
                    report["passed_tests"].append(name)
                    print("PASS SDK " + name, flush=True)
                report.pop("running_test")
                report["success"] = True
            print(
                "PASS Python SDK: Chat/Responses, incremental tool/UTF8 assembly, stream helper, usage, retention, queue errors and disconnect"
            )
        finally:
            process.stdin.close()
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdout.close()
            errors.seek(0)
            report["fixture_exit_code"] = process.returncode
            report["success"] = report["success"] and process.returncode == 0
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            assert (
                process.returncode == 0
            ), f"fixture exit {process.returncode}: {errors.read()}"


if __name__ == "__main__":
    main()
