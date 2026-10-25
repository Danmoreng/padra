# OpenAI-compatible API

Base URL: `http://127.0.0.1:8080/v1` (use your configured port).
Model ID: `mellum21-exl3-k3-research`, `mellum21-exl3-k4-research` or
`mellum21-exl3-k6-research`, selected by
the loaded checkpoint. `GET /v1/models` reports the active ID; requests must use
that ID. The examples below use K4.

Padra implements text/function subsets of Chat Completions and Responses.
See [API support](api-support.md) for endpoints, accepted options and exclusions.
MTP is selected at server startup without changing request schemas; see the
[MTP execution and accounting contract](mtp.md).

## Chat Completions

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mellum21-exl3-k4-research","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":256,"enable_thinking":false}'
```

Use `stream: true` for SSE deltas. Chat streams can include a final usage chunk
with `stream_options: {"include_usage": true}`. Reasoning appears separately as
`reasoning_content`.

Text-only content-part arrays and `developer` instructions are accepted. Chat
`store: false` requests no completion storage. Use `reasoning_effort: "none"` to
disable thinking; the other accepted effort names all select Mellum's enabled
thinking mode, without separate effort budgets. See [API support](api-support.md).

## Responses

```sh
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"mellum21-exl3-k4-research","input":"Hello","max_output_tokens":256,"store":false,"enable_thinking":false}'
```

Responses accepts text or supported message/function/reasoning items. Use
`stream: true` for Responses SSE events. With `store: true` (the default), an
eligible retained response can be continued with `previous_response_id` and read
or deleted at `/v1/responses/{id}`. Retention is bounded, process-local and expires;
it is not durable conversation storage. Top-level `instructions` applies to its
request and is not inherited by the next request.

## Pi Coding Agent

Pi can connect using its built-in `openai-responses` or `openai-completions` API.
Merge this provider into `~/.pi/agent/models.json`, using the server's actual port
and model ID:

```json
{
  "providers": {
    "padra": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-responses",
      "apiKey": "local-padra",
      "models": [{
        "id": "mellum21-exl3-k4-research",
        "name": "Padra Mellum K4",
        "reasoning": true,
        "thinkingLevelMap": {"off": "none"},
        "input": ["text"],
        "contextWindow": 131072,
        "maxTokens": 16384
      }]
    }
  }
}
```

```sh
pi --model padra/mellum21-exl3-k4-research
```

No provider extension, request adapter or proxy is required. The context window
must match the server capacity; input plus output budget must fit within it. Pi's
thinking levels share Mellum's single enabled mode, while `off` sends `none`.
Responses preserves Mellum's full plaintext reasoning across tool and user turns.
`summary: "auto"` uses that native presentation; a separate summary is not generated.
An encrypted-content include returns `encrypted_content: null`. Pi can replay the
plaintext reasoning item without opaque state. Cache keys are best-effort hints,
with no additional cache retention or KV reuse guarantee. See
[client qualification](../tests/client-contract.md) for the optional real Pi CLI smoke test.

## Python client

The pinned compatibility suite uses `openai==3.29.0`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="local-client",
    max_retries=0,
    timeout=600,
)
reply = client.responses.create(
    model="mellum21-exl3-k4-research",
    input="Hello",
    max_output_tokens=256,
    store=False,
    extra_body={"enable_thinking": False},
)
print(reply.output_text)
```

A Bearer value selects a local credential namespace; use the same value when
retrieving or continuing stored state. It is not external authentication. The
listener is loopback-only and has no public TLS/authentication deployment mode.
The client timeout is independent of the server's queue/generation deadlines.

## Function calls and continuation

Declare function tools in the selected API's format. Padra returns calls and
streams argument fragments; your application executes functions and submits their
outputs. Wait for successful completion before acting on streamed arguments.
Only offered names and valid completed JSON objects are accepted. Tool schemas
are prompt definitions, not constrained-decoding guarantees.

An incomplete tool response remains retrievable if stored, but cannot be used as
`previous_response_id`. Retry with enough output budget or continue from a valid
predecessor. See [error contracts](api-errors.md) for the precise state error.

## Operation

- [Supported endpoints and options](api-support.md)
- [Errors, retries and request IDs](api-errors.md)
- [Request lifecycle, limits, shutdown and metrics](server-lifecycle.md)
- [Store admission and lock contract](server-locks.md)
- [Client and native smoke tests](../tests/README.md)

[Server configuration and administration](server-configuration.md) describes
`--config`, resolved settings and the reserved health/readiness/metrics listener.
