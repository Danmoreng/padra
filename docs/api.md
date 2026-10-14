# OpenAI-compatible API

Base URL: `http://127.0.0.1:8080/v1` (use your configured port).
Model ID: `mellum21-exl3-k4-research`.

Padra implements text/function subsets of Chat Completions and Responses.
See [API support](api-support.md) for endpoints, accepted options and exclusions.

## Chat Completions

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mellum21-exl3-k4-research","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":256,"enable_thinking":false}'
```

Use `stream: true` for SSE deltas. Chat streams can include a final usage chunk
with `stream_options: {"include_usage": true}`. Reasoning appears separately as
`reasoning_content`.

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
