# API support

Padra supports the subset below for the loaded K4 or K6 checkpoint. Use the active
model ID from `GET /v1/models`: `mellum21-exl3-k4-research` or
`mellum21-exl3-k6-research`.
Compatibility with an OpenAI client does not imply support for all OpenAI APIs or
request options. Unknown fields and unsupported values are rejected.

## Endpoints

| Endpoint | Support |
| --- | --- |
| `POST /v1/chat/completions` | Text, reasoning, function calls; JSON or SSE |
| `POST /v1/responses` | Text/message/function/plaintext-reasoning input; JSON or SSE |
| `GET /v1/responses/{id}` | Retrieve a retained response in the same namespace |
| `DELETE /v1/responses/{id}` | Delete a retained response |
| `GET /v1/models` | Loaded model, artifact/build identity and profiles |
| `POST /v1/sessions` | Padra extension: create an explicit bounded session |
| `GET /v1/sessions/{id}` | Padra extension: session metadata and cache availability |
| `DELETE /v1/sessions/{id}` | Padra extension: discard a session |
| `GET /healthz`, `GET /readyz` | Liveness and readiness |
| `GET /metrics` | Prometheus metrics |
| Admin listener: `GET /healthz`, `/readyz`, `/metrics`, `/config` | Separate bounded loopback listener; configuration/build identity |

Embeddings, legacy text completions, audio, images/video, files/uploads, batches,
fine-tuning, assistants/threads, response cancellation and response input-item
listing endpoints are not implemented.

## Generation options

| Feature | Chat Completions | Responses |
| --- | --- | --- |
| Prompt | `messages` | `input`, optional text `instructions` |
| Output budget | `max_completion_tokens` or `max_tokens`, mutually exclusive | `max_output_tokens` |
| Streaming | `stream`; optional `stream_options.include_usage` | `stream`; usage in terminal response |
| Sampling and repetition control | `temperature`, `top_p`, `top_k`, `seed`, `repetition_penalty`, `frequency_penalty`, `presence_penalty` | Same |
| Stop strings | `stop`: one string or up to four strings | Same |
| Thinking control | Padra `enable_thinking` | Padra `enable_thinking`; no effort/summary control |
| Function definitions | Nested `tools[].function` | Flat `tools[]` with `type: "function"` |
| Tool selection | `tool_choice: "auto"` or `"none"` | Same |
| Multiple function calls | `parallel_tool_calls`; up to eight returned calls | Same |
| Session/state | Padra `session_id` from `/v1/sessions`; send full message history | `previous_response_id`; bounded `store` |
| Manual item replay | Chat message/tool history | Supported message, function and plaintext-reasoning items |
| Metadata | Not accepted | Up to 16 string pairs; key ≤64 bytes, value ≤512 bytes |
| Multiple choices | Only `n: 1` | One response |

All fields in the sampling row are supported by Padra on both endpoints. They
control decoding: temperature scales logits, top-k/top-p restrict candidate tokens,
seed selects the random sequence, and penalties adjust repeated tokens. They are
grouped together regardless of whether OpenAI exposes the same field.

For OpenAI request compatibility, `temperature` and `top_p` are standard fields on
both endpoints. `frequency_penalty`, `presence_penalty` and `seed` are listed for
[Chat Completions](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create),
but are Padra extensions on
[Responses](https://developers.openai.com/api/reference/python/resources/responses/methods/create).
`top_k` and `repetition_penalty` are Padra extensions on both. SDKs can send
extension fields using `extra_body`; see the [API guide](api.md). OpenAI model
support for standard fields can also vary by model.

Output budgets are integers 1–131072, further constrained by the admitted context:
prompt plus output budget must fit. Each stop string is 1–256 UTF-8 bytes. The
server admits at most 256 Chat messages or 256 Responses input items. Declared
function tools are limited to 16. Set `parallel_tool_calls: false` for at most one
returned call. This option does not enable concurrent GPU sequences or execute
functions on the server.

Sampling ranges are `temperature: 0` (greedy) or 0.0001–2, `top_p: (0, 1]`,
`top_k: 0–98304` (0 disables the filter), signed 64-bit `seed`,
`frequency_penalty`/`presence_penalty: -2–2`, and `repetition_penalty: 0.01–100`.
Null sampling fields select defaults. These are input/forwarding contracts;
synthetic client tests do not qualify stochastic model quality.

## Accepted restricted values

Responses accepts `truncation: "disabled"`, `background: false`, empty `include`,
and `text.format: {"type":"text"}`. Non-null reasoning effort/summary controls,
automatic truncation, background execution, nonempty includes and alternative text
formats are unsupported. Responses `stream_options` requires `stream: true` and
accepts only `include_obfuscation: false` (or an empty object); stream obfuscation
and Responses `stream_options.include_usage` are unsupported. Terminal Responses
events include usage without that option.

Function `strict` can be false or null; `strict: true`, required/forced/named tool
choice and constrained JSON-schema generation are unsupported. Function parameters
must be an object schema. Completed generated arguments must be a valid JSON
object; schema compliance is not enforced by constrained decoding.

## Input and output boundaries

Responses message input supports `user`, `system` and `assistant` roles with text.
Function results use `function_call_output` with matching `call_id` and text output.
Plaintext reasoning replay requires full `reasoning_text` content; encrypted
reasoning and summaries in place of full reasoning are unsupported. Media content
parts, nonempty annotations/logprobs, logit bias, logprob requests and structured
output/JSON-schema formats are unsupported.

Output text, reasoning and function items retain stable IDs/indices across SSE,
terminal output and stored retrieval. There is at most one message item, one
reasoning item and eight function items. Retention defaults to 32 records and
32 MiB with a 30-minute sliding TTL; it can evict eligible old records. Restart
clears retained state. Incomplete tool output is retrievable but not continuable;
valid text-only length endings remain continuable.

The server uses loopback HTTP and local credential namespaces. Public
authentication/TLS, durable conversation storage, hosted built-in tools and
multi-sequence GPU scheduling are not provided.

Chat message `content` is a string, or null for an assistant with function calls.
Responses additionally accepts the text-part/item forms described above. Chat
content-part arrays and developer roles are unsupported. Unknown/non-owned previous
response strings return `404/response_not_found`; nonstring IDs are invalid input.

## Contract verification

| Contract | Verification |
| --- | --- |
| Accepted/rejected fields, sampling ranges, restricted values, input forms, tool declaration/history limits | 247 HTTP cases in [the versioned field matrix](../tests/contracts/api-fields.json); both adapters' sampling values are also observed at the runtime interface |
| Unsupported endpoints and request/error IDs | `api-field-contract`, admission/error tests, SDK unsupported-option tests |
| Events, usage, output indices, raw argument bytes, terminal/GET equality | 32 exact wire goldens, output-order/partial-tool tests, SDK stream helpers |
| Retention, continuation, partial tools, deletion | Admission/partial-tool tests and SDK retention/GET/delete tests |
| Early tool deltas, UTF-8/stop, queue errors, disconnect, paused reader | SDK tests; barrier-controlled early tool completion and bounded client receive window |
| Heartbeats, slow uploads/connections, admin reachability, restart and drain | Separate HTTP operation and transport tests |

The supported SDK baseline is pinned to Python `openai==3.29.0`. The independent
[SDK update canary](../.github/workflows/sdk-canary.yml) runs on manual dispatch and
weekly against the version resolved by pip; it does not change that baseline or
act as a release gate. Both workflows retain actual SDK version, client settings,
case results and test output. Synthetic client/host tests and real-artifact native
smokes are distinct. See [client qualification](../tests/client-contract.md).

See the [API guide](api.md), [error contract](api-errors.md) and
[operation details](server-lifecycle.md) for usage and behavior.
