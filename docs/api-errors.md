# Generation errors and request identity

`POST /v1/chat/completions` and `POST /v1/responses` use the same admission error
contract. `X-Request-Id` is a server-generated `req_…` identity created before
request preparation. It is distinct from the completion/response object ID.
Admission refusal logs contain only that identity, route and bounded cause;
prompts, tools, session IDs and bearer values are not included.

| Admission cause | HTTP | `error.code` | `Retry-After` |
| --- | --- | --- | --- |
| Model/runtime not ready | 503 | `model_not_ready` | `1` |
| Server draining | 503 | `server_shutdown` | `1` |
| Preparation slots occupied (default four) | 503 | `preparation_limited` | `1` |
| Queue request limit | 503 | `queue_full` | `1` |
| Queue byte budget | 503 | `queue_bytes_exhausted` | `1` |
| Queue token budget | 503 | `queue_tokens_exhausted` | `1` |
| Missing/deleted session | 404 | `session_not_found` | absent |
| Session already queued/running | 409 | `session_busy` | `1` |
| Session history budget | 429 | `session_history_full` | `1` |
| Response record/byte budget | 429 | `response_store_full` | `1` |
| Protected/referenced root capacity | 429 | `session_unavailable` | `1` |
| Invalid admission request | 400 | `invalid_request_error` | absent |
| Internal admission failure | 500 | `admission_error` | absent |

An early successful check is provisional. Final admission checks queue bytes,
tokens, readiness, drain and session ownership atomically, after preparation.
If several constraints fail, drain precedes readiness, then queue count, bytes,
tokens and session state. Joint storage admission reports root capacity when
neither root nor record resources can fit. Some causes apply only to routes that
use the resource, for example response retention.

`Retry-After` describes transient availability, not guaranteed success. A request
that exceeds an individual byte/history budget needs a smaller request/output
budget or different retained state.

`mellum_admission_rejected_total{route="chat|responses|internal",cause="…"}`
counts each admission refusal once. Labels use fixed enums, never request/session
IDs. `mellum_queue_rejected_total` exposes the aggregate compatibility counter.
Parsing/policy/continuation-validation errors are separate from admission
refusals. An admitted job that subsequently fails or times out is a failed or
cancelled job, not a second admission refusal.

After admission, both adapters classify queue expiry as `503/queue_deadline`,
generation expiry as `504/generation_deadline`, and shutdown as
`503/server_shutdown`. Queue expiry and shutdown include `Retry-After: 1` when an
HTTP error can still be sent. Once streaming headers have been written, failures
are encoded in the protocol's error/failure events; the HTTP status is already
committed. `client_disconnected`, `session_discarded`, `stream_backpressure` and
`invalid_tool_output` distinguish other terminal causes. Socket loss can prevent
delivery of any final event.

A stored response with an incomplete tool call remains retrievable. Resolving it
as `previous_response_id` returns `400/response_not_continuable` before prompt
construction. This code is Padra-specific. Retry the original request or continue
from the last valid predecessor with enough output budget. Text-only length ends
remain continuable.
