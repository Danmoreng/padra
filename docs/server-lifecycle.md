# Request lifecycle and observability

Queue admission starts an independent five-second waiting deadline. Expiry,
disconnect and session deletion remove queued work and release its byte/token
reservation and session lease without waiting for the active GPU job. Generation
has its own deadline, starting when execution begins; the CLI default is 600
seconds. There is no separately configured total-request deadline.

Normal stop/EOS, length and tool completion are generation outcomes. Errors and
cancellations have a separate state. Streaming keeps 64 ordinary logical events and a
separate three-event completion lane for finish reason, optional usage and
`[DONE]`. A full delta/tool buffer produces an explicit error instead of reporting
success with a truncated protocol. Idle streams emit an SSE comment heartbeat
every 15 seconds by default; deployment-specific proxy behavior still needs testing.

Both APIs return `503/queue_deadline` or `503/server_shutdown` with `Retry-After: 1`,
and `504/generation_deadline`. SSE/Responses failures use the same domain codes.
Admission distinguishes readiness, drain, preparation, queue and session/storage
limits. Generation requests carry a server-generated `X-Request-Id`, including
early refusals. See [API errors](api-errors.md) for codes and retry behavior.
Response records, bytes and new physical roots are admitted together. Eviction
must actually free all references to a root before that root counts as an available
session slot. Manual sessions, active leases, pending and pinned records are
protected. Failed admission releases new roots and pending reservations. Cache
eviction is final, including when a later step fails. Record and physical-root
evictions have separate counters; TTL/deletion-only cleanup is excluded. See
[the lock contract](server-locks.md).
By default, at most four generation requests prepare/tokenize concurrently, with an early
queue/readiness check followed by final byte/token admission.

HTTP shutdown removes readiness, rejects new work and fails queued requests.
Active generation gets a five-second grace period before cancellation; backend
cancellation takes effect at its next cooperative boundary. The listener stays
open during this grace period so admitted responses can finish writing, and stops
when their transports finish or the grace deadline expires. Socket timeouts and
cooperative backend boundaries still govern connection/thread teardown; this is
not a hard process-exit deadline. Repeated HTTP stop calls are safe. Direct coordinator
shutdown cancels immediately. The GPU continues to execute one request at a time.

`/v1/models` reports validated loaded weight/manifest hashes, tokenizer hash,
build source fingerprint, and runtime math/kernel profiles. Missing artifact
identities are `null`. The build fingerprint identifies configure-time source
content; compiler/toolchain and build options are separate qualification inputs.
`/metrics` exposes fixed-bucket latency histograms, output-token totals, and prefill
chunk/eager counts. Queue time excludes preparation; output availability measures
internal text/tool readiness before transport. Job lifetime ends at coordinator
completion, so these metrics are not client-observed TTFT/end-to-end latency.
Transport disconnect/backpressure counters include failures after generation ends.
Admission refusals are counted once in `mellum_admission_rejected_total` with fixed
route/cause labels; `mellum_queue_rejected_total` is the aggregate counter.
Admission logs contain request identity, route and cause without request contents
or credentials.
`mellum_first_visible_write_seconds` measures handler entry to the first successful
body write containing text or tool identity/arguments; protocol-only and reasoning
events do not count. `mellum_response_write_seconds` measures handler entry through
transport completion or failure. Active transports, written responses, write failures
and successfully written body bytes are counted separately. These are server-side
socket-write observations, not confirmation of client receipt; request upload,
network transit and client buffering are excluded. Early admission errors and stored
GET responses are outside these generation-transport counters.

Tool identity is emitted after a complete offered function name is recognized,
then argument fragments stream during generation. Arguments retain their original
JSON bytes; the completed call still undergoes full JSON validation. Clients must
wait for successful completion before using the arguments. If arguments precede
the name, their bounded prefix waits for identification. A length-truncated call
remains partial and does not commit session history. If stored, its incomplete
envelope remains retrievable, but `previous_response_id` referencing it returns
HTTP 400 with Padra's own `response_not_continuable` code. Retry the original
request, or continue from the last valid predecessor, with enough output budget.
Unfinished argument JSON is never repaired or added to valid continuation history.
Text-only length endings remain continuable. Python SDK compatibility has
an optional pinned loopback suite; proxy/load qualification and authenticated
external deployment require separate qualification.

Responses output items keep their first-observed order across streaming deltas,
done events, terminal output and stored GET. There is at most one reasoning item,
one message item and eight tool items. Text/reasoning emitted again extends its
existing item, including text on both sides of a tool call. Nonstreaming requests
use the same logical ordering; items are not regrouped by type at completion.

See [server configuration](server-configuration.md) for validated queue/HTTP
settings, the reserved admin listener, resolved configuration, timing boundaries
and restart behavior.
