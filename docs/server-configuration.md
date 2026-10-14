# Server configuration and administration

The positional server CLI remains:

```sh
padra-server ARTIFACT TOKENIZER_JSON TOKENIZER_SHA256 PORT \
  [GENERATION_TIMEOUT_MS] [CONTEXT_CAPACITY] [PREFILL_CHUNK] [DEVICE_BUDGET_GIB] \
  [--config server.json]
```

The optional JSON file overrides the included queue/HTTP settings after positional
defaults are resolved. Omitted fields retain those defaults. It cannot change the
model, tokenizer, context or numerical execution profile. Invalid types, unknown
fields, unsupported values, duplicate keys and files above 64 KiB are rejected
before loading the model. Configuration is validated once and immutable while
running. Change it by restarting the process; there is no live reload endpoint.

Example:

```json
{
  "queue": {
    "requests": 8,
    "bytes": 67108864,
    "tokens": 1048576,
    "timeout_ms": 5000,
    "generation_timeout_ms": 600000,
    "drain_timeout_ms": 5000
  },
  "http": {
    "preparation_requests": 4,
    "threads": 16,
    "queued_connections": 32,
    "admin_threads": 2,
    "admin_queued_connections": 8,
    "admin_port": 0,
    "send_buffer_bytes": 16384,
    "keep_alive_seconds": 2,
    "keep_alive_requests": 16,
    "read_timeout_ms": 5000,
    "write_timeout_ms": 2000,
    "heartbeat_ms": 15000,
    "poll_ms": 20
  }
}
```

The CLI generation default is 600000 ms, context 131072, and queued tokens eight
times the selected context. The embedded coordinator defaults are 30000 ms,
8192 context in the HTTP constructor, and 65536 queued tokens. Other defaults are
shown above. An explicit positional generation timeout retains its meaning;
`queue.generation_timeout_ms`, if supplied, overrides it.

Queue/preparation limits bound admitted work separately from HTTP connection
limits. Request-body allowance is derived from context and reported as
`request_body_bytes`. Main HTTP workers are 1–64, pending connections 1–1024,
preparation slots 1–64, admin workers 1–8 and admin pending connections 1–64.
Read/write/heartbeat timeouts are 1–600000 ms; poll is 1–1000 ms and cannot exceed
the heartbeat interval. Queue/generation timeouts are 1–600000 ms and drain can
also be zero. Queue count is 1–1024, bytes 1–1 TiB and tokens 1–2³¹. Keep-alive is
1–60 seconds and 1–1024 requests; send buffer is 1024–1048576 bytes. These are
configuration bounds, not a qualification for every maximum-load combination.

## Reserved admin listener

The API and admin listeners bind to `127.0.0.1`. Startup JSON reports `listener`,
`admin_listener` and resolved `server_config`. `admin_port: 0` chooses an available
port; a fixed admin port must differ from the API port. Both ports are exclusive
to one live server instance and can be rebound after shutdown. Failure to bind
either listener fails startup and releases the other socket.

Use the admin listener for `GET /healthz`, `/readyz`, `/metrics` and `/config`.
It has its own bounded worker pool and connection queue. `/config` reports the
resolved nonsecret configuration and build/profile identity. `/v1/models` also
includes the resolved configuration with model and numerical profile metadata.
The original API-port health/readiness/metrics routes remain available in its
shared pool. Generation and state-changing routes are unavailable on admin.
Both listeners enforce local Host/Origin policy; they provide no public TLS/auth.

The reserved pool keeps admin reachable when the API pool is occupied by slow
uploads, queued connections or slow readers. It is not a guarantee under an
independent flood of the admin listener or a stalled process. Connection gauges
and rejection totals use fixed `listener="api"|"admin"` labels. Pending counts can
include the one connection being enqueued; successful admission to the pool still
obeys its configured queue bound.

## Timing and restart

Queue expiry runs independently of active inference. Generation time starts at
dequeue, including KV preparation; cancellation is cooperative at backend
boundaries. Read/write timeouts apply to socket operations, not to an entire
request upload/response lifetime. There is no end-to-end request deadline.

Idle SSE connections receive comment heartbeats at the configured interval,
including while model preparation/prefill has produced no token. Only successful
wire writes reset the idle interval. The polling interval bounds how soon the
provider checks disconnect/heartbeat state. Proxy buffering can alter client-side
observations.

Drain immediately makes readiness false while liveness stays alive. Queued jobs
are cancelled; admitted active generation and transport can finish within the
drain grace. Cancellation follows expiry of that grace. Socket timeouts and backend
cooperation govern final teardown, so drain is not a hard process-exit deadline.
Stop is idempotent, including before Bind/Listen and concurrent startup. A stopped
server object is not restarted in place; create a new server/runtime or process.
Sessions, response retention and KV identity are volatile and absent after restart.
