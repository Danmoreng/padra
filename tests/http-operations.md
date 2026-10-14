# HTTP operation verification

`padra-http-operation-tests` uses the production configuration, listener, pool and
transport implementation with a barrier-controlled synthetic backend. Its cases
are separate CTest entries and can be repeated individually:

- `http-heartbeat`: injected time values test the exact due/reset boundary; both
  APIs deliver a configured heartbeat while backend preparation is blocked before
  the first token. Admin readiness remains available; releasing the barrier lets
  the admitted request finish.
- `http-slow-bodies`: 16 partial uploads occupy all API workers, verified through
  connection counters. A main-port health request times out. More connections fill
  the 32-slot queue and cause bounded rejections. The separate admin health,
  readiness, metrics and config routes stay reachable within 300 ms client timeouts.
  Drain changes readiness to 503 and keeps liveness at 200. Closing clients frees
  all API workers/pending connections without admitting generation.
- `http-slow-reader`: a peer with a small receive window does not consume either
  API's SSE response. Bounded buffering and a short configured write timeout release
  generation, transport and any temporary root. Admin remains reachable.
- `http-restart`: stop before Bind/Listen, closing bound-but-unstarted API/admin
  sockets, occupied-port rejection, ten synchronized start/stop races, same-port
  restart with fresh runtime/server objects, and loss of volatile response records.
- `http-deadlines`: a blocked preparation overlaps a queued deadline. Queue expiry
  completes independently; generation then observes its own expired deadline at a
  cooperative boundary. Neither admitted timeout becomes an admission rejection.

`server-configuration` checks immutable copies, positional-default overrides,
resolved JSON, unknown fields, duplicate keys, type/range/size bounds and unsupported
context rejection. The CLI was also invoked with an invalid config and nonexistent
model paths: it rejected the configuration before loading tokenizer/model data.

Before the reserved listener was added, a local probe against commit `9c9d54a`
showed `/healthz` timing out after 0.301 s with 16 incomplete request bodies. This
was a shared-pool starvation finding, not a GPU failure. The operation test retains
the same API-pool saturation as the load condition and verifies the reserved path.

PR08 local verification, 2026-10-14:

- GCC 16 Debug: 20/20 CTest, including pinned OpenAI SDK 3.29.0 and 32 wire goldens.
- GCC 16 Release and Clang 23.1.1 Release: 18/18 each.
- Separate Clang 23.1.1 TSan: 18/18; each of five operation cases additionally
  passed 25 consecutive runs (250 synchronized startup races in the restart case).
- GCC 16 ASan/UBSan with leak detection: 18/18; the same 125 operation executions
  passed without suppression or retries after failure.
- CUDA 13.4 native-text build and real-artifact HTTP smoke passed with an explicit
  JSON config. The smoke checks applied overrides, resolved admin port, admin
  config/build identity and health/readiness/metrics, plus existing sessions,
  40 retained roots, both API streams, tool bytes and graceful process exit.

The operation profile does not qualify arbitrary proxies, external authentication,
model quality or maximum settings. See [configuration](../docs/server-configuration.md)
and [event contracts](../docs/server-events.md).
