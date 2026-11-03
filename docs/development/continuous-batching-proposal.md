# Proposal: configurable continuous batching for concurrent agents

Status: draft design request. The settings below are proposed, not implemented.
This document requests feedback on the runtime and scheduler direction before
an implementation is attempted.

## Problem and evidence

Padra's single-stream performance is strong, but several coding agents cannot
currently generate together on one loaded model. `Coordinator::Work()` removes
one FIFO job, sets `active_`, and calls `Execute(job)` before taking another.
`ModelRuntime` explicitly supports one preallocated session and rejects concurrent
prepare/generate/reset calls. Raising HTTP threads, preparation slots or queue
limits increases admitted/waiting work without adding active GPU sequences.

On an RTX 5070 Ti with 16 GB VRAM, K6/shared-Q8-head weights, BF16 KV,
131072-token capacity, 2048-token prefill chunks and EXL3 MTP D1, two short
burst rounds gave:

| Concurrent clients | Aggregate output tokens/s | Median request latency |
| ---: | ---: | ---: |
| 1 | 274.8 | 0.93 s |
| 2 | 287.0 | 1.34 s |
| 4 | 283.1 | 2.24 s |
| 8 | 280.6 | 4.08 s |

All 30 measured requests succeeded. Each had 1050 input tokens and 256 output
tokens, with no cached prefix. Output counts include reasoning. These are short,
repeated-code synthetic bursts, not sustained-load or model-quality results.
They show flat aggregate throughput and growing wait time, not a projected
speedup for any proposed implementation.

The [raw measurements](../performance/concurrency-rtx5070ti-2026-11-03.json)
record per-request usage and timing headers. Run the included standard-library
client against a configured server to repeat the workload:

```sh
python3 tools/bench_http_concurrency.py --rounds 2 --max-tokens 256 \
  --output concurrency-results.json
```

The evidence file records the exact engine/model revisions, toolchain, server
configuration and workload. Startup, model upload and warmup are excluded from
the measured burst waves; each wave's aggregate speed includes HTTP, queue,
prefill and decode time. There is one additional unmeasured warmup request.

## Proposed configuration

Follow the existing [server JSON configuration](../server-configuration.md)
passed through `--config`, with strict validation and immutable startup settings.
A separate scheduler section keeps active GPU work distinct from queue capacity:

```json
{
  "scheduler": {
    "max_running_sequences": 4,
    "max_batched_tokens": 4096
  }
}
```

These are candidate names and values for discussion, not working flags or
qualified hardware settings. Default `max_running_sequences` to 1 to preserve
the existing execution path. Define `max_batched_tokens` as the maximum number
of target input rows across prefill and decode/verification work in one scheduler
iteration. MTP proposals consume verifier-row budget; output-token budgets retain
their existing per-request meaning. Resolve the default and supported bounds
from the admitted runtime profile rather than promising these example values fit.

Keep `queue.requests`, `queue.tokens`, `http.threads` and
`http.preparation_requests` independent. Queued-token limits are host admission
bounds, not a GPU KV-memory budget. Report resolved scheduler settings through
startup JSON, admin `/config` and `/v1/models`.

JSON is the existing project convention. Environment-variable aliases can be
considered separately if maintainers prefer them; they should use the same
parser/validation and have documented precedence rather than a second settings
implementation. This proposal does not require an environment-variable API.

## Runtime and scheduling requirements

1. **Share weights, isolate sequences.** Load target/draft/shared-head weights
   once. Give each active sequence its own target global KV, local sliding-window
   rings, positions, sampler RNG, penalty history, stop/parser state and session
   identity. Keep checkpoint handles tied to their sequence and generation.
   Multiple host workers calling the current shared runtime would violate its
   contract; a worker-count switch alone is insufficient.
2. **Schedule bounded steps.** Replace whole-request execution with an active
   sequence set and bounded prefill/decode steps. Admit waiting requests in FIFO
   order when sequence and KV capacity permit, remove completed/cancelled
   sequences independently, and allow new arrivals to join between steps.
   Interleave bounded prefill chunks so a long prompt cannot monopolize the GPU
   while other sequences are decoding. Document fairness, queue-to-active
   transition and generation deadline boundaries.
3. **Execute across sequences.** Add sequence-aware attention descriptors and
   batched projection/MoE routing, expert dispatch and combination. Preserve
   causal masks and sliding-window wraparound for each sequence. A scheduler
   that merely alternates existing single-row launches may improve fairness;
   classify it separately from batched execution and measure both.
4. **Bound memory explicitly.** Account for all active KV, sequence state,
   scratch space, graph variants, checkpoints, and MTP rollback buffers while
   sharing weights. Retain the reserve and post-graph admission check. Use a
   bounded pool of KV pages or sequence slots; document the chosen policy and
   fragmentation costs. Reject incompatible startup configurations explicitly,
   and keep requests queued/reject admission according to a documented runtime
   policy when capacity is unavailable. Do not silently reduce context, change
   KV precision or offload weights. A 131K context limit per request does not
   promise that four independent 131K requests fit concurrently on 16 GB.
5. **Preserve isolation and lifecycle behavior.** Cancellation, disconnect,
   backpressure, output limits, session deletion and drain must release only the
   affected sequence's resources. Keep the existing one-in-flight-request lease
   per session. Drain and expiry must handle every active sequence rather than
   the current single `active_` job. A slow SSE reader must not stall unrelated
   generation; keep existing bounded transport behavior.

## MTP correctness

The [MTP execution contract](../mtp.md#execution-contract) already batches causal
verifier rows within one request and shares projection/expert work. That is
useful machinery to investigate, but those rows share one sequence's history.
It does not establish support for batching independent requests.

Cross-request execution needs independent draft rings, preceding target hidden
states, target/draft positions, verifier snapshots and rollback bookkeeping.
Rejected rows must restore only their sequence; no sequence may attend to another
sequence's rows or consume its RNG/penalty history. Only target-selected tokens
may reach callbacks and streams. Preserve target-sample coupling, EOS, stop and
short-budget tails described by the current contract.

For a fixed seed and numerical profile, batching membership/order should not
change a sequence's sampling state. Gate ordinary-versus-MTP and batch-size-one
compatibility against the current exactness contracts. If cross-sequence kernels
change numerical behavior, expose and qualify a new execution profile rather
than assuming equivalence. Unsupported MTP/KV/batch combinations must fail
explicitly. An initial ordinary-decoding-only stage is reasonable if selecting
MTP with concurrency greater than one is clearly rejected until it is supported.

## Suggested delivery and acceptance

- First define sequence state, memory admission, scheduler limits and fake-runtime
  tests while retaining the default single-sequence path.
- Add ordinary cross-sequence decode and bounded prefill, then qualify batched
  attention/MoE execution. Do not describe mere time slicing as a throughput gain.
- Add MTP with per-sequence verification/rollback and qualify supported depths and
  KV formats before enabling each combination.
- Export numeric active-sequence and waiting-request gauges, resident/reserved KV,
  prefill/decode batch occupancy, admission failures and aggregate output counts.
  Update the existing boolean active-work metric compatibly. Keep existing
  queue/TTFT/visible-output timing meanings documented; distinguish visible text
  from reasoning and do not infer per-token latency from speculative bursts.
- Validate staggered arrivals, mixed prompt/output lengths, session isolation,
  cancellation/disconnect, slow readers, queue deadlines, drain, capacity
  exhaustion and reclaim after EOS. Include local-ring rollover and mixed MTP
  acceptance/rejection with deterministic interleavings.
- Benchmark 1/2/4/8/16 clients with both short and long inputs, mixed lengths,
  cold and reused prefixes, ordinary and MTP decoding, and supported KV formats.
  Report successful/failed requests, actual peak active sequences, aggregate
  output tokens/s, per-sequence speed, TTFT/ITL and request p50/p95 latency, memory
  and MTP acceptance. Capture streaming timestamps for ITL; nonstreaming headers
  alone cannot measure it. Use sustained runs with explicit warmup and a
  batch-size-one regression baseline. Publish measured tradeoffs rather than a
  target speedup or a concurrency count inferred from client connections.

The request is configurable, bounded multi-sequence execution on one resident
model. CPU offload, distributed serving, speculative-algorithm changes and a
hierarchical multi-session prefix cache can be evaluated separately.
