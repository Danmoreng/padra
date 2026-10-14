# Store admission lock contract

Response admission plans record count, response bytes, additional physical session
slots and initial session bytes together. `ResponseStore::Reserve` takes the
response/session mutexes with `std::lock`; standalone session operations use the
same session mutex. It does not acquire the coordinator mutex. Prepared map nodes
and rollback ownership are allocated before the eviction commit. Failed planning
does not evict live records; expired/deleted cache cleanup remains final.

Only unpinned, nonpending response records are candidates. An active session lease
protects its records/root. A root frees a physical session only when all its record
references are retired and no external root references remain. Manual sessions
have no cache-owned root and cannot be selected. Record/byte and root constraints
must all fit before the plan commits. Pending reservation and new root are owned
by one RAII ticket; abandoning it releases both, including after failed Submit.
Cache evictions are final even if subsequent preparation or Submit fails.

Root discard callbacks and retired record destruction happen after both store
mutexes are unlocked. Callbacks may clear coordinator GPU-cache metadata or delete
a session; they cannot run within the joint store transaction. Core root deletion
precedes the optional notification, so a throwing callback cannot leak a root. Expiry/deletion
pruning in the other ResponseStore operations follows the same deferred-destruction
rule. Injected clocks must be nonreentrant, nonblocking time readers.

The coordinator may acquire a session mutex while holding its own mutex for final
admission/cache checks. No store operation acquires the coordinator mutex while
holding a store mutex. Transport accounting has a separate mutex: metrics may
take it under the coordinator mutex; transport callbacks never acquire the
coordinator mutex while holding it. HTTP stop is serialized independently.

Eviction metrics count actual cache record removals and physical roots freed by
those evictions, separately. TTL/deletion-only cleanup is not an eviction. The
existing entry/byte limits are unchanged; manual protected capacity can reject new
responses even when the response record cache itself has space.

Generation event slots own their fragment bytes. Worker result fields are published
by `Job::complete` while holding the job mutex; HTTP reads final result fields only
after observing completion through that mutex. Ordinary encoders read stable
request metadata plus their copied event, not mutable worker result fields.
Terminal failure classification is captured before publication. See
[generation events](server-events.md) for the bounded channel and adapter contract.

## Execution and ownership boundaries

`coordinator.cpp` owns admission, FIFO selection, independent queue expiry, metrics
and drain. `coordinator_execution.cpp` orchestrates KV preparation, output
processing, history commit and cleanup. `output_processing.cpp` owns the token
pipeline and logical event publication; it does not prepare KV, mutate stores or
write sockets. `job.cpp` owns the event channel, cancellation, terminal failure
snapshot and transport accounting. Chat and Responses encoders own protocol JSON.

`PrepareExecution` is worker-only and restores a matching retained checkpoint or
resets/prepares the runtime. `GenerateOutput` owns local decoder/parser storage and
reports runtime failures to the coordinator; parser failures do not by themselves
poison readiness. `CommitHistory` commits only valid complete tool history and
updates cache identity after commit. Partial tool output invalidates cache identity
and remains retrievable but not continuable. `Execute` catches failures from these
boundaries, clears touched cache identity, resets an unpoisoned runtime and
publishes a terminal failure. It then always calls `CompleteJob`.

| State dimension | Owner and publication | Resource release |
| --- | --- | --- |
| Queued/active/completed job | Coordinator FIFO/active pointer; completion under job mutex | Queue bytes/tokens removed at dequeue or expiry; active pointer cleared once |
| Generation result | Worker or expiry worker, then const adapter access after completion | Local parser storage unwinds on every success/failure |
| Session lease/history | Job holds one RAII lease until `CompleteJob` | Worker local lease is dropped first; job lease is reset before completion notification |
| Pending response/root reservation | HTTP request/provider owns RAII ticket | Drop rolls back uncommitted record/root; commit transfers retained state |
| Transport unfinished/written/failed | HTTP provider/release callback, idempotent atomics | `FinishTransport` decrements active transports once; generation completion does not close it |
| Retained continuability | Immutable stored record policy | Partial tools preserve retrievable envelope and last valid history without continuation |

Runtime execution is owned by one generation worker. The expiry worker never calls
GPU operations. Coordinator locks protect FIFO/cache identity and metrics; runtime
calls and token callbacks occur outside them. Job locks protect event indices and
completion publication. `CompleteJob` releases session ownership, records outcome
under the coordinator lock, releases that lock, then takes the job lock to publish.
No coordinator lock is held while waiting for HTTP drain or joining workers.
Callbacks may cancel through atomics without taking the job mutex.

`execution-boundaries` injects reset, preparation, generation, tool-validation and
two history-normalization failures. Each verifies inactive leases, unchanged
history, empty queue budgets, exactly one generation outcome, a separate pending
transport, and the ability to reserve the entire 32 MiB session budget afterward.

## HTTP boundaries

`http_server.cpp` registers routes and owns socket lifecycle and administrative
metadata. `http_generation.cpp` normalizes requests, performs early/final admission
and wires ownership to a response. `http_request.*` owns namespace validation and
the RAII preparation slot. `http_state.cpp` handles explicit sessions and retained
GET/delete. `response_retention.*` commits the final envelope and the permitted
history policy, while owning the ticket/root through provider completion.
`http_transport.cpp` owns error mapping, finite JSON providers and the shared SSE
pump, including disconnect polling, heartbeat writes and release callbacks.

Nonstream waiting acquires the job mutex only to observe completion. Streaming
checks that both lanes have drained under that mutex and releases it before
retention, final encoding or socket writes. There is no store or socket callback
under a job lock. The immutable published result keeps this safe. A retention
failure leaves the reservation owned until request/provider destruction; the
boundary test rejects an undersized commit and verifies that both pending record
and physical root are then released.
