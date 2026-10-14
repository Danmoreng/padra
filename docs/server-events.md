# Generation events and results

The coordinator publishes protocol-neutral `GenerationEvent` values. Text,
reasoning, tool identity and raw argument fragments are owned by fixed-capacity
slots; no decoder/parser string view crosses a thread boundary. Fragment payloads
are at most 1024 bytes, call IDs 128 bytes, and function names 64 bytes. UTF-8
characters are kept whole when splitting a fragment. Tool argument bytes are
preserved, including whitespace, escapes and property order.

Each job has 64 ordinary slots and a separate three-slot completion lane. Normal
completion publishes finish, optional usage and done events. Errors replace the
completion lane with one bounded error event. Ordinary backpressure cannot consume
completion capacity. The HTTP consumer drains ordinary slots first and accesses
completion events only after `complete` is published under the job mutex.

`GenerationResult` contains canonical output order, text/reasoning/tool output,
usage, finish reason and the captured generation failure cause. It is worker-owned
until completion; adapters and retention accept const references after that
publication. A later transport disconnect does not rewrite the generation result.
Transport completion, history continuability and generation completion are distinct.

`ChatStream` encodes events on the HTTP thread into its reserved SSE buffer.
`ResponseStream::OnEvent` consumes the same logical events directly and emits
Responses events. It does not parse serialized Chat chunks. Both adapters derive
terminal objects from the common result and its canonical output order. A
Responses adapter may emit multiple wire events for one logical item/fragment.

## Verification

`generation-events` checks source-buffer ownership, invalid/oversized slots,
ordinary saturation followed by all three completion records, worst-case JSON
escaping, and stable failure results after a transport disconnect. Existing mixed
output, partial-tool, retention, disconnect and drain tests exercise both adapters.
`api-wire-golden` compares 32 captured pre-refactor wire cases for both APIs,
streaming/JSON, text/reasoning/tools, stop, invalid and incomplete tools. Only
generated IDs and timestamps are normalized. Output content, argument bytes,
statuses, indices, sequence numbers and usage are compared exactly. The Python
wire test uses standard-library HTTP with explicit timeouts and no request retries.
It runs alongside the pinned SDK test with the synthetic fixture.

On local x86-64 GCC 16, the slot is 1256 bytes; 67 slots occupy 84152 bytes and
`sizeof(Job)` is 85088 bytes. The preceding serialized-slot representation at
commit `7cbbded` measured 8208, 549936 and 550832 bytes respectively with GCC 15.
These are in-object storage measurements, not total request RSS or throughput.
The allocation probe counts ordinary global `new`/`new[]`: 64 event constructions/
publications and 64 ordinary Chat encodes after buffer reservation allocate zero.
Usage/error/final-object construction can allocate. No speedup is claimed.

Local PR06 verification on 2026-10-14: GCC 16 Debug with pinned SDK and wire
goldens 13/13; GCC 16 Release, Clang 23 Release, GCC 15 TSan and GCC 16
ASan/UBSan each 11/11. Five admission/transport/disconnect tests also passed
20 consecutive runs each under TSan. CUDA 13.4 native-text build and the real
artifact HTTP smoke passed, including 40 retained roots, sessions, both streams,
raw tool output, metrics and graceful exit. This smoke is not model-quality or
numerical qualification.

The allocation probe runs in ordinary/ASan builds. TSan owns its allocator
interceptors, so TSan builds run the same functional event tests without replacing
`new`/`delete` and explicitly report that allocation measurement is disabled.
