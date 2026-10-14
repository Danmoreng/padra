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
