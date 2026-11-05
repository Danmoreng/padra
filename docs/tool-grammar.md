# Tool-output grammar

Requests with enabled function tools automatically constrain target token
selection. A lazy byte grammar activates at `<tool_call>` in visible assistant
content. Initial `<think>…</think>` reasoning remains unconstrained. Plain text
outside calls remains available, so `tool_choice: "auto"` does not force a call.
`tool_choice: "none"` disables tool parsing and the grammar.

The grammar emits one JSON object per `<tool_call>…</tool_call>` pair, with
`name` followed by `arguments`. Names must be offered functions; arguments must
be objects. Strings obey JSON escaping and UTF-8, including split token bytes
and paired Unicode surrogate escapes. EOS is excluded inside an unfinished
call. Parallel calls use separate marker pairs, up to eight per response (one
with `parallel_tool_calls: false`). The output parser still validates completed
calls and enforces its size/depth limits.

The compiler supports structural schema constraints for string, number, integer,
boolean, null, object properties, required fields, array items, const and enum.
Required object properties are emitted in `required` array order; optional
declared properties follow in lexical order. This selects an ordered subset of
possible objects. `anyOf` and `oneOf` select a union of alternatives; exclusive
`oneOf` matching is not checked. Unspecified types and unsupported compositions
such as `$ref`, `allOf` and `not` use generic JSON at that node. Other assertions,
including numeric bounds, string patterns, lengths and array cardinalities, are
not enforced. Full JSON Schema compliance is **not** guaranteed; `strict: true`
and structured response formats remain unsupported. Tools must validate their
arguments before execution.

The grammar cannot ensure useful arguments, working code or completion within
the output budget. Length limits, stop strings and cancellation can still end a
call incompletely. Clients must wait for successful completion before executing
tools; partial argument deltas are provisional. See [API errors](api-errors.md).

Each request owns a fresh grammar state, independently of KV prefix reuse.
The sampler masks forbidden tokens before top-k/top-p and before greedy
selection, preserves source logits, and advances the RNG once per selected
target token. MTP draft proposals are unconstrained; verification samples from
the constrained target distribution and accepts only matching proposals.
Only committed target tokens advance the grammar. Rejected proposals and
unvisited verification rows do not mutate it.

The CUDA sampler owns a 98,304-byte device mask, included in memory admission.
CPU masks are computed lazily and cached in a bounded per-request cache (at most
128 vocabulary masks). Ordinary CUDA graph launches and MTP verification use
the current mask. Constraint overhead is included in generation timings.
Schema compilation is bounded to depth 32, 128 declared properties per object,
8,192 rules and 65,536 byte terminals; exceeding these limits rejects the request.

Host tests cover escaped code, Unicode, wrong names/types, incomplete-call EOS,
multiple calls and independent request state. The model-free GPU sampler test
covers greedy/stochastic masking, top-k ordering, RNG steps, source-logit
preservation and changing masks in a captured graph. Hardware qualification and
end-to-end benchmark results are recorded separately under `docs/performance`.
