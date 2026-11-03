# Concurrent MTP D1 measurements, 2026-11-03

On one RTX 5070 Ti 16 GiB, original K6 MTP server source `b1abeef08e41ff553f5f5038bf913fd851465beb` was compared with concurrent-MTP candidate `e7597cb` (binary SHA-256 `edf97cf643ad5ae4f66255f7efb45e9ecb15d4537b2051f6a8f46c11122de061`). Builds used Release `-O3 -DNDEBUG`, CUDA architecture 120a, exact-fusions-v1, global-matrix-v1, packed-chunk-v1 and async64-v1. The EXL3 K6 target and EXL3 K4 draft used MTP depth 1 and compact draft dispatch. Both used 8192 context, BF16 KV, chunk 256, identical queue and HTTP limits, and stateless requests with zero cached tokens. A paused unrelated GPU process retained ~2.7 GiB. The candidate enabled 1, 2 or 4 runtime lanes. The previous [ordinary baseline](../performance-audit-2026-11-03/) used the same inputs and settings with MTP off. Configs, full model/build identity, per-request headers, usage, hashes and timings are in the accompanying JSON and logs.

The harness from the previous report used 6,170 input/1 output tokens for prefill-heavy, and 1,050 input/256 output tokens for decode-heavy; temperature 0.6, top-k 40 and seed 42 plus request index. Each workload had two warmups and three waves per offered concurrency. P is prefill-heavy whole-wave input tokens/wall second, and D is decode-heavy whole-wave output tokens/wall second, both including request overhead and, for D, prefill. p is median per-request prefill input tokens/server prefill second in prefill-heavy waves; d is median per-request (completion tokens minus first)/server decode second in decode-heavy waves. Values below are medians of the three wave-level results. Phase rates cannot be summed into concurrent throughput. Peak is maximum sampled active requests across waves, observed every 10 ms.

| Build / lanes | Clients | P | p | D | d | Peak |
|---|---:|---:|---:|---:|---:|---:|
| Original MTP / 1 | 1 | 7,503 | 7,540 | 268 | 317 | 1 |
| Candidate MTP / 1 | 1 | 7,487 | 7,530 | 268 | 317 | 1 |
| Candidate MTP / 2 | 2 | 8,643 | 4,337 | 337 | 201 | 2 |
| Candidate MTP / 2 | 4 | 8,605 | 4,313 | 340 | 203 | 2 |
| Candidate MTP / 2 | 8 | 8,573 | 4,289 | 334 | 198 | 2 |
| Candidate MTP / 4 | 2 | 8,555 | 4,381 | 333 | 200 | 2 |
| Candidate MTP / 4 | 4 | 8,729 | 2,190 | 380 | 117 | 4 |
| Candidate MTP / 4 | 8 | 8,676 | 2,173 | 379 | 115 | 4 |

At one client, the earlier ordinary baseline gave a median decode phase ~255 tokens/s, versus ~317 with MTP, while prefill fell from ~8,318 to ~7,530 tokens/s. At four clients, the prior ordinary four-lane build gave 10,004 prefill-heavy input tokens/s and 369 decode-heavy output tokens/s whole-wave, versus 8,729 and 380 with MTP. These short sequential runs are workload-specific; small differences may be noise. Candidate MTP output hashes and full usage matched original MTP for all 90 measured requests in each of the three lane configurations (270 comparisons), and the original MTP matched ordinary baseline for the same 90 inputs. No benchmark requests failed. D1 and D2 two-lane overlap, draft sharing, rejection, cancellation and freed-lane reuse probes passed; their raw results are included.

Separately, a two-lane D1 server with 32,768 context, chunk 2048 and BF16 KV started and completed a short two-client prefill/decode smoke, with peak active 2. At that point total device memory usage was ~15,013 MiB with the unrelated process resident, leaving ~921 MiB. This did not exercise a full 32,768-token input and is not a capacity guarantee. Two-lane 131,072 context using q8_0 failed startup admission by ~239 MiB in this setup. The original service remained configured at 131,072 BF16, one lane, and was restored after tests; the memory/context tradeoff requires an explicit deployment choice.
