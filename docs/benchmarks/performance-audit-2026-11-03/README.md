# K6 single and concurrent benchmark, 2026-11-03

These are measurements on one RTX 5070 Ti 16 GiB, not a general performance claim. Baseline is source `b1abeef08e41ff553f5f5038bf913fd851465beb` (server SHA-256 `43bea7dd3aa4cde9e102914094e608f9282029007f06c73e059936fddad1eac9`); integrated source is `39c53253ba72d02506b7e64fa155d1a58f9d11dd` (server SHA-256 `8075bf80e4b7a82d04748326aff6bb0f1663a0f595d27f3af5bb3525be947e09`). Both native builds used Release, C++ and CUDA `-O3 -DNDEBUG`, architecture `120a`, kernel `exact-fusions-v1`, prefill `global-matrix-v1`, Q8 prefill `packed-chunk-v1`, EXL3 transport `async64-v1`. Model was K6 native; tokenizer SHA-256 `58548a346eb073e5132bf7d8ad17dc6971bca36ade378ca4d2bfbc49bf60da2a`; checkpoint `5e9eb70fb97fc0915c3691314d7dce154e501d45b404d39309b3383d472530b1`, manifest `7b8f11faebcb9137f751e93ccb393c6919a34d6193e80794745777efb495f197`. Full resolved build/model identity is in each JSON `/v1/models` response.

Each server used context 8192, prefill chunk 256, BF16 KV, device budget 0, MTP off, queue 16 requests / 64 MiB / 1,048,576 tokens, queue and generation timeout 600 seconds, 8 preparation slots and 24 HTTP threads. Baseline has one running sequence; integrated has 1, 2 or 4. Requests are stateless, default caches disabled; every recorded request reports zero cached tokens. Two warmups per workload preceded three waves at each offered concurrency. Prefill-heavy uses 6,170 input tokens and one output token; decode-heavy uses 1,050 input tokens and 256 output tokens. Sampling is temperature 0.6, top-k 40, and seed 42 plus request index. The harness, configs, server logs, complete request timings/headers/usage and hashes are included here. A paused unrelated GPU process retained about 2.7 GiB; with the four-lane server loaded total GPU memory observed was approximately 13.2 GiB. The runs were sequential, not randomized; short-run variation and thermal drift limit small comparisons.

Each cell below is the median of the three per-wave values, in tokens/s. P is whole-wave prefill-heavy input tokens divided by wave wall time, including queue and request overhead. D is whole-wave decode-heavy output tokens divided by wave wall time, including prefill and request overhead. Neither is pure phase throughput. p is the median per-request prefill phase rate (prompt tokens / server prefill seconds) for the prefill-heavy wave. d is the median per-request decode phase rate ((completion tokens minus the first) / server decode seconds) for the decode-heavy wave. Phase rates describe individual requests and must not be summed to infer concurrent throughput. Peak is the maximum `mellum_active_requests` observed across the three waves for either workload, sampled every 10 ms.

| Build / lanes | Offered clients | P | p | D | d | Peak |
|---|---:|---:|---:|---:|---:|---:|
| Baseline / 1 | 1 | 8,275 | 8,318 | 226 | 255 | 1 |
| Baseline / 1 | 2 | 8,247 | 8,274 | 226 | 255 | 1 |
| Baseline / 1 | 4 | 8,231 | 8,257 | 226 | 255 | 1 |
| Baseline / 1 | 8 | 8,189 | 8,199 | 226 | 255 | 1 |
| Integrated / 1 | 1 | 8,245 | 8,311 | 226 | 256 | 1 |
| Integrated / 1 | 2 | 8,242 | 8,271 | 226 | 256 | 1 |
| Integrated / 1 | 4 | 8,250 | 8,260 | 226 | 256 | 1 |
| Integrated / 1 | 8 | 8,192 | 8,204 | 226 | 256 | 1 |
| Integrated / 2 | 1 | 8,192 | 8,238 | 226 | 256 | 1 |
| Integrated / 2 | 2 | 9,703 | 4,869 | 338 | 197 | 2 |
| Integrated / 2 | 4 | 9,647 | 4,835 | 337 | 196 | 2 |
| Integrated / 2 | 8 | 9,613 | 4,813 | 303 | 168 | 2 |
| Integrated / 4 | 1 | 8,184 | 8,236 | 226 | 256 | 1 |
| Integrated / 4 | 2 | 9,597 | 4,918 | 338 | 197 | 2 |
| Integrated / 4 | 4 | 10,004 | 2,509 | 369 | 109 | 4 |
| Integrated / 4 | 8 | 9,965 | 2,496 | 357 | 105 | 4 |

At four offered clients, four lanes increased whole-wave prefill-heavy input rate by 21.5% and decode-heavy end-to-end output rate by 63.6% versus baseline; individual requests' phase rates fell under concurrent GPU contention. The three four-client integrated decode waves varied from 335 to 373 output tokens/s. Baseline versus each integrated configuration matched output SHA-256 and full usage for all 90 corresponding measured requests (270 comparisons); hashes cover message and finish reason. No request errors occurred. This is a fixed-seed regression check for these inputs, not proof of general numerical equivalence. Single-lane MTP and the A-B-A snapshot cache exercise were not measured in this window.
