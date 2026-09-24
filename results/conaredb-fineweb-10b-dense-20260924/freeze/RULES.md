# S5/S6 rules, written 2026-09-24T00:50Z, before any dev sweep beyond the one 9k baseline and before any test query

- Queries: dev = q0-4999 (tuning only), test = q5000-99999 (95,000; first sent in S6, after the freeze commit). Split from S2 `splits.json`.
- Scoring: nova-storm 7d7d0d3 (conareai/supernova master ce6690ee), target conaredb, `/v2/search {vector, top_k: 10}`,
  S0 CSR expansion on the client (s5 NVMe copy, sha256 == S0), `copy_order: uuid` (nova-bf 'id' tie order; S1 shows the GT uses it in
  71,045/71,045 split groups). GT = Qdrant/FineWeb-10B@3aea0b8d gt_dense_k1000 (fp32), top-100 kept. tie_epsilon = nova-storm auto
  (2e-3, float16 data). Strict recall@10 (lower bound) is the headline; tie-aware (upper) second.
- Every run: `passes: 1` (each query exactly once), /tmp/BENCH_RUNNING held on A and s5, server identity (pid, exe sha256,
  server.json sha256, budgets) recorded before and after; a run whose identity changed is void and rerun.
- Dev sweep: leaves {9k, 12k, 18k} x (scan, route) threads {8+8, 4+4, 2+2} on the server binary chosen by a 9k dev A/B
  (conare main c136d6788 build vs the running 917245aa per-worker-pools build); per point c32 closed loop + r200 + r400 open loop.
- **Freeze rule (headline config):** among dev points with strict recall@10 >= 0.97 and 0 errors at c32, r200 and r400,
  the one with the lowest dev c32 p99. If none reaches 0.97, the point with the highest dev strict recall whose r200 p99 <= 100 ms.
  The frozen binary sha256 and server.json sha256 are committed to the research branch before the first test query.
- **S6 conditions (Qdrant/nova-storm style):** top_k 10; closed loop concurrency 32; open loop 200 and 400 rps (in-flight cap 256);
  3 repeats each, run order c32, r200, r400 per repeat; page cache warmed on one dev pass (c32) right before repeat 1, never on test.
  Each run = all 95,000 test queries once (c32 ~3 min, r200 475 s, r400 238 s).
- **Quoted value:** per condition and per metric, the median of the 3 repeats (recall, qps, p50, p95, p99 each taken separately);
  all 9 raw runs are published. Repeats 2-3 reuse the test queries (warmer page cache); disclosed.
