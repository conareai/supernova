> Copied from conareai/conare `RESEARCH/codex-postmortem-20260922/receipts/qdrant-10b-20260924/final/` (research branch, private repo). `../verify/cost/`, `s5/` and some `tools/` scripts referenced below live there only.

# Final lane (2026-09-24, 06:45–11:46Z): S3 expansion A/B, S6 grid, Qdrant fleet shape at top_k 100

Rules: `RULES.md`, pushed in c46e34eaf before any run of this lane. Raw runs: `runs/` (nova-storm `--json` output, the exact
yaml, the nova-storm log, per-request rows gzipped). Server identity every 30 s on A: `identlog.txt`; switches: `switch.log`,
`serve.log`; dev-tuning identity before/after each point: `tune-ident.txt`. Per-run table with the identity check:
`final-runs.tsv`; medians: `final-medians.json` (both from `tools/final_summarize.py`). Scripts: `tools/` (laptop + A),
`s5/` (the client-side copies that ran, with queue files and their stdout).

Client for every run: s5 (Standard_L32aos_v4), nova-storm sha256 `a5450143…` (conareai/supernova 7d7d0d3), target `conaredb`,
CSR uuid expansion `copy_order: uuid` in the measured path. Server: A (Standard_L96as_v4), exe `cd60f49a…` (conare main c136d6788).

## Summary

| Item | Result | Status |
|---|---|---|
| S3 expansion cost | mean c32 Δp50 **−0.09 ms**, Δp99 **−0.05 ms**; r200 Δp50 **+0.43 ms**, Δp99 +0.54 ms (on − off) | PASS: not material, client unchanged |
| S6 grid, frozen 12k, test, top_k 10 | c1, c16, c64, c256, 100/600/800 rps × 3, 95,000 requests each; 2 errors in 1,995,000 requests (HTTP 429 `queue_full`, c256 rep2, first 9 ms) | PASS |
| Throughput ceiling | closed loop c64/c256: 614/624 QPS (medians); 800 rps offered → 624 QPS achieved, above capacity | measured, reported |
| Qdrant fleet shape, frozen 12k (top_k 100, 160 in flight, 1,200 s) | strict r@100 **0.95836**, tie 0.98378, 628.1 QPS, p50/p95/p99 251.9/321.8/359.9 ms, 0 errors | measured |
| top_k 100 tuned on dev (18k leaves, rescore 1000; frozen in 5ba0ec3fe before its first test query) | strict r@100 **0.96892**, tie 0.98859, 397.0 QPS, p50/p95/p99 399.3/553.5/617.9 ms, 0 errors (dev had 0.97119: the test value is 0.0023 lower) | measured |

## Step 1-2: frozen server, dev warm, S3 expansion A/B (test, top_k 10)

A switched to the frozen server at 06:45:04Z (`switch.log`): pid 291953, exe `cd60f49a…`, server.json `4beb0731…`. It served
every top_k 10 test run from 06:45 to 10:35 without a restart (`identlog.txt`).

| Run | QPS | p50 | p95 | p99 | max | errors |
|---|---:|---:|---:|---:|---:|---:|
| c32 expand on (1) | 596.40 | 53.01 | 63.53 | 69.37 | 187.49 | 0 |
| c32 expand off (1) | 597.54 | 52.99 | 63.41 | 68.99 | 166.60 | 0 |
| r200 expand on | 199.99 | 26.02 | 32.15 | 36.69 | 104.19 | 0 |
| r200 expand off | 199.99 | 25.59 | 31.84 | 36.14 | 101.63 | 0 |
| c32 expand off (2) | 589.65 | 53.65 | 64.87 | 70.90 | 177.22 | 0 |
| c32 expand on (2) | 592.38 | 53.44 | 64.66 | 70.43 | 194.78 | 0 |

Δ (on − off), `s3-expansion-ab.json`: c32 pair 1 p50 +0.02 / p99 +0.38 ms; pair 2 p50 −0.20 / p99 −0.47 ms; mean c32 Δp50
**−0.09 ms**, Δp99 **−0.05 ms**; r200 Δp50 **+0.43 ms**, Δp99 +0.54 ms. **Not material** by the pre-set rule (mean c32 Δp50
> 1.0 ms, or Δp99 > 2.0 ms, or r200 Δp50 > 1.0 ms), so the client expansion was not changed. The "on" runs' recall is the frozen
value (strict 0.97299, tie 0.98608). The "off" runs return distinct-row ordinals, so their recall is meaningless by construction.
The c32 max moves by 18-21 ms with expansion on (single tail events; p99 does not move).

## Step 3: top_k 100 dev probe (frozen 12k, dev q0-4999, c160)

strict recall@100 **0.96092**, tie 2e-3 0.98518, 617.9 QPS, p50/p95/p99 252.7/368.4/414.3 ms, 0 errors. Strict < 0.97, so step 5
(dev-only top_k 100 tuning) ran after the S6 grid.

## Step 4: S6 grid (test q5000-99999, top_k 10, frozen 12k), 3 repeats, median per metric

| Load | achieved QPS | p50 ms | p95 ms | p99 ms | max ms | errors (rep1/2/3) | schedule lag s | above capacity |
|---|---:|---:|---:|---:|---:|---|---:|---|
| closed loop c1 | 42.9 | 22.6 | 27.8 | 49.9 | 74.6 | 0/0/0 | | |
| closed loop c16 | 441.7 | 36.0 | 44.3 | 48.5 | 124.5 | 0/0/0 | | |
| closed loop c32 (s156, earlier) | 595.2 | 53.2 | 64.1 | 69.8 | 172.0 | 0/0/0 | | |
| closed loop c64 | 614.1 | 102.2 | 147.4 | 168.6 | 267.2 | 0/0/0 | | |
| closed loop c256 | 623.5 | 407.8 | 496.5 | 539.7 | 698.6 | 0/**2**/0 | | |
| open loop 100 rps | 100.0 | 22.2 | 29.0 | 40.3 | 93.7 | 0/0/0 | 0.0 | no |
| open loop 200 rps (s156, earlier) | 200.0 | 25.3 | 31.4 | 35.3 | 99.8 | 0/0/0 | | no |
| open loop 400 rps (s156, earlier) | 399.9 | 34.1 | 42.4 | 46.6 | 123.5 | 0/0/0 | | no |
| open loop 600 rps | 599.9 | 52.3 | 100.3 | 151.4 | 286.8 | 0/0/0 | 0.0 | no |
| open loop 800 rps | **624.2** | 405.8 | 493.4 | 546.1 | 710.6 | 0/0/0 | 33.5 | **yes (3/3)** |

- Recall is identical in all 22 grid runs (21 valid + the void one below): strict 0.97299, tie 2e-3 0.98608, `missing_from_gt` 0 (c256 rep2 scores 94,998
  of its 95,000 queries: 0.9729942).
- **Errors:** c256 rep2 has 2 failed requests of 95,000. nova-storm logs only the first error text: `search HTTP 429 Too Many
  Requests: {"error":{"code":"queue_full","message":"request queue is full"}}`. Both failed rows are at t = 9 ms, the initial
  burst of 256 simultaneous requests (`runs/s6-c256-rep2.ts.jsonl.gz`). Every other grid run has 0 errors and 0 timeouts.
- **Capacity:** closed loop saturates at ~614-629 QPS (c64 and c256 differ only in queueing). At 800 rps offered, all three
  repeats achieved 614.7-629.3 QPS (< 0.99 × 800), so the row is **above capacity**, and its achieved QPS is the measured
  ceiling. nova-storm's in-flight cap (256) was full, and it measures latency from dispatch, so the ~33 s of schedule lag
  (wall − 95,000/800) is time waiting for a slot that the latency columns do not contain. No 429s at 800 rps: the server's
  admission queue (max_inflight 96 + read_queue 192) is larger than the client's 256 in flight.
- 600 rps sits just under the ceiling, so it is the noisiest row: rep1 p50/p99 74.2/285.5 ms, rep2 52.3/135.3, rep3 49.2/151.4
  (cause of the rep1 difference not isolated). Low load (c1, 100 rps) has a higher p99 (40-51 ms) than 200 rps had (35 ms);
  not investigated.
- **Void run, rerun:** `s6-r800-rep3` fails the 60 s identity window. At 10:35:32Z, 38 s after it ended, the on-disk
  server.json had already been rewritten for the first top_k 100 dev point (the serving pid was still the frozen 291953, which
  had loaded 4beb0731 at start; the next pid appears at 10:35:37). By the rule it is void; it was rerun as `s6-r800-rep3r` at
  10:58-11:01Z on a fresh frozen process (pid 322583, exe cd60f49a, cfg 4beb0731, after one dev warm pass) and the median uses
  rep1, rep2, rep3r. The void run is published; its values (625.8 QPS, p99 535.5 ms) would not change any median.
- Order and cache: per repeat c1, c16, c64, c256, 100, 600, 800 rps, repeats back to back (07:13-10:35Z). The earlier c32/200/400
  rows (s156) stay as they were.

## Step 5: top_k 100 tuning on dev only (`FREEZE-top100.json`, `tune-ident.txt`)

Grid leaves {12k, 18k, 24k, 36k} × rescore {400, 1000}, exact_centroids 1.25 × leaves, routing_candidates 32 × leaves, scan 4 +
route 4, all else as frozen; one dev pass (5,000 queries) at c160, top_k 100 per point, each on a fresh process started by
`gserve3.sh`, identity recorded before and after (`tune-ident.txt`).

| leaves | rescore | dev strict r@100 | tie 2e-3 | QPS | p50 | p95 | p99 | errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12,000 | 400 (= frozen) | 0.96092 | 0.98518 | 617.9 | 252.7 | 368.4 | 414.3 | 0 |
| 12,000 | 1000 | 0.96286 | 0.98567 | 558.2 | 278.9 | 404.5 | 456.9 | 0 |
| 18,000 | 400 | 0.96901 | 0.98934 | 415.9 | 377.8 | 487.4 | 546.5 | 0 |
| **18,000** | **1000** | **0.97119** | 0.98984 | 399.1 | 392.1 | 519.4 | **597.2** | 0 |
| 24,000 | 400 | 0.97438 | 0.99192 | 316.9 | 498.6 | 628.6 | 699.4 | 0 |
| 24,000 | 1000 | 0.97668 | 0.99231 | 307.4 | 515.1 | 647.7 | 713.7 | 0 |
| 36,000 | 400 | 0.97978 | 0.99468 | 212.9 | 746.4 | 934.2 | 1006.2 | 0 |
| 36,000 | 1000 | 0.98205 | 0.99498 | 207.8 | 755.5 | 961.6 | 1058.1 | 0 |

Rule (lowest dev p99 among strict ≥ 0.97 and 0 errors) → **18,000 leaves, rescore 1000**, server.json `02e65587…`
(`server-top100.redacted.json`; differs from the frozen config only in leaves, exact_centroids, routing_candidates, rescore),
exe unchanged `cd60f49a…`. `FREEZE-top100.json` was pushed in **5ba0ec3fe at 10:56:34Z**; the first test query on that config
went out at **11:24:06Z** (`runs/runs.log`; the 11:23:48Z pass before it is dev).

## Step 6: Qdrant's fleet shape (test, top_k 100)

Qdrant's `fineweb-10b` branch (6a0bc58) `configs/storm/fineweb_bf_k1000_fleet.yaml`: top_k 100, concurrency 16 × 10 replicated
workers, `duration_s: 1200`, batch 1, all 100k queries cycled. Ours: one nova-storm process at concurrency 160 (the same 160 in
flight; ten replicated workers would each start at query 0 and send the same query ten times at once), `duration_s: 1200`,
`passes: 0`, test file only (95,000 queries, cycled). Recall is the mean over all firings (the file is cycled ~8 times; the last
pass is partial). Each run was preceded by one dev pass on its own config.

| Config | requests | QPS | strict r@100 | tie 2e-3 r@100 | p50 ms | p95 ms | p99 ms | max ms | errors | timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen 12k (headline config) | 753,902 | 628.1 | 0.95836 | 0.98378 | 251.9 | 321.8 | 359.9 | 570.9 | 0 | 0 |
| top_k 100 tuned on dev (18k leaves, rescore 1000) | 476,554 | 397.0 | 0.96892 | 0.98859 | 399.3 | 553.5 | 617.9 | 893.2 | 0 | 0 |

Both runs: `missing_from_gt` 0, identity equal across the window (frozen: pid 322583 cfg 4beb0731; tuned: pid 328785 cfg 02e65587).
At 160 in flight the server is saturated, so latency is mostly queueing (Little's law: 160 / 628 QPS = 255 ms ≈ p50; 160 / 397 =
403 ms for the tuned config). The tuned config buys +0.0106 strict recall@100 for 37% less throughput. Neither top_k 100 row
reaches 0.97 strict on test. The exact-search ceiling at @100 with our regenerated query vectors was not measured (S1 is @10).

## Check: raw rows re-aggregate to every JSON (`tools/reagg_final.py` → `reagg-final.json`)

All 41 finished runs of this lane: row count = `requests`, ok rows = requests − errors, mean per-query strict recall from the
rows = the JSON's, nearest-rank p50/p99 over ok rows = the JSON's (≤ 0.02 ms), QPS from the rows' last timestamp = the JSON's.
`_all_match: true`.

## S7: stage binaries (`s7/`)

`s7/README.md` + `s7/binaries.tsv` (collected 06:40:37Z, nothing re-run): sha256 of every stage binary still on A. Build
`~/bin5/conaredb-bench` `04bc106a…` (conare main 411e597b3), spill + verify `~/bin8/conaredb-bench` `e1a563ec…` (main
004bf571b), import `~/bin9/conaredb-import` `e018eb15…` (004bf571b), dedupe `~/dedupe` `333bc916…` (C, source sha `913efd1c…`,
not in git), serve `cd60f49a…` (c136d6788). Not recoverable: the per-box dedupe binaries on l96-b/l96-c (both boxes deleted),
and the HF download → f16 prep, which was never timed end to end and was not re-run.

## Step 7: restore

11:46:14Z: A back on exe `917245aa…` at 9k leaves via `s156serve.sh conaredb-server.orig-917245aa 9000 8 8`
(= `gserve2.sh 9000 96 8 8`): pid 334520, server.json `d63a9dca…`, the same config sha as before the lane (`switch.log`
PRE-SWITCH). One real query: `/v2/search` q2000 top_k 10 → HTTP 200, 10 hits, 16.6 ms. identlog stopped; /tmp/BENCH_RUNNING
removed on A and s5; no client job left on s5.

## Cost

No new spend: A and s5 were already running. This lane used about 5.0 h of the pair (06:45-11:46Z); at the billed rates in
`../verify/cost/` ($8.256/h + $3.616/h) that is about $59 of box time attributed, not an Azure Cost Management query for today.
