# Final lane rules (S3 expansion A/B, S6 grid, Qdrant fleet shape, top_k 100), written 2026-09-24 ~06:55Z, before any run of this lane

Everything here extends `../s156/RULES.md` and `../s156/FREEZE.json`; nothing in them changes.

## Common to every run
- **Server:** the frozen process only: exe sha256 `cd60f49a…` (conare main c136d6788) + server.json sha256 `4beb0731…`
  (12,000 leaves, scan 4 + route 4, rescore 400), started by `s156serve.sh conaredb-server-main-c136d6788 12000 4 4`.
  The one exception is the top_k 100 tuned row (below), which runs on a config frozen and pushed before its first test query.
- **Identity:** `identlog.sh` on A appends `ident.sh` (pid, start time, exe sha256, server.json sha256, budgets, env) every 30 s for
  the whole lane. A run is valid only if every identity line from 60 s before its start to 60 s after its end is the same process with
  the expected exe and config sha. A run that fails this is void and rerun.
- **Client:** s5, nova-storm sha256 `a5450143…` (conareai/supernova 7d7d0d3, unchanged), target `conaredb`, `expand` with
  `copy_order: uuid` (S0 CSR on s5 NVMe), `timeout_s: 30`. Runs go through `queue.sh` → `storm2.sh` on s5 (nohup), one at a time.
- **Queries:** test = q5000-99999 (95,000; `storm-test.parquet` sha256 142b16f5…), dev = q0-4999 (`storm-dev.parquet` 6dfbde81…).
  No tuning ever reads a test result.
- **Lock:** /tmp/BENCH_RUNNING held on A and s5 for the whole lane.
- **Reported per run:** requests, errors, timeouts, first error text (nova-storm logs only the first error), missing_from_gt, strict
  and tie-aware recall, QPS, p50/p95/p99/max. Open loop: achieved QPS against the offered rate and the schedule lag
  (wall − requests/rate). nova-storm measures latency from dispatch, so when its in-flight cap (256) is full the wait for a slot is
  not in the latency; the lag shows it. An open-loop run whose achieved QPS is below 0.99 × the offered rate is reported as
  **above capacity**, with its achieved QPS as the measured ceiling. Nothing is censored.
- **Median rule** (as `../s156/RULES.md`): per condition and metric, the median of the 3 repeats; every raw run is published.

## Order
1. Switch A to the frozen server; one dev c32 pass (page-cache warm; dev only).
2. **S3 expansion A/B (test, top_k 10):** c32 on, c32 off, r200 on, r200 off, c32 off, c32 on (the second c32 pair reversed
   to cancel order and cache drift). "off" removes only the `expand` block; the server request and load are identical, hit ids stay
   distinct-row ordinals, and that run's recall is meaningless (latency only). **Material** = the mean c32 Δp50 (on − off) over the
   two pairs > 1.0 ms, or mean c32 Δp99 > 2.0 ms, or r200 Δp50 > 1.0 ms. If material, the client expansion is optimised (client
   code only; the engine config does not change), pushed, rebuilt, and the same six-run A/B repeated with the new client.
3. **top_k 100 dev probe:** frozen 12k, dev, closed loop c160, top_k 100, one pass (5,000 queries).
4. **S6 grid (test, top_k 10, frozen):** per repeat, in order c1, c16, c64, c256 (closed loop, one pass each) and 100, 600, 800 rps
   (open loop, in-flight cap 256, one pass each); repeats 1-3 back to back (21 runs). The existing c32, 200 and 400 rps rows stay.
5. **top_k 100 tuning, dev only, if the step-3 dev strict recall@100 < 0.97.** Grid: leaves {12k, 18k, 24k, 36k} × rescore
   {400, 1000} (exact_centroids 1.25 × leaves, routing_candidates 32 × leaves, scan 4 + route 4, every other field as frozen; written by
   `gserve3.sh`), each point dev c160 top_k 100 one pass. **Rule:** lowest dev c160 p99 among points with dev strict recall@100
   ≥ 0.97 and 0 errors; if none reaches 0.97, the highest dev strict recall@100. The chosen config's server.json sha256 and the
   exe sha256 (unchanged cd60f49a) are pushed as `FREEZE-top100.json` **before** its first test query.
6. **Qdrant's fleet shape (test, top_k 100):** their `fineweb-10b` branch config (`configs/storm/fineweb_bf_k1000_fleet.yaml`,
   6a0bc58): top_k 100, concurrency 16 × 10 workers, `duration_s: 1200`, batch 1, whole query file cycled. Ours: one nova-storm process
   at concurrency 160 (the same 160 in flight; ten replicated workers would each start at query 0 and send the same query ~10 times
   at once, which flatters any cache), `duration_s: 1200`, `passes: 0`, test file only. One run on the frozen 12k config, then (if
   step 5 ran) one on the top-100 config. Reported: recall@100 strict and tie-aware (2e-3 auto), p50/p95/p99, QPS, errors.
   Recall is the mean over firings (the file is cycled about N times; the last pass is partial), as in any timed nova-storm run.
7. Restore A to 917245aa at 9k (`gserve2.sh 9000 96 8 8` via `s156serve.sh`), confirm one real query answers.
