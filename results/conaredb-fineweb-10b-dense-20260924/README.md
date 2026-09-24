# ConareDB on Qdrant/FineWeb-10B, dense track (gte-multilingual-base), 2026-09-24

**Strict recall@10 = 0.9730** (lower bound, exact FineWeb-uuid matches) on 95,000 held-out test queries, with all
10,074,324,060 documents served from **one Azure Standard_L96as_v4**. The tie-aware upper bound is 0.9861 at nova-storm's
auto tolerance for float16 storage (2e-3), and 0.9756 at the 5e-6 float32 tolerance. Numbers are the median of 3 repeats
of a config frozen before any test query was sent. `missing_from_gt` is 0 in every run; the only errors in the 3,515,000 top_k 10
test requests of all 37 runs are 2 × HTTP 429 in one c256 run (below).

| Load (top_k 10, batch 1, one client) | QPS | p50 ms | p95 ms | p99 ms | max ms | errors |
|---|---:|---:|---:|---:|---:|---:|
| closed loop, concurrency 1 | 42.9 | 22.6 | 27.8 | 49.9 | 74.6 | 0 / 95,000 × 3 |
| closed loop, concurrency 16 | 441.7 | 36.0 | 44.3 | 48.5 | 124.5 | 0 / 95,000 × 3 |
| **closed loop, concurrency 32** | **595.2** | **53.2** | **64.1** | **69.8** | 172.0 | 0 / 95,000 × 3 |
| closed loop, concurrency 64 | 614.1 | 102.2 | 147.4 | 168.6 | 267.2 | 0 / 95,000 × 3 |
| closed loop, concurrency 256 | 623.5 | 407.8 | 496.5 | 539.7 | 698.6 | 2 / 95,000 × 3 (HTTP 429 `queue_full`, one run, first 9 ms) |
| open loop, 100 rps | 100.0 | 22.2 | 29.0 | 40.3 | 93.7 | 0 / 95,000 × 3 |
| open loop, 200 rps | 200.0 | 25.3 | 31.4 | 35.3 | 99.8 | 0 / 95,000 × 3 |
| open loop, 400 rps | 399.9 | 34.1 | 42.4 | 46.6 | 123.5 | 0 / 95,000 × 3 |
| open loop, 600 rps | 599.9 | 52.3 | 100.3 | 151.4 | 286.8 | 0 / 95,000 × 3 |
| open loop, 800 rps offered: **above capacity** | 624.2 achieved | 405.8 | 493.4 | 546.1 | 710.6 | 0 / 95,000 × 3 |

**Capacity:** one L96 saturates at **~620 QPS** at top_k 10 (c64 614, c256 624; 800 rps offered → 624 achieved in all 3
repeats). At 800 rps nova-storm's 256 in-flight slots stay full; it measures latency from dispatch, so the time waiting for a
slot (33.5 s of schedule lag over a run) is not in the latency columns. The server queued rather than rejected: the only 429s are
the 2 in the first 9 ms of one c256 run (the initial 256-request burst). c32/200/400 rows: `runs/` (first lane);
all other rows: `final/` (final lane, `final/README.md`).

### Qdrant's own load shape: top_k 100, 160 in flight, 1,200 s

Qdrant's `fineweb-10b` branch (6a0bc58) tests FineWeb-10B with `configs/storm/fineweb_bf_k1000_fleet.yaml`: top_k 100,
10 replicated workers × concurrency 16, 1,200 s, the query file cycled. We ran the same 160 in flight as one nova-storm process
(ten replicated workers would each start at query 0 and send the same query ten times at once), 1,200 s, on the 95,000 test
queries. One run per config; recall is the mean over all firings (the file cycles ~5-8 times).

| Config | requests | QPS | strict recall@100 | tie 2e-3 | p50 ms | p95 ms | p99 ms | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen headline config (12k leaves, rescore 400) | 753,902 | 628.1 | **0.9584** | 0.9838 | 251.9 | 321.8 | 359.9 | 0 |
| tuned for top_k 100 on dev only (18k leaves, rescore 1000) | 476,554 | 397.0 | **0.9689** | 0.9886 | 399.3 | 553.5 | 617.9 | 0 |

The top_k 100 config was picked on dev q0–4999 by a rule written before the lane (lowest dev p99 among 8 grid points with dev
strict recall@100 ≥ 0.97; it had 0.9712 on dev) and pushed (`final/FREEZE-top100.json`) before its first test query. Neither
config reaches 0.97 strict recall@100 on test. The box is saturated at 160 in flight, so these latencies are mostly queueing
(160 / QPS ≈ p50). The exact-search ceiling at @100 with our regenerated query vectors was not measured (the @10 one is 0.99878).

Recall@10 is identical in all 34 top_k 10 test runs with expansion on (same frozen config; the c256 run with 2 errors scored 94,998 queries and gets 0.9729942).

| Recall@10 on test (95,000 queries) | value |
|---|---:|
| **strict (lower bound)** | **0.97299** |
| tie-aware, 2e-3 relative (nova-storm auto, datatype float16) | 0.98608 |
| tie-aware, 5e-6 relative (nova-storm's float32 default) | 0.97558 |
| exact search with the same query vectors, strict (ceiling, see below) | 0.99878 |

The 2e-3 bound flatters: it also counts some real near-misses as ties (1.3e-3 to 1.9e-3 under the cutoff in a dev smoke).
The 5e-6 bound is below the score noise between our regenerated query vectors and Qdrant's (up to 2.3e-4), so neither
upper bound is a clean number. Quote the strict one.

## Hardware and cost

| Role | Azure SKU | CPU | RAM | Storage | Region | $/h billed |
|---|---|---|---|---|---|---:|
| Server | Standard_L96as_v4 | 96 vCPU, AMD EPYC 9V74 | 755 GiB | 12 × 1.7 TB local NVMe, RAID0 (21 TB) | eastus2, zone 1 | 8.256 |
| Load client | Standard_L32aos_v4 | 32 vCPU, AMD EPYC 9V74 | 251 GiB | 12 × 1.7 TB local NVMe, RAID0 | eastus2 | 3.616 |

- Client and server share one VNet and talk over private IPs. Accelerated networking is on for the server NIC and off for the client NIC.
- The $/h figures come from Azure Cost Management for 2026-09-22/23, compute only. They leave out Defender (+$0.027/h per VM), the public IP and the OS disk.
- **Server $ per 1M queries:** $3.85 at 595 QPS, $5.73 at 400 QPS, $11.47 at 200 QPS. Including the client box: $5.54, $8.25, $16.49.
- OS: Ubuntu 24.04.4, kernel 6.17.0-1022-azure.

## What was measured, pinned

| Item | Value |
|---|---|
| Corpus | HF `Qdrant/FineWeb-10B` `data/` (unchanged since 420f4686), `dense_embedding` 768-d float16, 10,074,324,060 rows |
| Ground truth | `queries/gt_dense_k1000.parquet` @ **3aea0b8d** (the fp32 run of 2026-09-19), sha256 `847989d235ecd8c7118b64709cfe5d54ff36b82ad8c70c9de31c5347633bbd81`. Scored against its top 10 (nova-storm truncates to `top_k`) |
| Query vectors | Qdrant's `queries/scripts/regenerate_queries.py` @ 420f4686 (sha256 `7bdd686a…`), all 100,000 queries in **one** run (no `--limit`). `verify_regeneration.py`: dense ok, 100,000 rows. gte-multilingual-base revision `9bbca17d`. Qdrant does not ship query vectors |
| Split | dev = q0–4999 (all tuning), test = q5000–99999 (95,000 queries). `splits.json` was pushed before any query was sent |
| Load generator and scorer | nova-storm from this fork: upstream master b8e5b07 + the `conaredb` target (`crates/nova-storm/src/targets/conaredb.rs`), commit 7d7d0d3 (merged as ce6690e), binary sha256 `a5450143…` (`runs/client-sha256.txt`) |
| Engine | ConareDB v2 `conaredb-server`, conareai/conare main `c136d6788`, sha256 `cd60f49a…`. Index built by `conaredb-bench build` main `411e597b3` (sha256 `04bc106a…`), spilled by main `004bf571b` (`e1a563ec…`), imported by `conaredb-import` `e018eb15…` (`final/s7/`) |
| Frozen server config | 12,000 leaves, exact_centroids 15,000, routing_candidates 384,000, rescore 400, max_inflight 96, 96 workers, scan 4 + route 4 threads. server.json sha256 `4beb0731…` (`freeze/server.redacted.json`; only the bearer token is redacted). The server has no result cache |
| Index | 2,557,787,738 distinct vectors plus 767,336,321 spill rows (30% selective spill), manifest sha256 `c45e971d…` |

## Protocol

1. **Rules before measurement.** The sweep grid, freeze rule, load conditions and median rule (`freeze/RULES.md`) were pushed at 2026-09-24 00:43:15Z.
2. **Tuned on dev only.** A 30-run dev sweep (`dev-sweep.tsv`) covered 9k/12k/18k leaves × 3 thread splits × {c32, r200, r400}. The fixed rule was: lowest dev c32 p99 among points with strict ≥ 0.97 and 0 errors. It picked 12k leaves with 4 + 4 threads.
3. **Frozen before test.** The binary sha and config sha (`freeze/FREEZE.json`) were pushed at 01:06:09Z (GitHub push event). The first test query went out at 01:07:14Z (`runs/runs.log`). Server pid, exe sha and config sha were recorded before and after every run (`runs/*.server`), and all 9 runs hit the same process.
4. **Cache.** Warmed with one dev pass (`runs/s6-warm-dev-c32.*`), never with test queries. Each run sends every test query exactly once (`passes: 1`). Repeats 2 and 3 reuse the test queries, so their page cache is warmer (c32 rep1 568.7 QPS vs about 595 for rep2 and rep3).
5. **Median of 3 per load shape**, each metric taken separately (`runs/s6-medians.json`). All 9 raw runs are included.
6. **Interruption.** The last run (r400 rep3) went out 3.6 h after the other eight, on the same server process. Its latency matches rep1 and rep2.
7. **Final lane** (06:45–11:46Z, rules `final/RULES.md` pushed first): the S6 grid c1/c16/c64/c256 and 100/600/800 rps × 3, the
   expansion on/off A/B, the top_k 100 dev tuning and the fleet-shape runs. Server identity was logged every 30 s; a run counts
   only if every line from 60 s before to 60 s after it shows the expected process and shas. One run (800 rps rep3) failed that
   window because the config file was rewritten for the next dev step 38 s after it ended; it is published, marked void, and was
   rerun (`rep3r`), which the median uses. Every run's JSON re-aggregates exactly from its raw rows (`final/reagg-final.json`).

## Disclosures that affect comparability

- **Duplicates are collapsed; ids are expanded in the client.** 74.6% of FineWeb-10B rows are byte-identical copies of another row's vector. The largest group has 55,725 copies; a 100k sampled merge audit found every merge byte-identical. The engine stores each distinct vector once (2,557,787,738) and returns distinct-row ids. Its 32-bit point ids cannot hold 10.07B uncollapsed points.
- **How the expansion works.** The `conaredb` target maps each returned row to its FineWeb uuids through a CSR table (offsets + 16-byte uuids, 181 GB on the client's NVMe). It fills the 10 slots in rank order. The lookups run **inside** the measured latency, on the client box. **Its cost, measured** (same server and queries, back to back, only `expand` toggled; `final/s3-expansion-ab.json`): c32 Δp50 −0.09 ms, Δp99 −0.05 ms (mean of two on/off pairs, order reversed in the second); 200 rps Δp50 +0.43 ms, Δp99 +0.54 ms.
- **Copy order inside a duplicate group: ascending uuid.** This is nova-bf's own tie-break for ids (`params.tiebreak: id` in `nova_bf/tiebreak.py`). The published ground truth follows it in 71,045 of 71,045 test groups split at the cutoff; corpus order reproduces 6,678. With corpus order the best possible strict recall@10 is 0.834, because any engine that returns one id per hit is capped there.
- **Query vectors are regenerated.** Ours differ from the ones used for the ground truth by up to 2.3e-4 in score (mean 1.3e-5). Exact search with our vectors therefore reaches strict 0.99878, not 1.0 (612 of 95,000 test queries have a reordered near-tie at the cutoff), and the engine loses 0.0258 against that ceiling (`ceiling.json`).
- **Index build is outside Supernova.** The index was built offline by ConareDB's own pipeline (global dedupe → build → spill → import on the same L96). It was not loaded with `nova-load`, so there is no `nova-load` load time. The stage wall times are 2:47:24 for the clean stages (≈ $23 at $8.256/h); the sha256 of every stage binary still on the box is in `final/s7/binaries.tsv`. Per-box dedupe on 3 × L96 and the transfer are in logs only partly, and their binaries' hashes are lost with the deleted boxes. **The HF download → f16 prep was never timed and was not re-run**, so its cost is in no figure here.
- **Load shapes.** One client box. top_k 10: closed loop c1–c256 and open loop 100–800 rps, 3 repeats each. top_k 100: Qdrant's fleet shape as one process at c160 (above). top_k 1000 was not run (the server's `max_top_k` is 100).
- **Recall@10 only.** RBO is described in the current docs, but nova-storm at b8e5b07 does not compute it, so it is not reported.

## Licences

- Corpus and ground truth: ODC-BY-1.0 (Qdrant/FineWeb-10B; FineWeb; Common Crawl terms).
- Query text is from MS MARCO, which allows non-commercial research use only. No query text or query vector is in this directory; the run configs point at local parquet files.

## Files

| Path | Content |
|---|---|
| `runs/s6-test-{c32,r200,r400}-rep{1,2,3}.json` | nova-storm `--json` summaries (schema_version 2), first lane |
| `final/runs/` | final lane, every run: `s6-{c1,c16,c64,c256,r100,r600,r800}-rep*` (grid), `s3-*` (expansion A/B), `fleet-k100-*` (Qdrant shape), `dev-k100-*` (top_k 100 dev tuning), `f-warm*` (dev warm passes); `.json`, `.yaml`, `.log`, `.ts.jsonl.gz`, `runs.log` |
| `final/README.md`, `final/RULES.md` | final lane write-up and the rules pushed before it ran |
| `final/final-runs.tsv`, `final/final-medians.json`, `final/reagg-final.json` | per-run table with identity check, medians, raw-row re-aggregation |
| `final/identlog.txt`, `final/tune-ident.txt`, `final/switch.log`, `final/serve.log` | server identity every 30 s, per tuning point, and at every switch |
| `final/FREEZE-top100.json`, `final/server-top100.redacted.json` | the top_k 100 config, frozen on dev before its test run |
| `final/s7/` | sha256 of every stage binary still on the server box |
| `runs/*.yaml` | the exact storm configs (the bearer token is an env var) |
| `runs/*.ts.jsonl.gz` | nova-storm per-request report rows: latency, ok, per-query recall |
| `runs/*.server`, `runs/*.log` | server identity before and after each run; nova-storm logs |
| `runs/runs.log`, `dev-sweep.tsv` | every run in time order, dev sweep and test runs |
| `runs/s6-medians.json`, `runs.tsv`, `result.json` | medians, per-run table, machine-readable headline |
| `freeze/` | FREEZE.json, RULES.md, server.json (token redacted), sha256 of binary, config and index manifest |
| `ceiling.json`, `splits.json` | exact-search ceiling and oracle bounds; the dev/test split |
| `verify/` | independent re-checks (below) |
| `SHA256SUMS` | sha256 of every file here |

## Independent verification (`verify/`)

- The raw per-request rows re-aggregate exactly to each run's JSON: mean strict 0.9729947, p50/p99 and QPS. The per-query recall histogram is identical in all 9 runs.
- Served rows for all 95,000 test queries were saved by a separate client on the frozen config. A re-score written from scratch expands them through the CSR and reads the GT parquet directly. It gives strict 0.9729947 and tie 2e-3 0.9860779, the same as nova-storm to the tenth. Its per-query histogram equals nova-storm's.
- Row → uuid table: 1,000 random corpus positions were read on 5 source boxes (their `ids.npy` and `docs-f16.npy`). For every one, the uuid sits in its distinct row's CSR list exactly once, the list length equals the copy count, and the vector bytes equal the vector the index was built from: 1,000 of 1,000.
- All 83,098,320 unique ground-truth uuids resolve to exactly one corpus position.
