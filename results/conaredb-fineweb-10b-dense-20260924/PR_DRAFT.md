# Draft upstream PR for qdrant-labs/supernova (NOT opened; the owner opens it)

How to open it (the #64 shape: target code + docs + example config + tests, no results):

```bash
git fetch https://github.com/conareai/supernova.git conaredb-v2-target-20260924:tgt submission-conaredb-20260924:sub
git switch -c conaredb-target tgt                    # 7d7d0d3: the target on top of upstream b8e5b07
git checkout sub -- docs/storm/overview.md           # the storm docs for the target; results/ stays out
git commit -m "docs(storm): conaredb target"
# push to the fork and open against qdrant-labs/supernova master with the title and body below
```

Why this shape (checked 2026-09-24): upstream master b8e5b07 has no `results/` or leaderboard directory, no CONTRIBUTING file,
no PR template, and no merged PR from any vendor carrying results. The one merged external backend, **#64** (OpenSearch,
2026-09-17), is target code, docs and an example config. So the result is self-published (the package on the fork, linked
from the PR body) and the PR carries only code.

---

## Title

Add ConareDB target for nova-storm (and a v1 bulk loader for nova-load)

## Body

`nova-storm` can load-test a ConareDB v2 server (`POST /v2/search {vector, top_k}`), and `nova-load` can bulk-load a
ConareDB namespace through the v1 bulk-vector frame. Both sit behind a new `conaredb` cargo feature, off by default, like
the other optional backends.

**Supported:** dense vectors, closed-loop and paced load, batch fan-out (N concurrent requests; the engine has no
multi-query endpoint), and tie-aware recall. The target reports per-hit cosine scores and declares datatype `float16`, so
the auto tie tolerance is 2e-3.

**Rejected with explicit errors:** `search_params` (the server owns every ANN budget), filters, sparse and multivector.

### The `expand` block (collapsed corpora)

FineWeb-10B is 74.6% byte-identical duplicate vectors (2,557,787,738 distinct of 10,074,324,060). A server that stores each
distinct vector once returns distinct-row ids; `expand` maps them back to corpus ids from a CSR table (offsets + 16-byte ids)
on the client. It fills `top_k` slots in rank order, and each copy carries its row's score. The positioned reads run
**inside** the measured latency window, and the log line and module docs call it a disclosed proxy, not the engine
returning ids. Measured cost on FineWeb-10B (same server, same 95,000 queries, back to back, expand on vs off): c32
Δp50 −0.09 ms / Δp99 −0.05 ms (mean of two pairs), 200 rps Δp50 +0.43 ms.

`copy_order: uuid` emits a group's copies in ascending id, which is nova-bf's `tiebreak: id` order; the published
`gt_dense_k1000` follows it in 71,045 of 71,045 top-10 groups split at the cutoff. `posting` (the default) keeps table order.

### Tests

Unit tests cover the CSR expansion (both copy orders), malformed hit ids and scores, the expand config's consistency check,
and the loader's CRBF0002 bulk-frame layout. `cargo clippy` is clean for the target.

### Result produced with this target (self-published, not part of this PR)

https://github.com/conareai/supernova/tree/submission-conaredb-20260924/results/conaredb-fineweb-10b-dense-20260924
(raw nova-storm JSON for every run, per-request rows, configs, server identity per run, SHA256SUMS).

One Azure Standard_L96as_v4 serves all 10,074,324,060 documents. 95,000 held-out queries (q5000–99999) against
`gt_dense_k1000` @3aea0b8d, configs frozen on dev before any test query (top_k 10 rows: median of 3 repeats):

- top_k 10: strict recall@10 **0.9730** (tie-aware upper bound 0.9861 at 2e-3). Closed loop c1 42.9 QPS p50/p99
  22.6/49.9 ms; c32 595 QPS 53.2/69.8 ms; saturation ~620 QPS (c64 614, c256 624). Open loop 100/200/400/600 rps
  p99 40.3/35.3/46.6/151.4 ms; 800 rps offered is above capacity (624 QPS achieved).
- Your fleet shape (top_k 100, 160 in flight, 1,200 s): frozen top-10 config strict recall@100 0.9584 at 628 QPS
  (p50/p99 252/360 ms); a top_k 100 config tuned on dev only: strict recall@100 0.9689 at 397 QPS (p50/p99 399/618 ms). One
  1,200 s run per config.

Query vectors were regenerated with `regenerate_queries.py` @420f4686 (Qdrant does not ship them); with them, exact search
reaches strict recall@10 0.99878, not 1.0.

### Notes for reviewers

- The `nova-load` store targets ConareDB's **v1** API. The 10B index above was built offline by the engine's own pipeline
  (dedupe → build → spill → import), not through `nova-load`.
- `.github/workflows/rust-binaries.yml` is unchanged. Adding `conaredb` to the release `--features` lists would ship it in
  release binaries, as #64 did for opensearch; left to the maintainers.
- The ConareDB server is not open source, so reproducing the result needs a server binary from us.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
