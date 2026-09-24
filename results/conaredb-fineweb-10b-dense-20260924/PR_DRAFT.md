# Draft PR description for qdrant-labs/supernova (NOT opened; the owner submits)

## How results reach Qdrant today (checked 2026-09-24)

There is no submission channel. qdrant-labs/supernova (master b8e5b07) has no `results/` or leaderboard directory,
no CONTRIBUTING file, no issue or PR templates, and no merged PR from any vendor carrying results. The HF dataset card
(@3aea0b8d) and both release posts (qdrant.tech/blog/qdrant-fineweb-10b-release, huggingface.co/blog/Qdrant/fineweb-10b-release)
describe the dataset and Supernova but give no submission process, and publish no Qdrant numbers. The HF dataset has
one discussion (the parquet-converter bot). The only external-vendor contribution that was merged is **#64 "Add OpenSearch
backend for nova-load and nova-storm"** (Franborat, merged by OckermanSethGVSU on 2026-09-17): target code, docs and
an example config, with **no results**.

Two options:

- **A (matches the #64 precedent):** open an upstream PR with the target code and docs only (below). Publish
  the result separately (a blog post or a HF dataset discussion) that links this directory. Use this if the
  goal is to have the target merged.
- **B:** open the PR from this branch, including `results/conaredb-fineweb-10b-dense-20260924/`. Upstream has no
  place for results, so expect to be asked to drop that directory.

To open option A, branch from `conaredb-v2-target-20260924` plus the docs commit from this branch, and leave out `results/`.

---

## Title

Add ConareDB target for nova-storm (and a v1 bulk loader for nova-load)

## Body

`nova-storm` can load-test a ConareDB v2 server (`POST /v2/search {vector, top_k}`), and `nova-load` can bulk-load a
ConareDB namespace through the v1 bulk-vector frame. Both sit behind a new `conaredb` cargo feature, off by default.

**Supported:** dense vectors, closed-loop and paced load, batch fan-out (N concurrent requests, since the engine has no
multi-query endpoint), and tie-aware recall. The target reports per-hit cosine scores and declares datatype
`float16`, so the auto tolerance is 2e-3.

**Rejected with explicit errors:** `search_params` (the server owns every ANN budget), filters, sparse and multivector.

### The `expand` block (collapsed corpora)

FineWeb-10B is 74.6% byte-identical duplicate vectors (2,557,787,738 distinct out of 10,074,324,060). A server that
stores each distinct vector once returns distinct-row ids. `expand` maps them back to corpus ids from a CSR table
(offsets + 16-byte ids) on the client. It fills `top_k` slots in rank order, and each copy carries its row's score.
The positioned reads run **inside** the measured latency window. Log lines and the module docs call it a disclosed
proxy, not the engine returning ids.

`copy_order: uuid` emits the copies of a group in ascending id. That is nova-bf's `tiebreak: id` order, and the published
`gt_dense_k1000` follows it in 71,045 of 71,045 top-10 groups split at the cutoff. `posting` (the default) keeps
table order.

### Tests

The unit tests cover the CSR expansion (both copy orders), malformed hit ids and scores, the expand config's
consistency check, and the loader's CRBF0002 bulk-frame layout. `cargo clippy` reports nothing in the target. The full 95,000-query FineWeb-10B runs were done with this code
(conareai/supernova 7d7d0d3). Their JSON reproduces from the raw per-request report rows, and an independent scorer
reading the GT parquet gets the same strict and tie-aware recall.

### Result produced with it (details and raw runs: `results/conaredb-fineweb-10b-dense-20260924/` on conareai/supernova branch `submission-conaredb-20260924`)

Strict recall@10 **0.9730** (tie-aware upper bound 0.9861 at 2e-3) on 95,000 held-out queries (q5000–99999) against
`gt_dense_k1000` @3aea0b8d. One Azure Standard_L96as_v4 serves all 10.07B documents: c32 closed loop 595 QPS at p50/p99
53.2/69.8 ms, 400 rps at p50/p99 34.1/46.6 ms, 200 rps at p50/p99 25.3/35.3 ms. Median of 3, 0 errors. The query
vectors were regenerated with `regenerate_queries.py` @420f4686; with them, exact search reaches strict 0.99878.

### Notes for reviewers

- The `nova-load` store targets ConareDB's **v1** API. The 10B index above was built offline by the engine's own
  pipeline (dedupe → build → spill → import), not through `nova-load`.
- `.github/workflows/rust-binaries.yml` is unchanged. Adding `conaredb` to the release `--features` lists would ship it in
  release binaries, as #64 did for opensearch. That is left to the maintainers.
- The ConareDB server is not open source, so reproducing the result needs a server binary from us.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
