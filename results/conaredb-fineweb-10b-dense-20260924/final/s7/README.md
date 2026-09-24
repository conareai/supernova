# §B.3 S7: stage binaries and wall times (collected on A 2026-09-24T06:40:37Z; nothing re-run)

`binaries.tsv` is the sha256, mtime and size of every binary and stage script still on A (`~/dedupe`, `~/keysample`, `~/bin*/`,
the two server copies in `/mnt/nvme/qdrant-10b/s156/`). The logs next to it are copies of the stage logs (`global.log`, `*.time`,
`*.out`, `*-request.json`) and of the scripts that ran each stage (`global-a.sh`, `global-a3.sh`, `gserve2.sh`, `serve_prep.py`).
Each binary's mtime is before the start of the stage that used it, and no stage script names a different path.

| Stage | Binary (path on A) | sha256 | Source | Wall | Log |
|---|---|---|---|---|---|
| per-box dedupe, A's rows (1,429,402,499 distinct) | `~/dedupe` | `333bc916a1f1379b9cdc847396a1892a2a6656718eb26d12719e1bb3e5abf992` | C program, source `~/dedupe.c` sha256 `913efd1c…` (not in git); built 2026-09-23 05:35:01Z | 18:45.84 | `a-perbox-dedupe.time`, `a-perbox-dedupe.out` |
| per-box dedupe, B's and C's rows | ran on l96-b and l96-c | **not recorded**: both boxes were deleted | | not in logs | `../../s0-chain/boxes/B.dedupe-stats.json` (counts only) |
| B, C → A transfer (nc + tar, 4 streams per box) | `~/bin5/recv.sh` (`834192f3…`) | | | start not logged; received 05:59:29Z (B) and 06:05:28Z (C), 4/4 streams each | `/mnt/nvme/in-{b,c}.recv.log` on A |
| global dedupe (4,284,692,459 → 2,557,787,738) | `~/dedupe` | `333bc916…` (the same binary) | as above | 23:52.23 (`/usr/bin/time`); 29:18 stage incl. manifest | `gdedupe.time`, `gdedupe.out`, `global.log` |
| copy counts | inline Python in `global-a.sh` (`dd55cb8b…`) | | | 1:19 | `global.log` |
| **index build** (23,744,385 leaves) | `~/bin5/conaredb-bench build` | **`04bc106a9f1f39bba5e2ff7a2f4e7a0700576b76cbd38da3982c97e90abacb7b`** | conare main **411e597b3** (`bin5-COMMIT`; ancestor of origin/main) | 1:13:57 | `build.time`, `build.out`, `build-request.json` |
| spill, attempt 1 (failed at 19:51) | `~/bin5/conaredb-bench spill` | `04bc106a…` | 411e597b3 | not counted | `global.log` |
| coded-spill detour (stopped, signal 15) | `~/bin8/conaredb-bench spill` | `e1a563ec…` | 004bf571b | not counted (14:26) | `spill-coded.time` |
| **spill** (exact, b512, 30 %, 767,336,321 rows) | `~/bin8/conaredb-bench spill` | **`e1a563ec4a38add6b1aad82e471f08203b16c5f7c4aa7894cba9a117a8cda351`** | conare main **004bf571b** (`global-a3.sh` header; ancestor of origin/main) | 56:19.41 | `spill.time`, `spill.out`, `spill-request.json` |
| verify | `~/bin8/conaredb-bench verify` | `e1a563ec…` | 004bf571b | 3:12 | `verify.out`, `global.log` |
| **import** of the serving tree `/mnt/nvme/gserve2` (the one every test run used) | `~/bin9/conaredb-import` | **`e018eb154b71f1eb1ec27597ceaa7fb5959af7ebfa0edc18eeb2bc7488d4f21b`** (byte-identical to `~/bin8/conaredb-import`) | 004bf571b | 3:32.31 | `gserve2-import.time`, `gserve2-import.out` |
| serve | `~/bin9/conaredb-server` while frozen | `cd60f49a…` | conare main c136d6788 | | `../../s156/FREEZE.json` |
| HF download → f16 prep (10 × L32) | | | | **never timed; not re-run** | |

**Erratum (SUBMISSION-READINESS.md, S7 row).** It said the build binary sha is `b016a0ac` (main 004bf571b), taken from
server.json `index_policy.binary_sha256`. That value is the sha256 of `~/bin8|bin9/conaredb-index-job`, which `serve_prep.py`
hashes into the server's policy for future index jobs. It is not the binary that built this index. The build ran
`~/bin5/conaredb-bench` (`04bc106a…`, main 411e597b3) and the spill ran `~/bin8/conaredb-bench` (`e1a563ec…`, main 004bf571b).

**HF → prep gap, stated plainly.** The FineWeb-10B `data/` parquet was downloaded and converted to per-shard `docs-f16.npy` +
`ids.npy` on ten L32 boxes during the earlier 10B program. Nobody timed that step end to end, and five of those ten L32s (s1, s2, s3, s4, s6) no longer exist. Its cost is therefore unknown and is not in any build-time or $ figure. It was not re-run for this submission: re-running
it means downloading and converting the whole `data/` split again on a fleet, which this lane did not do. A third party rebuilding from HF pays it.
