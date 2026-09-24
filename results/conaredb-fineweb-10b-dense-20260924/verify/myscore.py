#!/usr/bin/env python3
"""Adversarial verify, try 2 (2026-09-24). Independent re-score of the served distinct rows saved by the first verify lane
(verify/a/test-frozen-{5000,36667,68334}.served_rows.npy + test-frozen-err2 for the 2 queries that errored), against
HF gt_dense_k1000.parquet @3aea0b8d. Own code: vectorised CSR expansion, uuids compared as raw 16-byte values
(GT strings parsed to bytes), nova-storm recall_at_k semantics: strict = |ret ∩ gt10|/10; tolerant(eps) adds returned ids
not in gt10 whose served score s satisfies |s-c| <= eps*(1+|c|), c = gt score at rank 10, capped at 1.
usage: myscore.py <served dir> <out.json>"""
import sys, json, hashlib
import numpy as np, pyarrow.parquet as pq
D, OUT = sys.argv[1], sys.argv[2]
parts = [(5000, 36667), (36667, 68334), (68334, 100000)]
rows = np.full((95000, 10), -1, np.int64); scs = np.full((95000, 10), np.nan, np.float64)
sh = {}
for lo, hi in parts:
    r = np.load(f"{D}/test-frozen-{lo}.served_rows.npy"); s = np.load(f"{D}/test-frozen-{lo}.served_scores.npy")
    sh[f"test-frozen-{lo}.served_rows.npy"] = hashlib.sha256(open(f"{D}/test-frozen-{lo}.served_rows.npy","rb").read()).hexdigest()
    assert r.shape == (hi - lo, 10), (lo, r.shape)
    rows[lo-5000:hi-5000] = r; scs[lo-5000:hi-5000] = s
eq = np.load(f"{D}/test-err-qidx.npy"); er = np.load(f"{D}/test-frozen-err2.served_rows.npy"); es = np.load(f"{D}/test-frozen-err2.served_scores.npy")
print("err qidx", eq.tolist(), "err rows shape", er.shape, "rows at err before patch", [rows[q-5000].tolist() for q in eq])
for j, q in enumerate(eq):
    rows[q-5000] = er[j]; scs[q-5000] = es[j]
assert (rows >= 0).all(), "unfilled served rows"
off = np.memmap("/mnt/nvme/chain/csr-offsets.u64", dtype="<u8", mode="r")
cu = np.memmap("/mnt/nvme/chain/csr-uuids.bin", dtype=np.uint8, mode="r")
assert len(off) == 2557787739 and int(off[-1]) == 10074324060 and cu.shape[0] == 16 * 10074324060
t = pq.read_table("/mnt/nvme/qdrant-10b/gt/gt_dense_k1000.parquet", columns=["hit_fineweb_ids", "hit_scores"])
import pyarrow.compute as pc
GTI = [x for ch in t.column("hit_fineweb_ids").chunks for x in pc.list_slice(ch, 0, 10).to_pylist()]
GTS = [x for ch in t.column("hit_scores").chunks for x in pc.list_slice(ch, 0, 10).to_pylist()]
assert len(GTI) == 100000 == len(GTS)
del t
def top10(i):
    return GTI[i], GTS[i]
EPS = {"tie_2e-3": 2e-3, "tie_5e-6": 5e-6, "tie_abs_1e-5": None}  # None = the plan's 1e-5 ABSOLUTE rule
strict = np.zeros(95000); tol = {k: np.zeros(95000) for k in EPS}
short_ret = 0; missing_above = 0; maxd = 0.0; big_groups = 0
for qi in range(95000):
    q = qi + 5000
    g10, s10 = top10(q)
    gset = {bytes.fromhex(u.replace("-", "")) for u in g10}
    assert len(gset) == 10
    c = s10[9]
    ret = []; rsc = []
    for rank in range(10):
        g = int(rows[qi, rank]); a, b = int(off[g]), int(off[g + 1])
        need = 10 - len(ret)
        if need <= 0: break
        blk = np.asarray(cu[16 * a:16 * b]).reshape(-1, 16)
        if blk.shape[0] > 1:
            big_groups += 1
            # ascending uuid = lexicographic byte order: sort by the 16 bytes as a void view
            v = np.ascontiguousarray(blk).view(np.dtype((np.void, 16))).ravel()
            blk = blk[np.argsort(v, kind="stable")]
        for u in blk[:need]:
            ret.append(u.tobytes()); rsc.append(float(scs[qi, rank]))
    if len(ret) < 10: short_ret += 1
    hits = sum(1 for u in ret if u in gset)
    strict[qi] = hits / 10
    for k, e in EPS.items():
        nt = sum(1 for u, s in zip(ret, rsc) if u not in gset and (abs(s - c) <= 1e-5 if e is None else abs(s - c) <= e * (1 + abs(c))))
        tol[k][qi] = min(1.0, (hits + nt) / 10)
    missing_above += sum(1 for u, s in zip(ret, rsc) if u not in gset and s > c and abs(s - c) > 2e-3 * (1 + abs(c)))
res = {"queries": 95000, "strict": strict.mean(), **{k: v.mean() for k, v in tol.items()},
       "strict_tenths_sum": int(round(strict.sum() * 10)), "tie_2e-3_tenths_sum": int(round(tol["tie_2e-3"].sum() * 10)),
       "strict_hist_tenths": {str(i): int((np.rint(strict * 10) == i).sum()) for i in range(11)},
       "short_returns": short_ret, "missing_above_cutoff_2e-3": missing_above, "multi_copy_rows_expanded": big_groups,
       "served_rows_sha256": sh, "gt": "/mnt/nvme/qdrant-10b/gt/gt_dense_k1000.parquet (HF 3aea0b8d)"}
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res, indent=1))
