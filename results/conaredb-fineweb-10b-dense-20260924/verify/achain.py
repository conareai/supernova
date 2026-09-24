#!/usr/bin/env python3
"""verify try 2, A side: for each L32 sample (shard, position, uuid, vec sha): ordinal = start(shard) + position, with shard starts
recomputed from A's ids.npy copies (row counts cross-checked against the L32's own count); g = ord2dist[ordinal];
check uuid is in CSR list g exactly once, len(list g) == global-count[g], A's copy of ids.npy row == L32 uuid,
sha256(A global vector row g) == L32 vector sha."""
import sys, json, glob, hashlib, numpy as np
C = "/mnt/nvme/chain"
rows = {}
for k in range(10):
    rows[f"s{k}"] = np.load(f"{C}/in/s{k}.ids.npy", mmap_mode="r").shape[0]
start, acc = {}, 0
for k in range(10):
    start[f"s{k}"] = acc; acc += rows[f"s{k}"]
assert acc == 10074324060, acc
o2d = np.memmap(f"{C}/ord2dist.u32", dtype="<u4", mode="r"); assert o2d.shape[0] == acc
off = np.memmap(f"{C}/csr-offsets.u64", dtype="<u8", mode="r"); cu = np.memmap(f"{C}/csr-uuids.bin", dtype=np.uint8, mode="r")
gc = np.memmap("/mnt/nvme/global/global-count.u32", dtype="<u4", mode="r")
src = json.load(open("/mnt/nvme/global/source.json")); cst = np.cumsum([0] + [c["rows"] for c in src["chunks"]])
assert cst[-1] == 2557787738 == len(off) - 1 == gc.shape[0], (cst[-1], len(off), gc.shape)
res = {"shard_starts": start, "checked": 0, "rowcount_mismatch": [], "a_copy_uuid_mismatch": 0, "csr_not_found": 0,
       "csr_dup": 0, "len_ne_global_count": 0, "vector_mismatch": 0, "per_shard": {}}
for f in sorted(sys.argv[1:]):
    L = open(f).read().split("\n"); hdr = L[0].split(); K, n_l32 = hdr[1], int(hdr[3])
    if n_l32 != rows[K]: res["rowcount_mismatch"].append([K, n_l32, rows[K]])
    ida = np.load(f"{C}/in/{K}.ids.npy", mmap_mode="r").view(np.uint8).reshape(-1, 16)
    n = 0
    for line in L[1:]:
        if not line.strip(): continue
        _, p, u, vs = line.split(); p = int(p); ub = bytes.fromhex(u); o = start[K] + p
        if bytes(ida[p]) != ub: res["a_copy_uuid_mismatch"] += 1
        g = int(o2d[o]); a, b = int(off[g]), int(off[g + 1])
        blk = np.asarray(cu[16 * a:16 * b]).reshape(-1, 16)
        hits = int((blk == np.frombuffer(ub, np.uint8)).all(axis=1).sum())
        if hits == 0: res["csr_not_found"] += 1
        if hits > 1: res["csr_dup"] += 1
        if b - a != int(gc[g]): res["len_ne_global_count"] += 1
        ci = int(np.searchsorted(cst, g, side="right") - 1); ch = src["chunks"][ci]; r = g - int(cst[ci])
        with open(ch["path"], "rb") as fh:
            fh.seek(ch["vectors_offset"] + r * 1536); vb = fh.read(1536)
        if hashlib.sha256(vb).hexdigest() != vs: res["vector_mismatch"] += 1
        n += 1
    res["per_shard"][K] = n; res["checked"] += n
print(json.dumps(res, indent=1))
