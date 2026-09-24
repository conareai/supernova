#!/usr/bin/env python3
"""verify try 2: sample N random positions of one shard on its L32; print shard, position, uuid hex (raw 16 bytes), sha256 of the f16 vector row."""
import sys, hashlib, numpy as np
K, d, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ids = np.load(d + "/ids.npy", mmap_mode="r"); vec = np.load(d + "/docs-f16.npy", mmap_mode="r")
assert ids.dtype == np.dtype("S16") and vec.shape == (ids.shape[0], 768)
raw = ids.view(np.uint8).reshape(-1, 16)  # raw bytes: |S16 indexing strips trailing NULs
rng = np.random.default_rng(seed); pos = np.sort(rng.choice(ids.shape[0], n, replace=False))
print(f"#shard {K} rows {ids.shape[0]} seed {seed}")
for p in pos:
    print(K, int(p), bytes(raw[p]).hex(), hashlib.sha256(np.ascontiguousarray(vec[p]).tobytes()).hexdigest())
