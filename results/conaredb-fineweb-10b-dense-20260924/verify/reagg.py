#!/usr/bin/env python3
"""Re-aggregate nova-storm's raw per-request rows (report jsonl) for the 9 S6 test runs; compare with each run's JSON summary."""
import gzip, json, sys, glob, os, math
import numpy as np
D = sys.argv[1]; out = {}
for f in sorted(glob.glob(f"{D}/s6-test-*.ts.jsonl.gz")):
    n = os.path.basename(f).replace(".ts.jsonl.gz", "")
    L = [json.loads(x) for x in gzip.open(f, "rt")]
    lat = np.array([x["latency_ms"] for x in L]); ok = sum(x["ok"] for x in L)
    rec = np.array([r for x in L for r in x["recalls_full"]]); sh = sum(len(x["recalls_short"]) for x in L)
    s = json.load(open(f"{D}/{n}.json"))
    hist = {str(i): int((np.rint(rec * 10) == i).sum()) for i in range(11)}
    out[n] = {"rows": len(L), "ok": ok, "recall_n": len(rec), "short_n": sh, "strict_mean": rec.mean(),
              "strict_tenths_sum": int(np.rint(rec * 10).sum()), "hist": hist,
              "p50_np": float(np.percentile(lat, 50)), "p99_np": float(np.percentile(lat, 99)),
              "json_strict": s["full_recall"]["mean"], "json_tol": s["full_recall_tolerant"], "json_p50": s["p50_ms"], "json_p99": s["p99_ms"],
              "qps_from_rows": len(L) / max(x["t_s"] for x in L), "json_qps": s["qps"]}
hs = {json.dumps(v["hist"], sort_keys=True) for v in out.values()}
out["_all_9_histograms_identical"] = len(hs) == 1
print(json.dumps(out, indent=1))
