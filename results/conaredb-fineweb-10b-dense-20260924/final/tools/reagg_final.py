#!/usr/bin/env python3
"""Re-aggregate every final-lane run from its raw nova-storm rows (runs/*.ts.jsonl.gz) and compare with its JSON summary:
row count = requests, ok = requests - errors, mean per-query strict recall, nearest-rank p50/p99, QPS from the rows' last t_s.
usage: reagg_final.py FINAL_DIR -> FINAL_DIR/reagg-final.json"""
import gzip, json, glob, os, sys, math
D = sys.argv[1]; out = {}
def pct(v, p):  # nearest-rank
    return v[max(0, math.ceil(p / 100 * len(v)) - 1)]
for f in sorted(glob.glob(f"{D}/runs/*.ts.jsonl.gz")):
    n = os.path.basename(f)[:-len(".ts.jsonl.gz")]
    if not os.path.exists(f"{D}/runs/{n}.done"): continue
    lat, ok, rec, tmax, rows = [], 0, [], 0.0, 0
    for line in gzip.open(f, "rt"):
        x = json.loads(line); rows += 1; tmax = max(tmax, x["t_s"])
        if x["ok"]: ok += 1; lat.append(x["latency_ms"])
        rec.extend(x["recalls_full"])
    lat.sort(); s = json.load(open(f"{D}/runs/{n}.json"))
    r = {"rows": rows, "ok": ok, "json_requests": s["requests"], "json_errors": s["errors"],
         "strict_rows": sum(rec) / len(rec) if rec else None, "json_strict": s["full_recall"]["mean"] if s.get("full_recall") else None,
         "p50_rows_ok": pct(lat, 50), "json_p50": s["p50_ms"], "p99_rows_ok": pct(lat, 99), "json_p99": s["p99_ms"],
         "qps_rows": rows / tmax if tmax else None, "json_qps": s["qps"]}
    r["match"] = (rows == s["requests"] and ok == s["requests"] - s["errors"]
                  and (r["strict_rows"] is None or abs(r["strict_rows"] - (r["json_strict"] or 0)) < 1e-6)
                  and abs(r["p50_rows_ok"] - s["p50_ms"]) < 0.5 and abs(r["p99_rows_ok"] - s["p99_ms"]) < 1.0)
    out[n] = r
    print(f'{n:30} rows={rows} ok={ok} strict {r["strict_rows"]} vs {r["json_strict"]}  p50 {r["p50_rows_ok"]:.2f}/{s["p50_ms"]:.2f} p99 {r["p99_rows_ok"]:.2f}/{s["p99_ms"]:.2f} qps {r["qps_rows"]:.1f}/{s["qps"]:.1f} match={r["match"]}')
out["_all_match"] = all(v["match"] for k, v in out.items() if not k.startswith("_"))
json.dump(out, open(f"{D}/reagg-final.json", "w"), indent=1); print("all_match", out["_all_match"])
