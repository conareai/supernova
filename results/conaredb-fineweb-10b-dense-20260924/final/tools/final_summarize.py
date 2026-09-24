#!/usr/bin/env python3
"""Per-run table + medians + identity validity for the final lane.
usage: final_summarize.py DIR   (DIR holds runs/ from s5 and identlog.txt from A)
Writes DIR/final-runs.tsv and DIR/final-medians.json. Identity rule (RULES.md): every identlog line from 60 s before a run's
start to 60 s after its end must show one process (pid + start) with the expected exe and server.json sha."""
import json, re, statistics, sys
from datetime import datetime, timedelta
from pathlib import Path

D = Path(sys.argv[1]); R = D / "runs"
FROZEN = ("cd60f49ae1d2c70a1965415160a577aaff4eb64cfefcd0d19860d0d5ab6f7c89",
          "4beb073110d3ce3478ac8430a8e7bc8eead3e2bef41d94888eeb842677b4a548")
EXPECT = {}  # tag prefix -> (exe, cfg) for non-frozen runs; filled from FREEZE-top100.json if present
ft = D / "FREEZE-top100.json"
if ft.exists():
    f = json.loads(ft.read_text()); EXPECT["fleet-k100-tuned"] = (f["server_binary_sha256"], f["server_json_sha256"])
ts = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")

ident = []
for line in (D / "identlog.txt").read_text().splitlines():
    m = re.match(r"(\S+Z) pid=(\d+) start=(\S+) exe=(\w+) .*?cfg_sha=(\w+) (.*?) env=", line)
    if m:
        ident.append((ts(m[1]), m[2] + "@" + m[3], m[4], m[5], m[6]))

starts, ends = {}, {}
for line in (R / "runs.log").read_text().splitlines():
    p = line.split(" ", 3)
    if len(p) >= 3 and p[1] == "start": starts[p[2]] = ts(p[0])
    if len(p) >= 3 and p[1] == "end": ends[p[2]] = ts(p[0])

rows = []
for tag in sorted(ends, key=lambda t: ends[t]):
    js = R / f"{tag}.json"
    try: d = json.loads(js.read_text())
    except Exception: rows.append(dict(run=tag, status="NO JSON")); continue
    y = (R / f"{tag}.yaml").read_text()
    mode = re.search(r"mode=(\S+)", y)[1]; exp = re.search(r"expand=(\S+)", y)[1]; k = d.get("top_k")
    s, e = starts[tag], ends[tag]
    win = [x for x in ident if s - timedelta(seconds=60) <= x[0] <= e + timedelta(seconds=60)]
    want = next((v for p, v in EXPECT.items() if tag.startswith(p)), FROZEN)
    procs = {x[1] for x in win}
    ok_ident = bool(win) and len(procs) == 1 and all((x[2], x[3]) == want for x in win)
    log = (R / f"{tag}.log").read_text(errors="replace")
    err1 = next((re.sub(r"\x1b\[[0-9;]*m", "", l)[:300] for l in log.splitlines() if re.search(r"WARN|ERROR", l)), "")
    n = d["requests"]; qps = d["qps"]; wall = n / qps if qps else 0
    rate = float(mode[1:]) if mode.startswith("r") else None
    rows.append(dict(
        run=tag, mode=mode, expand=exp, top_k=k, requests=n, errors=d["errors"], timeouts=d["timeouts"],
        missing_from_gt=d.get("missing_from_gt"), strict=d["full_recall"]["mean"] if d.get("full_recall") else None,
        tie=d.get("full_recall_tolerant"), qps=qps, p50=d["p50_ms"], p95=d["p95_ms"], p99=d["p99_ms"], max=d["max_ms"],
        wall_s=round(wall, 1), offered_rps=rate,
        schedule_lag_s=round(wall - n / rate, 1) if rate else None,
        above_capacity=(qps < 0.99 * rate) if rate else None,
        start=s.isoformat() + "Z", end=e.isoformat() + "Z", ident_lines=len(win), ident_ok=ok_ident,
        process=";".join(sorted(procs)), first_warn_or_error=err1))

cols = ["run", "mode", "expand", "top_k", "requests", "errors", "timeouts", "missing_from_gt", "strict", "tie", "qps", "p50", "p95",
        "p99", "max", "wall_s", "offered_rps", "schedule_lag_s", "above_capacity", "start", "end", "ident_lines", "ident_ok", "process",
        "first_warn_or_error"]
fmt = lambda v: "" if v is None else (f"{v:.5f}" if isinstance(v, float) and v < 1.5 else (f"{v:.2f}" if isinstance(v, float) else str(v)))
with open(D / "final-runs.tsv", "w") as f:
    f.write("\t".join(cols) + "\n")
    for r in rows: f.write("\t".join(fmt(r.get(c)) for c in cols) + "\n")

# Runs voided by the identity rule and replaced by a rerun tagged <run>r (both published; the void one is not in any median).
VOID = {"s6-r800-rep3": "identity window: at 10:35:32Z, 38 s after the run ended, the on-disk server.json was already rewritten "
        "for the first top_k 100 dev point (the serving pid was still the frozen 291953); the 60 s rule voids the run -> rerun s6-r800-rep3r"}
groups = {}
for r in rows:
    m = re.match(r"(s6-.+)-rep\dr?$", r["run"])
    if m and r["run"] not in VOID: groups.setdefault(m[1], []).append(r)
med = {}
for g, rs in groups.items():
    med[g] = {"runs": [r["run"] for r in rs], "all_ident_ok": all(r["ident_ok"] for r in rs),
              "errors": [r["errors"] for r in rs], "timeouts": [r["timeouts"] for r in rs]}
    for mkey in ("strict", "tie", "qps", "p50", "p95", "p99", "max", "schedule_lag_s"):
        v = [r[mkey] for r in rs if r.get(mkey) is not None]
        if v: med[g][mkey] = statistics.median(v)
    med[g]["above_capacity"] = [r["above_capacity"] for r in rs]
med["_void"] = VOID
(D / "final-medians.json").write_text(json.dumps(med, indent=1))
for r in rows:
    print(f'{r["run"]:28} {r.get("mode",""):6} x={r.get("expand","")} k={r.get("top_k")} n={r.get("requests")} err={r.get("errors")} '
          f'to={r.get("timeouts")} strict={fmt(r.get("strict"))} tie={fmt(r.get("tie"))} qps={fmt(r.get("qps"))} '
          f'p50/95/99={fmt(r.get("p50"))}/{fmt(r.get("p95"))}/{fmt(r.get("p99"))} lag={r.get("schedule_lag_s")} '
          f'ident={r.get("ident_ok")}({r.get("ident_lines")}) {r.get("first_warn_or_error","")[:120]}')
