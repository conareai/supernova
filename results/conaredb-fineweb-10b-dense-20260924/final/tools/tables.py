#!/usr/bin/env python3
"""Markdown tables for the docs from final-runs.tsv + final-medians.json (run final_summarize.py first).
usage: tables.py FINAL_DIR"""
import csv, json, sys
from pathlib import Path

D = Path(sys.argv[1])
rows = {r["run"]: r for r in csv.DictReader(open(D / "final-runs.tsv"), delimiter="\t")}
med = json.loads((D / "final-medians.json").read_text()); med.pop("_void", None)
f1 = lambda x: f"{float(x):.1f}"

print("### S6 grid medians (test, top_k 10, frozen 12k; median of 3 per metric)\n")
print("| Load | offered | achieved QPS | p50 ms | p95 ms | p99 ms | max ms | errors | timeouts | schedule lag s | above capacity | identity ok |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
order = ["s6-c1", "s6-c16", "s6-c64", "s6-c256", "s6-r100", "s6-r600", "s6-r800"]
for g in order:
    if g not in med: continue
    m = med[g]; mode = g[3:]
    offered = f"{mode[1:]} rps" if mode.startswith("r") else "closed"
    lag = f1(m["schedule_lag_s"]) if "schedule_lag_s" in m else ""
    ac = "/".join("yes" if a else "no" for a in m["above_capacity"]) if mode.startswith("r") else ""
    print(f"| {'closed loop c' + mode[1:] if mode.startswith('c') else 'open loop ' + mode[1:] + ' rps'} | {offered} | {m['qps']:.1f} | "
          f"{m['p50']:.1f} | {m['p95']:.1f} | {m['p99']:.1f} | {m['max']:.1f} | {'/'.join(map(str, m['errors']))} | "
          f"{'/'.join(map(str, m['timeouts']))} | {lag} | {ac} | {'yes' if m['all_ident_ok'] else 'NO'} |")
print("\nRecall in every grid run (strict / tie 2e-3):",
      sorted({(rows[r]["strict"], rows[r]["tie"]) for g in order if g in med for r in med[g]["runs"]}))
print("\n### Every run of this lane\n")
cols = ["run", "mode", "expand", "top_k", "requests", "errors", "timeouts", "missing_from_gt", "strict", "tie", "qps", "p50", "p95", "p99", "max",
        "schedule_lag_s", "above_capacity", "ident_ok"]
print("| " + " | ".join(cols) + " |"); print("|" + "---|" * len(cols))
for r in rows.values():
    print("| " + " | ".join(r.get(c, "") for c in cols) + " |")
