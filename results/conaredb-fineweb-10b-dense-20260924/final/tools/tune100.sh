#!/bin/bash
# laptop driver, RULES.md step 5 (top_k 100 tuning, DEV ONLY). For each grid point: restart A with gserve3.sh LEAVES 4 4 RESCORE,
# wait 65 s (so the 30 s identlog window holds one process), record ident.sh, run one dev c160 top_k 100 pass on s5 (storm2.sh),
# record ident.sh again, wait 65 s. Every step is a short ssh call; nothing long-running lives on the laptop.
# usage: tune100.sh "12000:400 12000:1000 ..."
set -u
BX=$(dirname "$0")/bx; A=20.81.176.43; S5=20.96.201.24; IL=/mnt/nvme/qdrant-10b/final/tune-ident.txt
for p in $1; do
  L=${p%:*}; RS=${p#*:}; TAG=dev-k100-l$L-rs$RS-c160
  if [ "$($BX $S5 "cat ~/qdrant-10b/final/runs/$TAG.done 2>/dev/null")" = 0 ]; then echo "skip $TAG (done)"; continue; fi
  $BX $A "bash /mnt/nvme/qdrant-10b/final/gserve3.sh $L 4 4 $RS" || { echo "gserve3 failed at $p"; exit 1; }
  sleep 65
  $BX $A "echo \"\$(date -u +%FT%TZ) pre $TAG \$(bash /mnt/nvme/qdrant-10b/s156/ident.sh)\" >> $IL"
  timeout 900 $BX $S5 "~/qdrant-10b/final/storm2.sh $TAG dev c160 on 100 0"
  $BX $A "echo \"\$(date -u +%FT%TZ) post $TAG \$(bash /mnt/nvme/qdrant-10b/s156/ident.sh)\" >> $IL"
  $BX $S5 "tail -n 1 ~/qdrant-10b/final/runs/runs.log | cut -c1-330"
  sleep 65
done
