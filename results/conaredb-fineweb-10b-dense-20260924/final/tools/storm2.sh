#!/bin/bash
# s5: one nova-storm run against A (10.0.0.14:19305), v2 /search.
# usage: storm2.sh TAG SET(dev|test) MODE(cN = closed loop concurrency N | rR = open loop R rps, in-flight cap 256) EXPAND(on|off) TOPK DURATION_S
#   DURATION_S 0 = fixed work (passes: 1, every query exactly once); >0 = timed run (passes: 0, file cycled)
# Writes runs/TAG.{yaml,json,log,ts.jsonl}, then runs/TAG.done (rc).
set -u
TAG=$1; SET=$2; MODE=$3; EXP=$4; K=$5; DUR=$6; D=$HOME/qdrant-10b/final; R=$D/runs; mkdir -p $R
case $SET in dev) Q=storm-dev.parquet;; test) Q=storm-test.parquet;; *) exit 2;; esac
case $MODE in c*) C=${MODE#c}; RPS=0;; r*) C=256; RPS=${MODE#r};; *) exit 2;; esac
if [ "$DUR" = 0 ]; then LOAD="  passes: 1"; else LOAD="  duration_s: $DUR
  passes: 0"; fi
if [ "$EXP" = on ]; then XB="  expand:
    offsets: /mnt/nvme/qdrant-10b/chain/csr-offsets.u64
    postings: /mnt/nvme/qdrant-10b/chain/csr-uuids.bin
    id_offset: 1
    copy_order: uuid"
elif [ "$EXP" = off ]; then XB="  # expand: OFF (S3 A/B). Hit ids stay distinct-row ordinals, so this run's recall is meaningless; latency only."
else exit 2; fi
cat > $R/$TAG.yaml <<Y
# $TAG: set=$SET mode=$MODE expand=$EXP top_k=$K duration_s=$DUR; client s5 (Standard_L32aos_v4, 10.0.0.10) -> A (Standard_L96as_v4, 10.0.0.14) over the vnet
target:
  type: conaredb
  url: http://10.0.0.14:19305
  api_key: \${CONAREDB_AUTH_TOKEN}
  timeout_s: 30
$XB
query:
  top_k: $K
  source:
    uri: $HOME/qdrant-10b/queries-20260924/$Q
    column: dense_embedding
    limit: 100000
    ground_truth_column: hit_fineweb_ids
    ground_truth_score_column: hit_scores
load:
  concurrency: $C
  rps: $RPS
$LOAD
report:
  format: jsonl
  path: $R/$TAG.ts.jsonl
Y
export CONAREDB_AUTH_TOKEN=$(cat $HOME/serve.token)
echo "$(date -u +%FT%TZ) start $TAG bin=$(sha256sum $D/nova-storm | cut -c1-16) load=$(cut -d' ' -f1-3 /proc/loadavg)" >> $R/runs.log
$D/nova-storm --json $R/$TAG.yaml > $R/$TAG.json 2> $R/$TAG.log; rc=$?
echo "$(date -u +%FT%TZ) end $TAG rc=$rc $(cat $R/$TAG.json)" >> $R/runs.log
echo $rc > $R/$TAG.done
