#!/bin/bash
# s5: run a queue of storm2.sh runs, one at a time. usage: queue.sh QUEUE_FILE
#   lines: TAG SET MODE EXPAND TOPK DURATION_S ('#' = comment). A run whose .done holds 0 is skipped (resume-safe).
#   Stops before the next run if ~/qdrant-10b/final/STOP exists. Touches runs/queue-NAME.done at the end.
set -u
D=$HOME/qdrant-10b/final; Q=$1; N=$(basename $Q .txt); L=$D/runs/runs.log
echo "$(date -u +%FT%TZ) queue $N start" >> $L
while read -r TAG SET MODE EXP K DUR; do
  case "$TAG" in ''|\#*) continue;; esac
  [ -f $D/STOP ] && { echo "$(date -u +%FT%TZ) queue $N STOP before $TAG" >> $L; break; }
  [ "$(cat $D/runs/$TAG.done 2>/dev/null)" = 0 ] && continue
  $D/storm2.sh $TAG $SET $MODE $EXP $K $DUR
  sleep 5
done < $Q
echo "$(date -u +%FT%TZ) queue $N end" >> $L
touch $D/runs/queue-$N.done
