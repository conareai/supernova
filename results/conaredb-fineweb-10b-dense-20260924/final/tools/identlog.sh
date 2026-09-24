#!/bin/bash
# on A: append the serving identity (ident.sh) every 30 s until identlog.stop exists or 12 h pass.
D=/mnt/nvme/qdrant-10b/final; L=$D/identlog.txt; T0=$(date +%s)
while [ ! -f $D/identlog.stop ] && [ $(( $(date +%s) - T0 )) -lt 43200 ]; do
  echo "$(date -u +%FT%TZ) $(bash /mnt/nvme/qdrant-10b/s156/ident.sh) load=$(cut -d' ' -f1-3 /proc/loadavg | tr ' ' ,)" >> $L
  sleep 30
done
echo "$(date -u +%FT%TZ) identlog stop" >> $L
