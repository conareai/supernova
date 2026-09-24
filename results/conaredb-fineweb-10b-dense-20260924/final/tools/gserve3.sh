#!/bin/bash
# on A, top_k 100 tuning only: gserve2.sh with one more knob, RESCORE. Same prep (serve_prep.py), same server binary path
# (~/bin9/conaredb-server), same import; after prep, sets budget.rescore, then restarts the server and waits for one real /v2/search 200.
# usage: gserve3.sh LEAVES SCAN ROUTE RESCORE
set -u
L=$1; T=$2; R=$3; RS=$4; G=/mnt/nvme/gserve2; S=$G/serve; LOG=/mnt/nvme/qdrant-10b/final/serve.log
ulimit -n $(ulimit -Hn)
IDX=/mnt/nvme/gidx/idx-spill30-b512; IMPL=$(python3 -c "import json;print(json.load(open('$IDX/manifest.json'))['spill']['implementation'])")
python3 $HOME/bin/serve_prep.py 10.0.0.14 7 $L 96 $HOME/serve.token $IMPL $HOME/bin9/conaredb-index-job $IDX 512 $G 19305 >> $LOG || exit 1
python3 - <<PY
import json,os
p="$S/server.json"; c=json.load(open(p)); c["limits"].update(scan_threads=$T, route_threads=$R); c["engine"]["max_pinned_views"]=256
c["budget"]["rescore"]=$RS
fd=os.open(p,os.O_WRONLY|os.O_TRUNC); os.write(fd,json.dumps(c,indent=1).encode()); os.close(fd)
PY
python3 -c "import json;p='$S/import.json';c=json.load(open(p));c['engine']['max_pinned_views']=256;json.dump(c,open(p,'w'),indent=1)"
test -f $S/IMPORT.done || { echo "no IMPORT.done: refusing to import from this script" >> $LOG; exit 1; }
ps -eo pid,args | awk '$2=="/home/azureuser/bin9/conaredb-server" {print $1}' | xargs -r kill
for k in $(seq 60); do ps -eo args | grep -q '^/home/azureuser/bin9/conaredb-server' || break; sleep 1; done
echo "$(date -u +%FT%TZ) gserve3 start leaves=$L scan=$T route=$R rescore=$RS bin=$(sha256sum $HOME/bin9/conaredb-server | cut -c1-16)" >> $LOG
(exec setsid nohup $HOME/bin9/conaredb-server $S/server.json > $S/server.out 2>&1) >/dev/null 2>&1 </dev/null &
for k in $(seq 600); do ss -ltn | grep -q "10.0.0.14:19305" && break; sleep 1; done
python3 - >> $LOG 2>&1 <<'PY'
import http.client, json, time
tok = open("/home/azureuser/serve.token").read().strip(); q = json.load(open("/home/azureuser/queries-q2000.json"))[0]
t0 = time.time()
while time.time() - t0 < 600:
    try:
        c = http.client.HTTPConnection("10.0.0.14", 19305, timeout=30)
        c.request("POST", "/v2/search", body=json.dumps({"vector": q, "top_k": 100}), headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
        r = c.getresponse(); b = r.read()
        if r.status == 200: print("ready after", round(time.time() - t0, 1), "s, hits", len(json.loads(b)["hits"])); break
        else: print("status", r.status, b[:200])
    except Exception as e: pass
    time.sleep(2)
else: print("NOT READY")
PY
echo "$(date -u +%FT%TZ) gserve3 up $(bash /mnt/nvme/qdrant-10b/s156/ident.sh)" >> $LOG
tail -n 2 $LOG
