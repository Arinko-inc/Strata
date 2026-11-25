#!/bin/bash
# b70-read-run.sh TAG EXE [ENV=VAL ...]: the B70 stage worker alone (layers [30,48), the arguments and the environment of
# ~/b/svc-b70.sh, on 127.0.0.1:7861) and stage_probe --bench as its main process.  Only inside a GPU window
# (~/b/gpu-window, one line "until <unix seconds>"); never touches the production units.  The stage token is read from the
# production script when it runs and is not written anywhere.
#   SEQ="bench:chunk:reps ..."  the probe runs, in order (default: one chunk of 256, 1024, 2048, 3072, then 58368 in 3072s)
#   NEED=<seconds>              the window must have this much left (default 240)
#   PROBE=<stage_probe>         default ~/wt/b70-read/build-sycl/stage_probe
L=~/wt/b70-read-logs; mkdir -p $L
tag=$1; exe=$2; shift 2
PORT=7861
W=/home/arinko/b/gpu-window
[ -f $W ] || { echo "$tag: no gpu-window"; exit 3; }
left=$(( $(sed -n 's/.*until \([0-9]*\).*/\1/p' $W) - $(date +%s) ))
[ "$left" -gt "${NEED:-240}" ] || { echo "$tag: gpu-window: $left s left"; exit 3; }
for u in m1118-b70 strata-stage-b70; do [ "$(systemctl is-active $u)" = active ] && { echo "$tag: $u is active"; exit 4; }; done
ss -ltn 2>/dev/null | grep -q "127.0.0.1:$PORT " && { echo "$tag: another worker on $PORT"; exit 4; }   # (not pgrep -f: it matches a shell whose command line names the port)
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
export STRATA_WK_LEND_SINGLE=1 STRATA_MMVQ_LARGE_GRF=1 STRATA_MMVQ_ESIMD=1 STRATA_HC_REQ8=1 STRATA_SHEXP_ESIMD=1
export STRATA_VERIFY_NO_HOST=1 STRATA_VERIFY_DEVICE_PLAN=1 STRATA_STAGER_THREADS=8 ZES_ENABLE_SYSMAN=1 ONEAPI_DEVICE_SELECTOR=level_zero:0
export STRATA_REMOTE_TIMING=1 STRATA_SCORES_ESIMD=2 STRATA_EXPERT_Q4KQ5=1 STRATA_ATTN_ESIMD=1
export STRATA_STAGE_TOKEN=$(grep -o 'STRATA_STAGE_TOKEN=[^ ]*' /home/arinko/b/svc-b70.sh | head -1 | cut -d= -f2-)
for kv in "$@"; do export "$kv"; done
M=/data/Strata-data/models/orca-Q4_K_S/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00001-of-00003.gguf
ARGS="--serve --stage-worker $PORT --stage-begin 30 --stage-bind 127.0.0.1 --pack /data/Strata-data/packs/orca-q4_k_s --native $M --ple-gguf $M --ple-io mmap --expert-profile /home/arinko/wt/bf16/data/expert-profile.bin --expert-cache auto --prefill 3072 --spec 4 --max-context 262144 --kv int8 --kv-resident 32768 --stream-experts --vram-reserve-mib 1024 --batch 4 --vision"
(cd ~/wt/v9 && exec $exe $ARGS) > $L/worker-$tag.log 2>&1 &
WPID=$!
trap 'kill $WPID 2>/dev/null; wait $WPID 2>/dev/null' EXIT
for i in $(seq 1 300); do
  grep -q "listening on 127.0.0.1:$PORT" $L/worker-$tag.log && break
  kill -0 $WPID 2>/dev/null || { echo "$tag: the worker exited"; tail -5 $L/worker-$tag.log; exit 1; }
  sleep 1
done
echo "$(date +%T) $tag: listening after $i s ($*)"
: > $L/probe-$tag.log
for s in ${SEQ:-256:256:3 1024:1024:3 2048:2048:3 3072:3072:3 58368:3072:1}; do
  IFS=: read n c r <<< "$s"
  echo "## bench $n chunk $c reps $r" >> $L/probe-$tag.log
  echo "## bench $n chunk $c reps $r" >> $L/worker-$tag.log
  timeout 300 ${PROBE:-$HOME/wt/b70-read/build-sycl/stage_probe} 127.0.0.1:$PORT --bench $n --reps $r --chunk $c >> $L/probe-$tag.log 2>&1 || { echo "$tag: probe $s exit $?"; break; }
  sleep 0.3
done
echo "$(date +%T) $tag: done: $(grep -c 'bench rep .* chunk' $L/probe-$tag.log) chunks; $(grep 'tokens in' $L/probe-$tag.log | awk '{printf "%s/%s:%s ", $4, $7, $10}')"
