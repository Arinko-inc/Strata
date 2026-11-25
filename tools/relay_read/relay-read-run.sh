#!/bin/bash
# relay-read-run.sh TAG EXE [ENV=VAL ...]: the relay worker alone (layers [18,30), as ~/b/svc-7900-3st.sh, on 127.0.0.1:7852),
# its next worker a stage_sink (127.0.0.1:7853, rows returned unchanged), stage_probe_bench --bench as its main process.
# Only inside a GPU window (~/b/gpu-window); never touches the production unit.  SIZES="256 1024 2048 3072" REPS=5 LONG=12288
L=~/wt/relay-read-logs; mkdir -p $L
T=~/wt/relay-read
tag=$1; exe=$2; shift 2
W=/home/arinko/b/gpu-window
[ -f $W ] || { echo "no gpu-window"; exit 3; }
left=$(( $(sed -n 's/.*until \([0-9]*\).*/\1/p' $W) - $(date +%s) ))
[ "$left" -gt ${NEED:-420} ] || { echo "gpu-window: $left s left"; exit 3; }
[ "$(systemctl is-active strata-stage)" = active ] && { echo "strata-stage is active"; exit 4; }
pgrep -f "stage-worker 7852" >/dev/null && { echo "another worker on 7852"; exit 4; }
SP=/home/arinko/rocm-venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SP/_rocm_sdk_core/lib:$SP/_rocm_sdk_libraries_gfx110X_dgpu/lib HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0
D=/mnt/llmsv02-strata-data
M=$D/models/orca-Q4_K_S/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00001-of-00003.gguf
export STRATA_STAGE_TOKEN=relay-read-bench
export STRATA_AUX_CPUS=auto STRATA_WK_LEND_SINGLE=1 STRATA_WK_REFILL_EARLY=1 STRATA_GR_RDNA=1 STRATA_RDNA_Q6K=1 STRATA_HC_REQ8=1
for kv in "$@"; do export "$kv"; done
$T/stage_sink 7853 --chunk 3072 > $L/sink-$tag.log 2>&1 &
SPID=$!
ARGS="--serve --stage-worker 7852 --stage-begin 18 --stage-end 30 --stage-next 127.0.0.1:7853 --stage-bind 127.0.0.1 --pack $D/packs/orca-q4_k_s --native $M --ple-gguf $M --ple-io mmap --expert-profile /home/arinko/strata-fork/data/expert-profile.bin --expert-cache auto --prefill 3072 --spec 4 --max-context 262144 --kv int8 --kv-resident 32768 --mmap-experts --vram-reserve-mib 360 --batch 4 --vision"
(cd /home/arinko/strata-v9 && exec flock /home/arinko/b/gpu-bench.lock timeout 590 $exe $ARGS) > $L/worker-$tag.log 2>&1 &
WPID=$!
trap 'kill $WPID $SPID 2>/dev/null; sleep 1; for p in $(pgrep -f "stage-worker 7852"); do kill $p 2>/dev/null; done; wait 2>/dev/null' EXIT
for i in $(seq 1 300); do
  grep -q "listening on 127.0.0.1:7852" $L/worker-$tag.log && break
  kill -0 $WPID 2>/dev/null || { echo "$tag: the worker exited"; tail -5 $L/worker-$tag.log | grep -v TOKEN; exit 1; }
  sleep 1
done
echo "$(date +%T) $tag: listening after $i s; dpm $(cat /sys/class/drm/card*/device/power_dpm_force_performance_level 2>/dev/null | tr '\n' ' ')"
: > $L/probe-$tag.log
for n in ${SIZES:-256 1024 2048 3072}; do
  echo "== bench $n chunk $n" >> $L/probe-$tag.log
  echo "== bench $n chunk $n" >> $L/worker-$tag.log
  timeout 200 $T/stage_probe_bench 127.0.0.1:7852 --bench $n --reps ${REPS:-5} --chunk $n >> $L/probe-$tag.log 2>&1 || echo "$tag: probe $n exit $?"
done
if [ "${LONG:-12288}" -gt 0 ]; then
  echo "== bench ${LONG:-12288} chunk 3072" >> $L/probe-$tag.log
  echo "== bench ${LONG:-12288} chunk 3072" >> $L/worker-$tag.log
  timeout 200 $T/stage_probe_bench 127.0.0.1:7852 --bench ${LONG:-12288} --reps 2 --chunk 3072 >> $L/probe-$tag.log 2>&1 || echo "$tag: probe long exit $?"
fi
echo "$(date +%T) $tag: done"
grep "tokens in\|^==" $L/probe-$tag.log | tr '\n' ';' | sed 's/;==/\n==/g'; echo
