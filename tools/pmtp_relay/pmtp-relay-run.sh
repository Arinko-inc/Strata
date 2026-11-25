#!/bin/bash
# pmtp-relay-run.sh TAG EXE [ENV=VAL ...]: the relay worker alone (layers [18,30), production's arguments, on
# 127.0.0.1:7862), its next worker a stage_sink_dfr (127.0.0.1:7863, rows returned unchanged), then as its main process
#   STEPS has "bench": stage_probe_bench --bench 3072 (the prompt chunk's hash: production's is ad680b8c4f428e89)
#   STEPS has "dfr":   stage_probe_dfr (the deferred windows against the commit graphs; DFR_ARGS, e.g. --no-deferred)
# Only inside a GPU window (~/b/gpu-window: "until <unix seconds>"); everything it starts ends before the window does.
# Never touches the production unit.  The token is read from ~/wt/relay-read-run.sh (not kept here).
L=~/wt/pmtp-relay-logs; mkdir -p $L
T=~/wt/pmtp-relay
tag=$1; exe=$2; shift 2
W=/home/arinko/b/gpu-window
[ -f $W ] || { echo "no gpu-window"; exit 3; }
left=$(( $(sed -n 's/.*until \([0-9]*\).*/\1/p' $W) - $(date +%s) ))
[ "$left" -gt ${NEED:-150} ] || { echo "gpu-window: $left s left"; exit 3; }
[ "$(systemctl is-active strata-stage)" = active ] && { echo "strata-stage is active"; exit 4; }
pgrep -f "stage-worker 7862" >/dev/null && { echo "another worker on 7862"; exit 4; }
cap=$(( left - 30 )); [ $cap -gt 590 ] && cap=590
SP=/home/arinko/rocm-venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SP/_rocm_sdk_core/lib:$SP/_rocm_sdk_libraries_gfx110X_dgpu/lib HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0
D=/mnt/llmsv02-strata-data
M=$D/models/orca-Q4_K_S/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00001-of-00003.gguf
eval "$(grep '^export STRATA_STAGE_TOKEN=' ~/wt/relay-read-run.sh)"
export STRATA_AUX_CPUS=auto STRATA_WK_LEND_SINGLE=1 STRATA_WK_REFILL_EARLY=1 STRATA_GR_RDNA=1 STRATA_RDNA_Q6K=1 STRATA_HC_REQ8=1 STRATA_HIP_WMMA=1 STRATA_MMQ_KQ=1 STRATA_RDNA_MMVQ_WAVE=3 STRATA_RELAY_REPLY_BUFS=4 STRATA_WMMA_GEMM=1
for kv in "$@"; do export "$kv"; done
vram() { for f in /sys/class/drm/card*/device/mem_info_vram_used; do [ -r $f ] && echo -n "$(( $(cat $f) / 1048576 ))/$(( $(cat ${f%_used}_total) / 1048576 )) MiB "; done; }
timeout $cap $T/stage_sink_dfr 7863 --chunk 3072 > $L/sink-$tag.log 2>&1 &
SPID=$!
ARGS="--serve --stage-worker 7862 --stage-begin 18 --stage-end 30 --stage-next 127.0.0.1:7863 --stage-bind 127.0.0.1 --pack $D/packs/orca-q4_k_s --native $M --ple-gguf $M --ple-io mmap --expert-profile /home/arinko/strata-fork/data/expert-profile.bin --expert-cache auto --prefill 3072 --spec 4 --max-context 262144 --kv int8 --kv-resident 32768 --mmap-experts --vram-reserve-mib 360 --batch 4 --vision"
echo "$(date +%T) $tag: VRAM before the worker: $(vram)"
(cd /home/arinko/strata-v9 && exec flock /home/arinko/b/gpu-bench.lock timeout $cap $exe $ARGS) > $L/worker-$tag.log 2>&1 &
WPID=$!
trap 'kill $WPID $SPID 2>/dev/null; sleep 1; for p in $(pgrep -f "stage-worker 7862"); do kill $p 2>/dev/null; done; for i in $(seq 1 40); do pgrep -f "stage-worker 7862" >/dev/null || break; sleep 1; done; pkill -f "stage_sink_dfr 7863" 2>/dev/null; wait 2>/dev/null' EXIT
for i in $(seq 1 $cap); do
  grep -q "listening on 127.0.0.1:7862" $L/worker-$tag.log && break
  kill -0 $WPID 2>/dev/null || { echo "$tag: the worker exited"; tail -5 $L/worker-$tag.log | grep -v TOKEN; exit 1; }
  sleep 1
done
echo "$(date +%T) $tag: listening after $i s; VRAM $(vram)"
STEPS=${STEPS:-bench dfr}
: > $L/probe-$tag.log
case " $STEPS " in *" bench "*)
  for n in ${SIZES:-3072}; do
    echo "== bench $n chunk $n" >> $L/probe-$tag.log
    timeout 120 $T/stage_probe_bench 127.0.0.1:7862 --bench $n --reps ${REPS:-2} --chunk $n >> $L/probe-$tag.log 2>&1 || echo "$tag: bench $n exit $?"
  done
  echo "$(date +%T) $tag: bench done; VRAM $(vram)";;
esac
case " $STEPS " in *" dfr "*)
  echo "== dfr $DFR_ARGS" >> $L/probe-$tag.log
  timeout ${DFR_TIMEOUT:-240} $T/stage_probe_dfr 127.0.0.1:7862 $DFR_ARGS >> $L/probe-$tag.log 2>&1; echo "$(date +%T) $tag: dfr exit $?; VRAM $(vram)";;
esac
case " $STEPS " in *" bench2 "*)
  echo "== bench 3072 again (after the windows)" >> $L/probe-$tag.log
  timeout 120 $T/stage_probe_bench 127.0.0.1:7862 --bench 3072 --reps 2 --chunk 3072 >> $L/probe-$tag.log 2>&1 || echo "$tag: bench2 exit $?"
  echo "$(date +%T) $tag: bench2 done; VRAM $(vram)";;
esac
kill -0 $WPID 2>/dev/null && echo "$tag: the worker is still up" || echo "$tag: THE WORKER EXITED"
grep -h "tokens in\|^==\|PASS\|FAIL\|all passed\|^  [a-z0-9-]* *all \|kStageBatchDeferred\|stage_probe" $L/probe-$tag.log | cut -c1-260
grep -h "VRAM free\|borrows\|saved rows\|WK_PIPED_MTP\|deferred batch windows:\|deferred commits\|never committed\|100% VRAM\|out of VRAM\|out of memory\|rror" $L/worker-$tag.log | grep -v TOKEN | cut -c1-300 | head -40
echo "sink: $(grep -c 'sink: run' $L/sink-$tag.log) runs ($(grep -c 'flags=2' $L/sink-$tag.log) deferred), $(grep -c 'sink: dcommit' $L/sink-$tag.log) deferred commits, $(grep -c 'sink: commit' $L/sink-$tag.log) plain commits"
