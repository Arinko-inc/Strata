#!/bin/bash
# relay-read-mmqb.sh: after queue2 (DONE-queue2.txt), inside the GPU window: relay_mmq_bench (about 20 s)
L=~/wt/relay-read-logs; W=/home/arinko/b/gpu-window
for i in $(seq 1 1100); do [ -f $L/DONE-queue2.txt ] && break; sleep 10; done
[ -f $W ] || { echo "no window" > $L/mmqb.log; exit 3; }
left=$(( $(sed -n 's/.*until \([0-9]*\).*/\1/p' $W) - $(date +%s) ))
[ "$left" -gt 200 ] || { echo "window: $left s left" > $L/mmqb.log; exit 3; }
[ "$(systemctl is-active strata-stage)" = active ] && exit 4
SP=/home/arinko/rocm-venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SP/_rocm_sdk_core/lib:$SP/_rocm_sdk_libraries_gfx110X_dgpu/lib HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 STRATA_MMQ_KQ=1
flock /home/arinko/b/gpu-bench.lock timeout 70 ~/wt/relay-read/relay_mmq_bench > $L/mmqb.log 2>&1
echo "exit $?" >> $L/mmqb.log
