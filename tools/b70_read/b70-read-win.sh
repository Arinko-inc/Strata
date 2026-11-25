#!/bin/bash
# b70-read-win.sh QUEUE: waits for ~/b/gpu-window (30 s polls, 70 min at most), then runs the queue file's lines one by one
# (each line is a shell command in which $R is b70-read-run.sh, e.g.  NEED=150 SEQ="3072:3072:3" $R TAG EXE ENV=VAL).  Stops
# starting runs when the window has under WNEED (default 150) seconds left; every run is under flock, 10 minutes at most.
Q=$1; L=~/wt/b70-read-logs; mkdir -p $L
W=/home/arinko/b/gpu-window
for i in $(seq 1 140); do [ -f $W ] && break; sleep 30; done
[ -f $W ] || { echo "$(date +%T) no window in 70 min" >> $L/win.log; exit 1; }
sleep 8
echo "$(date +%T) window: $(cat $W) queue $Q" >> $L/win.log
pre=""
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  case "$line" in *'$R '*) ;; *) pre="$pre $line;"; continue;; esac   # a line without $R: variables for the lines after it
  [ -f $W ] || { echo "$(date +%T) the window is gone" >> $L/win.log; break; }
  left=$(( $(sed -n 's/.*until \([0-9]*\).*/\1/p' $W) - $(date +%s) ))
  [ "$left" -gt "${WNEED:-150}" ] || { echo "$(date +%T) $left s left: stop before: $line" >> $L/win.log; break; }
  flock /home/arinko/b/gpu-bench.lock timeout 600 bash -c "R=$(dirname $0)/b70-read-run.sh; $pre $line" >> $L/win.log 2>&1 < /dev/null
done < $Q
echo "$(date +%T) queue done" >> $L/win.log
touch $L/DONE-$(basename $Q)
