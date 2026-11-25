#!/bin/bash
# relay-read-win.sh QUEUE: waits for ~/b/gpu-window (polls each minute, up to 150 min), then runs the queue file's lines
# (each: "TAG EXE [ENV=VAL ...]" for relay-read-run.sh; "#" lines skipped), stopping 2 min + one run before `until`.
L=~/wt/relay-read-logs; q=$1; W=/home/arinko/b/gpu-window
for i in $(seq 1 150); do [ -f $W ] && break; sleep 60; done
[ -f $W ] || { echo "$(date +%T) no window in 150 min" >> $L/win.log; exit 3; }
echo "$(date +%T) window: $(cat $W)" >> $L/win.log
while read -r line; do
  case "$line" in ''|\#*) continue;; esac
  [ -f $W ] || { echo "$(date +%T) the window closed" >> $L/win.log; break; }
  left=$(( $(sed -n 's/.*until \([0-9]*\).*/\1/p' $W) - $(date +%s) ))
  [ "$left" -gt 420 ] || { echo "$(date +%T) $left s left: stopping before: $line" >> $L/win.log; break; }
  eval "~/wt/relay-read-run.sh $line" >> $L/win.log 2>&1 < /dev/null
done < $q
echo "$(date +%T) queue $q finished" >> $L/win.log
touch $L/DONE-$(basename $q)
