#!/bin/bash
# pmtp-win1.sh QUEUE: waits (30 s polls, at most 70 min) for ~/b/gpu-window, then runs the queue's lines inside it
L=~/wt/pmtp-relay-logs; mkdir -p $L; q=$1; M=${2:-120}; W=/home/arinko/b/gpu-window
rm -f $L/$(basename $q).done
for i in $(seq 1 140); do [ -f $W ] && break; sleep 30; done
[ -f $W ] || { echo "$(date +%T) no window in 70 min" >> $L/win.log; echo nowindow > $L/$(basename $q).done; exit 3; }
echo "$(date +%T) win $q: $(cat $W)" >> $L/win.log
while read -r line; do
  case "$line" in ''|\#*) continue;; esac
  [ -f $W ] || { echo "$(date +%T) the window closed" >> $L/win.log; break; }
  left=$(( $(sed -n 's/.*until \([0-9]*\).*/\1/p' $W) - $(date +%s) ))
  [ "$left" -gt "$M" ] || { echo "$(date +%T) $left s left: stopping before: $line" >> $L/win.log; break; }
  bash -c "$line" >> $L/win.log 2>&1 < /dev/null
done < $q
echo "$(date +%T) win $q finished" >> $L/win.log
echo done > $L/$(basename $q).done
