#!/bin/bash
# relay-read-win3.sh QUEUE [MARGIN s, default 150]: inside ~/b/gpu-window only: each queue line is a shell command
# (usually "VAR=.. ~/wt/relay-read-run.sh TAG EXE ENV=VAL ..."), run while more than MARGIN seconds are left.
L=~/wt/relay-read-logs; q=$1; M=${2:-150}; W=/home/arinko/b/gpu-window
[ -f $W ] || { echo "no window"; exit 3; }
echo "$(date +%T) win3 $q: $(cat $W)" >> $L/win.log
while read -r line; do
  case "$line" in ''|\#*) continue;; esac
  [ -f $W ] || { echo "$(date +%T) the window closed" >> $L/win.log; break; }
  left=$(( $(sed -n 's/.*until \([0-9]*\).*/\1/p' $W) - $(date +%s) ))
  [ "$left" -gt "$M" ] || { echo "$(date +%T) $left s left: stopping before: $line" >> $L/win.log; break; }
  NEED=$((M - 40)) bash -c "$line" >> $L/win.log 2>&1 < /dev/null
done < $q
echo "$(date +%T) win3 $q finished" >> $L/win.log
