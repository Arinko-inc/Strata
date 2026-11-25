#!/bin/bash
# chain7.sh: after queue6 (the 07:02 window) and with no-build gone: build, copy the exe to strata.mt6, queue7 in the next window
L=~/wt/b70-read-logs; D=$(dirname $0)
for i in $(seq 1 500); do [ -f $L/DONE-queue6.txt ] && break; sleep 10; done
for i in $(seq 1 400); do [ -f /home/arinko/b/no-build ] || break; sleep 10; done
[ -f /home/arinko/b/no-build ] && { echo "$(date +%T) chain7: no-build stayed" >> $L/win.log; exit 1; }
~/wt/b70-read.sh strata
grep -q "exit 0" ~/wt/b70-read/build-sycl.done || { echo "$(date +%T) chain7: build failed" >> $L/win.log; exit 1; }
cp ~/wt/b70-read/build-sycl/strata ~/wt/b70-read/strata.mt6
echo "$(date +%T) strata.mt6 $(sha256sum ~/wt/b70-read/strata.mt6 | cut -c1-16)" >> $L/win.log
for i in $(seq 1 60); do [ -f /home/arinko/b/gpu-window ] || break; sleep 10; done
WNEED=80 $D/b70-read-win.sh $D/queue7.txt
