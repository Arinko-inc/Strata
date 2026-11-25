#!/bin/bash
# first.sh: waits for ~/b/no-build to go, builds stage_probe (--chunk), runs queue0 in the first window, then builds strata
# (again only while no-build is absent).
L=~/wt/b70-read-logs; D=$(dirname $0)
for i in $(seq 1 400); do [ -f /home/arinko/b/no-build ] || break; sleep 10; done
[ -f /home/arinko/b/no-build ] && { echo "$(date +%T) no-build stayed" >> $L/win.log; exit 1; }
~/wt/b70-read.sh stage_probe; echo "$(date +%T) stage_probe: $(cat ~/wt/b70-read/build-sycl.done)" >> $L/win.log
WNEED=100 $D/b70-read-win.sh $D/queue0.txt
for i in $(seq 1 400); do [ -f /home/arinko/b/no-build ] || break; sleep 10; done
[ -f /home/arinko/b/no-build ] && exit 1
~/wt/b70-read.sh strata stage_probe; echo "$(date +%T) strata: $(cat ~/wt/b70-read/build-sycl.done)" >> $L/win.log
