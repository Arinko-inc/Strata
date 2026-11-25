#!/bin/bash
# chain3.sh: after queue2 and the build, copies the exe to strata.mq2 and runs queue3 in what is left of the window
L=~/wt/b70-read-logs; D=$(dirname $0)
for i in $(seq 1 200); do [ -f $L/DONE-queue2.txt ] && [ -f ~/wt/b70-read/build-sycl.done ] && break; sleep 5; done
grep -q "exit 0" ~/wt/b70-read/build-sycl.done || { echo "$(date +%T) chain3: build failed" >> $L/win.log; exit 1; }
cp ~/wt/b70-read/build-sycl/strata ~/wt/b70-read/strata.mq2
WNEED=80 $D/b70-read-win.sh $D/queue3.txt
