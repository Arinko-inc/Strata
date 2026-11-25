#!/usr/bin/env python3
"""sink_check.py SINK_LOG: every deferred commit the relay sent on names (by its tag) a deferred SlotRun of the same
connection whose slots are exactly the slots that keep rows; prints the counts and the mismatches."""
import re, sys
runs, bad, n_dc, committed = {}, 0, 0, set()
for line in open(sys.argv[1], errors="replace"):
    m = re.match(r"sink: run n=(\d+) flags=(\d+) slots=([\d,]+)", line)
    if m:
        n = int(m.group(1))
        if n == 1:
            runs, committed = {}, set()   # a new connection
        runs[n] = (int(m.group(2)), sorted(set(int(x) for x in m.group(3).split(","))), m.group(3).split(","))
        continue
    m = re.match(r"sink: dcommit tag=(\d+) keep=([-\d,]+)", line)
    if m:
        n_dc += 1
        tag, keep = int(m.group(1)), [int(x) for x in m.group(2).split(",")]
        r = runs.get(tag)
        slots = [s for s, k in enumerate(keep) if k > 0]
        ok = r is not None and r[0] == 2 and r[1] == slots and tag not in committed and all(
            1 <= keep[s] <= r[2].count(str(s)) for s in slots)
        committed.add(tag)
        if not ok:
            bad += 1
            print("MISMATCH:", line.strip(), "run:", r)
print(f"{n_dc} deferred commits, {bad} mismatches")
sys.exit(1 if bad else 0)
