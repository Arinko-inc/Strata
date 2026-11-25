#!/usr/bin/env python3
# pft.py TAG...: the worker's own time per chunk and the STRATA_PREFILL_TIMING phases, from ~/wt/b70-read-logs/worker-TAG.log.
# One line per (bench section, T, position): the median of "read in" over the reps (the first rep of a section left out when
# there are three or more), and the phases of the last timeline line of that (T, position).
import re, sys, os, statistics
L = os.path.expanduser('~/wt/b70-read-logs')
for tag in sys.argv[1:]:
    sec = ''; rd = {}; ph = {}; grp = {}; last_tl = None; last_grp = None; order = []
    for ln in open(f'{L}/worker-{tag}.log', errors='replace'):
        if ln.startswith('## bench'): sec = ln[3:].strip(); continue
        m = re.search(r'expert grouping: the host waited for the GPU (\d+) ms, grouped \(GPU idle\) (\d+) ms, launched the experts and the next layer (\d+) ms', ln)
        if m: last_grp = tuple(int(x) for x in m.groups()); continue
        m = re.search(r'prefill timing: (\d+) tokens, GPU timeline (\d+) ms, wall (\d+) ms, host staging (\d+) ms: (.*)', ln)
        if m:
            d = {k.strip(): int(v) for k, v in re.findall(r'([a-z+/ ]+?) (\d+) \(', m.group(5))}
            d['_tl'] = int(m.group(2)); d['_wall'] = int(m.group(3)); last_tl = d; continue
        m = re.search(r'chunk T=(\d+) at (\d+): read in (\d+) ms', ln)
        if m:
            k = (sec, int(m.group(1)), int(m.group(2)))
            if k not in rd: order.append(k)
            rd.setdefault(k, []).append(int(m.group(3)))
            if last_tl: ph[k] = last_tl
            if last_grp: grp[k] = last_grp
            last_tl = last_grp = None
    print(f'== {tag}')
    for k in order:
        v = rd[k]; use = v[1:] if len(v) >= 3 else v
        s = f'{k[0]:28s} T={k[1]:5d} at {k[2]:6d}: read {statistics.median(use):6.0f} ms (n={len(v)}: {",".join(map(str, v))})'
        if k in grp: s += f' | host wait/grp/launch {grp[k][0]}/{grp[k][1]}/{grp[k][2]}'
        print(s)
        if k in ph:
            d = ph[k]
            print('      tl %d wall %d: ' % (d['_tl'], d['_wall']) + ', '.join(f'{n} {x}' for n, x in sorted(d.items(), key=lambda kv: -kv[1]) if not n.startswith('_') and x > 0))
