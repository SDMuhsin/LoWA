#!/usr/bin/env python3
"""P9b: gate-level window runs vs the Verilator RTL reference (tb/power/run_ref_verilator.sh).

    python3 tb/power/cmp_vref.py <gls_logs_dir> <stim_root> <plan>...

<gls_logs_dir>/<window>.out and <window>.oplog: sim.out and oplog.txt of each gate-level run (fetched from the
server); <stim_root>/<stim>/vref.log and vref_oplog.txt: the reference.  For each window: every heartbeat of the
gate-level run (cycle, unit busy flags, DRAM write beats, write-stream signature) must appear identically in the
reference, and the gate-level op log must equal the reference's events up to the same cycle."""
import os
import sys


def hb(path):
    return [ln.strip() for ln in open(path) if ln.startswith("HB cyc")]


def ops(path):
    return [ln.split() for ln in open(path) if len(ln.split()) == 3]


logs, stim = sys.argv[1], sys.argv[2]
nbad = 0
for plan in sys.argv[3:]:
    for ln in open(plan):
        p = ln.split()
        if not p or p[0].startswith("#"):
            continue
        n, st = p[0], p[1]
        sd = os.path.join(stim, "l1" if st == "l1" else st)
        g, r = os.path.join(logs, n + ".out"), os.path.join(sd, "vref.log")
        if not (os.path.exists(g) and os.path.exists(r)):
            print(f"CMP {n} missing")
            continue
        hg, hr = hb(g), set(hb(r))
        same = sum(1 for x in hg if x in hr)
        og, orf = ops(os.path.join(logs, n + ".oplog")), ops(os.path.join(sd, "vref_oplog.txt"))
        last = int(hg[-1].split()[2]) if hg else 0
        orf2 = [e for e in orf if int(e[2]) <= max([int(e[2]) for e in og] + [0])]
        ok = same == len(hg) and og == orf2
        nbad += not ok
        print(f"CMP {n:10s} {st:6s} heartbeats {same}/{len(hg)} identical to cycle {last}"
              f" ({hg[-1].split(' ', 4)[4] if hg else '-'}); op log {len(og)} events {'same' if og == orf2 else 'DIFF'}"
              f" -> {'OK' if ok else 'MISMATCH'}")
print(f"CMP_VREF {'ALL OK' if nbad == 0 else f'{nbad} MISMATCH'}")
