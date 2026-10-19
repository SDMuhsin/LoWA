#!/usr/bin/env python3
"""P9b: compare two op logs (tb_top / tb_top_gls +oplog: "S|D|R unit cycle" lines) up to a cycle limit.

    python3 tb/power/cmp_oplog.py <ref_oplog> <test_oplog> [max_cycle] [ref_offset]

ref_offset is added to the reference cycles (e.g. a gate-level run that starts at a later op).  Prints the number
of events compared, the first mismatch if any, and CMP_OPLOG SAME / DIFF."""
import sys


def ev(path, lim, off=0):
    out = []
    for ln in open(path):
        p = ln.split()
        if len(p) == 3:
            c = int(p[2]) + off
            if c <= lim:
                out.append((p[0], int(p[1]), c))
    return out


lim = int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 62
off = int(sys.argv[4]) if len(sys.argv) > 4 else 0
a, b = ev(sys.argv[1], lim, off), ev(sys.argv[2], lim)
# the test log may stop inside the window: compare the common prefix in time
last = min(a[-1][2] if a else 0, b[-1][2] if b else 0)
a2 = sorted(x for x in a if x[2] <= last)
b2 = sorted(x for x in b if x[2] <= last)
n = min(len(a2), len(b2))
bad = next((i for i in range(n) if a2[i] != b2[i]), None)
print(f"events compared {n} (ref {len(a2)}, test {len(b2)}) up to cycle {last}")
if bad is not None or len(a2) != len(b2):
    i = bad if bad is not None else n
    print(f"first difference at event {i}: ref {a2[i] if i < len(a2) else None} test {b2[i] if i < len(b2) else None}")
    print("CMP_OPLOG DIFF")
else:
    print("CMP_OPLOG SAME")
