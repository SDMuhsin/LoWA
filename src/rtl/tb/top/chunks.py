#!/usr/bin/env python3
"""P9a: training-step cycles composed from isolated program chunks run on the full chip RTL.

Each chunk is the compiled program with the unit ops before static op j replaced by NOPs and END after static op k
(run_e2e.py --nop-before j --stop-after k; output dir l{layers}_b{B}_s{L}_n1_nop{j}[_stop{k}]).  Every chunk starts
at a drain point of the in-situ schedule (all engines idle, nothing in flight; tb/top/drain.py), so its timing from
its first op on equals the in-situ timing (the schedule does not depend on data values; the DRAM model has fixed
latency / bandwidth).  Contribution of a chunk [j..k] to the step:
  next chunk starts at k (shared op)   S(k) - S(j)
  next chunk starts at k + 1           R(k) - S(j) + gap(k -> k+1), gap = S(k+1) - R(k) from an in-situ run
                                       (sequencer FENCE + descriptor fetch, shape-independent; at most ~100 cycles)
  last chunk of the program            IRQ - S(j)
  first chunk                          S(j) replaced by the cycle the sequencer started (IRQ cycle - ACCT cyc)
Shared ops are simulated twice (end of one chunk, start of the next): equal durations confirm the isolation.

  python3 tb/top/chunks.py --batch 32 --seq 128 [--insitu1 DIR --insitu2 DIR] [--json out.json]
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import drain as DR  # noqa: E402
import seg as SG  # noqa: E402

# (name, layers, j, k, mode)  mode: 'shared' / 'next' / 'last'
L1_CHUNKS = [("C1", 1, 0, 6, "shared"), ("C2", 1, 6, 10, "shared"), ("C3", 1, 10, 11, "next"),
             ("C4", 1, 12, 12, "next"), ("C5", 1, 13, 31, "shared"), ("C6", 1, 31, 32, "next"),
             ("C7", 1, 33, 33, "next"), ("C8", 1, 34, 39, "shared"), ("C9", 1, 39, 44, "shared"),
             ("C10", 1, 44, None, "last")]
BODY_CHUNKS = [("D1a", 2, 30, 31, "next"), ("D1b", 2, 32, 32, "next"), ("D2", 2, 33, 33, "next"),
               ("D3", 2, 34, 39, "shared"), ("D4", 2, 39, 45, "next"), ("D5a", 2, 46, 46, "next"),
               ("D5b", 2, 47, 48, "next")]     # D5b: op 47 (LoWA SG) runs beside op 48 (GEMM_ACC2); 48 retires last
F_FIRST, HEAD_FIRST, B0_FIRST, TAIL_FIRST = 3, 14, 30, 46   # static ops of the 1-layer program (segment starts)


def cdir(out, layers, B, L, j, k):
    return os.path.join(out, f"l{layers}_b{B}_s{L}_n1_nop{j}" + (f"_stop{k}" if k is not None else ""))


def info(d, j):
    """{static op: (S, R)}, IRQ cycle, ACCT dict for a chunk dir (static index = dynamic index + j)"""
    tl = DR.timeline(d)
    ops = {i + j: (S, R) for i, _, _, _, _, S, R in tl}
    irq, acct = None, {}
    for ln in open(os.path.join(d, "sim.log")):
        m = re.match(r"STEP 1 IRQ at cycle (\d+)", ln)
        if m:
            irq = int(m.group(1))
        if ln.startswith("ACCT 1 "):
            t = ln.split()[2:]
            acct = {t[i]: int(t[i + 1]) for i in range(0, len(t) - 1, 2)}
    return ops, irq, acct


def gaps(d):
    """{static k: S(k+1) - R(k)} from an in-situ run (static program order of its first step)"""
    tl = DR.timeline(d)
    by = {}
    for i, st, _, _, _, S, R in tl:
        by.setdefault(st, (S, R))
    return {k: by[k + 1][0] - by[k][1] for k in by if k + 1 in by and by[k][1] is not None and by[k + 1][0] is not None}


def compose(chunks, out, B, L, gap):
    """rows, total, shared-op checks, pos = {static op: composed start cycle relative to the step start}"""
    rows, tot, checks, pos = [], 0, [], {}
    prev = None
    for idx, (nm, lay, j, k, mode) in enumerate(chunks):
        d = cdir(out, lay, B, L, j, k)
        if not os.path.exists(os.path.join(d, "sim.log")):
            rows.append((nm, j, k, None, "missing"))
            tot = None
            continue
        ops, irq, acct = info(d, j)
        start = ops[j][0]
        if idx == 0 and j == 0:
            start = irq - acct["cyc"] if irq and acct else None
        try:
            if mode == "shared":
                c = ops[k][0] - start
            elif mode == "next":
                c = ops[k][1] - start + gap[k]
                mr = max(R for S_, R in ops.values() if R is not None)
                if mr != ops[k][1]:
                    print(f"WARN {nm}: an op retires after op {k} ({mr} > {ops[k][1]})")
            else:
                c = irq - start
        except (KeyError, TypeError):
            c = None
        if tot is not None and start is not None:
            for s_, (S_, _) in ops.items():
                if S_ is not None and (k is None or s_ <= k):
                    pos[s_] = tot + S_ - start
        if c is None or (mode == "last" and not irq):
            rows.append((nm, j, k, None, "running"))
            tot = None
        else:
            rows.append((nm, j, k, c, ""))
            if tot is not None:
                tot += c
        if prev and prev[1] == j:              # shared op simulated in both chunks
            a, b = prev[0], ops[j]
            if a[0] is not None and a[1] is not None and b[1] is not None:
                checks.append((j, a[1] - a[0], b[1] - b[0]))
        prev = (ops.get(k, (None, None)), k) if k is not None else None
    return rows, tot, checks, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "e2e"))
    ap.add_argument("--insitu1", required=True, help="in-situ 1-layer run (any shape) for the inter-op gaps")
    ap.add_argument("--insitu2", required=True, help="in-situ 2-layer run (any shape) for the inter-op gaps")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    g1, g2 = gaps(a.insitu1), gaps(a.insitu2)
    res = {}
    for nm, ch, g in (("l1", L1_CHUNKS, g1), ("body", BODY_CHUNKS, g2)):
        rows, tot, checks, pos = compose(ch, a.out, a.batch, a.seq, g)
        for r in rows:
            print(f"CHUNK {r[0]:4s} ops {r[1]}..{r[2]} contribution {r[3]} {r[4]}")
        for j, d1, d2 in checks:
            print(f"SHARED op {j}: end of previous chunk {d1} / start of next chunk {d2} {'EQUAL' if d1 == d2 else 'DIFF %+d' % (d2 - d1)}")
        print(f"TOTAL {nm} {tot}")
        res[nm] = dict(rows=rows, total=tot, checks=checks)
        if nm == "l1":
            b = [("pre", 0, F_FIRST), ("F0", F_FIRST, HEAD_FIRST), ("head", HEAD_FIRST, B0_FIRST), ("B0", B0_FIRST, TAIL_FIRST)]
            for sn, x, y in b:
                if x in pos and y in pos:
                    res[sn] = (pos[y] if y else 0) - (pos[x] if x else 0)
            if tot is not None and TAIL_FIRST in pos:
                res["tail"] = tot - pos[TAIL_FIRST]
            for sn in ("pre", "F0", "head", "B0", "tail"):
                if sn in res:
                    print(f"SEG {sn:5s} {res[sn]}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
