#!/usr/bin/env python3
"""P9a: per-segment cycle split of an e2e run (tb/top/run_e2e.py output dir) from its op log.

The program (sw/compile.py) is   pre (SETCSR, EMB, DMA_COPY, LOWA_SCAL) ; LOOP up {forward body} ; head fwd + loss +
head bwd ; LOOP down (l = n-1 .. 1) {backward body} ; layer-0 backward ; tail (GNORM x 29, DMA_COPY clip, ADAMW).
The sequencer issues unit ops in program order, so the start cycles (op log 'S' events) are monotone in program
order.  A segment's duration is the start of its first op to the start of the next segment's first op (the last
segment ends at the step's IRQ), so the segment durations add up exactly to the step; ops that overlap a boundary
(e.g. a LOWA_SG running beside the next layer) are charged to the segment in which they start.
  python3 tb/top/seg.py build/e2e/l2_b32_s128_n1 [--step 1] [--json out.json]
Prints one line per segment: name, start cycle, duration, unit-op count."""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "sw"))
import isa  # noqa: E402

UIDX = {isa.UNIT_DMA: 0, isa.UNIT_GEMM: 1, isa.UNIT_VEC: 2, isa.UNIT_LOWA: 3}


def walk_seg(words):
    """program-order list of (unit, segment, op name) for the unit ops; segments as in the module docstring."""
    term = [0] * 16
    strd = [0] * 16
    pc, loop, out, nloops = 0, None, [], 0
    seg = "pre"
    for _ in range(100000):
        w0 = int(words[pc // 4])
        nsl = ((w0 >> 21) & 3) + 1
        h, name, f, pay = isa.decode_words(words[pc // 4:pc // 4 + 16 * nsl], term)
        nxt = pc + 64 * nsl
        if name == "LOOP":
            nloops += 1
            loop = dict(start=nxt, end=nxt + f["body"], left=f["count"], dir=f["dir"])
            term[isa.T_LIDX] = f["start"]
            term[1:13] = [0] * 12
        elif name == "SETCSR":
            if 1 <= f["csr"] <= 12:
                strd[f["csr"]] = f["value"]
        elif name == "END":
            return out
        elif name not in ("NOP", "FENCE"):
            if seg == "tail":
                pass
            elif loop:
                seg = ("B%d" if loop["dir"] else "F%d") % term[isa.T_LIDX]
            elif nloops == 1:
                seg = "head"
            elif nloops >= 2:
                seg = "B0"
            if name == "GNORM" and seg in ("B0", "head") and nloops >= 1:
                seg = "tail"
            out.append((UIDX[h["unit"]], seg, name))
        pc = nxt
        if loop and pc == loop["end"]:
            loop["left"] -= 1
            if loop["left"] > 0:
                s = -1 if loop["dir"] else 1
                term[isa.T_LIDX] += s
                for k in range(1, 13):
                    term[k] = (term[k] + s * strd[k]) & 0xFFFFFFFF
                pc = loop["start"]
            else:
                loop = None
    raise RuntimeError("no END")


def load_prog(d):
    regs = [ln.split() for ln in open(os.path.join(d, "dram.reg"))]
    raw = np.fromfile(os.path.join(d, "dram.bin"), ">u4")
    o = 0
    for b, n in regs:
        b, n = int(b, 16), int(n, 16)
        if b == 0:
            return raw[o:o + n].astype(np.uint32)
        o += n
    raise RuntimeError("no program region")


def segments(d, step=1, nb0=None):
    """{segment: [start, duration, nops]} for one step; nb0 = op count of the layer-0 backward (needed when the
    program has no LOOP down, i.e. 1 layer: then head and B0 are not separated by a loop)."""
    ops = walk_seg(load_prog(d))
    if nb0 is not None and not any(s.startswith("B") and s != "B0" for _, s, _ in ops):
        idx = [i for i, (_, s, _) in enumerate(ops) if s == "head"]
        for i in idx[len(idx) - nb0:]:
            ops[i] = (ops[i][0], "B0", ops[i][2])
    ev = defaultdict(list)
    for ln in open(os.path.join(d, "oplog.txt")):
        k, u, c = ln.split()
        if k == "S":
            ev[int(u)].append(int(c))
    per_unit = defaultdict(int)
    for u, _, _ in ops:
        per_unit[u] += 1
    seen = defaultdict(int)
    seg = {}
    order = []
    for u, s, _ in ops:
        i = (step - 1) * per_unit[u] + seen[u]
        seen[u] += 1
        if i >= len(ev[u]):
            continue                                   # op not started (yet)
        t = ev[u][i]
        if s not in seg:
            seg[s] = [t, 0, 0]
            order.append(s)
        seg[s][0] = min(seg[s][0], t)
        seg[s][2] += 1
    irq = None
    log = os.path.join(d, "sim.log")
    if os.path.exists(log):
        for ln in open(log):
            m = re.match(r"STEP (\d+) IRQ at cycle (\d+)", ln)
            if m and int(m.group(1)) == step:
                irq = int(m.group(2))
    order.sort(key=lambda s: seg[s][0])
    for a, b in zip(order, order[1:]):
        seg[a][1] = seg[b][0] - seg[a][0]
    if order:
        seg[order[-1]][1] = (irq - seg[order[-1]][0]) if irq else -1     # -1: step not finished
    return order, seg, ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--nb0", type=int, default=None, help="op count of the layer-0 backward (1-layer programs)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    order, seg, ops = segments(a.dir, a.step, a.nb0)
    tot = defaultdict(int)
    for _, s, _ in ops:
        tot[s] += 1
    for s in order:
        st, du, n = seg[s]
        print(f"SEG {s:6s} start {st:12d} dur {du:12d} ops {n}/{tot[s]}")
    if a.json:
        json.dump({s: seg[s] for s in order}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
