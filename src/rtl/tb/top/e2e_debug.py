#!/usr/bin/env python3
"""Debug aid for run_e2e.py failures: attribute every mismatching word of a step dump to the last op (program order,
sw/compile.py ranges()) whose write range covers it, and list ops in program order with their mismatch counts.
  python3 tb/top/e2e_debug.py --layers 1 --batch 2 --seq 32 --steps 2 [--step 1]
Golden = the interpreter (re-run here).  Not a test: prints a table only."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_e2e as E  # noqa: E402

CP, isa, interp = E.CP, E.isa, E.interp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--step", type=int, default=1)
    a = ap.parse_args()
    c = CP.Cfg(a.layers, a.batch, a.seq)
    bdir = os.path.join(E.ROOT, "build", "e2e", f"l{a.layers}_b{a.batch}_s{a.seq}_n{a.steps}")
    r = CP.compile_all(c)
    s0 = int(r["inp"]["step_s"])
    items = CP.program(c, CP.EPOCHS * int(r["inp"]["n_batches"]))
    flat = [x for x in CP.flatten(items, c, None, s0) if isa.OPS[x[0]][1] != isa.UNIT_SEQ]
    wregs = [tuple(int(v, 16) * (4 if i == 0 else 4) for i, v in enumerate(l.split())) for l in open(f"{bdir}/dump.reg")]
    wregs = [(b, b + n) for b, n in wregs]
    ndump = sum(hi - lo for lo, hi in wregs) // 4
    d = E.read_hex(f"{bdir}/dump_{a.step}.hex", ndump)
    m = interp.Machine(CP.IMG_END, r["csr"]["SEED"], s0, r["csr"]["SLOT_DELTA"], 0)
    m.load(r["img"])
    m.run(r["csr"]["PROG_BASE"])
    full = np.zeros(CP.IMG_END // 4, np.uint32)
    o = 0
    for lo, hi in wregs:
        full[lo // 4:hi // 4] = d[o:o + (hi - lo) // 4]
        o += (hi - lo) // 4
    last = {}
    owner = np.full(CP.IMG_END // 4, -1, np.int32)
    for i, (name, f, _, _) in enumerate(flat):
        _, Wr = CP.ranges(name, f, c)
        for lo, hi in Wr:
            owner[lo // 4:hi // 4] = i
    bad = np.zeros(len(flat), np.int64)
    tot = np.zeros(len(flat), np.int64)
    for lo, hi in wregs:
        sl = slice(lo // 4, hi // 4)
        ow = owner[sl]
        mis = full[sl] != m.mem[sl]
        for i in np.unique(ow):
            if i >= 0:
                sel = ow == i
                tot[i] += int(sel.sum())
                bad[i] += int((mis & sel).sum())
    for i, (name, f, ii, j) in enumerate(flat):
        extra = isa.VOPS[f["vop"]] if "vop" in f else (f"m{f['m']} n{f['n']} k{f['k0']}" if "m" in f and "k0" in f else "")
        print(f"{i:4d} {name:9s} {extra:22s} last-written {tot[i]:8d} mismatches {bad[i]:8d}")


if __name__ == "__main__":
    main()

