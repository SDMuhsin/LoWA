#!/usr/bin/env python3
"""P9a: the full-size (B 32, L 128) 12-layer step composed from the chunk runs (tb/top/chunks.py), per op class and
per phase against docs/calc/budget.py, array utilisation and DRAM traffic.

  python3 tb/top/p9a_summary.py --insitu1 build/e2e/l1_b2_s32_n2 --insitu2 build/e2e/l2_b2_s32_n1 \
      --tail12 build/e2e/l12_b2_s32_n1

DRAM traffic: every chunk run's ACCT read / write beats (32 B each, counted while the sequencer runs).  Ops simulated in
two chunks are subtracted once using their single-op runs (S6, S10, S31, S39, SB39; op 44, a LoWA SG, by the
equivalent single-SG chunk D5a); the forward body's beats (ops 3..13) are C1 - S0_2 + C2 - S6 + C3 - S10 + C4 + S10
(op 13 is the same LN_FWD shape as op 10)."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "docs", "calc"))
import budget as BU  # noqa: E402
import chunks as CH  # noqa: E402
import seg as SG  # noqa: E402

B, L = 32, 128
OUT = os.path.join(ROOT, "build", "e2e")
MHZ = 1000 / 5.5


def acct(lay, j, k):
    return CH.info(CH.cdir(OUT, lay, B, L, j, k), j)[2]


def beats(lay, j, k):
    a = acct(lay, j, k)
    return a["rbeats"], a["wbeats"]


def sub(x, y):
    return x[0] - y[0], x[1] - y[1]


def add(*xs):
    return sum(x[0] for x in xs), sum(x[1] for x in xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insitu1", required=True)
    ap.add_argument("--insitu2", required=True)
    ap.add_argument("--tail12", required=True)
    a = ap.parse_args()
    g1, g2 = CH.gaps(a.insitu1), CH.gaps(a.insitu2)
    r1, t1, _, pos1, op1, _ = CH.compose(CH.L1_CHUNKS, OUT, B, L, g1)
    rb, tb, _, posb, opb, _ = CH.compose(CH.BODY_CHUNKS, OUT, B, L, g2)
    F = pos1[CH.HEAD_FIRST] - pos1[CH.F_FIRST]
    _, s1, _ = SG.segments(a.insitu1, 1, nb0=CH.TAIL_FIRST - CH.B0_FIRST)
    _, s12, _ = SG.segments(a.tail12, 1)
    dtail = s12["tail"][1] - s1["tail"][1]
    step = t1 + 11 * (F + tb) + dtail
    print(f"1-layer step {t1:,d}; forward body {F:,d}; backward body {tb:,d}; tail growth (11 layers) {dtail:,d}")
    print(f"12-layer step {step:,d} cycles = {step / MHZ / 1e6:.3f} s at 5.5 ns ({MHZ:.1f} MHz)")
    # segments of the 12-layer step
    segs = [("pre (EMB, X0 copy, LoWA scalars)", pos1[CH.F_FIRST]), ("forward bodies x 12", 12 * F),
            ("head fwd + loss + head bwd", pos1[CH.B0_FIRST] - pos1[CH.HEAD_FIRST]), ("backward bodies x 11", 11 * tb),
            ("layer-0 backward", pos1[CH.TAIL_FIRST] - pos1[CH.B0_FIRST]),
            ("tail (grad norm, clip, AdamW) 12 layers", t1 - pos1[CH.TAIL_FIRST] + dtail)]
    assert sum(v for _, v in segs) == step
    for n, v in segs:
        print(f"SEG {n:42s} {v:15,d} {v / step:6.1%}")
    # per op class: start -> retire per op (overlapping ops counted in full, so classes add up to more than the step)
    cF = CH.classes({s: v for s, v in op1.items() if CH.F_FIRST <= s < CH.HEAD_FIRST}, "F")
    cl1, cb = CH.classes(op1, "l1"), CH.classes(opb, "body")
    cls = {}
    for src, m in ((cl1, 1), (cF, 11), (cb, 11)):
        for c, (meas, mod, _) in src.items():
            r = cls.setdefault(c, [0, 0.0])
            r[0] += m * meas
            r[1] += m * mod
    cls.setdefault("tail growth (GNORM, coef AdamW)", [dtail, 0.0])
    for c, (meas, mod) in sorted(cls.items()):
        print(f"CLASS {c:32s} measured {meas:15,d} per-op budget model {mod:15,.0f} {'ratio %.3f' % (meas / mod) if mod else ''}")
    cfg = BU.cfg_list()[0]
    st_u = BU.step(cfg, B, L, cls_prune=False)
    st_p = BU.step(cfg, B, L, cls_prune=True)
    for nm, st in (("no CLS pruning (as compiled)", st_u), ("CLS-pruned (paper)", st_p)):
        print(f"BUDGET {nm}: gemm {st['cyc_gemm'] / 1e6:.1f} M vec {st['cyc_vec'] / 1e6:.1f} M opt+misc {st['cyc_opt'] / 1e6:.1f} M "
              f"(lowa {st['cyc_lowa'] / 1e6:.1f} M hidden) overlapped {st['cyc_overlap'] / 1e6:.1f} M = {st['cyc_overlap'] / MHZ / 1e6:.2f} s; "
              f"measured {step / st['cyc_overlap'] - 1:+.2%}; MACs {st['macs'] / 1e9:.1f} G, array utilisation {st['macs'] / (256 * st['cyc_overlap']):.3f}")
    print(f"MEASURED array utilisation (MACs of the compiled program / (256 PEs x measured step)) "
          f"{st_u['macs'] / (256 * step):.3f}")
    # DRAM traffic
    try:
        C = {nm: beats(lay, j, k) for nm, lay, j, k, _ in CH.L1_CHUNKS + CH.BODY_CHUNKS}
        S = {"S6": beats(1, 6, 6), "S10": beats(1, 10, 10), "S31": beats(1, 31, 31), "S39": beats(1, 39, 39),
             "SB39": beats(2, 39, 39), "S0_2": beats(1, 0, 2)}
    except (FileNotFoundError, KeyError) as e:
        print(f"DRAM: incomplete ({e})")
        return
    l1 = sub(add(*(C[n] for n, *_ in CH.L1_CHUNKS)), add(S["S6"], S["S10"], S["S31"], S["S39"], C["D5a"]))
    body = sub(add(*(C[n] for n, *_ in CH.BODY_CHUNKS)), S["SB39"])
    fb = add(sub(C["C1"], S["S0_2"]), sub(C["C2"], S["S6"]), sub(C["C3"], S["S10"]), C["C4"], S["S10"])
    tot = add(l1, (11 * fb[0], 11 * fb[1]), (11 * body[0], 11 * body[1]))
    for nm, (r, w), cyc in (("1-layer step", l1, t1), ("forward body", fb, F), ("backward body", body, tb),
                            ("12-layer step (tail growth not counted)", tot, step)):
        by = 32 * (r + w)
        print(f"DRAM {nm:40s} read {32 * r / 1e9:8.2f} GB write {32 * w / 1e9:7.2f} GB  {by / cyc:5.2f} B/cycle = "
              f"{by / cyc * MHZ * 1e6 / 1e9:5.2f} GB/s at {MHZ:.1f} MHz ({by / cyc / 32:.1%} of the 32 B/cycle port)")
    print(f"budget.py DRAM traffic per step: {st_u['dram_total'] / 1e9:.2f} GB (no pruning), {st_p['dram_total'] / 1e9:.2f} GB (pruned)")


if __name__ == "__main__":
    main()
