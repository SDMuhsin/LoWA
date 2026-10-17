#!/usr/bin/env python3
"""P7a cycle accounting of an e2e run (tb/top/run_e2e.py output dir): every unit op's measured start -> engine done ->
retire (oplog.txt, matched per unit in program order with the reference walk of the program) against the
docs/calc/budget.py model of the same op at the same shape, summed per op class; plus the per-step ACCT counters.
  python3 tb/top/acct.py build/e2e/l12_b2_s32_n2 --layers 12 --batch 2 --seq 32 [--step 1]"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "sw"))
sys.path.insert(0, os.path.join(ROOT, "docs", "calc"))
sys.path.insert(0, os.path.join(ROOT, "tb", "ctrl"))
import budget as BU  # noqa: E402
import isa  # noqa: E402
import run_tests as CT  # noqa: E402  (tb/ctrl: the reference program walk)

CFG = BU.cfg_list()[0]                                   # full16
VMAP = {"LN_FWD": "LN_FWD", "LN_BWD": "LN_BWD", "SMAX_FWD": "SMAX_FWD", "SMAX_REC": "SMAX_REC", "SMAX_BWD": "SMAX_BWD",
        "EMB": "EMB", "CE": "CE", "MSE": "CE", "GN_ROWS": "GN", "GN_FIN": "GN", "E_TANH": "TANH", "E_TANHB": "TANH"}


def fields(u, pay):
    name = {1: "GEMM", 2: "VEC", 3: "LOWA_SG", 0: "DMA_COPY"}[u]
    ws = np.array(pay, np.uint32)
    f = {}
    for i, spec in enumerate(isa.UNIT_FIELDS[{1: isa.UNIT_GEMM, 2: isa.UNIT_VEC, 3: isa.UNIT_LOWA, 0: isa.UNIT_DMA}[u]]):
        if isinstance(spec, str):
            f[spec] = int(ws[i])
        else:
            for nm, lsb, w in spec:
                f[nm] = (int(ws[i]) >> lsb) & ((1 << w) - 1)
    return name, f


def model(u, f, hdr_op):
    """(class, budget cycles) of one op"""
    if u == 1:
        M, N, K, nb = f["m"], f["n"], f["k0"] + f["k1"] + f["k2"], f["nb1"] * f["nb2"]
        ep = BU.EP_RATE_GELU if f["ep_gelu"] else BU.EP_RATE_GELU_BWD if f["ep_gelu_bwd"] else None
        g = BU.gemm_geom(CFG)
        cyc = max(BU.gemm_cycles(CFG, M, N, K, ep, nb), (g["bpm"] * M * N * K + 4 * M * N) * nb / (BU.DRAM_BPC * BU.EFF_GEMM))
        cls = "GEMM W-stream (K,N>=768)" if min(K, N) >= 768 or max(K, N) >= 3072 else "GEMM attention/head/small"
        return cls, cyc
    if u == 2:
        vop = isa.VOPS[f["vop"]]
        key = VMAP.get(vop, "ELEM")
        return f"VEC {vop}", BU.vec_cycles(key, f["rows"], f["cols"])
    if u == 3:
        op = f["op"]
        if op == 0:
            return "LOWA_SG", f["len"] * (BU.K_ATOM * BU.WIN + BU.D + BU.LOWA_TOK_OVH)
        if op == 2:
            return "ADAMW", f["len"] * BU.OPT_CYC_EL
        return "LOWA_SCAL", 0
    words = f["n0"] * f["n1"] * f["n2"]
    return "DMA_COPY", words * 4 / BU.DRAM_BPC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--step", type=int, default=1)
    a = ap.parse_args()
    prog = None
    regs = [l.split() for l in open(os.path.join(a.dir, "dram.reg"))]
    raw = np.fromfile(os.path.join(a.dir, "dram.bin"), ">u4")
    o = 0
    for b, n in regs:
        b, n = int(b, 16), int(n, 16)
        if b == 0:
            prog = raw[o:o + n].astype(np.uint32)
        o += n
    walk = [w for w in CT.walk(prog, 0) if w[0] == "U"]
    per_unit = defaultdict(list)
    for w in walk:
        per_unit[w[1]].append(w)
    ev = defaultdict(lambda: defaultdict(list))
    for ln in open(os.path.join(a.dir, "oplog.txt")):
        k, u, c = ln.split()
        ev[int(u)][k].append(int(c))
    rows = defaultdict(lambda: [0, 0, 0.0, 0])               # class -> [n, start->done, model, start->retire]
    for u, ops in per_unit.items():
        n = len(ops)
        S, D, R = ev[u]["S"][(a.step - 1) * n:a.step * n], ev[u]["D"], ev[u]["R"][(a.step - 1) * n:a.step * n]
        if u != 0:
            D = D[(a.step - 1) * n:a.step * n]
        for i, w in enumerate(ops):
            if i >= min(len(S), len(R), len(D) if u != 0 else len(R)):
                print(f"unit {u}: only {i} of {n} ops logged (oplog incomplete)")
                break
            _, f = fields(u, w[2])
            cls, mc = model(u, f, w[3]["op"])
            r = rows[cls]
            r[0] += 1
            r[1] += (D[i] - S[i]) if u != 0 else (R[i] - S[i])
            r[2] += mc
            r[3] += R[i] - S[i]
    print(f"{'class':30s} {'ops':>4s} {'measured':>12s} {'budget':>12s} {'ratio':>6s} {'retire':>12s}")
    tot = defaultdict(lambda: [0, 0.0])
    for cls in sorted(rows):
        n, meas, mod, ret = rows[cls]
        print(f"{cls:30s} {n:4d} {meas:12,d} {mod:12,.0f} {meas / mod if mod else 0:6.2f} {ret:12,d}")
        un = cls.split()[0].split("_")[0]
        tot[un][0] += meas
        tot[un][1] += mod
    for un, (m, b) in tot.items():
        print(f"TOTAL {un:10s} measured {m:12,d} budget {b:12,.0f} ratio {m / b if b else 0:.3f}")
    st = BU.step(CFG, a.batch, a.seq, cls_prune=False)
    st_p = BU.step(CFG, a.batch, a.seq, cls_prune=True)
    print(f"budget.py step (B {a.batch}, L {a.seq}; no CLS pruning, as compiled): gemm {st['cyc_gemm']:,.0f} vec "
          f"{st['cyc_vec']:,.0f} lowa {st['cyc_lowa']:,.0f} opt+misc {st['cyc_opt']:,.0f} overlapped {st['cyc_overlap']:,.0f}"
          f" (CLS-pruned: {st_p['cyc_overlap']:,.0f})")
    res = json.load(open(os.path.join(a.dir, "result.json"))) if os.path.exists(os.path.join(a.dir, "result.json")) else {}
    for ln in res.get("log", []):
        if ln.startswith(f"ACCT {a.step} "):
            print(ln)


if __name__ == "__main__":
    main()
