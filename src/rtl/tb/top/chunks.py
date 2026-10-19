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
    g = {}
    for a, b in zip(tl, tl[1:]):                 # dynamic neighbours whose static indices are k, k + 1
        if b[1] == a[1] + 1 and a[6] is not None and b[5] is not None:
            g.setdefault(a[1], b[5] - a[6])
    return g


def compose(chunks, out, B, L, gap):
    """rows, total, shared-op checks, pos = {static op: composed start cycle relative to the step start}"""
    rows, tot, checks, pos, per_op, beats = [], 0, [], {}, {}, {}
    prev = None
    for idx, (nm, lay, j, k, mode) in enumerate(chunks):
        d = cdir(out, lay, B, L, j, k)
        if not os.path.exists(os.path.join(d, "sim.log")):
            rows.append((nm, j, k, None, "missing"))
            tot = None
            continue
        ops, irq, acct = info(d, j)
        if acct:
            beats[nm] = (acct["rbeats"], acct["wbeats"], acct["cyc"])
        for s_, (S_, R_) in ops.items():
            if S_ is not None and R_ is not None and s_ not in per_op:
                per_op[s_] = (R_ - S_, d)
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
    return rows, tot, checks, pos, per_op, beats


def op_models(d, j):
    """{static op: (class, budget.py per-op model cycles)} of the real ops of a chunk program"""
    sys.path.insert(0, os.path.join(ROOT, "sw"))
    sys.path.insert(0, os.path.join(ROOT, "tb", "ctrl"))
    import acct as AC
    import isa
    import run_tests as CT
    words = SG.load_prog(d)
    pc, k, out = 0, -1, {}
    while True:
        w0 = int(words[pc // 4])
        nsl = ((w0 >> 21) & 3) + 1
        h = isa.header(w0)
        if (w0 & 0xFF) == isa.OPS["END"][0]:
            return out
        if h["unit"] != isa.UNIT_SEQ:
            k += 1
            u = SG.UIDX[h["unit"]]
            _, _, _, pay = isa.decode_words(words[pc // 4:pc // 4 + 16 * nsl], [0] * 16)
            _, f = AC.fields(u, pay[:CT.NPAY[u]])
            out[k + j] = AC.model(u, f, None)
        pc += 64 * nsl


def classes(per_op, nm):
    """per op class: [measured start->retire, budget model] over the ops of one composition (each op once)"""
    cls = {}
    cache = {}
    for s_, (dur, d) in sorted(per_op.items()):
        j = int(re.search(r"_nop(\d+)", d).group(1))
        if d not in cache:
            cache[d] = op_models(d, j)
        c, mc = cache[d].get(s_, ("?", 0))
        c = c.split()[0] if c.startswith(("GEMM", "VEC")) else c
        r = cls.setdefault(c, [0, 0.0, []])
        r[0] += dur
        r[1] += mc
        r[2].append(s_)
    return cls


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
        rows, tot, checks, pos, per_op, beats = compose(ch, a.out, a.batch, a.seq, g)
        for r in rows:
            print(f"CHUNK {r[0]:4s} ops {r[1]}..{r[2]} contribution {r[3]} {r[4]}")
        for j, d1, d2 in checks:
            print(f"SHARED op {j}: end of previous chunk {d1} / start of next chunk {d2} {'EQUAL' if d1 == d2 else 'DIFF %+d' % (d2 - d1)}")
        print(f"TOTAL {nm} {tot}")
        res[nm] = dict(rows=rows, total=tot, checks=checks, beats=beats)
        res[nm + "_classes"] = classes(per_op, nm)
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
    for nm in ("l1", "body"):
        for c, (m, b, ops) in sorted(res[nm + "_classes"].items()):
            print(f"CLASS {nm:4s} {c:10s} ops {len(ops):2d} measured {m:12,d} budget-model {b:14,.0f} ratio {m / b if b else 0:.3f}")
        bt = res[nm]["beats"]
        rb, wb, cy = (sum(v[i] for v in bt.values()) for i in range(3))
        if cy:
            print(f"DRAM {nm}: chunk runs {cy:,d} cycles, R beats {rb:,d}, W beats {wb:,d} = {(rb + wb) * 32 / cy:.2f} B/cycle "
                  f"(32 B/cycle peak); shared ops counted twice")
    l1, body = res["l1"]["total"], res["body"]["total"]
    if l1 and body and "F0" in res:
        per_layer = res["F0"] + body
        res["step12"] = l1 + 11 * per_layer
        sys.path.insert(0, os.path.join(ROOT, "docs", "calc"))
        import budget as BU
        cfg = BU.cfg_list()[0]
        nl0 = BU.NL
        bud = {}
        for n in (1, 2, 12):
            BU.NL = n
            bud[n] = BU.step(cfg, a.batch, a.seq, cls_prune=False)["cyc_overlap"]
        BU.NL = nl0
        pruned = BU.step(cfg, a.batch, a.seq, cls_prune=True)["cyc_overlap"]
        print(f"MEASURED 1-layer step {l1:,d}  budget.py (no CLS pruning) {bud[1]:,.0f}  delta {l1 / bud[1] - 1:+.2%}")
        print(f"MEASURED per-layer increment (forward body F0 {res['F0']:,d} + backward body {body:,d}) {per_layer:,d}  "
              f"budget.py {bud[2] - bud[1]:,.0f}  delta {per_layer / (bud[2] - bud[1]) - 1:+.2%}")
        print(f"COMPOSED 12-layer step = 1-layer + 11 x per-layer = {res['step12']:,d}  budget.py no pruning {bud[12]:,.0f} "
              f"({res['step12'] / bud[12] - 1:+.2%}); budget.py CLS-pruned (paper) {pruned:,.0f} ({res['step12'] / pruned - 1:+.2%})")
        for mhz in (181.818,):
            print(f"STEP TIME at {mhz} MHz: {res['step12'] / mhz / 1e6:.2f} s")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
