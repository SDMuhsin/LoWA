#!/usr/bin/env python3
"""P9b: full-size step energy on P9a's measured schedule (docs/p9a_report.md): every op of P9a's isolated
full-size chunks (B 32, L 128; their op logs) at the gate-level power of its op type (P9b windows, PrimePower summary).

    python3 tb/power/p9b_energy_p9a.py <p9a_full.json> <p9a e2e dir> <summary_gls_tt.txt> [<summary2> ...]

Model: chip power at any time = idle floor + the sum, over the ops running then, of (window power - floor).  The
floor is the ADAMW window minus its LoWA-engine increase.  The additivity holds where it was measured: LOWA_SG beside
GEMM_ACC2 raised u_lw by the same 0.017 W as LOWA_SG alone, and u_gm was unchanged.  The step is P9a's composition,
step(12) = step(1) + 11 x (forward body + backward body) + tail growth, with the energy of each part from the ops of
the chunks that make it up (P9a rows: a chunk [j..k] contributes its ops j..k-1 when the next chunk starts at k, else
j..k).  Op types -> windows:
  GEMM: batched over heads (nb1 > 1: attention) -> g_att; FFN1 (GELU epilogue) -> g_ffn2; every other GEMM, dense
        forward / backward dX / head / GEMM_ACC2 -> mean of the dense GEMM windows
  VEC:  EMB, SMAX_FWD, LN_FWD, LN_BWD, SMAX_REC, SMAX_BWD, GN_* -> their windows (2-layer windows for the backward
        ones); element-wise ops and LOSS -> the mean of the vector windows
  LOWA_SG -> l2_sg; ADAMW -> o_adamw; DMA_COPY, LOWA_SCAL -> floor (no window; 4.9 M of 319 M cycles)"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "sw"))
sys.path.insert(0, HERE)
import compile as CP  # noqa: E402
import isa  # noqa: E402
import p9b_energy as PE  # noqa: E402

FMHZ = 181.818e6


def static_ops(layers, B=32, L=128):
    """unit ops in static program order (loop bodies once), as run_e2e --nop-before / --stop-after count them"""
    c = CP.Cfg(layers, B, L)
    items = CP.program(c, 1000)
    out = []
    for it in items:
        ops = it.body_fn(list(it.layers)[0], lambda v, sel: v) if isinstance(it, CP.Loop) else [it]
        for op in ops:
            if isa.OPS[op.name][1] != isa.UNIT_SEQ:
                out.append((op.name, CP.resolve(op.f, {isa.T_STEP: 0})))
    return out


def op_window(name, f):
    if name in ("GEMM", "GEMM_ACC2"):
        if f.get("nb1", 1) > 1:
            return "g_att"
        if f.get("ep_gelu", 0):
            return "g_ffn2"
        return "gdense"
    if name in ("VEC", "LOSS", "GNORM"):
        v = isa.VOPS[f["vop"]]
        return {"EMB": "v_emb", "SMAX_FWD": "v_smax", "LN_FWD": "v_ln", "LN_BWD": "v2_lnb", "SMAX_REC": "v2_smaxr",
                "SMAX_BWD": "v2_smaxb", "GN_ROWS": "v_gnorm", "GN_FIN": "v_gnorm"}.get(v, "vmean")
    return {"LOWA_SG": "l2_sg", "ADAMW": "o_adamw"}.get(name, "floor")


def chunk_ops(ddir, j, k, ops):
    """[(op index, window, cycles)] of the ops j..k (k None: to the end) of one chunk, from its op log"""
    ev = [ln.split() for ln in open(os.path.join(ddir, "oplog.txt")) if len(ln.split()) == 3]
    unit_of = {isa.UNIT_DMA: 0, isa.UNIT_GEMM: 1, isa.UNIT_VEC: 2, isa.UNIT_LOWA: 3}
    idx = list(range(j, (k if k is not None else len(ops) - 1) + 1))
    queue = {u: [i for i in idx if unit_of[isa.OPS[ops[i][0]][1]] == u] for u in range(4)}
    st, out = {}, []
    for t, u, cyc in ev:
        u, cyc = int(u), int(cyc)
        if t == "S":
            st[u] = (queue[u].pop(0), cyc)
        elif t == "R" and u in st:
            i, s = st.pop(u)
            out.append((i, op_window(*ops[i]), cyc - s))
    return out


def main():
    js, edir = sys.argv[1], sys.argv[2]
    tot = {}
    for sfile in sys.argv[3:]:
        t, _, h = PE.parse(sfile, ("g_qkv", "k_ffnb", "k_qkv"))
        tot.update(t)
        if "o_adamw" in t:
            hier = h
    floor = tot["o_adamw"] - (hier["o_adamw"]["u_lw"][3] - hier["g_k"]["u_lw"][3])
    dense = [w for w in ("g_qkv2", "g_k", "g_o", "g_ffn2f", "g2_ffn2dx", "g2_ffn1dx", "g2_odx") if w in tot]
    vecs = [w for w in ("v_smax", "v_ln", "v2_lnb", "v2_smaxr", "v2_smaxb") if w in tot]
    P = dict(tot, gdense=sum(tot[w] for w in dense) / len(dense), vmean=sum(tot[w] for w in vecs) / len(vecs),
             floor=floor)
    d = json.load(open(js))
    res = {}
    for prog, key in ((1, "l1"), (2, "body")):
        ops = static_ops(prog)
        rows = d[key]["rows"]
        allops = []
        for n, (name, j, k, _, _) in enumerate(rows):
            nxt = rows[n + 1][1] if n + 1 < len(rows) else None
            dd = os.path.join(edir, f"l{prog}_b32_s128_n1_nop{j}" + (f"_stop{k}" if k is not None else ""))
            co = chunk_ops(dd, j, k, ops)
            keep = range(j, k) if (nxt is not None and nxt == k) else range(j, (k if k is not None else 10 ** 9) + 1)
            allops += [x for x in co if x[0] in keep]
        res[key] = allops
    E_l1 = P["floor"] * d["l1"]["total"] + sum((P[w] - floor) * c for _, w, c in res["l1"])
    f0 = [x for x in res["l1"] if 3 <= x[0] <= 13]
    E_f0 = P["floor"] * d["F0"] + sum((P[w] - floor) * c for _, w, c in f0)
    E_body = P["floor"] * d["body"]["total"] + sum((P[w] - floor) * c for _, w, c in res["body"])
    E12 = E_l1 + 11 * (E_f0 + E_body) + P["floor"] * d["dtail"]
    E12 /= FMHZ
    T = d["step12"] / FMHZ
    # time per window type over the 12-layer step (busy cycles, overlaps counted in each op)
    busy = {}
    for part, mult in ((res["l1"], 1), (f0, 11), (res["body"], 11)):
        for _, w, c in part:
            busy[w] = busy.get(w, 0) + mult * c
    print(f"floor {floor:.4f} W; dense GEMM {P['gdense']:.4f} W ({', '.join(dense)}); vector mean {P['vmean']:.4f} W")
    print("busy cycles per window type over the 12-layer step (M): " +
          ", ".join(f"{w} {c / 1e6:.1f}" for w, c in sorted(busy.items(), key=lambda x: -x[1])))
    print(f"parts (J): step(1) {E_l1 / FMHZ:.3f}, forward body {E_f0 / FMHZ:.3f}, backward body {E_body / FMHZ:.3f}")
    print(f"P9A STEP: {d['step12']:,} cycles = {T:.2f} s, energy {E12:.2f} J, average {E12 / T:.3f} W")


if __name__ == "__main__":
    main()
