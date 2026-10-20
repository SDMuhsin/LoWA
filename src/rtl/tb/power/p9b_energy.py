#!/usr/bin/env python3
"""P9b: per-window and per-phase chip power from the PrimePower summary of the gate-level SAIF windows
(build/tsmc65/pt_gls/<tag>/summary_gls_tt.txt), and the energy of a training step.

    python3 tb/power/p9b_energy.py <summary.txt> [--gemm M --vec M --lowa M --opt M] [--fmhz 181.818] [--src text]

Window names: tb/power/win/p9b_plan*.txt.  Phase power = weighted mean of window powers:
  GEMM    forward dense (Q / K / V / O / FFN2: g_qkv2, g_k, g_o, g_ffn2f), FFN1 (g_ffn2), attention GEMMs forward and
          backward (g_att), backward dX (g2_ffn2dx, g2_ffn1dx, g2_odx: layer 0 of the 2-layer program, dense
          gradients), weighted by their full-size MACs (budget.py's GEMM list: 12 layers, B 32, L 128, CLS-pruned)
  vector  SMAX_FWD (v_smax), LN_FWD (v_ln), SMAX_REC (v2_smaxr), SMAX_BWD (v2_smaxb), LN_BWD (v2_lnb), weighted by
          budget.py's full-size vector cycles (VEC_RATE x elements)
  LoWA    LOWA_SG (l2_sg) runs beside the GEMMs: its extra power dP = u_lw in l2_sg - u_lw in the GEMM windows
  opt     ADAMW (o_adamw)
Phase cycles default to budget.py's full-size step (docs/p8_report.md section 8): GEMM 2,777.9 M, vector 506.7 M,
LoWA 261.8 M (overlapped with GEMM), optimiser + misc 5.97 M.  E = P_gemm t_gemm + P_vec t_vec + P_opt t_opt +
dP_lowa t_lowa.  Also prints the time-weighted average over the reduced one-layer step (B 2, L 32) that the 1-layer
windows come from, with each op at the power of its own window type."""
import argparse
import re
from collections import defaultdict

D, F, H, DH, NL = 768, 3072, 12, 64, 12
B, L = 32, 128
NTOK, E = B * L, B * H * L * L
R = {"LN_FWD": 13.51, "LN_BWD": 23.75, "SMAX_FWD": 23.44, "SMAX_REC": 22.4, "SMAX_BWD": 14.69}
VCLS = {"v_smax": R["SMAX_FWD"] * E, "v_ln": 3 * R["LN_FWD"] * NTOK * D, "v2_smaxr": R["SMAX_REC"] * E,
        "v2_smaxb": R["SMAX_BWD"] * E, "v2_lnb": 2 * R["LN_BWD"] * NTOK * D}
GWIN = {"fwd_dense": ["g_qkv2", "g_k", "g_o", "g_ffn2f"], "fwd_ffn1": ["g_ffn2"], "attn": ["g_att"],
        "bwd_dx": ["g2_ffn2dx", "g2_ffn1dx", "g2_odx"]}
# reduced one-layer step (RTL op log, docs/p9b_report.md): (window, cycles) per op, in program order
RED = [("v_emb", 130177), ("idle", 37658), ("idle", 347), ("g_qkv2", 158939), ("g_k", 158766), ("g_qkv2", 158937),
       ("g_att", 16252), ("v_smax", 51163), ("g_att", 16106), ("g_o", 165082), ("v_ln", 56373), ("g_ffn2", 651280),
       ("g_ffn2f", 619546), ("v_ln", 56367), ("vsmall", 2741), ("g_att", 76946), ("vsmall", 19680), ("vsmall", 2743),
       ("g_att", 3645), ("vsmall", 540), ("g_att", 3236 + 370 + 3236), ("vsmall", 2759 + 4895), ("g_att", 152332),
       ("g_att", 3219), ("g_att", 89153), ("vsmall", 2744), ("v_lnb", 96672), ("vsmall", 21139), ("g_ffnb2", 652000),
       ("g_ffnb2", 619546), ("v_lnb", 96675), ("vsmall", 21139), ("g_ffnb2", 158766), ("g_att", 16252),
       ("v_smaxb", 41959), ("g_att", 16252), ("v_smaxb", 34638), ("vsmall", 10397), ("g_att", 16106 + 16106),
       ("l_sg", 179920), ("l_sg", 179919), ("v_gnorm", 1275 + 1271 + 191831 + 2831 + 2839 + 247 + 209),
       ("idle", 90), ("o_adamw", 4815), ("o_adamw", 5329393)]
RED_TOTAL = 10359455


def gemm_macs():
    m = defaultdict(int)
    N = NTOK
    for layer in range(NL):
        last = layer == NL - 1
        Nq, Lq = (B, 1) if last else (N, L)
        m["fwd_dense"] += Nq * D * D + 2 * N * D * D + Nq * D * D + Nq * D * F     # q, k, v, o, ffn2
        m["fwd_ffn1"] += Nq * F * D
        m["attn"] += (2 + 4) * Lq * L * DH * B * H                                  # qk, pv; recompute, dP, dV, dQ
        m["bwd_dx"] += Nq * F * D + Nq * D * F + Nq * D * D                         # ffn2_dx, ffn1_dx, o_dx
        if layer > 0:
            m["attn"] += L * DH * Lq * B * H                                         # dK
            m["bwd_dx"] += N * D * 3 * D                                             # qkv_dx
    return m


def parse(path, exclude=()):
    tot, comp, hier = {}, defaultdict(dict), defaultdict(dict)
    for ln in open(path):
        p = ln.split()
        if len(p) > 1 and p[1] in exclude:
            continue
        if ln.startswith("power ") and "Total Power" in ln:
            tot[p[1]] = float(p[p.index("=") + 1])
        elif ln.startswith("power ") and re.search(r"(Net Switching|Cell Internal|Cell Leakage) Power", ln):
            comp[p[1]][p[2] + "_" + p[3]] = float(p[p.index("=") + 1])
        elif ln.startswith("power ") and p[2] == "clock_network":
            comp[p[1]]["clock"] = float(p[6])
        elif ln.startswith("power_hier "):
            # power_hier <win> <inst> (<ref>) internal switching leakage total pct; level 2: the units come first
            if p[2] not in hier[p[1]]:
                hier[p[1]][p[2]] = [float(x) for x in p[4:8]]
    return tot, comp, hier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--gemm", type=float, default=2777.9)
    ap.add_argument("--vec", type=float, default=506.7)
    ap.add_argument("--lowa", type=float, default=261.8)
    ap.add_argument("--opt", type=float, default=5.97)
    ap.add_argument("--fmhz", type=float, default=181.818)
    ap.add_argument("--src", default="budget.py (docs/p8_report.md section 8)")
    ap.add_argument("--exclude", default="g_qkv,k_ffnb,k_qkv", help="windows to ignore (g_qkv: no SAIF written; k_*: check runs)")
    a = ap.parse_args()
    tot, comp, hier = parse(a.summary, a.exclude.split(","))
    units = ["u_gm", "u_vu", "u_lw", "u_mm", "u_seq"]
    print("| window | total W | switching | internal | leakage | clock network | " + " | ".join(units) + " |")
    print("|---" * (6 + len(units)) + "|")
    for w in tot:
        c = comp[w]
        print(f"| {w} | **{tot[w]:.3f}** | {c.get('Net_Switching', 0):.3f} | {c.get('Cell_Internal', 0):.3f} | "
              f"{c.get('Cell_Leakage', 0):.3f} | {c.get('clock', 0):.3f} | "
              + " | ".join(f"{hier[w].get(u, [0, 0, 0, 0])[3]:.3f}" for u in units) + " |")
    mean = lambda ws, k=None: sum((tot[w] if k is None else hier[w][k][3]) for w in ws) / len(ws)
    GM = gemm_macs()
    miss = [w for ws in GWIN.values() for w in ws if w not in tot] + [w for w in VCLS if w not in tot]
    miss += [w for w in ("l2_sg", "o_adamw") if w not in tot]
    if miss:
        print("missing windows:", miss)
    gcls = {k: [w for w in ws if w in tot] for k, ws in GWIN.items()}
    gcls = {k: ws for k, ws in gcls.items() if ws}
    wg = sum(GM[k] for k in gcls)
    pg = sum(mean(ws) * GM[k] for k, ws in gcls.items()) / wg
    lw_g = sum(mean(ws, "u_lw") * GM[k] for k, ws in gcls.items()) / wg
    vv = [w for w in VCLS if w in tot]
    wv = sum(VCLS[w] for w in vv)
    pv = sum(tot[w] * VCLS[w] for w in vv) / wv
    lsg = "l2_sg" if "l2_sg" in tot else "l_sg"
    po = tot["o_adamw"]
    dpl = hier[lsg]["u_lw"][3] - lw_g
    f = a.fmhz * 1e6
    tg, tv, tl, to = (x * 1e6 / f for x in (a.gemm, a.vec, a.lowa, a.opt))
    T = tg + tv + to
    Eg, Ev, Eo, El = pg * tg, pv * tv, po * to, dpl * tl
    Et = Eg + Ev + Eo + El
    print()
    print(f"phase cycles from {a.src}: GEMM {a.gemm} M, vector {a.vec} M, LoWA {a.lowa} M (beside GEMM), opt {a.opt} M")
    print("GEMM classes (full-size MAC share): " + "; ".join(
        f"{k} {GM[k] / wg:.3f}: {', '.join(f'{w} {tot[w]:.3f}' for w in ws)} -> {mean(ws):.3f}" for k, ws in gcls.items()))
    print(f"GEMM   P = {pg:.3f} W, t = {tg:.2f} s, E = {Eg:.2f} J")
    print(f"vector P = {pv:.3f} W (cycle share {', '.join(f'{w} {tot[w]:.3f} x {VCLS[w] / wv:.3f}' for w in vv)}), "
          f"t = {tv:.2f} s, E = {Ev:.2f} J")
    print(f"LoWA  dP = {dpl:.4f} W (u_lw {hier[lsg]['u_lw'][3]:.4f} in {lsg} - {lw_g:.4f} in GEMM; {lsg} total "
          f"{tot[lsg]:.3f} W), t = {tl:.2f} s beside GEMM, E = {El:.3f} J")
    print(f"opt    P = {po:.3f} W, t = {to:.3f} s, E = {Eo:.3f} J")
    print(f"FULL STEP: {T:.2f} s, energy {Et:.2f} J, average power {Et / T:.3f} W "
          f"(GEMM {Eg / Et:.1%}, vector {Ev / Et:.1%}, LoWA {El / Et:.1%}, opt {Eo / Et:.1%})")
    # reduced one-layer step, time-weighted (small vector ops at the vector mean, gaps and DMA at the idle floor)
    floor = min(tot[w] for w in ("o_adamw", "l_sg") if w in tot)
    pw = dict(tot, idle=floor, vsmall=sum(tot[w] for w in ("v_ln", "v_lnb", "v_smax", "v_smaxb")) / 4)
    cyc = sum(c for _, c in RED)
    er = sum(pw[w] * c for w, c in RED) + floor * (RED_TOTAL - cyc)
    print(f"REDUCED STEP (1 layer, B 2, L 32, {RED_TOTAL:,} cycles = {RED_TOTAL / f * 1e3:.1f} ms): average "
          f"{er / RED_TOTAL:.3f} W, energy {er / f * 1e3:.1f} mJ (ADAMW of the head is {5329393 / RED_TOTAL:.0%} of it)")


if __name__ == "__main__":
    main()
