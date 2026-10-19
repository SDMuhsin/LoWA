#!/usr/bin/env python3
"""P9b: per-window and per-phase chip power from the PrimePower summary of the gate-level SAIF windows
(build/tsmc65/summary/ptgls_<tag>_tt.txt, fetched by `sync.sh results`), and the energy of a training step.

    python3 tb/power/p9b_energy.py <summary.txt> [--gemm M --vec M --lowa M --opt M] [--fmhz 181.818]

Phases (window names, tb/power/win/p9b_plan.txt): GEMM = g_qkv2 / g_ffn2 / g_ffnb2 weighted by the full-size MAC mix
of budget.py's GEMM list (forward dense GEMMs other than FFN1 and forward attention -> g_qkv2; FFN1 -> g_ffn2; backward
dX GEMMs and backward attention -> g_ffnb2; head GEMMs ignored), vector = v_* weighted by the full-size
vector-cycle mix of docs/calc/budget.py (SMAX_FWD, LN_FWD, SMAX_REC + SMAX_BWD, LN_BWD; v_emb and v_gnorm
are reported but carry no full-size weight: EMB runs once per step, GNORM in the optimiser tail), LoWA = l_sg,
optimiser = o_adamw.  Phase cycles default to budget.py's full-size step (full16, 12 layers, B 32, L 128):
GEMM 2,777.9 M, vector 506.7 M, LoWA 261.8 M (beside the GEMMs: overlapped), optimiser + misc 6.0 M.
Energy = P_gemm t_gemm + P_vec t_vec + P_opt t_opt + dP_lowa t_lowa, where dP_lowa = u_lw power in the LOWA_SG window
minus u_lw power in the GEMM windows (the LoWA engine's extra power while it runs beside a GEMM)."""
import argparse
import re
from collections import defaultdict

# budget.py VEC_RATE (cycles per element per lane) and the per-layer element counts of the full-size step
D, NTOK, H, L, B = 768, 32 * 128, 12, 128, 32
E = B * H * L * L
R = {"LN_FWD": 13.51, "LN_BWD": 23.75, "SMAX_FWD": 23.44, "SMAX_REC": 22.4, "SMAX_BWD": 14.69}
VMIX = {"v_smax": R["SMAX_FWD"] * E, "v_ln": 3 * R["LN_FWD"] * NTOK * D,
        "v_smaxb": (R["SMAX_REC"] + R["SMAX_BWD"]) * E, "v_lnb": 2 * R["LN_BWD"] * NTOK * D}


# full-size GEMM MACs per class (budget.py step(): 12 layers, B 32, L 128, CLS-pruned last layer)
def gemm_mix(B=32, L=128, D=768, H=12, DH=64, F=3072, NL=12):
    N = B * L
    m = {"g_qkv2": 0, "g_ffn2": 0, "g_ffnb2": 0}
    for layer in range(NL):
        last = layer == NL - 1
        Nq, Lq = (B, 1) if last else (N, L)
        m["g_qkv2"] += Nq * D * D * 2 + N * D * D * 2 + Nq * D * F + 2 * Lq * L * DH * B * H
        m["g_ffn2"] += Nq * F * D
        m["g_ffnb2"] += Nq * F * D + Nq * D * F + Nq * D * D + 4 * Lq * L * DH * B * H
        if layer > 0:
            m["g_ffnb2"] += L * DH * Lq * B * H + N * D * 3 * D
    return m


def parse(path, exclude=()):
    tot, comp, hier, cyc = {}, defaultdict(dict), defaultdict(dict), {}
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
            # power_hier <win> <inst> (<ref>) internal switching leakage total pct
            # (level 2: the units come first, their children after them; keep the first line per name = the unit)
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
    ap.add_argument("--exclude", default="g_qkv", help="windows to ignore (g_qkv: its SAIF was never written)")
    a = ap.parse_args()
    tot, comp, hier = parse(a.summary, a.exclude.split(","))
    units = ["u_gm", "u_vu", "u_lw", "u_mm", "u_seq"]
    print("| window | total W | switching | internal | leakage | clock network | " + " | ".join(units) + " |")
    print("|---" * (6 + len(units)) + "|")
    for w in sorted(tot):
        c = comp[w]
        print(f"| {w} | **{tot[w]:.4f}** | {c.get('Net_Switching', 0):.4f} | {c.get('Cell_Internal', 0):.4f} | {c.get('Cell_Leakage', 0):.4f} | {c.get('clock', 0):.4f} | "
              + " | ".join(f"{hier[w].get(u, [0, 0, 0, 0])[3]:.4f}" for u in units) + " |")
    GM = gemm_mix()
    g = [w for w in GM if w in tot]
    v = [w for w in VMIX if w in tot]
    if not g or not v or "o_adamw" not in tot or "l_sg" not in tot:
        print("phases incomplete:", sorted(tot))
        return
    wg = sum(GM[w] for w in g)
    pg = sum(tot[w] * GM[w] for w in g) / wg
    wv = sum(VMIX[w] for w in v)
    pv = sum(tot[w] * VMIX[w] for w in v) / wv
    po = tot["o_adamw"]
    lw_g = sum(hier[w]["u_lw"][3] * GM[w] for w in g) / wg
    dpl = hier["l_sg"]["u_lw"][3] - lw_g
    f = a.fmhz * 1e6
    tg, tv, tl, to = a.gemm * 1e6 / f, a.vec * 1e6 / f, a.lowa * 1e6 / f, a.opt * 1e6 / f
    T = tg + tv + to
    Eg, Ev, Eo, El = pg * tg, pv * tv, po * to, dpl * tl
    Et = Eg + Ev + Eo + El
    print()
    print(f"phase cycles from {a.src}: GEMM {a.gemm} M, vector {a.vec} M, LoWA {a.lowa} M (overlapped), opt {a.opt} M")
    print(f"GEMM  P = {pg:.4f} W (full-size MAC mix {', '.join(f'{w} {tot[w]:.4f} x {GM[w] / wg:.3f}' for w in g)}), t = {tg:.3f} s, E = {Eg:.3f} J")
    print(f"vector P = {pv:.4f} W (full-size mix {', '.join(f'{w} {VMIX[w] / wv:.3f}' for w in v)}), t = {tv:.3f} s, E = {Ev:.3f} J")
    print(f"LoWA  dP = {dpl:.4f} W (u_lw {hier['l_sg']['u_lw'][3]:.4f} in l_sg - {lw_g:.4f} in GEMM; l_sg total {tot['l_sg']:.4f} W), t = {tl:.3f} s, E = {El:.3f} J")
    print(f"opt   P = {po:.4f} W, t = {to:.3f} s, E = {Eo:.3f} J")
    print(f"STEP: {T:.2f} s, energy {Et:.2f} J, average power {Et / T:.4f} W")


if __name__ == "__main__":
    main()
