#!/usr/bin/env python
"""Hardware cost of an adapter SWAP on merged weights, LoWA (code name SGWA)
against every frequency-domain baseline of the paper.

  CUDA_VISIBLE_DEVICES=0 env/bin/python scripts/hw_cost.py

Deterministic (fixed seeds).  Writes results/sgwa/HW_cost.md and the paper's
table llmdocs/paper/v1_lowa_iscas27/tables/hwcost.tex.  Every number in the
paper's "Hardware cost" paragraph comes from this file's output.

What is MEASURED here (from the repo's own adapter implementations):
  nnz(dW)      instantiate each adapter with random coefficients, build dW in
               float64, count entries with |dW| > 1e-10 max|dW|.
  bursts       distinct 64-byte bursts (16 fp32 words, row-major, aligned)
               that hold a nonzero of dW -- the read-modify-write footprint of
               W <- W - dW_old + dW_new.  For LoWA this is read from the actual
               atom index sets (`eflat`), not from a formula.
What is DERIVED:
  build flops  the repo's frozen op counter (`bench_adapter_cost`, the source of
               Table III), or the support-restricted synthesis where cheaper.
  energy       Horowitz, ISSCC 2014, Fig. 1.1.9 (45 nm, 0.9 V): DRAM 1.3-2.6 nJ
               per 64-bit access (the LOW end, 1.3 nJ, is used), 3.7 pJ fp32
               mult, 0.9 pJ fp32 add.  Decoded from the paper's own figure.
SWAP MODEL.  All fixed-support adapters are linear in their coefficients, so
dW_new - dW_old = build(c_new - c_old): ONE build, one RMW of supp(dW), plus
reading both coefficient vectors.  LoCA learns its locations per task, so it
builds both updates (2x build flops).  Builds are assumed to stay on chip (no
spill traffic is charged), which favours the dense-buffer methods.
"""
import json, math, os, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import bench_adapter_cost as BC                       # noqa: E402
from sgwa_adapter import SGWALinear, wavelet_kernel    # noqa: E402
from merged_fourierft import MergedFourierFTLinear     # noqa: E402
from haar_adapter import HaarLinear, haar_matrix       # noqa: E402
from loca_adapter import LoCALinear                    # noqa: E402
from qwha_adapter import QWHALinear, _wht_sync_free    # noqa: E402
from spectral_adapter import SpectralAdapterLinear     # noqa: E402

K_COEF, K_HOP, SEED = 256, 3, 777
S_ATOM = 2 * K_HOP * (K_HOP + 1) + 1
WIDTHS = [768, 2048]
MODULES = {768: 24, 2048: 36}            # roberta-base q,v x 12; gemma-2b q,o x 18
WORD, BURST = 4, 64                      # fp32; DDR4 x64 BL8 burst = 64 B
WPB = BURST // WORD
# Horowitz ISSCC'14 Fig. 1.1.9, 45 nm, 0.9 V: DRAM 1.3-2.6 nJ per 64-bit access
E_DRAM64, E_MUL, E_ADD = 1.3e-9, 3.7e-12, 0.9e-12
E_DRAM = E_DRAM64 / 2                    # per 32-bit word moved
E_FLOP = 0.5 * (E_MUL + E_ADD)           # a flop = half a fp32 MAC
# A40 / GA102 (NVIDIA Ampere GA102 whitepaper): 128 KB L1/shared per SM, 6 MB L2
L1_BYTES, L2_BYTES = 128 * 1024, 6 * 1024 * 1024
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
ORDER = ["fourierft", "wave1", "wave2", "loca", "qwha", "lyra", "lowa"]
TITLE = {"fourierft": "FourierFT", "wave1": r"WaveFT ($\mu{=}1$)",
         "wave2": r"WaveFT ($\mu{=}2$)", "loca": "LoCA", "qwha": "QWHA",
         "lyra": "LYRA", "lowa": r"\textbf{LoWA}"}
BCKEY = {"fourierft": "fftm", "wave1": "wave1", "wave2": "wave2", "loca": "loca",
         "qwha": "qwha", "lyra": "lyra"}
QWHA_CHECK = {}
FACTORED = {"wave1", "wave2", "qwha", "lyra"}   # Table I dagger rows (LoWA excluded: merged)


def base(d, dtype=torch.float64, bias=False):
    torch.manual_seed(0)
    b = nn.Linear(d, d, bias=bias).to(DEV, dtype)
    for p in b.parameters():
        p.requires_grad_(False)
    return b


def build(name, d):
    """(module, number of trained scalars) at width d, float64."""
    b = base(d)
    if name == "lowa":
        m = SGWALinear(b, k=K_COEF, K=K_HOP, seed=SEED)
    elif name == "fourierft":
        m = MergedFourierFTLinear(b, n_frequency=K_COEF, scaling=100.0, random_loc_seed=SEED)
    elif name in ("wave1", "wave2"):
        m = HaarLinear(b, n_frequency=K_COEF, mu=1 if name == "wave1" else 2,
                       support_seed=SEED, fourierft_scaling=150.0, init_std=1.0)
    elif name == "loca":
        m = LoCALinear(b, n_frequency=K_COEF, scale=0.25, learn_location_iter=-1, init_seed=SEED)
        m.spectrum_indices.requires_grad_(False)
    elif name == "qwha":
        m = QWHALinear(b, n_frequency=K_COEF, scaling=26.5165, random_loc_seed=SEED,
                       init_weights=False, sync_free=True)
    elif name == "lyra":
        m = SpectralAdapterLinear(b, p=16, q=16, scaling=0.05, d_initial=0.07,
                                  freq_mode="geometric", freq_exponent=2.0)
    m = m.to(DEV).double()
    g = torch.Generator().manual_seed(1234)
    for pn, p in m.named_parameters():
        if pn.startswith("base_layer") or pn == "spectrum_indices":
            continue
        with torch.no_grad():
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64).to(p.device))
    ntrain = 3 * K_COEF if name == "loca" else K_COEF
    return m, ntrain


@torch.no_grad()
def delta(name, m, d):
    if name == "qwha":
        # QWHA never builds dW: its forward is (x H)(W0 H + S)^T / n, so
        # dW = S M^T / n (M the unnormalised Hadamard map, M M^T = n I).  Built from the
        # authors' own WHT, then CHECKED against the forward probe, whose fp32
        # sqrt(n) scalar leaves ~1e-9 roundoff on every entry (so it is not counted).
        vals = m.spectrum * m.scaling / math.sqrt(m.out_features)
        S = torch.zeros(d, d, dtype=torch.float64, device=DEV).index_put(
            (m.indices[0], m.indices[1]), vals, accumulate=True)
        M = _wht_sync_free(torch.eye(d, dtype=torch.float64, device=DEV))  # x -> x M
        dW = S @ M.T / m.in_features      # M M^T = n I; M != M^T for the 12-block at 768
        eye = torch.eye(d, dtype=torch.float64, device=DEV)
        probe = (m(eye) - F.linear(eye, m.base_layer.weight)).T
        QWHA_CHECK[d] = float((probe - dW).abs().max() / dW.abs().max())
        return dW
    if name in ("wave1", "wave2"):
        return m.get_delta_weight(torch.float64)
    return m.get_delta_weight().double()


def nnz_flat(dW):
    a = dW.abs()
    return torch.nonzero((a > 1e-10 * a.max()).reshape(-1)).squeeze(1)


def bursts_of(flat):
    return int(torch.unique(flat // WPB).numel())


def haar_supp(d):
    H = haar_matrix(d)
    return (H != 0).sum(1)                          # support of every Haar row


@torch.no_grad()
def build_flops(name, m, d, flat):
    """(flops, energy J) to construct dW_new - dW_old, and how it is counted."""
    if name == "lowa":
        ks = m.eflat.numel()
        return ks + ks, ks * E_MUL + ks * E_ADD, "k s mult + k s add (scatter)"
    if name in ("wave1", "wave2"):
        sr, sc = haar_supp(d).to(DEV), haar_supp(d).to(DEV)
        outer = int((sr[m.rows] * sc[m.cols]).sum())   # entries written by k mu outer products
        dense = BC.arm_build_flops(BCKEY[name], d)
        f = min(2 * outer, dense)
        return f, f * E_FLOP, "sparse outer products" if 2 * outer < dense else "dense inverse Haar"
    if name == "qwha":
        rows = int(torch.unique(m.indices[0]).numel())
        sparse = rows * d * math.log2(d) + rows * d     # WHT per touched row, adds only
        dense = BC.arm_build_flops("qwha", d) + d * d
        f = min(sparse, dense)
        return f, f * E_ADD, "WHT of the touched rows (adds)"
    f = BC.arm_build_flops(BCKEY[name], d) + d * d     # build + fold, as Table III
    if name == "loca":
        f = 2 * BC.arm_build_flops("loca", d) + d * d  # both tasks' locations differ
    return f, f * E_FLOP, "repo op counter (Table III)"


def working_set(name, m, d):
    """Live on-chip bytes to construct dW, for the build counted above."""
    k = K_COEF
    if name == "lowa":
        return WORD * (k + k + S_ATOM), "k coef + k centres + s taps"
    if name == "fourierft":
        return 2 * WORD * d * d, "complex m x n ifft2 buffer"
    if name in ("wave1", "wave2"):
        mu = 1 if name == "wave1" else 2
        return WORD * (k + mu * k), "k coef + mu k cells"
    if name == "loca":
        return WORD * (d * d + 3 * k), "m x n accumulator + 3k"
    if name == "qwha":
        return WORD * (d + 2 * k), "one length-n WHT row + k coef/cells"
    if name == "lyra":
        p = q = 16
        return WORD * (d * p + p * q + q * d), "U, D, V factors"


def per_token_overhead(name, d):
    """Unmerged per-token adapter flops (marginal in b) and their energy."""
    a, kw = BC._ARM_TO_COUNTER(BCKEY[name])
    f = (BC.theoretical_flops(a, d, d, 2, K_COEF, **kw)["fwd_adapter"]
         - BC.theoretical_flops(a, d, d, 1, K_COEF, **kw)["fwd_adapter"])
    e = f * (E_ADD if name == "qwha" else E_FLOP)
    return f, e


@torch.no_grad()
def verify_lowa(d):
    """(interior atoms, max |atom - H| over them, conv-equivalence max abs err)."""
    b = base(d)
    m = SGWALinear(b, k=K_COEF, K=K_HOP, seed=SEED).to(DEV).double()
    H = wavelet_kernel(K_HOP).to(DEV)
    w = 2 * K_HOP + 1
    from sgwa_adapter import sample_centers
    flat = sample_centers(d, d, K_COEF, SEED)
    ci, cj = (flat // d).tolist(), (flat % d).tolist()
    worst, n_int = 0.0, 0
    for j in range(K_COEF):
        if not (K_HOP <= ci[j] < d - K_HOP and K_HOP <= cj[j] < d - K_HOP):
            continue
        n_int += 1
        m.coef.zero_(); m.coef[j] = 1.0
        psi = m.get_delta_weight() / m.scaling
        win = psi[ci[j] - K_HOP:ci[j] + K_HOP + 1, cj[j] - K_HOP:cj[j] + K_HOP + 1]
        rest = psi.abs().sum() - win.abs().sum()
        worst = max(worst, float((win - H).abs().max()), float(rest))
    # conv form: dW = (impulse map, half-sample symmetric pad K) * H, valid
    g = torch.Generator().manual_seed(99)
    c = torch.randn(K_COEF, generator=g, dtype=torch.float64).to(DEV)
    m.coef.copy_(c)
    dW = m.get_delta_weight()
    Z = np.zeros((d, d))
    Z[ci, cj] = (m.scaling * c).cpu().numpy()
    Zp = torch.from_numpy(np.pad(Z, K_HOP, mode="symmetric")).to(DEV)
    conv = F.conv2d(Zp[None, None], torch.flip(H, (0, 1))[None, None])[0, 0]
    err = float((conv - dW).abs().max())
    return n_int, worst, err, float(dW.abs().max())


def fmt_e(x):
    return f"{x:.3g}"


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    out = {"assumptions": dict(word=WORD, burst=BURST, E_dram64_nJ=E_DRAM64 * 1e9,
                               E_mul_pJ=E_MUL * 1e12, E_add_pJ=E_ADD * 1e12,
                               L1=L1_BYTES, L2=L2_BYTES, k=K_COEF, K=K_HOP, s=S_ATOM),
           "rows": {}, "verify": {}}
    timing_path = os.path.join(ROOT, "results", "sgwa", "HW_swap_timing.json")
    timing = json.load(open(timing_path)) if os.path.exists(timing_path) else {}
    for d in WIDTHS:
        out["verify"][d] = dict(zip(("interior_atoms", "max_err_interior_vs_H",
                                     "conv_equiv_max_abs_err", "max_abs_dW"),
                                    verify_lowa(d)))
        for name in ORDER:
            m, ntrain = build(name, d)
            dW = delta(name, m, d)
            flat = nnz_flat(dW)
            nnz = int(flat.numel())
            bursts = bursts_of(flat)
            coef_b = math.ceil(ntrain * WORD / BURST)
            fl, e_ops, how = build_flops(name, m, d, flat)
            # DRAM: read + write each touched burst, read old and new coefficients
            dram_words = 2 * bursts * WPB + 2 * coef_b * WPB
            e_dram = dram_words * E_DRAM
            ws, ws_how = working_set(name, m, d)
            row = dict(nnz=nnz, nnz_frac=nnz / d / d, bursts=bursts,
                       acc32=2 * nnz + 2 * ntrain, dram_bursts=2 * bursts + 2 * coef_b,
                       dram_bytes=dram_words * WORD, build_flops=fl, build_how=how,
                       E_dram_uJ=e_dram * 1e6, E_ops_uJ=e_ops * 1e6,
                       E_swap_uJ=(e_dram + e_ops) * 1e6,
                       E_model_mJ=(e_dram + e_ops) * MODULES[d] * 1e3,
                       ws_bytes=ws, ws_how=ws_how, ws_fits=("L1" if ws <= L1_BYTES else
                                                            "L2" if ws <= L2_BYTES else "DRAM"))
            if name == "lowa":
                row["ks"] = int(m.eflat.numel())
                row["max_bursts_per_atom"] = max(
                    bursts_of(m.eflat.view(K_COEF, -1)[j]) for j in range(K_COEF))
            if name in FACTORED:
                ftok, etok = per_token_overhead(name, d)
                e_unmerged = 2 * coef_b * WPB * E_DRAM      # load k new coefficients
                row.update(tok_flops=ftok, tok_E_pJ=etok * 1e12,
                           E_swap_unmerged_uJ=e_unmerged * 1e6)
            t = timing.get(str(d), {}).get(name)
            row["swap_us"] = t
            out["rows"].setdefault(d, {})[name] = row
            del m, dW
            torch.cuda.empty_cache()
        lw = out["rows"][d]["lowa"]
        for name in FACTORED:
            r = out["rows"][d][name]
            r["breakeven_tokens"] = ((lw["E_swap_uJ"] - r["E_swap_unmerged_uJ"]) * 1e-6
                                     / (r["tok_E_pJ"] * 1e-12))
    # ---------------------------------------------------------------- report
    L = ["# HW cost: adapter swap on merged weights (scripts/hw_cost.py)", "",
         f"Assumptions: fp32 words, {BURST}-B bursts (DDR4 x64 BL8), Horowitz'14 45 nm: "
         f"DRAM {E_DRAM64*1e9} nJ/64b (low end of 1.3-2.6), mult {E_MUL*1e12} pJ, add {E_ADD*1e12} pJ; "
         f"k={K_COEF}, K={K_HOP}, s={S_ATOM}; GA102 L1 {L1_BYTES//1024} KiB/SM, L2 {L2_BYTES>>20} MiB.", ""]
    for d in WIDTHS:
        v = out["verify"][d]
        L += [f"## d = {d} ({MODULES[d]} modules)", "",
              f"LoWA checks: {v['interior_atoms']} interior atoms equal H, max err "
              f"{v['max_err_interior_vs_H']:.2e}; conv(sym-pad impulse map, H) vs dW max abs err "
              f"{v['conv_equiv_max_abs_err']:.2e} (max|dW| {v['max_abs_dW']:.2e}).", "",
              "| method | nnz | nnz/mn | bursts touched | DRAM bursts (RMW+coef) | 32-bit acc | build flops (how) | E_dram uJ | E_ops uJ | E_swap uJ/module | E_swap mJ/model | working set B (fits) | swap us (GPU) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for n in ORDER:
            r = out["rows"][d][n]
            L.append(f"| {n} | {r['nnz']} | {r['nnz_frac']:.4f} | {r['bursts']} | {r['dram_bursts']} | "
                     f"{r['acc32']} | {r['build_flops']:.4g} ({r['build_how']}) | {r['E_dram_uJ']:.4g} | "
                     f"{r['E_ops_uJ']:.4g} | {r['E_swap_uJ']:.4g} | {r['E_model_mJ']:.4g} | "
                     f"{r['ws_bytes']} ({r['ws_fits']}; {r['ws_how']}) | {r['swap_us']} |")
        L += ["", "Unmerged (factored) alternative: swap = load k coefficients; pays per-token flops forever.", "",
              "| method | E_swap unmerged uJ | flops/token | E/token pJ | break-even tokens vs LoWA merged swap |",
              "|---|---|---|---|---|"]
        for n in ORDER:
            r = out["rows"][d][n]
            if n in FACTORED:
                L.append(f"| {n} | {r['E_swap_unmerged_uJ']:.4g} | {r['tok_flops']:.0f} | "
                         f"{r['tok_E_pJ']:.4g} | {r['breakeven_tokens']:.0f} |")
        lw = out["rows"][d]["lowa"]
        L += ["", f"LoWA ks={lw['ks']}, max bursts per atom {lw['max_bursts_per_atom']}.", ""]
    L += ["## Headline (quoted in the paper)", ""]
    for d in WIDTHS:
        R = out["rows"][d]; lw = R["lowa"]
        for n in ORDER[:-1]:
            r = R[n]
            L.append(f"- d={d} {n}: bursts {r['bursts']/lw['bursts']:.2f}x LoWA, energy "
                     f"{r['E_swap_uJ']/lw['E_swap_uJ']:.2f}x LoWA, time "
                     + (f"{r['swap_us']/lw['swap_us']:.2f}x LoWA" if r['swap_us'] else "n/t")
                     + (f", break-even {r['breakeven_tokens']:.0f} tokens" if n in FACTORED else ""))
        L.append(f"- d={d} LoWA: {lw['E_swap_uJ']:.1f} uJ/module, {lw['E_model_mJ']:.2f} mJ/model "
                 f"({MODULES[d]} modules), nnz {lw['nnz']} <= ks {lw['ks']}, bursts/atom mean "
                 f"{lw['bursts']/K_COEF:.1f}")
    L.append("")
    L += [f"QWHA structural dW = S M^T / n vs forward probe, max rel err: {QWHA_CHECK}", ""]
    out["verify"]["qwha_probe_rel_err"] = QWHA_CHECK
    md = os.path.join(ROOT, "results", "sgwa", "HW_cost.md")
    open(md, "w").write("\n".join(L) + "\n")
    json.dump(out, open(os.path.join(ROOT, "results", "sgwa", "HW_cost.json"), "w"), indent=1)
    write_table(out)
    print("\n".join(L))


def write_table(out):
    d1, d2 = WIDTHS
    rows = []
    for n in ORDER:
        a, b = out["rows"][d1][n], out["rows"][d2][n]
        t = (lambda r: "--" if r["swap_us"] is None else f"{r['swap_us']:.0f}")
        rows.append(f"{TITLE[n]} & {a['bursts']/1e3:.1f} & {b['bursts']/1e3:.1f} & "
                    f"{a['E_swap_uJ']:.0f} & {b['E_swap_uJ']:.0f} & "
                    f"{kib(b['ws_bytes'])} & {t(a)} & {t(b)} \\\\")
    tex = r"""\begin{table}[!t]
\centering
\caption{Task swap on one merged module of width $d$: %(B)d-byte bursts rewritten, energy, build buffer, and A40 time of each method's merge path.}
\label{tab:hwcost}
\footnotesize
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{@{}l rr rr r rr@{}}
\toprule
 & \multicolumn{2}{c}{Bursts (k)} & \multicolumn{2}{c}{Energy ($\mu$J)} & Buffer & \multicolumn{2}{c}{Time ($\mu$s)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){7-8}
Method & %(d1)d & %(d2)d & %(d1)d & %(d2)d & %(d2)d & %(d1)d & %(d2)d \\
\midrule
%(rows)s
\bottomrule
\end{tabular}
\end{table}
""" % dict(B=BURST, d1=d1, d2=d2, rows="\n".join(rows))
    p = os.path.join(ROOT, "llmdocs", "paper", "v1_lowa_iscas27", "tables", "hwcost.tex")
    open(p, "w").write(tex)


def kib(b):
    if b >= 1 << 20:
        return f"{b / (1 << 20):.1f}\\,MiB"
    return f"{b / 1024:.1f}\\,KiB"


if __name__ == "__main__":
    main()
