#!/usr/bin/env python3
"""[A.1] reader -- LoWA (SGWA) ablation tables, llmdocs/A1_lowa_ablation_prereg.md.

  env/bin/python scripts/a1_read.py [--md results/sgwa/A1_table.md]

Convention ([S.3]'s / the papers'): max-over-30-epochs primary metric (STS-B Pearson, CoLA MCC), median over
seeds, x100 at one decimal round-half-up.  Reference rung (K = 3, gamma = 0.552427) = [S.3]'s own cells.
For each ablation x task the table uses the largest seed set in {42-46, 42-44} that is COMPLETE for every rung
(PROCESS 1.4: partial seeds are never quoted).  Marks are paired against the reference: at n = 5 * beats it
5/5, o loses 5/5 (the gate); at n = 3 the marks are (+)/(-) = directional only (sign floor 0.125).
"""
import argparse, csv, json, os, statistics, sys
from decimal import Decimal, ROUND_HALF_UP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
A1 = os.path.join(ROOT, "scratchpad", "a1")
S3 = os.path.join(ROOT, "scratchpad", "s3", "csv")
TASKS = [("stsb", "STS-B", "pearson"), ("cola", "CoLA", "matthews_correlation")]
FLOOR = {"stsb": 0.0, "cola": 0.0}
G = 0.552427
K_RUNGS = [0, 1, 2, 3, 4, 8]
G_RUNGS = [("d8", "÷8", G / 8), ("d4", "÷4", G / 4), ("d2", "÷2", G / 2), ("ref", "reported", G),
           ("x2", "×2", G * 2), ("x4", "×4", G * 4), ("x8", "×8", G * 8)]
SEEDSETS = [[42, 43, 44, 45, 46], [42, 43, 44]]


def q(x):
    return Decimal(repr(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def _row(path):
    return list(csv.DictReader(open(path)))[-1]


def path(t, tag, s):
    if tag in ("K3", "gref"):
        return os.path.join(S3, f"{t}-sgwa-seed{s}.csv")
    return os.path.join(A1, "csv", f"a1-{t}-{tag}-seed{s}.csv")


def done(t, tag, s):
    if tag in ("K3", "gref"):
        return os.path.exists(path(t, tag, s))
    return os.path.exists(os.path.join(A1, "done", f"a1-{t}-{tag}-seed{s}"))


def walls():
    w = {}
    p = os.path.join(A1, "walls.txt")
    if os.path.exists(p):
        for line in open(p):
            lab, dt, _ = line.split()
            w[lab] = int(dt)
    return w


def block(tags, labels, extra_cols=None):
    """tags: rung tags; labels: printed rung labels. Returns markdown lines."""
    W = walls()
    hdr = "| rung | " + " | ".join(lbl for _, lbl, _ in TASKS)
    if extra_cols:
        hdr += " | " + " | ".join(h for h, _ in extra_cols)
    hdr += " | as-run wall s/cell (STS-B, CoLA) |"
    out = [hdr, "|---" * (hdr.count("|") - 1) + "|"]
    info, cols = [], {}
    for t, lbl, m in TASKS:
        seeds = next((ss for ss in SEEDSETS if all(done(t, g, s) for g in tags for s in ss)), None)
        if seeds is None:
            have = sum(done(t, g, s) for g in tags for s in SEEDSETS[0])
            info.append(f"- {lbl}: incomplete ({have}/{len(tags) * 5} rung-seeds done); not reported")
            cols[t] = None
            continue
        vals = {g: [float(_row(path(t, g, s))[m]) for s in seeds] for g in tags}
        eps = {g: [int(_row(path(t, g, s))["best_epoch"]) for s in seeds] for g in tags}
        cols[t] = (seeds, vals, eps)
        info.append(f"- {lbl}: n = {len(seeds)} (seeds {seeds[0]}–{seeds[-1]})"
                    + ("" if len(seeds) == 5 else " — marks directional only"))
    ref = "K3" if "K3" in tags else "gref"
    for g, rl in zip(tags, labels):
        cells = []
        for t, _, _ in TASKS:
            c = cols[t]
            if c is None:
                cells.append("·"); continue
            seeds, vals, _ = c
            v, r = vals[g], vals[ref]
            n = len(seeds)
            w = sum(a > b for a, b in zip(v, r)); l = sum(b > a for a, b in zip(v, r))
            mk = "" if g == ref else (("⋆" if w == n else "◦" if l == n else "") if n == 5
                                      else ("(+)" if w == n else "(−)" if l == n else ""))
            deg = sum(x <= FLOOR[t] + 1e-9 for x in v)
            cells.append(f"{q(100 * statistics.median(v))}{mk}" + (f" [{deg} degen.]" if deg else ""))
        row = f"| {rl} | " + " | ".join(cells)
        if extra_cols:
            row += " | " + " | ".join(f(g) for _, f in extra_cols)
        ws = []
        for t, _, _ in TASKS:
            xs = [W[f"a1-{t}-{g}-seed{s}"] for s in (42, 43, 44, 45, 46) if f"a1-{t}-{g}-seed{s}" in W]
            ws.append(f"{statistics.mean(xs):.0f}" if xs else ("S.3" if g == ref else "·"))
        out.append(row + f" | {', '.join(ws)} |")
    detail = []
    for t, lbl, _ in TASKS:
        if cols[t] is None:
            continue
        seeds, vals, eps = cols[t]
        for g, rl in zip(tags, labels):
            detail.append(f"- {lbl} {rl}: " + ", ".join(f"{x:.4f}" for x in vals[g])
                          + f" | argmax epochs {eps[g]}")
    return out, info, detail


def k_struct():
    import torch
    from sgwa_adapter import SGWALinear, wavelet_kernel
    st = {}
    for K in K_RUNGS:
        a = SGWALinear(torch.nn.Linear(768, 768), k=256, K=K, seed=777)
        H = wavelet_kernel(K)
        st[f"K{K}"] = (a.hval.numel(), a.eflat.unique().numel(), float(H.abs().max() ** 2))
    return st


def table():
    md = ["# [A.1] LoWA ablations (RoBERTa-base, STS-B Pearson / CoLA MCC ×100, median over seeds)", "",
          "Prereg `llmdocs/A1_lowa_ablation_prereg.md`. Reference rung (K = 3, γ = 0.552427, η = 0.166667) = "
          "`[S.3]` cells. ⋆/◦ = beats/loses vs reference on 5/5 paired seeds; (+)/(−) = same at n = 3 "
          "(directional, not significant). Wall = as-run, 3 cells/GPU, shared box: not a speed claim.", ""]
    st = k_struct()
    ktags = [f"K{K}" for K in K_RUNGS]
    klab = [("K = 0 (un-filtered)" if K == 0 else f"K = {K}" + (" (reported)" if K == 3 else "")) for K in K_RUNGS]
    out, info, det = block(ktags, klab, [("nnz/atom", lambda g: str(st[g][0])),
                                         ("ΔW entries/module", lambda g: f"{st[g][1]:,}"),
                                         ("centre energy", lambda g: f"{st[g][2]:.3f}")])
    md += ["## A. Atom footprint at fixed budget (k = 256/module, 6,144 adapter params; γ, η fixed ⇒ P fixed)", ""]
    md += out + [""] + info + [""]
    gtags = [f"g{t}" for t, _, _ in G_RUNGS]
    glab = [f"{l} ({v:.6g})" + (" ← derived 150/√(2mn)" if t == "d4" else "") for t, l, v in G_RUNGS]
    out, info2, det2 = block(gtags, glab, [("P = η·γ", lambda g: f"{0.166667 * dict((f'g{t}', v) for t, _, v in G_RUNGS)[g]:.4g}")])
    md += ["## B. Scale γ over a range of 64, η = 0.166667 fixed (SCoRA's Table `scale` design)", ""]
    md += out + [""] + info2 + [""]
    # PROCESS 1.2 reproduction
    rp = os.path.join(A1, "csv", "a1-cola-ref-seed42.csv")
    if os.path.exists(os.path.join(A1, "done", "a1-cola-ref-seed42")):
        a = float(_row(rp)["matthews_correlation"]); b = float(_row(path("cola", "K3", 42))["matthews_correlation"])
        md += [f"Reproduction (PROCESS 1.2): CoLA reference seed 42 re-run {a:.4f} vs `[S.3]` {b:.4f} (Δ {a - b:+.4f}).", ""]
    md += ["## Per-seed detail", ""] + det + det2
    return "\n".join(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md")
    a = ap.parse_args()
    md = table()
    print(md)
    if a.md:
        open(a.md, "w").write(md + "\n")
