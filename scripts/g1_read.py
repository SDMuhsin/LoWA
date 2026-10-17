#!/usr/bin/env python3
"""[G.1] SGWA on gemma-2b: the Table II(b)-equivalent for RTE / MRPC / STS-B / CoLA.

  env/bin/python scripts/g1_read.py

Baselines are REUSED from the paper's source, /workspace/lora_research_signal/results/fir_final_gemma2b.csv
(fir, H100, stage 05): median over seeds 42-46 of the max over epochs, per-seed values in seed order.
SGWA from scratchpad/g1/csv/g1-<task>-sgwa-seed<s>.csv (this box, A40).  Marks: ⋆ beats FourierFT (merged)
on 5/5 paired seeds, ◦ loses 5/5 (the paper's RoBERTa convention, applied here to the decoder).
Full fine-tuning is the paper's Table II(b) row (tables/main.tex), not recomputed.
Writes results/sgwa/G1_table.md.
"""
import csv, os, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "/workspace/lora_research_signal/results/fir_final_gemma2b.csv"
TASKS = ["rte", "mrpc", "stsb", "cola"]
HDR = ["RTE", "MRPC", "STS-B", "CoLA"]
METRIC = {"rte": "accuracy", "mrpc": "f1", "stsb": "pearson", "cola": "matthews_correlation"}
FLOOR = {"rte": 0.527076, "mrpc": 0.8122, "stsb": 0.0, "cola": 0.0}     # majority / degenerate floors
SEEDS = [42, 43, 44, 45, 46]
ARMS = [("fftm", "FourierFT (merged)"), ("fftstock", "FourierFT (additive)"), ("wave1", "WaveFT (μ=1)"),
        ("wave2", "WaveFT (μ=2)"), ("loca", "LoCA †"), ("qwha", "QWHA"), ("lyra", "LYRA"),
        ("scora", "SCoRA (derived)"), ("scora2", "SCoRA (swept)")]
FULL_FT = {"rte": 85.2, "mrpc": 91.6, "stsb": 89.8, "cola": 65.6}          # paper Table II(b)
D, B, K, NNZ, N_MOD, HEAD = 2048, 32 * 128, 256, 25, 36, 4096


def last(p):
    return list(csv.DictReader(open(p)))[-1]


def main():
    base = {}
    for r in csv.DictReader(open(BASE)):
        if r["task"] in TASKS:
            base[(r["task"], r["arm"])] = r
    sg = {t: {} for t in TASKS}
    sg_cost = {t: [] for t in TASKS}
    for t in TASKS:
        for s in SEEDS:
            p = os.path.join(ROOT, f"scratchpad/g1/csv/g1-{t}-sgwa-seed{s}.csv")
            if os.path.exists(p):
                r = last(p)
                sg[t][s] = (float(r[METRIC[t]]), int(r["best_epoch"]))
                sg_cost[t].append(r)

    vals = {}
    for a, _ in ARMS:
        for t in TASKS:
            vals[(a, t)] = [float(x) for x in base[(t, a)]["values"].split()]
    for t in TASKS:
        vals[("sgwa", t)] = [sg[t][s][0] for s in SEEDS] if len(sg[t]) == 5 else None

    def mark(a, t):
        if a == "fftm" or vals[(a, t)] is None:
            return ""
        w = sum(x > y for x, y in zip(vals[(a, t)], vals[("fftm", t)]))
        l = sum(x < y for x, y in zip(vals[(a, t)], vals[("fftm", t)]))
        return "⋆" if w == 5 else "◦" if l == 5 else ""

    rows = [("full", "Full fine-tuning")] + ARMS + [("sgwa", "**SGWA (RoBERTa HPs, no gemma search)**")]
    med = {}
    for a, _ in rows:
        for t in TASKS:
            if a == "full":
                med[(a, t)] = FULL_FT[t]
            elif vals[(a, t)] is not None:
                med[(a, t)] = 100 * statistics.median(vals[(a, t)])
    best = {t: max(med[(a, t)] for a, _ in rows if a != "full" and (a, t) in med) for t in TASKS}
    out = ["## G.1 task metrics: gemma-2b, median over seeds 42–46 of the max over epochs (× 100)\n",
           "| Method | " + " | ".join(HDR) + " | Avg. (4) |", "|---|" + "---|" * (len(TASKS) + 1)]
    for a, name in rows:
        cells = []
        for t in TASKS:
            if (a, t) not in med:
                cells.append(f"({len(sg[t])}/5)"); continue
            v = f"{med[(a, t)]:.1f}"
            if a != "full" and abs(med[(a, t)] - best[t]) < 1e-9:
                v = f"**{v}**"
            cells.append(v + (mark(a, t) if a != "full" else ""))
        avg = f"{statistics.fmean(med[(a, t)] for t in TASKS):.1f}" if all((a, t) in med for t in TASKS) else "–"
        out.append(f"| {name} | " + " | ".join(cells) + f" | {avg} |")
        if a == "full" or a == "lyra":
            out.append("|  |" + " |" * (len(TASKS) + 1))
    sds = []
    for t in TASKS:
        per = [statistics.stdev(vals[(a, t)]) for a, _ in ARMS]
        sds.append(f"{100 * statistics.median(per):.2f}")
    out.append("| *seed s.d. (baselines, median)* | " + " | ".join(sds) + " | |")
    out.append("\n⛔ MRPC is in-sample for every baseline (their HPs were selected on gemma-2b MRPC); SGWA had no gemma "
               "search at all. † LoCA trains 31,744 parameters, every other row 13,312. ⋆/◦ = 5/5 paired seeds vs "
               "FourierFT (merged).\n")
    # per-seed detail
    out.append("SGWA per seed (42..46), best epoch, degenerate (at the task floor):")
    for t in TASKS:
        if len(sg[t]) == 5:
            v = [sg[t][s][0] for s in SEEDS]; e = [sg[t][s][1] for s in SEEDS]
            n_ep = int(sg_cost[t][0]["num_train_epochs"])
            out.append(f"- {t}: " + ", ".join(f"{x:.4f}" for x in v) + f" | epochs {e} of 0–{n_ep - 1} | degenerate "
                       f"{sum(x <= FLOOR[t] + 1e-6 for x in v)}/5 | argmax at last epoch {sum(x == n_ep - 1 for x in e)}/5")
    # ---- computational ----
    out.append("\n## G.1 computational metrics (per cell; baselines measured on fir H100, SGWA on this box's A40)\n")
    out.append("| Method | trainable params | adapter flops/token merged | unmerged | peak MiB (RTE) | s/step RTE | s/step MRPC | s/step STS-B | s/step CoLA | GPU |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for a, name in ARMS:
        r = {t: base[(t, a)] for t in TASKS}
        un = float(r["rte"]["adapter_flops_per_token_unmerged"] or "nan")
        mg = float(r["rte"]["adapter_flops_per_token_merged"] or "nan")
        out.append(f"| {name} | {int(r['rte']['trainable_params']):,} | {mg:,.0f} | {un:,.0f} | {float(r['rte']['peak_mem_mib']):,.0f} | "
                   + " | ".join(f"{float(r[t]['sec_per_step']):.3f}" for t in TASKS) + " | H100 |")
    sgf = (K + 2 * K * NNZ + D * D) / B                         # merged: build + fold-in, per token
    sgu = 2 * K * 7 * 7 + 2 * K * 7                            # factored: gather-mix-scatter, K=3 box
    if all(sg_cost[t] for t in TASKS):
        steps = {t: int(base[(t, "fftm")]["train_steps"]) for t in TASKS}
        sps = {t: statistics.median(float(r["total_training_time_sec"]) for r in sg_cost[t]) / steps[t] for t in TASKS}
        pk = statistics.median(float(r["peak_mem_mib"]) for r in sg_cost["rte"])
        out.append(f"| **SGWA** | {K * N_MOD + HEAD:,} | {sgf:,.0f} | {sgu:,} | {pk:,.0f} | "
                   + " | ".join(f"{sps[t]:.3f}" for t in TASKS) + " | **A40** |")
        out.append(f"| *A40/H100 step-time ratio, SGWA vs FourierFT (merged)* | | | | | "
                   + " | ".join(f"{sps[t] / float(base[(t, 'fftm')]['sec_per_step']):.2f}×" for t in TASKS) + " | |")
    out.append("\n⚠ s/step = median cell wall-clock (incl. per-epoch eval) / training steps. SGWA ran on a different GPU "
               "(A40, fp32) from the baselines (H100, fp32), so its step time is not comparable to theirs; the ratio row "
               "shows the hardware gap. Flops/token are per adapted module (d=2048, 4,096 tokens/step): baselines from the fir cost "
               "model, SGWA from the S.1 formulas (merged = build + fold-in, SGWA's training path here; unmerged = "
               "the factored gather/mix/scatter).")
    text = "\n".join(out)
    print(text)
    open(os.path.join(ROOT, "results/sgwa/G1_table.md"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
