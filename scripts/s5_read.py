#!/usr/bin/env python3
"""[S.5] reader: the two-GPU block-owned SGWA (S.4c arm picked by the prereg rule) on every RoBERTa column.

  env/bin/python scripts/s5_read.py --arm S

Task row = rank-0 CSV of each torchrun pair (both ranks hold the identical model after the epoch-end
exchange; the reader asserts they agree): max over epochs of the primary metric, median over seeds.
RTE = the S.4c cells of the picked arm.  Reference = [S.3]'s full-search single-process cells ([S.2] for RTE).
Computational metrics per run: wall-clock (as run, PAIRS pairs concurrent, cuda:1 shared), peak MiB per
GPU, bytes exchanged, exchange time per epoch; quiet-GPU per-epoch costs from scratchpad/s5/timing/summary.tsv
when present.  Writes results/sgwa/S5_table.md.
"""
import argparse, csv, os, re, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [42, 43, 44, 45, 46]
ORDER = ["rte", "cb", "mrpc", "stsb", "cola", "boolq", "sst2"]
NAMES = {"rte": "RTE", "cb": "CB", "mrpc": "MRPC", "stsb": "STS-B", "cola": "CoLA", "boolq": "BoolQ", "sst2": "SST-2"}
PRIMARY = {"cola": "matthews_correlation", "stsb": "pearson", "mrpc": "f1", "cb": "f1"}
S3_ROW = {"rte": 70.8, "cb": 93.1, "mrpc": 91.0, "stsb": 88.6, "cola": 53.5, "boolq": 73.8, "sst2": 92.9}   # S3_table.md, full search
FFT_ROW = {"rte": 79.4, "cb": 92.3, "mrpc": 91.4, "stsb": 89.1, "cola": 56.4, "boolq": 73.6, "sst2": 92.9}  # FourierFT merged


def last(p):
    return list(csv.DictReader(open(p)))[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="S.4c arm letter picked by the rule (E or S)")
    a = ap.parse_args()
    task_rows, comp = {}, {}
    for t in ORDER:
        pk = PRIMARY.get(t, "accuracy")
        vals, eps, times, peaks, mib, syncs, spe, ntr = [], [], [], [], [], [], None, None
        for s in SEEDS:
            if t == "rte":
                base = os.path.join(ROOT, f"scratchpad/s4c/csv/rte-sync{a.arm}-seed{s}")
                log = os.path.join(ROOT, f"scratchpad/s4c/log/rte-sync{a.arm}-seed{s}.log")
            else:
                base = os.path.join(ROOT, f"scratchpad/s5/csv/{t}-sync-seed{s}")
                log = os.path.join(ROOT, f"scratchpad/s5/log/{t}-sync-seed{s}.log")
            ps = [f"{base}-b{g}.csv" for g in (0, 1)]
            if not all(os.path.exists(p) for p in ps):
                continue
            r = [last(p) for p in ps]
            assert abs(float(r[0][pk]) - float(r[1][pk])) < 1e-6, f"{t} seed {s}: ranks disagree beyond fp32 atomics (STS-B Pearson differs by 1.2e-7 across devices)"
            vals.append(float(r[0][pk])); eps.append(int(r[0]["best_epoch"]))
            times.append(float(r[0]["train_time_s"])); peaks.append(max(float(x["peak_mem_mib"]) for x in r))
            mib.append(int(r[0]["sync_bytes"]) / 2**20)
            spe, ntr = int(r[0]["steps_per_epoch"]), int(r[0]["n_train"])
            syncs.append(statistics.fmean(float(x) for x in re.findall(r"sync ([0-9.]+)s", open(log).read())))
        # reference per-seed cells
        ref = []
        for s in SEEDS:
            p = os.path.join(ROOT, f"scratchpad/s2/csv/sgwa-D-c0-seed{s}.csv" if t == "rte" else f"scratchpad/s3/csv/{t}-sgwa-seed{s}.csv")
            if os.path.exists(p):
                rr = last(p); ref.append((float(rr[pk]), float(rr.get("total_training_time_sec", rr.get("train_time_s", 0)) or 0)))
        task_rows[t] = dict(vals=vals, eps=eps, ref=[x[0] for x in ref])
        comp[t] = dict(times=times, peaks=peaks, mib=mib, syncs=syncs, spe=spe, ntr=ntr, ref_time=[x[1] for x in ref])

    out = ["## S.5 task metrics (median over seeds 42–46 of the max over epochs, × 100; ⋆/◦ = 5/5 paired vs FourierFT merged not computed here)\n",
           "| Method | " + " | ".join(NAMES[t] for t in ORDER) + " | Avg. |", "|---|" + "---|" * (len(ORDER) + 1)]
    out.append("| FourierFT (merged) [R.310] | " + " | ".join(f"{FFT_ROW[t]:.1f}" for t in ORDER) + f" | {statistics.fmean(FFT_ROW.values()):.1f} |")
    out.append("| SGWA (full search), one GPU, one process [S.3] | " + " | ".join(f"{S3_ROW[t]:.1f}" for t in ORDER) + f" | {statistics.fmean(S3_ROW.values()):.1f} |")
    cells, done = [], []
    for t in ORDER:
        v = task_rows[t]["vals"]
        if len(v) == 5:
            cells.append(f"{100 * statistics.median(v):.1f}"); done.append(100 * statistics.median(v))
        else:
            cells.append(f"({len(v)}/5)")
    avg = f"{statistics.fmean(done):.1f}" if len(done) == len(ORDER) else "–"
    out.append(f"| **SGWA-2GPU (block-owned, exchange arm {a.arm})** | " + " | ".join(cells) + f" | {avg} |")
    out.append("")
    out.append("| task | seeds 42..46 (× 100) | mean ± sd | [S.3] mean ± sd | Δ mean | Δ / sd | best epochs |")
    out.append("|---|---|---|---|---:|---:|---|")
    for t in ORDER:
        v, ref = task_rows[t]["vals"], task_rows[t]["ref"]
        if len(v) < 5 or len(ref) < 5:
            out.append(f"| {NAMES[t]} | ({len(v)}/5) | | | | | |"); continue
        m, sd = 100 * statistics.fmean(v), 100 * statistics.stdev(v); rm, rs = 100 * statistics.fmean(ref), 100 * statistics.stdev(ref)
        out.append(f"| {NAMES[t]} | " + ", ".join(f"{100 * x:.1f}" for x in v) + f" | {m:.2f} ± {sd:.2f} | {rm:.2f} ± {rs:.2f} | {m - rm:+.2f} | "
                   f"{(m - rm) / rs if rs else float('nan'):+.1f} | {task_rows[t]['eps']} |")
    out.append("")
    out.append("## S.5 computational metrics (per run = one torchrun pair; as run with concurrent pairs and a shared cuda:1)\n")
    out.append("| task | train ex. per GPU | steps/epoch per GPU | wall-clock s, median (2 GPUs) | [S.3] one-GPU cell s, median (3 workers/GPU) | peak MiB per GPU | exchanged MiB per run | exchange s per epoch |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for t in ORDER:
        c = comp[t]
        if not c["times"]:
            continue
        rt = f"{statistics.median(c['ref_time']):,.0f}" if c["ref_time"] else "–"
        out.append(f"| {NAMES[t]} | {c['ntr']} | {c['spe']} | {statistics.median(c['times']):,.0f} | {rt} | {statistics.median(c['peaks']):,.0f} | "
                   f"{statistics.fmean(c['mib']):.1f} | {statistics.fmean(c['syncs']):.2f} |")
    tp = os.path.join(ROOT, "scratchpad/s5/timing/summary.tsv")
    if os.path.exists(tp):
        rows = list(csv.DictReader(open(tp), delimiter="\t"))
        out.append("\nQuiet-GPU per-epoch cost (cuda:0 alone, 2-epoch runs, same code path): full single-process cell vs one block-owned branch without exchange\n")
        out.append("| task | full: train s | full: eval s | full: peak MiB | branch: train s | branch: eval s | branch: peak MiB | branch/full train |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        d = {(r["task"], r["mode"]): r for r in rows}
        for t in ORDER:
            f, b = d.get((t, "full")), d.get((t, "branch"))
            if f and b:
                out.append(f"| {NAMES[t]} | {f['train_s']} | {f['eval_s']} | {f['peak_mib']} | {b['train_s']} | {b['eval_s']} | {b['peak_mib']} | "
                           f"{float(b['train_s']) / float(f['train_s']):.2f} |")
    text = "\n".join(out)
    print(text)
    open(os.path.join(ROOT, "results/sgwa/S5_table.md"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
