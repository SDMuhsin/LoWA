#!/usr/bin/env python3
"""[S.4b] merge + read: branch-train-merge in the wavelet domain (prereg llmdocs/S4b_btm_prereg.md).

  env/bin/python scripts/s4b_read.py [--device cuda:0] [--epochs 30]

For every seed, arm and epoch e: load both branches' epoch-e states from scratchpad/s4b/state,
merge coefficients (disjoint: sum = concatenation; shared: mean; plus the other weight as a readout),
average the two heads, score RTE dev, take the max over e (the protocol applied to the merged
model).  Branch-alone rows come from the branches' own CSVs.  Scores S4-P6..P9 against [S.2]'s
full-data single-process cells.  Writes results/sgwa/S4b_table.md.
"""
import argparse, csv, glob, os, statistics, sys, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
SEEDS = [42, 43, 44, 45, 46]
S2 = "scratchpad/s2/csv/sgwa-D-c0-seed{s}.csv"
FULL_CELL_EPOCH_S = 9.5          # S.4 §3: one train_glue cell, train+eval per epoch, quiet cuda:0


def last(path):
    return list(csv.DictReader(open(path)))[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--state_dir", default="scratchpad/s4b/state")
    ap.add_argument("--csv_dir", default="scratchpad/s4b/csv")
    a = ap.parse_args()
    import torch
    from datasets import load_dataset
    import evaluate
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    from sgwa_parallel import SGWAPopModel
    dev = torch.device(a.device)

    tok = AutoTokenizer.from_pretrained("roberta-base")
    raw = load_dataset("glue", "rte")["validation"]
    enc = raw.map(lambda ex: tok(ex["sentence1"], ex["sentence2"], padding=False, max_length=128, truncation=True),
                  batched=True)
    ev_ids = [torch.tensor(x) for x in enc["input_ids"]]
    ev_lab = torch.tensor(enc["label"])
    metric = evaluate.load("glue", "rte")
    pad = tok.pad_token_id

    def collate(seqs):
        T = max(s.numel() for s in seqs)
        ids = torch.full((len(seqs), T), pad, dtype=torch.long); am = torch.zeros(len(seqs), T, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :s.numel()] = s; am[i, :s.numel()] = 1
        return ids.to(dev), am.to(dev)

    cfg = AutoConfig.from_pretrained("roberta-base", num_labels=2)
    hf = AutoModelForSequenceClassification.from_pretrained("roberta-base", config=cfg).to(dev)
    model = SGWAPopModel(hf, ["query", "value"], 1, k=256, K=3, scaling=[0.552427], seed=777,
                         mode="merged", num_labels=2, head_seeds=[0]).to(dev)
    del hf.classifier
    model.eval()
    batches = [collate(ev_ids[lo:lo + 64]) for lo in range(0, len(ev_ids), 64)]
    rid0 = lambda n: torch.zeros(n, dtype=torch.long, device=dev)

    @torch.no_grad()
    def score(coef, head):
        for p, c in zip(model.pops, coef):
            p.coef[0].copy_(c.to(dev))
        for t, v in zip(model.head.params(), head):
            t[0].copy_(v.to(dev))
        preds = []
        for ids, am in batches:
            preds.append(model(ids, am, rid0(ids.shape[0]))[0].argmax(-1).cpu())
        return metric.compute(predictions=torch.cat(preds).numpy(), references=ev_lab.numpy())["accuracy"]

    def load(arm, g, seed, e):
        p = os.path.join(ROOT, a.state_dir, f"rte-{arm}-b{g}-seed{seed}_ep{e:02d}.pt")
        return torch.load(p, map_location="cpu", weights_only=True) if os.path.exists(p) else None

    rows = {}          # (arm, variant) -> {seed: (best, epoch)}
    alone = {}         # (arm, g) -> {seed: (best, epoch, train_time_s)}
    for arm in ("disjoint", "shared"):
        for g in (0, 1):
            for s in SEEDS:
                p = os.path.join(ROOT, a.csv_dir, f"rte-{arm}-b{g}-seed{s}.csv")
                if os.path.exists(p):
                    r = last(p)
                    alone.setdefault((arm, g), {})[s] = (float(r["accuracy"]), int(r["best_epoch"]), float(r["train_time_s"]))
        for s in SEEDS:
            curves = {}
            for e in range(a.epochs):
                st = [load(arm, g, s, e) for g in (0, 1)]
                if any(x is None for x in st):
                    break
                head = [(h0 + h1) / 2 for h0, h1 in zip(st[0]["head"], st[1]["head"])]
                for w in (1.0, 0.5):
                    coef = w * (st[0]["coef"] + st[1]["coef"])
                    curves.setdefault(w, []).append(score(coef, head))
                # diagnostics: which half of the merge hurts?  sum / half adapter with branch 0's OWN head,
                # branch 0's own adapter with the averaged head, and the sanity check (own adapter, own head)
                curves.setdefault("sum+head0", []).append(score(st[0]["coef"] + st[1]["coef"], st[0]["head"]))
                curves.setdefault("half+head0", []).append(score(0.5 * (st[0]["coef"] + st[1]["coef"]), st[0]["head"]))
                curves.setdefault("own0+avghead", []).append(score(st[0]["coef"], head))
                curves.setdefault("own0+head0", []).append(score(st[0]["coef"], st[0]["head"]))
            for w, cv in curves.items():
                if len(cv) == a.epochs:
                    b = max(range(a.epochs), key=lambda i: cv[i])
                    rows.setdefault((arm, w), {})[s] = (cv[b], b)
            if len(curves.get("own0+head0", [])) == a.epochs and s in alone.get((arm, 0), {}):
                v, e, _ = alone[(arm, 0)][s]
                print(f"[s4b] sanity {arm} seed {s}: branch-0 CSV best {v:.4f}@{e} vs re-scored {max(curves['own0+head0']):.4f}"
                      f"@{curves['own0+head0'].index(max(curves['own0+head0']))}", flush=True)
            print(f"[s4b] {arm} seed {s}: {len(curves.get(1.0, []))}/{a.epochs} epochs merged", flush=True)

    ref = [float(last(os.path.join(ROOT, S2.format(s=s)))["accuracy"]) for s in SEEDS]
    ref_mean, ref_sd = statistics.fmean(ref), statistics.stdev(ref)
    out = []
    out.append("| row | seeds 42..46 | median | mean ± sd | best epochs |")
    out.append("|---|---|---:|---|---|")

    def line(name, d):
        if len(d) < 5:
            out.append(f"| {name} | ({len(d)}/5 done) | | | |"); return None
        v = [d[s][0] for s in SEEDS]
        out.append(f"| {name} | " + ", ".join(f"{x:.4f}" for x in v) +
                   f" | {statistics.median(v):.4f} | {statistics.fmean(v):.4f} ± {statistics.stdev(v):.4f} | "
                   f"{[d[s][1] for s in SEEDS]} |")
        return statistics.fmean(v)

    out.append(f"| [S.2] full data, one process, 256 atoms | " + ", ".join(f"{x:.4f}" for x in ref) +
               f" | {statistics.median(ref):.4f} | {ref_mean:.4f} ± {ref_sd:.4f} | |")
    m = {}
    for arm in ("disjoint", "shared"):
        for g in (0, 1):
            m[(arm, "alone", g)] = line(f"{arm} branch {g} alone (half data{', 128 atoms' if arm == 'disjoint' else ''})",
                                        alone.get((arm, g), {}))
        for w, tag in ((1.0, "sum"), (0.5, "mean")):
            prim = (arm == "disjoint") == (w == 1.0)
            m[(arm, w)] = line(f"**{arm} merge: {tag}**{' (primary)' if prim else ' (readout)'}", rows.get((arm, w), {}))
        for w, tag in (("sum+head0", "diag: summed adapter, branch-0 head"), ("half+head0", "diag: half adapter, branch-0 head"),
                       ("own0+avghead", "diag: branch-0 adapter, averaged head"), ("own0+head0", "diag: branch 0 re-scored")):
            line(f"{arm} {tag}", rows.get((arm, w), {}))
    out.append("")
    # ---- predictions ----
    def ok(c): return "✅ PASS" if c else "⛔ FAIL"
    if all(m.get(k) is not None for k in [("disjoint", "alone", 0), ("disjoint", "alone", 1), ("disjoint", 1.0), ("shared", 0.5)]):
        ba = (m[("disjoint", "alone", 0)] + m[("disjoint", "alone", 1)]) / 2
        out.append(f"* S4-P6 disjoint branch alone ≤ full − 1 sd: branches mean {ba:.4f} vs {ref_mean - ref_sd:.4f}  {ok(ba <= ref_mean - ref_sd)}")
        d = m[("disjoint", 1.0)]
        out.append(f"* S4-P7 disjoint sum > branches ({d:.4f} > {ba:.4f}) {ok(d > ba)} and within 2 sd of full "
                   f"(|{d - ref_mean:+.4f}| ≤ {2 * ref_sd:.4f}) {ok(abs(d - ref_mean) <= 2 * ref_sd)}; within 1 sd: {abs(d - ref_mean) <= ref_sd}")
        out.append(f"* S4-P8 disjoint sum ≥ shared mean: {d:.4f} vs {m[('shared', 0.5)]:.4f}  {ok(d >= m[('shared', 0.5)])}")
    t0 = [alone[("disjoint", 0)][s][2] for s in SEEDS if s in alone.get(("disjoint", 0), {})]
    if t0:
        per_epoch = statistics.median(t0) / 30
        out.append(f"* S4-P9 branch wall-clock on cuda:0 per epoch (train+eval) {per_epoch:.1f} s vs full cell {FULL_CELL_EPOCH_S} s "
                   f"= {per_epoch / FULL_CELL_EPOCH_S:.2f}×  {ok(per_epoch <= 0.6 * FULL_CELL_EPOCH_S)}")
        t1 = [alone[("disjoint", 1)][s][2] for s in SEEDS if s in alone.get(("disjoint", 1), {})]
        out.append(f"  (cuda:1, shared with another tenant, not a measurement: {statistics.median(t1) / 30:.1f} s per epoch)")
    text = "\n".join(out)
    print(text)
    with open(os.path.join(ROOT, "results/sgwa/S4b_table.md"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
