#!/usr/bin/env python3
"""[S.4c] reader: block-owned branches with periodic exact exchange (prereg llmdocs/S4c_btm_sync_prereg.md).
Rows from scratchpad/s4c/csv/rte-sync{E,S}-seed{s}-b{0,1}.csv (both ranks must agree).  Scores S4-P10..P12
and prints the [S.5] rule's pick.  Writes results/sgwa/S4c_table.md."""
import csv, os, statistics, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [42, 43, 44, 45, 46]
FULL = [float(list(csv.DictReader(open(os.path.join(ROOT, f"scratchpad/s2/csv/sgwa-D-c0-seed{s}.csv"))))[-1]["accuracy"]) for s in SEEDS]
ONESHOT = [0.6065, 0.6282, 0.6209, 0.6137, 0.6498]      # S4b_table.md, disjoint merge: sum
out = ["| row | seeds 42..46 | median | mean ± sd | best epochs | exchange MiB / run | sync s / epoch |", "|---|---|---:|---|---|---:|---:|"]
fm, fs = statistics.fmean(FULL), statistics.stdev(FULL)
out.append(f"| [S.2] full data, one process | " + ", ".join(f"{x:.4f}" for x in FULL) + f" | {statistics.median(FULL):.4f} | {fm:.4f} ± {fs:.4f} | | 0 | 0 |")
out.append(f"| [S.4b] one-shot sum (0 exchange) | " + ", ".join(f"{x:.4f}" for x in ONESHOT) + f" | {statistics.median(ONESHOT):.4f} | {statistics.fmean(ONESHOT):.4f} ± {statistics.stdev(ONESHOT):.4f} | | 0 | 0 |")
means = {"one-shot": statistics.fmean(ONESHOT)}
for arm, name in (("E", "arm E: exchange at epoch end"), ("S", "arm S: exchange every 10 steps")):
    v, ep, mib, sync = [], [], [], []
    for s in SEEDS:
        ps = [os.path.join(ROOT, f"scratchpad/s4c/csv/rte-sync{arm}-seed{s}-b{g}.csv") for g in (0, 1)]
        if not all(os.path.exists(p) for p in ps):
            continue
        r = [list(csv.DictReader(open(p)))[-1] for p in ps]
        assert abs(float(r[0]["accuracy"]) - float(r[1]["accuracy"])) < 1e-9, f"ranks disagree seed {s}"
        v.append(float(r[0]["accuracy"])); ep.append(int(r[0]["best_epoch"])); mib.append(int(r[0]["sync_bytes"]) / 2**20)
        log = open(os.path.join(ROOT, f"scratchpad/s4c/log/rte-sync{arm}-seed{s}.log")).read()
        sync.append(statistics.fmean(float(x) for x in re.findall(r"sync ([0-9.]+)s", log)))
    if len(v) < 5:
        out.append(f"| {name} | ({len(v)}/5 done) " + (", ".join(f"{x:.4f}" for x in v)) + " | | | | | |"); continue
    means[arm] = statistics.fmean(v)
    out.append(f"| **{name}** | " + ", ".join(f"{x:.4f}" for x in v) + f" | {statistics.median(v):.4f} | {means[arm]:.4f} ± {statistics.stdev(v):.4f} | {ep} | {statistics.fmean(mib):.1f} | {statistics.fmean(sync):.2f} |")
out.append("")
ok = lambda c: "✅ PASS" if c else "⛔ FAIL"
if "E" in means:
    e = means["E"]
    out.append(f"* S4-P10 arm E ≥ one-shot + 0.03 ({e:.4f} vs {means['one-shot'] + 0.03:.4f}) {ok(e >= means['one-shot'] + 0.03)}; "
               f"within 2 sd of full (|{e - fm:+.4f}| ≤ {2 * fs:.4f}) {ok(abs(e - fm) <= 2 * fs)}; within 1 sd: {abs(e - fm) <= fs}")
if "E" in means and "S" in means:
    out.append(f"* S4-P11 arm S ≥ arm E: {means['S']:.4f} vs {means['E']:.4f} {ok(means['S'] >= means['E'])}")
out.append("* S4-P12 exchange < 1 s per epoch (arm E): see the sync column")
if len(means) == 3:
    best = max(means, key=means.get)
    out.append(f"* **[S.5] rule picks: {best}** (means one-shot {means['one-shot']:.4f}, E {means['E']:.4f}, S {means['S']:.4f})")
text = "\n".join(out)
print(text)
open(os.path.join(ROOT, "results/sgwa/S4c_table.md"), "w").write(text + "\n")
