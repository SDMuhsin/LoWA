#!/usr/bin/env python3
"""[S.3] reader -- the SCoRA paper's left (RoBERTa) table with SGWA's two rows.

  env/bin/python scripts/s3_read.py [--md OUT.md]

Conventions are the paper's (`build_prl_tables`): median over seeds 42-46 of the
max-over-30-epochs primary metric (CB/MRPC F1, STS-B Pearson, CoLA MCC, else
accuracy), 0-100 at one decimal round-half-up; marks two-sided paired against
FourierFT merged (⋆ beats it 5/5, ◦ loses 5/5); Avg. = mean of PRINTED medians.
Baselines: results/sgwa/baselines_r310_all.json (frozen [R.305]/[R.310] values).
SGWA RTE: [S.2] (both rows).  Others: scratchpad/s3/csv.
"""
import argparse, csv, glob, json, os, statistics, sys
from decimal import Decimal, ROUND_HALF_UP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = [("rte", "RTE"), ("cb", "CB"), ("mrpc", "MRPC"), ("stsb", "STS-B"),
         ("cola", "CoLA"), ("boolq", "BoolQ"), ("sst2", "SST-2")]
SEEDS = [42, 43, 44, 45, 46]
ROWS = [("fftm", "FourierFT (merged)"), ("fftstock", "FourierFT (additive)"),
        ("wave1", "WaveFT (mu=1)"), ("wave2", "WaveFT (mu=2)"), ("loca", "LoCA †"),
        ("qwha", "QWHA"), ("lyra", "LYRA"), ("scora", "SCoRA (derived)"),
        ("scora2", "SCoRA (swept)"),
        ("sgwa", "SGWA (full search)"), ("sgwaA", "SGWA (grid-only)")]
S2_RTE = {  # [S.2] §2 and §7.1, seeds 42-46
    "sgwa": "scratchpad/s2/csv/sgwa-D-c0-seed{s}.csv",
    "sgwaA": "scratchpad/s2a/csv/sgwaA-D-p2-s2-seed{s}.csv",
}


def q(x):
    return Decimal(repr(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def _row(path):
    return list(csv.DictReader(open(path)))[-1]


def collect():
    base = json.load(open(os.path.join(ROOT, "results", "sgwa", "baselines_r310_all.json")))
    vals = {t: dict(v) for t, v in base["vals"].items()}
    eps = {}
    for arm, pat in S2_RTE.items():
        ps = [os.path.join(ROOT, pat.format(s=s)) for s in SEEDS]
        if all(os.path.exists(p) for p in ps):
            vals["rte"][arm] = [float(_row(p)["accuracy"]) for p in ps]
    for t, _ in TASKS[1:]:
        m = base["metric"][t]
        for arm in ("sgwa", "sgwaA"):
            ps = [os.path.join(ROOT, "scratchpad", "s3", "csv", f"{t}-{arm}-seed{s}.csv") for s in SEEDS]
            have = [p for p in ps if os.path.exists(p)]
            if len(have) == len(SEEDS):
                rows = [_row(p) for p in ps]
                vals[t][arm] = [float(r[m]) for r in rows]
                eps[(t, arm)] = [int(r["best_epoch"]) for r in rows]
            elif have:
                vals[t][arm] = None
                eps[(t, arm)] = len(have)
    return vals, base["floor"], eps


def table():
    vals, floor, eps = collect()
    hdr = "| Method | " + " | ".join(lbl for _, lbl in TASKS) + " | Avg. |"
    out = [hdr, "|---" * (len(TASKS) + 2) + "|"]
    best = {t: max(statistics.median(v) for v in vals[t].values() if v) for t, _ in TASKS}
    for arm, lbl in ROWS:
        cells, printed = [], []
        for t, _ in TASKS:
            v = vals[t].get(arm)
            if v is None:
                done = eps.get((t, arm))
                cells.append(f"({done}/5)" if isinstance(done, int) else "·")
                continue
            ref = vals[t]["fftm"]
            w = sum(a > b for a, b in zip(v, ref))
            l = sum(b > a for a, b in zip(v, ref))
            mk = "" if arm == "fftm" else ("⋆" if w == 5 else "◦" if l == 5 else "")
            m = statistics.median(v)
            printed.append(q(100 * m))
            s = f"{q(100 * m)}"
            cells.append((f"**{s}**" if m == best[t] else s) + mk)
        avg = (f"{(sum(printed) / len(printed)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}"
               if len(printed) == len(TASKS) else "")
        out.append(f"| {lbl} | " + " | ".join(cells) + f" | {avg} |")
    sd = []
    for t, _ in TASKS:
        sds = [statistics.stdev(v) for a, v in vals[t].items() if v and not a.startswith("sgwa")]
        sd.append(f"{100 * statistics.median(sds):.2f}")
    out.append("| *seed s.d. (baselines)* | " + " | ".join(sd) + " | |")
    notes = ["", "SGWA detail (per seed 42..46, argmax epoch of 0-29, degenerate = at/below the task floor):"]
    for t, _ in TASKS[1:]:
        for arm in ("sgwa", "sgwaA"):
            v = vals[t].get(arm)
            if v:
                ep = eps[(t, arm)]
                deg = sum(x <= floor[t] + 1e-9 for x in v)
                late = sum(e >= 27 for e in ep)
                notes.append(f"- {t} {arm}: " + ", ".join(f"{x:.4f}" for x in v)
                             + f" | epochs {ep} | degenerate {deg}/5 | argmax in last 10%: {late}/5")
    return "\n".join(out + notes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md")
    a = ap.parse_args()
    md = table()
    print(md)
    if a.md:
        open(a.md, "w").write(md + "\n")
