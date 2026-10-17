#!/usr/bin/env python3
"""[S.4] reader -- the population run's RTE rows against [S.2]'s per-seed confirmation cells.

  env/bin/python scripts/s4_read.py [--dir scratchpad/s4/csv]

Scores S4-P2: for each S.2 row, the population's 5-seed mean must lie within one
seed s.d. (the larger of the two sets') of the single-process mean.  Seeds are
paired by label but NOT expected to match individually (different RNG streams).
"""
import argparse, csv, glob, os, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S2 = {"sgwa": "scratchpad/s2/csv/sgwa-D-c0-seed{s}.csv",
      "sgwaA": "scratchpad/s2a/csv/sgwaA-D-p2-s2-seed{s}.csv"}
SEEDS = [42, 43, 44, 45, 46]


def last(path):
    return list(csv.DictReader(open(path)))[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="scratchpad/s4/csv")
    a = ap.parse_args()
    print("| row | source | seeds 42..46 | mean ± sd | best epochs |")
    print("|---|---|---|---|---|")
    for arm in ("sgwa", "sgwaA"):
        ref = [float(last(os.path.join(ROOT, S2[arm].format(s=s)))["accuracy"]) for s in SEEDS]
        pop_paths = [os.path.join(ROOT, a.dir, f"rte-{arm}-seed{s}.csv") for s in SEEDS]
        if not all(os.path.exists(p) for p in pop_paths):
            print(f"| {arm} | population | ({sum(os.path.exists(p) for p in pop_paths)}/5 written) | | |")
            continue
        rows = [last(p) for p in pop_paths]
        pop = [float(r["accuracy"]) for r in rows]
        sd = max(statistics.stdev(ref), statistics.stdev(pop))
        for name, v, ep in (("[S.2] single-process", ref, None),
                            ("[S.4] population R=10", pop, [int(r["best_epoch"]) for r in rows])):
            print(f"| {arm} | {name} | " + ", ".join(f"{x:.4f}" for x in v) +
                  f" | {statistics.fmean(v):.4f} ± {statistics.stdev(v):.4f} | {ep or ''} |")
        d = statistics.fmean(pop) - statistics.fmean(ref)
        print(f"| {arm} | **S4-P2** | Δmean = {d:+.4f} vs one sd {sd:.4f} | "
              f"{'✅ PASS' if abs(d) <= sd else '⛔ FAIL'} | |")
    for r in sorted(glob.glob(os.path.join(ROOT, a.dir, "rte-*.csv")))[:1]:
        row = last(r)
        print(f"\npopulation wall-clock {float(row['train_time_s']) / 3600:.2f} h for R={row['pop_R']} "
              f"({row['pop_mode']}, chunks {row['pop_chunks']}), peak {float(row['peak_mem_mib']):,.0f} MiB"
              " -- measured on a SHARED GPU (S.3 running): not S4-P3's number.")


if __name__ == "__main__":
    main()
