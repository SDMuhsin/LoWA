#!/usr/bin/env python
"""[S.1] PAIRED latency for the sub-millisecond tier, where r308's protocol
(marginal of min block-medians from separate blocks) did not reproduce across
three sessions on this box (results/sgwa/timing{,_s2,_s3}.csv).

Each ROUND times the frozen Linear and every arm back to back (order shuffled
per round) with r308's own `_time_block`, and records arm - frozen WITHIN the
round, so slow drift (clocks, co-tenants) cancels.  Reported: median over
rounds of the per-round marginal, its IQR, and how often each arm beats each
other arm in the same round.  Same builders, input, dtype and sync gate as r308.

  env/bin/python scripts/sgwa_paired_timing.py [--rounds 40]
"""
import argparse, csv, os, random, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import sgwa_timing  # noqa: F401  (installs the SGWA builders into r308)
import r308_timing as T

ARMS = ["sgwa", "scora", "fftm", "lyra", "scora_factored", "sgwa_factored"]
OUT = os.path.join(ROOT, "results", "sgwa", "timing_paired.csv")


def main(rounds, device="cuda:0"):
    import torch
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    x = torch.randn(T.B_TOKENS, T.D_MODEL, device=device)
    mods = {a: T.build(a, device, torch.float32) for a in ["frozen"] + ARMS}
    for a, m in mods.items():
        assert T.sync_count(m, x) == 0, f"{a} fails the 0-sync gate"
    rows = []
    for r in range(rounds):
        order = ["frozen"] + ARMS
        random.Random(r).shuffle(order)
        t = {a: T._time_block(mods[a], x, "fwdbwd", 20, 5) for a in order}
        for a in ARMS:
            rows.append(dict(round=r, arm=a, ms=t[a], frozen_ms=t["frozen"],
                             marginal=t[a] - t["frozen"]))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = {a: [r["marginal"] for r in rows if r["arm"] == a] for a in ARMS}
    print(f"{'arm':16s} {'median marg ms':>15s} {'IQR':>17s}   (fwd+bwd, tf32, {rounds} rounds)")
    for a in sorted(ARMS, key=lambda a: statistics.median(by[a])):
        q = statistics.quantiles(by[a], n=4)
        print(f"{a:16s} {statistics.median(by[a]):15.4f}   [{q[0]:.4f}, {q[2]:.4f}]")
    print("\nfraction of rounds the ROW arm is faster than the COLUMN arm:")
    print(" " * 16 + "".join(f"{b[:10]:>11s}" for b in ARMS))
    for a in ARMS:
        cells = []
        for b in ARMS:
            if a == b:
                cells.append(f"{'-':>11s}")
                continue
            w = sum(x < y for x, y in zip(by[a], by[b])) / rounds
            cells.append(f"{w:11.2f}")
        print(f"{a:16s}" + "".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
    main(ap.parse_args().rounds)
