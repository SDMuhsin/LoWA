#!/usr/bin/env python
"""Module-level wall-clock and activation memory for SGWA beside every arm of
the SCoRA cost table, by `scripts/r308_timing.py`'s OWN harness, unmodified.

  env/bin/python scripts/sgwa_timing.py --run      # measure (writes CSV)
  env/bin/python scripts/sgwa_timing.py            # read

Every hygiene rule is r308's: CUDA events, warmup discarded, median over REPS,
the whole sweep repeated REPEATS times in randomised arm order, MIN block-median
as the estimator, the 0-sync gate at measurement time, marginal over a frozen
nn.Linear of the same shape (d=768, k=256, b=32x128 tokens).  Only the arm list
and the output path differ: the baselines are re-timed IN THE SAME SESSION as
SGWA, so the latency column compares like with like.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import r308_timing as T

OUT = os.path.join(ROOT, "results", "sgwa", "timing.csv")
SGWA_ARMS = ["sgwa", "sgwa_factored"]
T.TITLES.update({"sgwa": "SGWA (merged: index_add into W0 copy)",
                 "sgwa_factored": "SGWA (gather / mix / scatter, no dW)"})
_build = T.build


def build(arm, device, dtype):
    if arm in SGWA_ARMS:
        import torch.nn as nn
        from sgwa_adapter import SGWALinear
        base = nn.Linear(T.D_MODEL, T.D_MODEL, bias=True).to(device=device, dtype=dtype)
        for p in base.parameters():
            p.requires_grad_(False)
        return SGWALinear(base, k=T.K, K=3, seed=777,
                          materialise=(arm == "sgwa")).to(device)
    return _build(arm, device, dtype)


T.build = build
T.OUT = OUT
T.ARM_ORDER = T.ARM_ORDER + SGWA_ARMS

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--session", default="", help="independent re-measurement: writes timing_<session>.csv")
    a = ap.parse_args()
    if a.session:
        OUT = T.OUT = os.path.join(ROOT, "results", "sgwa", f"timing_{a.session}.csv")
    if a.run:
        T.run(a.device)
    rows = T.load(OUT)
    if not rows:
        print("no measurement yet -- run with --run")
        sys.exit(0)
    est, marg = T.estimate(rows, 1), T.marginal(rows, 1)
    mem = {r["arm"]: float(r["mem_bytes"]) / 1024 ** 2 for r in rows
           if r["mem_bytes"] and int(r["tf32"]) == 1}
    print(f"{'arm':40s} {'marg fwd':>9s} {'marg f+b':>9s} {'spread':>8s} {'act MiB':>8s}")
    for arm in sorted(marg, key=lambda x: marg[x]["fwdbwd"]):
        print(f"{T.TITLES.get(arm, arm):40s} {marg[arm]['fwd']:9.4f} {marg[arm]['fwdbwd']:9.4f} "
              f"{est[arm]['fwdbwd'][1]:8.4f} {mem.get(arm, float('nan')):8.3f}")
