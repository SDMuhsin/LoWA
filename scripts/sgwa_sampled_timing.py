#!/usr/bin/env python
"""Dense (autograd G^T X) vs SAMPLED (SDDMM over the atom supports) backward for
the merged SGWA/LoWA module, by `sgwa_paired_timing.py`'s PAIRED protocol and
r308's own `_time_block` / sync gate (fwd+bwd, tf32 on, as Table III).

Each round times the frozen Linear, `sgwa` (dense) and `sgwa_sampled` back to
back in shuffled order and records arm - frozen within the round.  Peak memory:
max_memory_allocated over ONE fwd+bwd, minus the frozen Linear's own, input
excluded.  d = 768 and 2048, b = 4096 tokens, k = 256, K = 3, seed 777.

  CUDA_VISIBLE_DEVICES=0 env/bin/python scripts/sgwa_sampled_timing.py [--rounds 40]
"""
import argparse, csv, gc, os, random, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import r308_timing as T

OUT_CSV = os.path.join(ROOT, "results", "sgwa", "timing_sampled.csv")
OUT_TXT = os.path.join(ROOT, "results", "sgwa", "sampled_grad_timing.txt")
ARMS = ["sgwa", "sgwa_sampled"]


def build(arm, d, device):
    import torch, torch.nn as nn
    from sgwa_adapter import SGWALinear
    base = nn.Linear(d, d, bias=True).to(device)
    for p in base.parameters():
        p.requires_grad_(False)
    if arm == "frozen":
        return base
    return SGWALinear(base, k=T.K, K=3, seed=777, materialise=True,
                      sampled_grad=(arm == "sgwa_sampled")).to(device)


def peak_fwdbwd(mod, x):
    import torch
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    xr = x.detach().requires_grad_(True)
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    mod(xr).square().mean().backward()
    torch.cuda.synchronize()
    pk = torch.cuda.max_memory_allocated() - before
    mod.zero_grad(set_to_none=True)
    return pk


def main(rounds, device="cuda:0"):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rows, lines = [], []
    lines.append(f"[LoWA sampled backward] paired fwd+bwd marginal over frozen nn.Linear, tf32, "
                 f"b={T.B_TOKENS}, k={T.K}, K=3, {rounds} rounds, {torch.cuda.get_device_name(device)}")
    for d in (768, 2048):
        torch.manual_seed(0)
        x = torch.randn(T.B_TOKENS, d, device=device)
        mods = {a: build(a, d, device) for a in ["frozen"] + ARMS}
        for a, m in mods.items():
            assert T.sync_count(m, x) == 0, f"{a} fails the 0-sync gate"
        mem = {a: peak_fwdbwd(mods[a], x) for a in ["frozen"] + ARMS}
        for r in range(rounds):
            order = ["frozen"] + ARMS
            random.Random(1000 * d + r).shuffle(order)
            t = {a: T._time_block(mods[a], x, "fwdbwd", 20, 5) for a in order}
            for a in ARMS:
                rows.append(dict(d=d, round=r, arm=a, ms=t[a], frozen_ms=t["frozen"],
                                 marginal=t[a] - t["frozen"],
                                 peak_bytes=mem[a], frozen_peak_bytes=mem["frozen"]))
        by = {a: [r_["marginal"] for r_ in rows if r_["d"] == d and r_["arm"] == a] for a in ARMS}
        raw = {a: [r_["ms"] for r_ in rows if r_["d"] == d and r_["arm"] == a] for a in ARMS}
        fz = [r_["frozen_ms"] for r_ in rows if r_["d"] == d and r_["arm"] == ARMS[0]]
        lines.append(f"\nd={d}  frozen fwd+bwd median {statistics.median(fz):.4f} ms, "
                     f"frozen peak {mem['frozen'] / 2**20:.2f} MiB")
        lines.append(f"  {'arm':14s} {'marg ms (med)':>14s} {'IQR':>20s} {'abs ms':>8s} {'marg peak MiB':>14s}")
        for a in ARMS:
            q = statistics.quantiles(by[a], n=4)
            lines.append(f"  {a:14s} {statistics.median(by[a]):14.4f}   [{q[0]:.4f}, {q[2]:.4f}] "
                         f"{statistics.median(raw[a]):8.4f} {(mem[a] - mem['frozen']) / 2**20:14.2f}")
        w = sum(s < dd for s, dd in zip(by["sgwa_sampled"], by["sgwa"])) / rounds
        lines.append(f"  sampled faster than dense in {w:.2f} of rounds")
        del mods, x
        gc.collect(); torch.cuda.empty_cache()
    with open(OUT_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    txt = "\n".join(lines) + "\n"
    open(OUT_TXT, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    main(a.rounds, a.device)
