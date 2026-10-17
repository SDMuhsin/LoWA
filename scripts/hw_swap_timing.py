#!/usr/bin/env python
"""Measured task-swap time on merged weights, W <- W - dW_old + dW_new, in place,
fp32, one module, LoWA's sparse index_add_ against each baseline's own dense
merge path.  cuda:0 ONLY.

  CUDA_VISIBLE_DEVICES=0 env/bin/python scripts/hw_swap_timing.py

Hygiene: TF32 off; every arm gets WARMUP discarded swaps; then REPS rounds, arm
order shuffled per round (fixed seed), each swap timed by CUDA events with a
synchronize before and after; the median over rounds is reported.  Each round times one isolated swap
(launch-bound, kept as `<d>_single`) and INNER back-to-back swaps between one
event pair (per-swap time, the reported figure: launch latency amortised equally).  Writes
results/sgwa/HW_swap_timing.json (read by scripts/hw_cost.py).
"""
import json, math, os, random, statistics, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import hw_cost as H                                     # noqa: E402
from haar_adapter import haar_matrix                    # noqa: E402
from qwha_adapter import _wht_sync_free                 # noqa: E402

WARMUP, REPS, INNER = 30, 300, 20
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "cuda:0 only"


def arms(d):
    """name -> zero-arg callable performing one in-place swap of W."""
    torch.manual_seed(0)
    out, keep = {}, []
    for name in H.ORDER:
        m, _ = H.build(name, d)
        m = m.float()
        W = m.base_layer.weight.data.clone()
        keep.append((m, W))
        if name == "lowa":
            dc = torch.randn(H.K_COEF, device=H.DEV)
            idx, hv, g = m.eflat, m.hval, m.scaling
            def f(W=W, dc=dc, idx=idx, hv=hv, g=g):
                W.view(-1).index_add_(0, idx, ((g * dc)[:, None] * hv).reshape(-1))
        elif name == "fourierft":
            def f(W=W, m=m):
                W.add_(m.get_delta_weight())
        elif name in ("wave1", "wave2"):
            Hm = haar_matrix(d, torch.float32).to(H.DEV)
            vals = (m.spectrum * m.scaling)[m.pidx]
            def f(W=W, m=m, Hm=Hm, vals=vals):
                C = torch.zeros(d, d, device=H.DEV).index_put((m.rows, m.cols), vals)
                W.add_(Hm.T @ C @ Hm)
        elif name == "loca":
            def f(W=W, m=m):                     # locations differ: build both tasks
                W.sub_(m.get_delta_weight())
                W.add_(m.get_delta_weight())
        elif name == "qwha":
            vals = m.spectrum * m.scaling / math.sqrt(d)
            def f(W=W, m=m, vals=vals):
                S = torch.zeros(d, d, device=H.DEV).index_put(
                    (m.indices[0], m.indices[1]), vals, accumulate=True)
                W.add_(_wht_sync_free(S) / d)
        elif name == "lyra":
            def f(W=W, m=m):
                W.add_(m.get_delta_weight())
        out[name] = f
    return out, keep


@torch.no_grad()
def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    res = {"device": torch.cuda.get_device_name(0), "warmup": WARMUP, "reps": REPS}
    rng = random.Random(7)
    for d in H.WIDTHS:
        fns, keep = arms(d)
        for f in fns.values():
            for _ in range(WARMUP):
                f()
        torch.cuda.synchronize()
        t = {n: [] for n in fns}
        t1 = {n: [] for n in fns}
        names = list(fns)
        for _ in range(REPS):
            rng.shuffle(names)
            for n in names:
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                a.record(); fns[n](); b.record()
                torch.cuda.synchronize()
                t1[n].append(a.elapsed_time(b) * 1e3)         # us, one swap (launch-bound)
                torch.cuda.synchronize()
                a.record()
                for _ in range(INNER):
                    fns[n]()
                b.record()
                torch.cuda.synchronize()
                t[n].append(a.elapsed_time(b) * 1e3 / INNER)  # us per swap, back to back
        res[str(d)] = {n: statistics.median(v) for n, v in t.items()}
        res[str(d) + "_single"] = {n: statistics.median(v) for n, v in t1.items()}
        res[str(d) + "_iqr"] = {n: (statistics.quantiles(v, n=4)[0], statistics.quantiles(v, n=4)[2])
                                for n, v in t.items()}
        print(d, {n: round(v, 1) for n, v in res[str(d)].items()}, flush=True)
        del fns, keep
        torch.cuda.empty_cache()
    json.dump(res, open(os.path.join(ROOT, "results", "sgwa", "HW_swap_timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
