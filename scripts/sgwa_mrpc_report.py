#!/usr/bin/env python
"""[S.1] The joint task x cost table for SGWA on RoBERTa / MRPC, in the SCoRA
letter's conventions.  Every number is read from a file, never typed:

  task   [R.310] per-seed MRPC F1, frozen in results/sgwa/baselines_r310_mrpc.json
         SGWA: results/sgwa/mrpc/csv/*.csv (this study)
  cost   latency + activations: results/sgwa/timing.csv (r308 harness, all arms
         re-timed in one session, tf32 on, marginal fwd+bwd over a frozen Linear)
         flops: bench_adapter_cost's merged accounting, b = 4096, d = 768
Conventions (build_prl_tables): median of 5 seeds of the max-over-epochs F1, on
0-100 at one decimal, round-half-up; two-sided paired sign marks vs `fftm`.

  env/bin/python scripts/sgwa_mrpc_report.py [--md OUT.md] [--json OUT.json]
"""
import argparse, csv, glob, json, math, os, statistics, sys
from decimal import Decimal, ROUND_HALF_UP

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import bench_adapter_cost as BC
import r308_timing as T

SEEDS = [42, 43, 44, 45, 46]
D, B, K_ATOMS, NNZ = 768, 4096, 256, 25
FLOOR = 0.8122                                   # all-positive F1 on MRPC dev
REF = "fftm"
ROWS = [  # key, label, cost path, timing key
    ("fftm", "FourierFT (merged)", "merged", "fftm"),
    ("fftstock", "FourierFT (additive)", "additive", None),
    ("wave1", "WaveFT (mu=1)", "factored", "wave1"),
    ("wave2", "WaveFT (mu=2)", "factored", "wave2"),
    ("loca", "LoCA", "additive", "loca"),
    ("qwha", "QWHA", "factored", "qwha"),
    ("lyra", "LYRA", "factored", "lyra"),
    ("scora", "SCoRA (derived)", "merged", "scora"),
    ("scora2", "SCoRA (swept)", "merged", "scora"),
    ("sgwa", "**SGWA**", "merged", "sgwa"),
]
EXTRA_COST = [("scora", "SCoRA", "factored", "scora_factored"),
              ("sgwa", "**SGWA**", "factored", "sgwa_factored")]


def pct(v):
    return float(Decimal(repr(100 * v)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def sgwa_merged_flops():
    """Build = gamma*c (k) + c x h (k*25) + index_add (k*25); fold-in d^2, per token."""
    return (K_ATOMS + 2 * K_ATOMS * NNZ + D * D) / B


def sgwa_factored_flops():
    w = 7                                         # 2K+1 at K=3, box as implemented
    return 2 * K_ATOMS * w * w + K_ATOMS * w + K_ATOMS * w


def flops(key, path):
    if key.startswith("sgwa"):
        return sgwa_merged_flops() if path == "merged" else sgwa_factored_flops()
    return BC.arm_merged_per_token("scora" if key == "scora2" else key, D, B)


def load_sgwa():
    vals, eps = {}, {}
    for p in glob.glob(os.path.join(ROOT, "results", "sgwa", "mrpc", "csv", "*.csv")):
        for r in csv.DictReader(open(p)):
            vals[int(r["seed"])] = float(r["f1"])
            eps[int(r["seed"])] = int(r["best_epoch"])
    return vals, eps


def collect():
    base = json.load(open(os.path.join(ROOT, "results", "sgwa", "baselines_r310_mrpc.json")))
    f1 = {a: list(v) for a, v in base["f1"].items()}
    params = dict(base["params"])
    sv, se = load_sgwa()
    if set(sv) >= set(SEEDS):
        f1["sgwa"] = [sv[s] for s in SEEDS]
    params["sgwa"] = 6144
    rows = T.load(os.path.join(ROOT, "results", "sgwa", "timing.csv"))
    # r308 protocol, three independent sessions: median and range per arm
    sess = [T.marginal(T.load(os.path.join(ROOT, "results", "sgwa", f)), 1)
            for f in ("timing.csv", "timing_s2.csv", "timing_s3.csv")]
    marg = {a: {"fwdbwd": statistics.median(s_[a]["fwdbwd"] for s_ in sess),
                "lo": min(s_[a]["fwdbwd"] for s_ in sess),
                "hi": max(s_[a]["fwdbwd"] for s_ in sess)}
            for a in sess[0] if all(a in s_ for s_ in sess)}
    mem = {r["arm"]: float(r["mem_bytes"]) / 1024 ** 2 for r in rows
           if r["mem_bytes"] and int(r["tf32"]) == 1}
    rec = T.marginal(T.load(os.path.join("/workspace/lora_research_signal", "scratchpad",
                                         "phaseR", "r308", "timing.csv")), 1)
    pr = list(csv.DictReader(open(os.path.join(ROOT, "results", "sgwa", "timing_paired.csv"))))
    paired = {}
    for r in pr:
        paired.setdefault(r["arm"], []).append(float(r["marginal"]))
    paired = {a: statistics.median(v) for a, v in paired.items()}
    return f1, params, marg, mem, rec, se, paired


def mark(v, ref):
    w = sum(a > b for a, b in zip(v, ref))
    l = sum(b > a for a, b in zip(v, ref))
    return "⋆" if w == 5 else ("◦" if l == 5 else ""), w


def table():
    f1, params, marg, mem, rec, sgwa_eps, paired = collect()
    prank = {a: i + 1 for i, a in enumerate(sorted(paired, key=paired.get))}
    ref = f1[REF]
    meds = {a: statistics.median(v) for a, v in f1.items()}
    best = max(meds.values())
    out = ["| Method | Params | MRPC F1 (median) | wins vs FourierFT | mean ± sd | path | latency ms, median [range] of 3 sessions | paired ms (rank) | activations MiB | flops/token (k) |",
           "|---|---:|---:|:-:|---:|---|---:|---:|---:|---:|"]
    js = {}
    for key, lab, path, tk in ROWS:
        v = f1.get(key)
        if v is None:
            out.append(f"| {lab} | {params[key]:,} | (running) | | | {path} | | | |")
            continue
        mk, w = mark(v, ref) if key != REF else ("", None)
        m = pct(meds[key])
        cell = (f"**{m:.1f}**" if meds[key] == best else f"{m:.1f}") + mk
        lat = (f"{marg[tk]['fwdbwd']:.3f} [{marg[tk]['lo']:.3f}, {marg[tk]['hi']:.3f}]"
               if tk in marg else "n/t")
        pc = f"{paired[tk]:.3f} ({prank[tk]})" if tk in paired else "n/m"
        act = f"{mem[tk]:.3f}" if tk in mem else "n/t"
        fl = flops(key, path) / 1000
        out.append(f"| {lab} | {params[key]:,}{' †' if key == 'loca' else ''} | {cell} | "
                   f"{'ref' if key == REF else f'{w}/5'} | {100*statistics.mean(v):.2f} ± "
                   f"{100*statistics.stdev(v):.2f} | {path} | {lat} | {pc} | {act} | {fl:.2f} |")
        js[key] = dict(f1=v, median=meds[key], wins_vs_fftm=w, mark=mk,
                       latency_ms=marg.get(tk, {}).get("fwdbwd"), act_mib=mem.get(tk),
                       flops_per_token=flops(key, path), path=path)
    out.append("")
    out.append("| other execution paths | path | latency ms, median [range] | paired ms (rank) | activations MiB | flops/token (k) |")
    out.append("|---|---|---:|---:|---:|---:|")
    for key, lab, path, tk in EXTRA_COST:
        out.append(f"| {lab} | {path} | {marg[tk]['fwdbwd']:.3f} [{marg[tk]['lo']:.3f}, "
                   f"{marg[tk]['hi']:.3f}] | {paired[tk]:.3f} ({prank[tk]}) | {mem[tk]:.3f} | "
                   f"{flops(key, path)/1000:.2f} |")
    out.append("")
    out.append("Latency reproducibility vs the recorded [R.308] run (marginal fwd+bwd ms, tf32):")
    out.append("")
    out.append("| arm | this box, median [range] of 3 sessions | recorded R.308 |")
    out.append("|---|---:|---:|")
    for a in ["fftm", "scora", "scora_factored", "lyra", "loca", "wave1", "wave2", "qwha"]:
        if a in marg and a in rec:
            out.append(f"| {a} | {marg[a]['fwdbwd']:.3f} [{marg[a]['lo']:.3f}, {marg[a]['hi']:.3f}] "
                       f"| {rec[a]['fwdbwd']:.3f} |")
    if "sgwa" in f1:
        out.append("")
        out.append(f"SGWA per seed (42..46): " + ", ".join(f"{x:.4f}" for x in f1["sgwa"])
                   + f" | best epoch: {[sgwa_eps[s] for s in SEEDS]} | degenerate (<= {FLOOR}): "
                   + str(sum(x <= FLOOR for x in f1["sgwa"])))
    return "\n".join(out), js


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    md, js = table()
    print(md)
    if a.md:
        open(a.md, "w").write(md + "\n")
    if a.json:
        json.dump(js, open(a.json, "w"), indent=1)
