#!/usr/bin/env python3
"""`make synth-rng` (P6b): Yosys + OpenSTA of the on-chip dropout-mask generator rn_mkgen (MKW 16, STAGES 3) in the
vector-rate (NCORE 9, 16 keep bits / cycle) and GEMM-rate (NCORE 3, 16 bits / 3 cycles) configurations inside
flow/synth/wrap_rn_mkgen.v, through flow/synth/synth_vec.sh (same libraries, SDC and repair step as P4-P6a), plus the
STAGES 2 sensitivity.  Writes build/synth/rng_table.md; exit 1 if a default configuration misses 40 ns at ss after repair."""
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "build", "synth")
CFGS = [("vector rate", "NCORE=9 STAGES=3", True), ("GEMM rate", "NCORE=3 STAGES=3", True),
        ("vector rate, STAGES 2", "NCORE=9 STAGES=2", False)]


def g(txt, pat):
    m = re.search(pat, txt)
    return float(m.group(1)) if m else float("nan")


def run(cfg):
    label, params, gate = cfg
    d = os.path.join(OUT, "rn_mkgen_" + params.replace(" ", "_").replace("=", ""))
    t = time.time()
    r = subprocess.run(f"flow/synth/synth_vec.sh rn_mkgen {d} {params}", shell=True, cwd=ROOT, capture_output=True, text=True)
    txt = open(os.path.join(d, "summary.txt")).read() if os.path.exists(os.path.join(d, "summary.txt")) else ""
    stat = open(os.path.join(d, "stat.txt")).read() if os.path.exists(os.path.join(d, "stat.txt")) else ""
    ss, tt = g(txt, r"STA ss_125C_3v00 wns_setup (-?[\d.]+)"), g(txt, r"STA tt_025C_3v30 wns_setup (-?[\d.]+)")
    rep = g(txt, r"wns_setup (-?[\d.]+) wns_hold -?[\d.]+ \(after repair")
    hold = g(txt, r"STA ss_125C_3v00 wns_setup -?[\d.]+ wns_hold (-?[\d.]+)")
    area = g(stat, r"Chip area for module '\\wrap_rn_mkgen': ([\d.]+)")
    cells = g(stat, r"(?m)^\s+(\d+)\s+[\d.E+]+\s+cells")
    flops = len(re.findall(r"gf180mcu_\w+?__df", open(os.path.join(d, "nl.v")).read())) if os.path.exists(os.path.join(d, "nl.v")) else -1
    ok = r.returncode == 0 and area == area and (rep if rep == rep else ss) >= 0
    return label, params, gate, cells, flops, area, ss, rep, tt, hold, time.time() - t, ok


def main():
    os.makedirs(OUT, exist_ok=True)
    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(run, CFGS))
    lines = ["| configuration | parameters | cells | flops | area tt (um^2) | slack ss raw (ns) | slack ss repaired (ns) | "
             "slack tt raw (ns) | hold ss (ns) | runtime |", "|---|---|---|---|---|---|---|---|---|---|"]
    bad = 0
    for label, params, gate, cells, flops, area, ss, rep, tt, hold, dt, ok in res:
        bad += gate and not ok
        lines.append(f"| {label} | MKW=16 {params} | {cells:,.0f} | {flops:,} | {area:,.0f} | {ss:+.2f} | {rep:+.2f} | {tt:+.2f} | "
                     f"{hold:+.2f} | {dt / 60:.1f} min |" + ("" if ok or not gate else " **FAIL**"))
    lines.append("\nArea includes the wrapper's input flops (seed / step / site / layer / request: 129 flops).\n")
    open(os.path.join(OUT, "rng_table.md"), "w").write("\n".join(lines))
    print("\n".join(lines))
    print("SYNTH-RNG", "PASS" if bad == 0 else "FAIL")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
