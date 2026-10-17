#!/usr/bin/env python3
"""`make synth-ctrl` (P6c): Yosys + OpenSTA of the sequencer ct_seq (+ ct_dma) inside flow/synth/wrap_ct_seq.v (all
inputs registered), through flow/synth/synth_vec.sh (same libraries, SDC and repair step as P4-P6b).  Writes
build/synth/ctrl_table.md; exit 1 if it misses 40 ns at ss after repair."""
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "build", "synth")
CFGS = [("sequencer", "", True)]


def g(txt, pat):
    m = re.search(pat, txt)
    return float(m.group(1)) if m else float("nan")


def run(cfg):
    label, params, gate = cfg
    d = os.path.join(OUT, "ct_seq")
    t = time.time()
    r = subprocess.run(f"flow/synth/synth_vec.sh ct_seq {d} {params}", shell=True, cwd=ROOT, capture_output=True, text=True)
    txt = open(os.path.join(d, "summary.txt")).read() if os.path.exists(os.path.join(d, "summary.txt")) else ""
    stat = open(os.path.join(d, "stat.txt")).read() if os.path.exists(os.path.join(d, "stat.txt")) else ""
    ss, tt = g(txt, r"STA ss_125C_3v00 wns_setup (-?[\d.]+)"), g(txt, r"STA tt_025C_3v30 wns_setup (-?[\d.]+)")
    rep = g(txt, r"wns_setup (-?[\d.]+) wns_hold -?[\d.]+ \(after repair")
    hold = g(txt, r"STA ss_125C_3v00 wns_setup -?[\d.]+ wns_hold (-?[\d.]+)")
    area = g(stat, r"Chip area for module '\\wrap_ct_seq': ([\d.]+)")
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
        lines.append(f"| {label} | GP 36, VP 25, LP 10 | {cells:,.0f} | {flops:,} | {area:,.0f} | {ss:+.2f} | {rep:+.2f} | {tt:+.2f} | "
                     f"{hold:+.2f} | {dt / 60:.1f} min |" + ("" if ok or not gate else " **FAIL**"))
    lines.append("\nArea includes the wrapper's input flops (CSR bus, read beat, handshakes: ~320 flops).\n")
    open(os.path.join(OUT, "ctrl_table.md"), "w").write("\n".join(lines))
    print("\n".join(lines))
    print("SYNTH-CTRL", "PASS" if bad == 0 else "FAIL")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
