#!/usr/bin/env python3
"""make test-mem-banks (P8a): the mm_sram_bank port contract on every technology backend (generic, gf180, tsmc65 with the
behavioural clock gate), tb/mem/tb_mm_sram_bank.sv vs its reference array, in Icarus and Verilator, three bank shapes.
gf180 is checked on read data only (its macros update q on a write to the last-read row; the hold check is off).
Every run has a fixed op count and a wall-clock timeout.  Summary: build/test-banks/summary.txt, exit 1 on any FAIL."""
import glob
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "build", "test-banks")
HW = " ".join(os.path.relpath(f, ROOT) for f in sorted(glob.glob(os.path.join(ROOT, "hw", "**", "*.v"), recursive=True)))
TECH = {"generic": 0, "gf180": 1, "tsmc65": 2}
SHAPES = [(768, 32, 10), (256, 32, 8), (300, 24, 9)]
N = 100000


def run(job):
    sim, tech, (w, wd, aw) = job
    tag = f"{sim}_{tech}_{w}x{wd}"
    d = f"-DMM_TECH={TECH[tech]}" + (" -DNO_HOLD_CHECK" if tech == "gf180" else "")
    if sim == "icarus":
        exe = os.path.join(OUT, tag + ".vvp")
        b = (f"iverilog -g2012 {d} -Ptb_mm_sram_bank.WORDS={w} -Ptb_mm_sram_bank.WIDTH={wd} -Ptb_mm_sram_bank.AW={aw} "
             f"-Ptb_mm_sram_bank.N={N} -o {exe} tb/mem/tb_mm_sram_bank.sv {HW}")
        r_cmd = f"vvp -n {exe}"
    else:
        mdir = os.path.join(OUT, tag)
        b = (f"verilator --binary -j 2 --timing -O2 -Wno-fatal -Wno-lint -Wno-style {d.replace('-D', '+define+')} "
             f"-GWORDS={w} -GWIDTH={wd} -GAW={aw} -GN={N} --top-module tb_mm_sram_bank -Mdir {mdir} tb/mem/tb_mm_sram_bank.sv {HW}")
        r_cmd = os.path.join(mdir, "Vtb_mm_sram_bank")
    log = os.path.join(OUT, tag + ".log")
    try:
        rb = subprocess.run(b, shell=True, cwd=ROOT, capture_output=True, text=True, timeout=1800)
        rr = subprocess.run(r_cmd, shell=True, cwd=ROOT, capture_output=True, text=True, timeout=1800) if rb.returncode == 0 else rb
        txt = rb.stdout + rb.stderr + rr.stdout + rr.stderr
    except subprocess.TimeoutExpired:
        txt = "FAIL timeout"
    open(log, "w").write(txt)
    m = re.search(r"^(PASS|FAIL) mm_sram_bank.*$", txt, re.M)
    return (sim, tech, f"{w}x{wd} AW{aw}", m.group(0) if m else "FAIL (no result line)")


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(s, t, sh) for s in ("icarus", "verilator") for t in TECH for sh in SHAPES]
    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(run, jobs))
    lines = ["| simulator | backend | shape | result |", "|---|---|---|---|"] + [f"| {a} | {b} | {c} | {r} |" for a, b, c, r in rows]
    bad = sum(not r[3].startswith("PASS") for r in rows)
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("TEST-MEM-BANKS", "PASS" if bad == 0 else f"FAIL ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
