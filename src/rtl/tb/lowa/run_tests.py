#!/usr/bin/env python3
"""make test-lowa: tb_lw_top cases (Verilator; Icarus smoke on the small ones), goldens from L1 via gen_lowa.py.
   run_tests.py [--quick] [--cases a,b] [--icarus-only]   Summary: build/test-lowa/summary.txt, exit 1 on any FAIL."""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = sys.executable
OUT = os.path.join(ROOT, "build", "test-lowa")
HW = sorted(glob.glob(os.path.join(ROOT, "hw", "**", "*.v"), recursive=True))
ALL = ["merge_unit", "sg_unit", "sg_edge", "scal_s0", "scal_peak", "adamw_s0", "adamw_clip", "adamw_noclip",
       "adamw_real_coef", "adamw_real_head", "merge_real_row", "merge_real_trn", "sg_real_l11q", "sg_real_l0q",
       "nomerge_gemm", "merge_gemm"]
GEMM = ["nomerge_gemm", "merge_gemm"]            # tb_lw_gemm (gm_top + lw_top), MEMW 23
QUICK = ALL[:8]
ICARUS = ["sg_edge", "scal_s0", "adamw_clip", "merge_unit"]


def sh(cmd, log):
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
    with open(log, "w") as f:
        f.write(cmd + "\n" + r.stdout + r.stderr)
    return r


def build_vl(name, params, top="tb_lw_top"):
    mdir = os.path.join(OUT, "sim", name)
    exe = os.path.join(mdir, "V" + top)
    g = " ".join(f"-G{k}={v}" for k, v in params.items())
    r = sh(f"verilator --binary -j 8 --timing -O3 -Wno-fatal -Wno-lint -Wno-style {g} --top-module {top} -Mdir {mdir} "
           f"tb/lowa/{top}.sv {' '.join(os.path.relpath(f, ROOT) for f in HW)}", mdir + ".build.log")
    if r.returncode or not os.path.exists(exe):
        raise RuntimeError(f"verilator build failed: {mdir}.build.log")
    return exe


def build_iv(name, params):
    out = os.path.join(OUT, "sim", name + ".vvp")
    p = " ".join(f"-Ptb_lw_top.{k}={v}" for k, v in params.items())
    r = sh(f"iverilog -g2012 {p} -o {out} tb/lowa/tb_lw_top.sv {' '.join(os.path.relpath(f, ROOT) for f in HW)}", out + ".build.log")
    if r.returncode:
        raise RuntimeError(f"iverilog failed: {out}.build.log")
    return f"vvp -n {out}"


def run(exe, case, tag, extra=""):
    d = os.path.join(OUT, "vec")
    log = os.path.join(OUT, f"{case}_{tag}.log")
    t = time.time()
    r = sh(f"{exe} +desc={d}/{case}.desc +mem={d}/{case}.mem.hex +exp={d}/{case}.exp.hex {extra}", log)
    txt = r.stdout
    m = re.search(r"^(PASS|FAIL) lw_(top|gemm) (.*)$", txt, re.M)
    cyc = re.search(r"CYCLES (\d+)", txt)
    mg = re.search(r"MERGE (.*)", txt)
    return (m.group(0) if m else "FAIL (no result line)", int(cyc.group(1)) if cyc else -1, mg.group(1) if mg else "", time.time() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cases")
    ap.add_argument("--icarus-only", action="store_true")
    a = ap.parse_args()
    cases = a.cases.split(",") if a.cases else (QUICK if a.quick else ALL)
    os.makedirs(os.path.join(OUT, "sim"), exist_ok=True)
    rows, bad = [], 0
    for c in cases:
        r = sh(f"{PY} tb/lowa/gen_lowa.py {c} {OUT}/vec", os.path.join(OUT, f"gen_{c}.log"))
        if r.returncode:
            rows.append((c, "gen", "FAIL (generator)", -1, "", 0)); bad += 1
            print(open(os.path.join(OUT, f"gen_{c}.log")).read()[-800:])
    if not a.icarus_only:
        if [c for c in cases if c not in GEMM]:
            exe = build_vl("vl", {"MEMW": 23})
        for c in [c for c in cases if c not in GEMM]:
            for tag, extra in (("lat24", ""), ("stall30", "+lat=7 +stall=30")) if c in QUICK else (("lat24", ""),):
                res = run(exe, c, "vl_" + tag, extra)
                rows.append((c, "verilator " + tag) + res); bad += not res[0].startswith("PASS")
                print(c, tag, res[0], res[1], res[2], f"{res[3]:.0f}s", flush=True)
        if [c for c in cases if c in GEMM]:
            exe = build_vl("vlg", {"MEMW": 23}, "tb_lw_gemm")
        for c in [c for c in cases if c in GEMM]:
            res = run(exe, c, "vl_lat24")
            rows.append((c, "verilator lat24 (gm_top + lw_top)") + res); bad += not res[0].startswith("PASS")
            print(c, "lat24", res[0], res[1], res[2], f"{res[3]:.0f}s", flush=True)
    for c in [x for x in ICARUS if x in cases]:
        exe = build_iv("iv", {"MEMW": 15})
        res = run(exe, c, "iv", "+lat=5")
        rows.append((c, "icarus MEMW15") + res); bad += not res[0].startswith("PASS")
        print(c, "icarus", res[0], res[1], res[2], f"{res[3]:.0f}s", flush=True)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write("| case | sim | result | cycles | merge stats | time |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]:.0f}s |\n")
        f.write("TEST-LOWA " + ("FAIL" if bad else "PASS") + "\n")
    print("TEST-LOWA", "FAIL" if bad else "PASS")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
