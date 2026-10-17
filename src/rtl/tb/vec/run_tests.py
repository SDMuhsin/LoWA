#!/usr/bin/env python3
"""`make test-vec`: L2 tests of hw/vec against L1 (bit-exact), docs/p4_report.md.
  1. vu_top V = 16 (Verilator, memory model): the unit cases (every op, directed corners + random)
     and the end-to-end roberta-base cases from the L1 step-B replay (tb/vec/gen_vec.py), cycles
     vs docs/calc/budget.py's vector model
  2. unit cases with random memory stalls / latencies (Verilator)
  3. vu_top V = 4 Icarus smoke (small cases, small testbench memories)
Prints a table, writes build/test-vec/summary.txt; exit 1 on any FAIL / missing result.
  run_tests.py [--quick] [--no-real] [--jobs N] [--rebuild]
"""
import argparse
import concurrent.futures as cf
import glob
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = sys.executable
VEC = os.path.join(ROOT, "build", "vec")
OUT = os.path.join(ROOT, "build", "test-vec")
SIM = os.path.join(ROOT, "build", "sim", "vec")
sys.path.insert(0, os.path.join(ROOT, "docs", "calc"))
import budget  # noqa: E402

UNIT_CASES = ["elem", "ln_fwd", "ln_bwd", "smax", "emb", "loss", "gnorm"]
REAL_CASES = ["real_loss", "real_gnorm", "real_ln_fwd", "real_ln_bwd", "real_emb", "real_smax_bwd", "real_smax"]
QUICK_REAL = ["real_loss", "real_gnorm", "real_ln_bwd"]
STRESS = ["+stall=25 +lat=3", "+stall=40 +lat=60", "+lat=1"]
ICARUS_CASES = ["elem", "loss", "smax_small", "ln_small"]
HW = sorted(glob.glob(os.path.join(ROOT, "hw", "**", "*.v"), recursive=True))


def sh(cmd, log=None, env=None):
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, env=env)
    if log:
        with open(log, "w") as f:
            f.write(r.stdout + r.stderr)
    return r


def newest_src():
    return max(os.path.getmtime(f) for f in HW + glob.glob(os.path.join(ROOT, "tb", "vec", "*.sv")))


def build_verilator(name, params, rebuild, threads=1):
    mdir = os.path.join(SIM, name)
    exe = os.path.join(mdir, "Vtb_vu_top")
    if not rebuild and os.path.exists(exe) and os.path.getmtime(exe) > newest_src():
        return exe
    g = " ".join(f"-G{k}={v}" for k, v in params.items())
    thr = f"--threads {threads}" if threads > 1 else ""
    r = sh(f"verilator --binary -j 16 --timing -O3 -Wno-fatal -Wno-lint -Wno-style {thr} {g} --top-module tb_vu_top "
           f"-Mdir {mdir} tb/vec/tb_vu_top.sv {' '.join(os.path.relpath(f, ROOT) for f in HW)}", log=mdir + ".build.log")
    if r.returncode or not os.path.exists(exe):
        raise RuntimeError(f"verilator build failed: {mdir}.build.log")
    return exe


def build_icarus(name, params):
    out = os.path.join(SIM, name + ".vvp")
    if os.path.exists(out) and os.path.getmtime(out) > newest_src():
        return f"vvp -n {out}"
    p = " ".join(f"-Ptb_vu_top.{k}={v}" for k, v in params.items())
    r = sh(f"iverilog -g2012 {p} -o {out} tb/vec/tb_vu_top.sv {' '.join(os.path.relpath(f, ROOT) for f in HW)}",
           log=out + ".build.log")
    if r.returncode:
        raise RuntimeError(f"iverilog failed: {out}.build.log")
    return f"vvp -n {out}"


def result_of(txt):
    m = re.search(r"^(PASS|FAIL) (\S+) (.*)$", txt, re.M)
    if not m:
        return "FAIL", 0, -1, "no result line"
    if m.group(1) == "PASS":
        return "PASS", int(m.group(3).split()[0]), 0, ""
    mm = re.match(r"(\d+)/(\d+)", m.group(3))
    return "FAIL", int(mm.group(2)) if mm else 0, int(mm.group(1)) if mm else -1, m.group(3)


def gen_case(case, d, v=16):
    os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, case + ".desc")):
        return
    r = sh(f"{PY} tb/vec/gen_vec.py {case} {d} --v {v}", env=dict(os.environ, OMP_NUM_THREADS="8"))
    if r.returncode:
        raise RuntimeError(f"gen {case}: {r.stderr[-800:]}")


def run_case(exe, d, case, extra="", maxcyc=100000000):
    p = os.path.join(d, case)
    log = os.path.join(OUT, f"{os.path.basename(d)}_{case}_{extra.replace('+', '').replace(' ', '_').replace('=', '')}.log")
    t = time.time()
    r = sh(f"{exe} +desc={p}.desc +mem={p}.mem.hex +exp={p}.exp.hex +mask={p}.mask.hex +maxcyc={maxcyc} {extra}", log=log)
    st, n, mm, note = result_of(r.stdout)
    cyc = re.search(r"^CYCLES (\d+) RUNS (\d+) ELEMENTS (\d+) MODEL (\d+)", r.stdout, re.M)
    info = dict(cycles=int(cyc.group(1)), runs=int(cyc.group(2)), elements=int(cyc.group(3))) if cyc else {}
    return st, n, mm, note, info, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="the small real cases only")
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    for d in (OUT, SIM, VEC):
        os.makedirs(d, exist_ok=True)
    rows, bad = [], 0
    dunit = os.path.join(VEC, "unit")
    dreal = os.path.join(VEC, "real")
    dsmall = os.path.join(VEC, "small")
    real = [] if a.no_real else (QUICK_REAL if a.quick else REAL_CASES)
    with cf.ThreadPoolExecutor(3) as ex:
        fut = [ex.submit(gen_case, c, dunit) for c in UNIT_CASES]
        fut += [ex.submit(gen_case, c, dsmall, 4) for c in ("smax_small", "ln_small")]
        exe16 = build_verilator("vl16", {"MEMW": 25}, a.rebuild, threads=4)
        exe4 = build_verilator("vl4", {"V": 4, "VW": 2, "MEMW": 20}, a.rebuild)
        for f in fut:
            f.result()
        for c in real:
            gen_case(c, dreal)
    # ---------------------------------------------------------------- 1. V = 16 unit + real
    jobs = [(dunit, c, "") for c in UNIT_CASES] + [(dreal, c, "") for c in real]
    with cf.ThreadPoolExecutor(max(1, a.jobs // 2)) as ex:
        res = list(ex.map(lambda j: (j, run_case(exe16, j[0], j[1], j[2])), jobs))
    for (d, c, extra), (st, n, mm, note, info, dt) in res:
        bad += st != "PASS"
        cyc = ""
        if info:
            mc = budget.vec_cycles_case(c, info["elements"])
            cyc = f"{info['cycles']:,} ({info['elements']:,} el, {info['runs']} ops; model {mc:,.0f}, {info['cycles'] / max(mc, 1):.2f}x)"
        rows.append(("vu_top", "V=16", "verilator", c, n, mm, st, note, cyc, dt))
    # ---------------------------------------------------------------- 2. stress (unit cases, V = 16 and V = 4)
    jobs = [(dunit, c, s) for c in ("elem", "smax", "loss", "gnorm") for s in STRESS]
    with cf.ThreadPoolExecutor(max(1, a.jobs // 2)) as ex:
        res = list(ex.map(lambda j: (j, run_case(exe16, j[0], j[1], j[2])), jobs))
    for (d, c, extra), (st, n, mm, note, info, dt) in res:
        bad += st != "PASS"
        rows.append(("vu_top", "V=16", "verilator", f"{c} {extra}", n, mm, st, note, "", dt))
    jobs = [(dsmall, c, s) for c in ("smax_small", "ln_small") for s in ("", "+stall=30 +lat=7")]
    with cf.ThreadPoolExecutor(max(1, a.jobs // 2)) as ex:
        res = list(ex.map(lambda j: (j, run_case(exe4, j[0], j[1], j[2])), jobs))
    for (d, c, extra), (st, n, mm, note, info, dt) in res:
        bad += st != "PASS"
        rows.append(("vu_top", "V=4", "verilator", f"{c} {extra}", n, mm, st, note, "", dt))
    # ---------------------------------------------------------------- 3. Icarus smoke, V = 4
    cmd = build_icarus("iv4", {"V": 4, "VW": 2, "MEMW": 14, "MASKW": 13})   # small tb memories: iverilog time (P3)
    for c in ("smax_small", "ln_small"):
        st, n, mm, note, info, dt = run_case(cmd, dsmall, c, "+lat=5", maxcyc=4000000)
        bad += st != "PASS"
        rows.append(("vu_top", "V=4", "icarus", f"{c} +lat=5", n, mm, st, note, f"{info.get('cycles', 0):,}", dt))
    # ---------------------------------------------------------------- table
    lines = ["| unit | config | simulator | set | vectors | mismatches | result | cycles (model) | time |",
             "|---|---|---|---|---|---|---|---|---|"]
    for u, cfg, sim, s, n, mm, st, note, cyc, dt in rows:
        lines.append(f"| {u} | {cfg} | {sim} | {s} | {n:,} | {mm} | {st} {note} | {cyc} | {dt:.0f} s |")
    lines.append("")
    lines.append("TEST-VEC " + ("PASS" if bad == 0 else "FAIL"))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
