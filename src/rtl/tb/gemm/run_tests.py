#!/usr/bin/env python3
"""`make test-gemm`: L2 tests of hw/gemm against L1 (bit-exact), docs/p3_report.md.

  1. gm_pe: directed + random chunk vectors (tb/gemm/gen_pe.py, L1 fma chains), Verilator at the
     production FMA_STAGES = 2 and an Icarus sweep over FMA_STAGES 2 / 3 / 4
  2. gm_top 16 x 16 (Verilator, memory model): every unit case (K chunking, partial tiles, batches,
     operand modes, K segments, every epilogue option) and the end-to-end roberta-base GEMMs with
     real data (tb/gemm/gen_gemm.py), cycles vs docs/calc/budget.py's model
  3. gm_top 4 x 4: the unit cases with random memory stalls / latencies (Verilator) and an Icarus
     smoke run (reduced set)
Prints a table, writes build/test-gemm/summary.txt; exit 1 on any FAIL / missing result.
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
VEC = os.path.join(ROOT, "build", "vec", "gemm")
OUT = os.path.join(ROOT, "build", "test-gemm")
SIM = os.path.join(ROOT, "build", "sim", "gemm")
sys.path.insert(0, os.path.join(ROOT, "docs", "calc"))
import budget  # noqa: E402

UNIT_CASES = ["chunk_k1", "chunk_k63", "chunk_k64", "chunk_k65", "chunk_k128", "chunk_k4096", "chunk_k4160",
              "partial", "partial_small", "batched", "mode_at", "mode_bt", "mode_abt", "segs", "segs2",
              "ep_bias", "ep_scale", "ep_bias_scale", "ep_gelu", "ep_gelu_nobias", "ep_drop", "ep_resid",
              "ep_bias_drop_resid", "ep_gelu_bwd", "ep_all", "ep_gelu_k64", "ep_gelu_bwd_k64",
              # P7a: short-K multi-block GEMMs (drain RMW tail vs epilogue reads; reduced-L attention P.V)
              "custom_32_64_32_2_12_0_1", "custom_64_128_16_1_1_0_0"]
REAL_CASES = ["real_qproj", "real_scores", "real_pv", "real_attnout", "real_ffn1_256", "real_dv", "real_dgelu",
              "real_ffn1"]
QUICK_REAL = ["real_scores", "real_pv", "real_dv", "real_ffn1_256", "real_dgelu"]
SMALL_CASES = ["chunk_k1", "chunk_k65", "chunk_k128", "partial", "partial_small", "batched", "mode_at", "mode_bt",
               "mode_abt", "segs", "segs2", "ep_bias_scale", "ep_bias_drop_resid", "ep_resid", "ep_gelu",
               "ep_gelu_bwd", "ep_all"]
STRESS = ["+stall=25 +lat=3", "+stall=40 +lat=60", "+lat=1"]
ICARUS_CASES = ["chunk_k65", "partial_small", "mode_abt", "ep_bias_drop_resid", "ep_gelu"]
HW = sorted(glob.glob(os.path.join(ROOT, "hw", "**", "*.v"), recursive=True))


def sh(cmd, log=None, env=None):
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, env=env)
    if log:
        with open(log, "w") as f:
            f.write(r.stdout + r.stderr)
    return r


def newest_src():
    return max(os.path.getmtime(f) for f in HW + glob.glob(os.path.join(ROOT, "tb", "gemm", "*.sv")))


def build_verilator(name, top, params, rebuild, threads=1):
    mdir = os.path.join(SIM, name)
    exe = os.path.join(mdir, f"V{top}")
    if not rebuild and os.path.exists(exe) and os.path.getmtime(exe) > newest_src():
        return exe
    g = " ".join(f"-G{k}={v}" for k, v in params.items())
    thr = f"--threads {threads}" if threads > 1 else ""
    r = sh(f"verilator --binary -j 16 --timing -O3 -Wno-fatal -Wno-lint -Wno-style {thr} {g} --top-module {top} "
           f"-Mdir {mdir} tb/gemm/{top}.sv {' '.join(os.path.relpath(f, ROOT) for f in HW)}", log=mdir + ".build.log")
    if r.returncode or not os.path.exists(exe):
        raise RuntimeError(f"verilator build failed: {mdir}.build.log")
    return exe


def build_icarus(name, top, params):
    out = os.path.join(SIM, name + ".vvp")
    if os.path.exists(out) and os.path.getmtime(out) > newest_src():
        return f"vvp -n {out}"
    p = " ".join(f"-P{top}.{k}={v}" for k, v in params.items())
    r = sh(f"iverilog -g2012 {p} -o {out} tb/gemm/{top}.sv {' '.join(os.path.relpath(f, ROOT) for f in HW)}",
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


def gen_case(case, d, rows, cols):
    os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, case + ".desc")):
        return
    r = sh(f"{PY} tb/gemm/gen_gemm.py {case} {d} --rows {rows} --cols {cols}",
           env=dict(os.environ, OMP_NUM_THREADS="12"))
    if r.returncode:
        raise RuntimeError(f"gen {case}: {r.stderr[-500:]}")


def run_case(exe, d, case, extra="", maxcyc=400000000):
    p = os.path.join(d, case)
    mask = f"+mask={p}.mask.hex" if os.path.exists(p + ".mask.hex") else ""
    log = os.path.join(OUT, f"{os.path.basename(d)}_{case}_{extra.replace('+', '').replace(' ', '_').replace('=', '')}.log")
    t = time.time()
    r = sh(f"{exe} +desc={p}.desc +mem={p}.mem.hex +exp={p}.exp.hex {mask} +maxcyc={maxcyc} {extra}", log=log)
    st, n, mm, note = result_of(r.stdout)
    cyc = re.search(r"^CYCLES (\d+) ACTIVE (\d+) MACS (\d+) MODEL (\d+) (\d+) (\d+) (\d+)", r.stdout, re.M)
    info = dict(cycles=int(cyc.group(1)), active=int(cyc.group(2)), macs=int(cyc.group(3)),
                m=int(cyc.group(4)), n=int(cyc.group(5)), k=int(cyc.group(6)), nb=int(cyc.group(7))) if cyc else {}
    return st, n, mm, note, info, time.time() - t


def model_cycles(rows, cols, m, n, k, nb):
    c = dict(name="x", R=rows, C=cols, V=16, dbuf=2, lane_macros=4, xdeep=4)
    return budget.gemm_cycles(c, m, n, k) * nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer random PE vectors, the small real cases only")
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SIM, exist_ok=True)
    os.makedirs(VEC, exist_ok=True)
    rows = []
    bad = 0

    # ---------------------------------------------------------------- 1. PE
    pe_dir = os.path.join(VEC, "pe.dir.hex")
    if not os.path.exists(pe_dir):
        sh(f"{PY} tb/gemm/gen_pe.py directed 0 1 {pe_dir}")
    nrand = 2000 if a.quick else 20000
    pe_rnd = os.path.join(VEC, f"pe.r{nrand}.hex")
    if not os.path.exists(pe_rnd):
        sh(f"{PY} tb/gemm/gen_pe.py random {nrand} 7 {pe_rnd}", env=dict(os.environ, OMP_NUM_THREADS="8"))
    exe = build_verilator("vl_pe", "tb_gm_pe", {"FMA_STAGES": 2}, a.rebuild)
    for f, gap in ((pe_dir, 0), (pe_rnd, 0), (pe_rnd, 5)):
        r = sh(f"{exe} +vec={f} +gap={gap}", log=os.path.join(OUT, f"pe_{os.path.basename(f)}_g{gap}.log"))
        st, n, mm, note = result_of(r.stdout)
        bad += st != "PASS"
        rows.append(("gm_pe", "FMA_STAGES=2", "verilator", f"{os.path.basename(f)} gap{gap}", n, mm, st, note, ""))
    for s in (2, 3, 4):
        cmd = build_icarus(f"iv_pe{s}", "tb_gm_pe", {"FMA_STAGES": s})
        r = sh(f"{cmd} +vec={pe_dir} +gap=3", log=os.path.join(OUT, f"pe_iv{s}.log"))
        st, n, mm, note = result_of(r.stdout)
        bad += st != "PASS"
        rows.append(("gm_pe", f"FMA_STAGES={s}", "icarus", "pe.dir.hex gap3", n, mm, st, note, ""))

    # ---------------------------------------------------------------- 2. gm_top 16 x 16
    d16 = os.path.join(VEC, "s16")
    dreal = os.path.join(VEC, "real")
    real = [] if a.no_real else (QUICK_REAL if a.quick else REAL_CASES)
    with cf.ThreadPoolExecutor(2) as ex:
        fut = [ex.submit(gen_case, c, d16, 16, 16) for c in UNIT_CASES]
        fut += [ex.submit(gen_case, c, dreal, 16, 16) for c in real]
        exe16 = build_verilator("vl_top16", "tb_gm_top", {"MEMW": 25}, a.rebuild, threads=8)
        for f in fut:
            f.result()
    jobs = [(d16, c, "") for c in UNIT_CASES] + [(dreal, c, "") for c in real]
    with cf.ThreadPoolExecutor(max(1, a.jobs // 4)) as ex:
        res = list(ex.map(lambda j: (j, run_case(exe16, j[0], j[1], j[2])), jobs))
    for (d, c, extra), (st, n, mm, note, info, dt) in res:
        bad += st != "PASS"
        cyc = ""
        if info:
            mc = model_cycles(16, 16, info["m"], info["n"], info["k"], info["nb"])
            util = info["macs"] / (info["cycles"] * 256)
            cyc = f"cycles {info['cycles']:,} model {mc:,} ({info['cycles'] / mc:.3f}x) util {util:.3f} active {info['active'] / info['cycles']:.3f}"
        rows.append(("gm_top", "16x16", "verilator", c, n, mm, st, note, cyc + f" [{dt:.0f}s]"))

    # ---------------------------------------------------------------- 3. gm_top 4 x 4 stress + Icarus
    d4 = os.path.join(VEC, "s4")
    for c in SMALL_CASES:
        gen_case(c, d4, 4, 4)
    exe4 = build_verilator("vl_top4", "tb_gm_top", {"ROWS": 4, "COLS": 4, "NSFU": 2, "MEMW": 18}, a.rebuild)
    jobs = [(d4, c, e) for c in SMALL_CASES for e in STRESS]
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda j: (j, run_case(exe4, j[0], j[1], j[2], 20000000)), jobs))
    agg = {}
    for (d, c, extra), (st, n, mm, note, info, dt) in res:
        bad += st != "PASS"
        v = agg.setdefault(extra, [0, 0, []])
        v[0] += n
        v[1] += max(mm, 0)
        if st != "PASS":
            v[2].append(f"{c}: {note}")
    for extra, (n, mm, notes) in agg.items():
        rows.append(("gm_top", "4x4", "verilator", f"{len(SMALL_CASES)} cases {extra}", n, mm,
                     "PASS" if not notes else "FAIL", "; ".join(notes[:3]), ""))
    # ivl's compile time scales with the memory-model array sizes (2^24 keep bits = hours, 2 x 2^16 = 5 min):
    # the Icarus cases use < 2^13 words and <= 2,627 keep bits, so MEMW 14 / MASKW 12
    t_iv = time.time()
    iv = build_icarus("iv_top4", "tb_gm_top", {"ROWS": 4, "COLS": 4, "NSFU": 2, "MEMW": 14, "MASKW": 12})
    t_iv = time.time() - t_iv
    n_iv = mm_iv = 0
    notes = []
    for c in ICARUS_CASES:
        st, n, mm, note, info, dt = run_case(iv, d4, c, "+lat=5", 4000000)
        n_iv += n
        mm_iv += max(mm, 0)
        if st != "PASS":
            notes.append(f"{c}: {note}")
            bad += 1
    rows.append(("gm_top", "4x4", "icarus", f"{len(ICARUS_CASES)} cases +lat=5", n_iv, mm_iv,
                 "PASS" if not notes else "FAIL", "; ".join(notes[:3]), f"[compile {t_iv:.0f}s]"))

    lines = [f"{'unit':7s} {'config':12s} {'simulator':10s} {'set':34s} {'elements':>10s} {'mism':>5s}  status"]
    for u, cfg, sim, s, n, mm, st, note, extra in rows:
        lines.append(f"{u:7s} {cfg:12s} {sim:10s} {s:34s} {n:10,d} {mm:5d}  {st} {note} {extra}")
    lines.append(f"TEST-GEMM {'PASS' if bad == 0 else 'FAIL'}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
