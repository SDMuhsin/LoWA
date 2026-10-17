#!/usr/bin/env python3
"""`make test-fp`: L2 tests of hw/fp against the L1 golden vectors (tb/fp/gen_vectors.py).

  1. vectors into build/vec/fp (deterministic seeds; regenerated only if missing)
  2. Verilator, production configuration of every unit: all directed sets + the full random sets
     (fma/add/mul 10 x 1M, cmp/i2f 2 x 1M, sfu 4 x 250k per function + 1M mixed-function)
  3. Icarus smoke: every stage-count configuration on the directed sets (+ a mixed SFU set)
  4. --gls: the synthesised netlists of flow/synth (gls.v + PDK cell models), Verilator
Prints a table and writes build/test-fp/summary.txt; exit 1 on any FAIL / missing result.
  run_tests.py [--quick] [--gls] [--jobs N]
"""
import argparse
import concurrent.futures as cf
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = sys.executable
VEC = os.path.join(ROOT, "build", "vec", "fp")
OUT = os.path.join(ROOT, "build", "test-fp")
SIM = os.path.join(ROOT, "build", "sim", "fp")
HW_COMMON = ["hw/common/cm_delay.v", "hw/fp/fp_norm.v", "hw/fp/fp_round.v"]
DEPS = {"fma": ["hw/fp/fp_fma.v"], "add": ["hw/fp/fp_add.v"], "mul": ["hw/fp/fp_mul.v"],
        "cmp": ["hw/fp/fp_cmp.v"], "i2f": ["hw/fp/fp_i2f.v"],
        "sfu": ["hw/fp/fp_fma.v", "hw/fp/fp_sfu_ucode.v", "hw/fp/fp_sfu.v"]}
# production configurations (docs/p2_report.md) and the Icarus sweep
PROD = {"fma": {"STAGES": 2}, "add": {"STAGES": 2}, "mul": {"STAGES": 1}, "cmp": {"STAGES": 1},
        "i2f": {"STAGES": 1}, "sfu": {"LF": 2, "NTHR": 4}}
SWEEP = {"fma": [{"STAGES": s} for s in (1, 2, 3, 4, 6)],
         "add": [{"STAGES": s} for s in (1, 2, 3, 4, 5)],
         "mul": [{"STAGES": s} for s in (1, 2, 3)],
         "cmp": [{"STAGES": s} for s in (1, 2)],
         "i2f": [{"STAGES": s} for s in (1, 2, 3)],
         "sfu": [{"LF": lf, "NTHR": nt} for lf in (1, 2, 3, 4) for nt in sorted({1, lf + 2})]}
FNS = ["exp", "recip", "rsqrt", "sqrt", "tanh", "erf", "log"]


def sh(cmd, log=None, env=None):
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, env=env)
    if log:
        with open(log, "w") as f:
            f.write(r.stdout + r.stderr)
    return r


def vec_jobs(quick):
    """(unit, kind, n, seed, file) for every vector file."""
    j = []
    for u in ("fma", "add", "mul", "cmp", "i2f"):
        j.append((u, "directed", 0, 1, f"{u}.dir.hex"))
    nrand = {"fma": 10, "add": 10, "mul": 10, "cmp": 2, "i2f": 2}
    for u, k in nrand.items():
        for i in range(1 if quick else k):
            j.append((u, "random", 20000 if quick else 1000000, 100 + i, f"{u}.r{i}.hex"))
    for fn in FNS:
        j.append((f"sfu:{fn}", "directed", 0, 2, f"sfu.dir.{fn}.hex"))
        for i in range(1 if quick else 4):
            j.append((f"sfu:{fn}", "random", 5000 if quick else 250000, 200 + i, f"sfu.r{i}.{fn}.hex"))
    j.append(("sfu", "mixed", 3000 if quick else 1000000 // 7 + 1, 300, "sfu.mix.hex"))
    j.append(("sfu", "mixed", 3000, 301, "sfu.mixsmall.hex"))
    return j


def gen_vectors(quick, jobs):
    os.makedirs(VEC, exist_ok=True)
    todo = [j for j in vec_jobs(quick) if not os.path.exists(os.path.join(VEC, j[4]))]
    def run(j):
        u, k, n, s, f = j
        tmp = os.path.join(VEC, f + ".tmp")
        r = sh(f"{PY} tb/fp/gen_vectors.py {u} {k} {n} {s} {tmp}")
        if r.returncode:
            raise RuntimeError(r.stderr)
        os.replace(tmp, os.path.join(VEC, f))
    # the directed SFU sets run exhaustive per-binade scans (OpenMP): fewer at a time
    with cf.ThreadPoolExecutor(max(1, jobs // 6)) as ex:
        list(ex.map(run, todo))


def tag(params):
    return "_".join(f"{k}{v}" for k, v in params.items())


def build_verilator(unit, params, gls=None):
    name = f"vl_{unit}_{tag(params)}" + ("_gls" if gls else "")
    mdir = os.path.join(SIM, name)
    gflags = " ".join(f"-G{k}={v}" for k, v in params.items())
    if gls:
        cells = glob.glob(os.path.join(os.environ["PDK_ROOT"],
                          "gf180mcuD/libs.ref/gf180mcu_as_sc_mcu7t3v3/verilog/gf180mcu_as_sc_mcu7t3v3.v"))[0]
        srcs = f"{gls} {cells}"
        gflags += " -DGLS"
    else:
        srcs = " ".join(HW_COMMON + DEPS[unit])
    r = sh(f"verilator --binary --timing -O3 -Wno-fatal -Wno-lint -Wno-style {gflags} "
           f"--top-module tb_fp_{unit} -Mdir {mdir} tb/fp/tb_fp_{unit}.sv {srcs}",
           log=mdir + ".build.log")
    exe = os.path.join(mdir, f"Vtb_fp_{unit}")
    if r.returncode or not os.path.exists(exe):
        raise RuntimeError(f"verilator build failed: {mdir}.build.log")
    return exe


def build_icarus(unit, params):
    out = os.path.join(SIM, f"iv_{unit}_{tag(params)}.vvp")
    pflags = " ".join(f"-Ptb_fp_{unit}.{k}={v}" for k, v in params.items())
    srcs = " ".join(HW_COMMON + DEPS[unit])
    r = sh(f"iverilog -g2012 {pflags} -o {out} tb/fp/tb_fp_{unit}.sv {srcs}", log=out + ".build.log")
    if r.returncode:
        raise RuntimeError(f"iverilog failed: {out}.build.log")
    return f"vvp -n {out}"


def run_one(cmd, vecfile, log, extra=""):
    r = sh(f"{cmd} +vec={vecfile} {extra}", log=log)
    m = re.search(r"^(PASS|FAIL) (\S+) (.*)$", r.stdout, re.M)
    if not m:
        return ("FAIL", 0, -1, "no result line")
    if m.group(1) == "PASS":
        n = int(m.group(3).split()[0])
        return ("PASS", n, 0, "")
    mm = re.match(r"(\d+)/(\d+)", m.group(3))
    return ("FAIL", int(mm.group(2)) if mm else 0, int(mm.group(1)) if mm else -1, m.group(3))


def files_for(unit, sets):
    out = []
    for f in sorted(os.listdir(VEC)):
        if not f.endswith(".hex") or not f.startswith(unit + "."):
            continue
        is_dir = ".dir." in f
        if ("dir" in sets and is_dir) or ("rand" in sets and not is_dir and "mixsmall" not in f) \
                or ("mixsmall" in sets and "mixsmall" in f):
            out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gls", action="store_true", help="only the gate-level runs (after synth-fp)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    global VEC
    if a.quick:
        VEC = VEC + "_quick"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SIM, exist_ok=True)
    gen_vectors(a.quick, a.jobs)
    rows = []          # (unit, config, sim, set, vectors, mismatches, status)
    runs = []          # (unit, config, sim, file, cmd, extra)

    if a.gls:
        for unit, params in PROD.items():
            d = os.path.join(ROOT, "build", "synth", f"fp_{unit}_prod")
            exe = build_verilator(unit, params, gls=os.path.join(d, "gls.v"))
            sets = ["dir", "mixsmall"] if unit == "sfu" else ["dir"]
            for f in files_for(unit, sets):
                runs.append((unit, tag(params), "verilator-GLS", f, exe, "+gap=3" if "mix" in f else ""))
            if unit != "sfu":
                runs.append((unit, tag(params), "verilator-GLS", f"{unit}.r0.hex", exe, ""))
    else:
        with cf.ThreadPoolExecutor(8) as ex:
            exes = dict(zip(PROD, ex.map(lambda u: build_verilator(u, PROD[u]), PROD)))
            ivs = {}
            for unit, cfgs in SWEEP.items():
                for p in cfgs:
                    ivs[(unit, tag(p))] = ex.submit(build_icarus, unit, p)
            ivs = {k: v.result() for k, v in ivs.items()}
        for unit, params in PROD.items():
            for f in files_for(unit, ["dir", "rand"]):
                runs.append((unit, tag(params), "verilator", f, exes[unit], "+gap=3" if "mix" in f else ""))
        for (unit, t), cmd in ivs.items():
            sets = ["dir", "mixsmall"] if unit == "sfu" else ["dir"]
            for f in files_for(unit, sets):
                if unit == "sfu" and ".dir." in f and f not in ("sfu.dir.exp.hex", "sfu.dir.tanh.hex"):
                    continue           # Icarus SFU sweep: 2 directed sets + the mixed set per config
                runs.append((unit, t, "icarus", f, cmd, "+gap=3" if "mix" in f else ""))

    def go(rr):
        unit, t, sim, f, cmd, extra = rr
        log = os.path.join(OUT, f"{sim}_{unit}_{t}_{f}.log")
        return rr, run_one(cmd, os.path.join(VEC, f), log, extra)
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(go, runs))
    agg = {}
    bad = 0
    for (unit, t, sim, f, _, _), (st, n, mm, note) in res:
        kind = "directed" if ".dir." in f else ("mixed" if "mix" in f else "random")
        k = (unit, t, sim, kind)
        v = agg.setdefault(k, [0, 0, 0, []])
        v[0] += n
        v[1] += max(mm, 0)
        v[2] += 1
        if st != "PASS":
            bad += 1
            v[3].append(f"{f}: {note}")
    lines = [f"{'unit':5s} {'config':12s} {'simulator':14s} {'set':9s} {'files':>5s} {'vectors':>11s} "
             f"{'mismatches':>10s}  status"]
    for k in sorted(agg):
        n, mm, nf, notes = agg[k]
        lines.append(f"{k[0]:5s} {k[1]:12s} {k[2]:14s} {k[3]:9s} {nf:5d} {n:11,d} {mm:10d}  "
                     + ("PASS" if not notes else "FAIL " + "; ".join(notes[:3])))
    tot = sum(v[0] for v in agg.values())
    lines.append(f"TOTAL {tot:,} vectors, {sum(v[1] for v in agg.values())} mismatches, "
                 f"{bad} failing runs of {len(res)}")
    lines.append("TEST-FP " + ("GLS " if a.gls else "") + ("PASS" if bad == 0 and res else "FAIL"))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "summary_gls.txt" if a.gls else "summary.txt"), "w") as f:
        f.write(txt + "\n")
    sys.exit(0 if bad == 0 and res else 1)


if __name__ == "__main__":
    main()
