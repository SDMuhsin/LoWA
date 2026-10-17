#!/usr/bin/env python3
"""make test-rng (P6b): the on-chip dropout mask generator (hw/rng) bit-exact vs model/rng.py.
  1. rn_tf (+ rn_ks): Random123 KAT + >= 1M random (counter, key) vs rng.threefry2x32_13; STAGES sweep.
  2. rn_mkgen on its mk channel vs rng.keep_mask: the real step-B attention-probability mask of layer 0 in the
     vector unit's request order, the real step-B FFN-output (hidden-state) mask of layer 5 in the GEMM epilogue's
     order, random odd / even indices on random configurations; MKW / NCORE / STAGES variants, request gaps;
     throughput (cycles per request = ceil((MKW/2 + 1) / NCORE)) checked on the gap-free runs.
  3. End to end with the generator as the engine's keep-bit source (+define+MK_GEN): vu_top real_smax (attn L0),
     real_smax_bwd (attn L11, the backward regeneration), real_emb (emb, hidden state); gm_top 16x16 real_attnout
     (attn_out L0, epilogue dropout, NCORE 3) and ep_bias_drop_resid; outputs vs L1 and every keep bit vs L1.
run_tests.py [--quick]   Summary: build/test-rng/summary.txt; exit 1 on any FAIL."""
import argparse
import glob
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = sys.executable
OUT = os.path.join(ROOT, "build", "test-rng")
VEC, SIM = os.path.join(OUT, "vec"), os.path.join(OUT, "sim")
HW = " ".join(os.path.relpath(f, ROOT) for f in sorted(glob.glob(os.path.join(ROOT, "hw", "**", "*.v"), recursive=True)))
SITE = {"emb": 0, "attn": 1, "attn_out": 2, "ffn_out": 3, "cls1": 4, "cls2": 5}


def sh(cmd, log, env=None):
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, env=env)
    with open(log, "w") as f:
        f.write(cmd + "\n" + r.stdout + r.stderr)
    return r


def build(name, top, srcs, params="", defs="", threads=""):
    mdir = os.path.join(SIM, name)
    exe = os.path.join(mdir, "V" + top)
    r = sh(f"verilator --binary -j 16 --timing -O3 -Wno-fatal -Wno-lint -Wno-style {threads} {params} {defs} "
           f"+incdir+tb/rng +incdir+tb/mem --top-module {top} -Mdir {mdir} {srcs} {HW}", mdir + ".build.log")
    if r.returncode or not os.path.exists(exe):
        raise RuntimeError(f"verilator build failed: {mdir}.build.log")
    return exe


def cfg_args(seed, step, site, layer):
    return f"+mkseed={seed:016x} +mkstep={step:x} +mksite={SITE[site]:x} +mklayer={layer:x}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="no real end-to-end engine cases")
    a = ap.parse_args()
    for d in (OUT, VEC, SIM):
        os.makedirs(d, exist_ok=True)
    rows, bad = [], 0
    z = np.load(os.path.join(ROOT, "model", "runs", "stepB_l0f32.npz"))
    seedB, stepB = int(z["seed"]), int(z["step_s"])

    # ---------------------------------------------------------------- vectors + builds
    gens = [("tf_1m", f"tf {VEC}/tf_1m.vec 1048576 7"), ("tf_64k", f"tf {VEC}/tf_64k.vec 65536 11"),
            ("rand16", f"mk {VEC}/rand16.vec rand 16"), ("rand4", f"mk {VEC}/rand4.vec rand 4"),
            ("rand2", f"mk {VEC}/rand2.vec rand 2"), ("real_attn0", f"mk {VEC}/real_attn0.vec real_attn0 16"),
            ("real_ffn5", f"mk {VEC}/real_ffn5.vec real_ffn5 16")]
    tf_st = [3, 1, 2, 7, 14]
    mk_v = [(16, 9, 3), (16, 3, 3), (16, 1, 3), (16, 5, 2), (16, 9, 1), (4, 3, 3), (4, 2, 3), (4, 1, 3), (2, 1, 3), (2, 2, 3)]
    real = not a.quick
    with ThreadPoolExecutor(8) as ex:
        gf = {k: ex.submit(sh, f"{PY} tb/rng/gen_rng.py {c}", os.path.join(OUT, f"gen_{k}.log")) for k, c in gens}
        tfb = {s: ex.submit(build, f"tf_s{s}", "tb_rn_tf", "tb/rng/tb_rn_tf.sv", f"-GSTAGES={s}") for s in tf_st}
        mkb = {v: ex.submit(build, f"mk_{v[0]}_{v[1]}_{v[2]}", "tb_rn_mkgen", "tb/rng/tb_rn_mkgen.sv",
                            f"-GMKW={v[0]} -GNCORE={v[1]} -GSTAGES={v[2]}") for v in mk_v}
        if real:
            # engine end-to-end: case vectors from the P3 / P4 generators (L1 goldens), generator-mode tbs
            dv, dg = os.path.join(VEC, "vec"), os.path.join(VEC, "gemm")
            os.makedirs(dv, exist_ok=True); os.makedirs(dg, exist_ok=True)
            eg = [ex.submit(sh, f"{PY} tb/vec/gen_vec.py {c} {dv} --v 16", os.path.join(OUT, f"gen_vu_{c}.log"),
                            dict(os.environ, OMP_NUM_THREADS="8")) for c in ("real_smax", "real_smax_bwd", "real_emb")
                  if not os.path.exists(os.path.join(dv, c + ".desc"))]
            eg += [ex.submit(sh, f"{PY} tb/gemm/gen_gemm.py {c} {dg} --rows 16 --cols 16", os.path.join(OUT, f"gen_gm_{c}.log"),
                             dict(os.environ, OMP_NUM_THREADS="12")) for c in ("real_attnout", "ep_bias_drop_resid")
                   if not os.path.exists(os.path.join(dg, c + ".desc"))]
            vub = ex.submit(build, "vu_gen", "tb_vu_top", "tb/vec/tb_vu_top.sv", "-GMEMW=25", "+define+MK_GEN", "--threads 4")
            gmb = ex.submit(build, "gm_gen", "tb_gm_top", "tb/gemm/tb_gm_top.sv", "-GMEMW=25", "+define+MK_GEN +define+MKGEN_NC=3",
                            "--threads 8")
        for k, f in gf.items():
            if f.result().returncode:
                raise RuntimeError(f"gen_rng {k} failed")
        tfb = {k: f.result() for k, f in tfb.items()}
        mkb = {k: f.result() for k, f in mkb.items()}
        if real:
            for f in eg:
                if f.result().returncode:
                    raise RuntimeError("case generation failed: " + f.result().args)
            vub, gmb = vub.result(), gmb.result()

    # ---------------------------------------------------------------- unit runs
    jobs = [("rn_tf", f"STAGES={s}", "tf_1m" if s == 3 else "tf_64k", f"{tfb[s]} +vec={VEC}/{'tf_1m' if s == 3 else 'tf_64k'}.vec "
             f"+gap={10 if s == 3 else 30} +maxcyc=20000000") for s in tf_st]
    for v in mk_v:
        mkw, nc, st = v
        nseq = -(-(mkw // 2 + 1) // nc)
        for case in (["rand16"] if mkw == 16 else [f"rand{mkw}"]) + (["real_attn0"] if v == (16, 9, 3) else []) + \
                (["real_ffn5"] if v == (16, 3, 3) else []):
            cfg = open(f"{VEC}/{case}.vec.cfg").read().strip()
            for gap in ((0, 30) if case.startswith("rand") else (0,)):
                jobs.append(("rn_mkgen", f"MKW={mkw} NCORE={nc} STAGES={st}", f"{case} gap {gap}",
                             f"{mkb[v]} +vec={VEC}/{case}.vec {cfg} +gap={gap} +maxcyc=20000000", nseq if gap == 0 else None))

    def run(j):
        unit, par, case, cmd = j[:4]
        nseq = j[4] if len(j) > 4 else None
        t = time.time()
        log = os.path.join(OUT, f"{unit}_{par}_{case}.log".replace(" ", "_").replace("=", ""))
        r = sh(cmd, log)
        m = re.search(rf"^(PASS|FAIL) {unit} (.*)$", r.stdout, re.M)
        res = m.group(0) if m else "FAIL (no result line)"
        note = ""
        cy = re.search(r"CYCLES (\d+) requests (\d+) latency (\d+)", r.stdout)
        if cy:
            c, n, lat = map(int, cy.groups())
            note = f"{c / n:.3f} cycles/request, latency {lat}"
            if nseq is not None and c > n * nseq + 40:                # throughput contract on the gap-free runs
                res = f"FAIL throughput {c} cycles for {n} requests > {nseq}/request"
        return (unit, par, case, res, note, time.time() - t)
    with ThreadPoolExecutor(8) as ex:
        for row in ex.map(run, jobs):
            rows.append(row); bad += not row[3].startswith("PASS")

    # ---------------------------------------------------------------- engine end-to-end with the generator
    if real:
        dv, dg = os.path.join(VEC, "vec"), os.path.join(VEC, "gemm")
        ejobs = [("vu_top", "real_smax", vub, dv, cfg_args(seedB, stepB, "attn", 0), 40000000),
                 ("vu_top", "real_smax_bwd", vub, dv, cfg_args(seedB, stepB, "attn", 11), 25000000),
                 ("vu_top", "real_emb", vub, dv, cfg_args(seedB, stepB, "emb", 0), 25000000),
                 ("gm_top", "real_attnout", gmb, dg, cfg_args(seedB, stepB, "attn_out", 0), 40000000),
                 ("gm_top", "ep_bias_drop_resid", gmb, dg, cfg_args(777, 3, "attn_out", 5), 400000)]

        def erun(j):
            unit, case, exe, d, cfg, mx = j
            t = time.time()
            p = os.path.join(d, case)
            r = sh(f"{exe} +desc={p}.desc +mem={p}.mem.hex +exp={p}.exp.hex +mask={p}.mask.hex +maxcyc={mx} {cfg}",
                   os.path.join(OUT, f"e2e_{case}.log"))
            m = re.search(rf"^(PASS|FAIL) {unit} (.*)$", r.stdout, re.M)
            res = m.group(0) if m else "FAIL (no result line)"
            mc = re.search(r"MKCHECK (\d+) responses (\d+) bit", r.stdout)
            cy = re.search(r"CYCLES (\d+)", r.stdout)
            note = (f"keep responses {int(mc.group(1)):,} ({int(mc.group(2))} bit mismatches vs L1)" if mc else "no MKCHECK") + \
                (f", cycles {int(cy.group(1)):,}" if cy else "")
            if not mc or int(mc.group(2)) or not int(mc.group(1)):
                res = "FAIL " + res
            return (unit + " + rn_mkgen", "MK_GEN", case, res, note, time.time() - t)
        with ThreadPoolExecutor(5) as ex:
            for row in ex.map(erun, ejobs):
                rows.append(row); bad += not row[3].startswith("PASS")

    lines = ["| unit | parameters | case | result | notes | time |", "|---|---|---|---|---|---|"]
    lines += [f"| {u} | {p} | {c} | {r} | {n} | {t:.0f}s |" for u, p, c, r, n, t in rows]
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("TEST-RNG", "PASS" if bad == 0 else f"FAIL ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
