#!/usr/bin/env python3
"""make test-mem (P6a): memory-system unit tests (tb_mm_unit vs tb_dram, goldens from the python reference in gen_mm.py on a
memory_map-placed image) and integration (gm_top / vu_top / lw_top on mm_top + tb_dram, goldens from L1 via the P3-P5
generators, re-placed by gen_integ.py).  run_tests.py [--quick] [--tech generic|gf180|tsmc65]
--tech picks the mm_sram_bank backend (PLAN R11a; default generic): tsmc65 runs the latch register file with the behavioural
clock gate hw/mem/tech/mm_icg.v.  Summary: build/test-mem[-<tech>]/summary.txt, exit 1 on any FAIL."""
import argparse
import glob
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = sys.executable
OUT = os.path.join(ROOT, "build", "test-mem")
TECHS = {"generic": 0, "gf180": 1, "tsmc65": 2}
TECHDEF = ""                                                                 # +define+MM_TECH=<n> for a non-default backend
HW = " ".join(os.path.relpath(f, ROOT) for f in sorted(glob.glob(os.path.join(ROOT, "hw", "**", "*.v"), recursive=True)))
UNIT = ["burst", "pattern", "write", "order", "outst"]
NOM = "+lat=24 +wlat=8 +bw=3200 +wfair=1"                                  # 800 MB/s at 25 MHz, ~1 us read latency
STRESS = "+lat=100 +wlat=40 +bw=1600 +stall=30 +reorder=1 +ilv=1 +rstall=30"
MKID = {"gm": 2, "vu": 2, "gmg": 2}                                            # ARID of the keep-bit channel (NR = 2)
# per-case cycle limits (watchdog), ~4x the nominal-DRAM cycle count
MAXCYC = {"ep_bias_drop_resid": 100000, "elem": 400000, "emb": 3000000, "adamw_real_head": 25000000,
          "real_ffn1_256": 12000000, "real_ln_fwd": 25000000}
# gmg = gm_top on mm_top with the on-chip Threefry generator as the keep-bit source (MK_DRAM = 0, P6b), no stand-in
MKCFG = {"gmg": "+mkseed=0000000000000309 +mkstep=3 +mksite=2 +mklayer=5"}      # gen_gemm: keep_mask(777, 3, attn_out, 5)
INTEG = [("gm", "ep_bias_drop_resid", False), ("gmg", "ep_bias_drop_resid", False), ("vu", "elem", False), ("vu", "emb", False),
         ("lw", "adamw_real_head", True), ("gm", "real_ffn1_256", True), ("vu", "real_ln_fwd", True)]


def sh(cmd, log):
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
    with open(log, "w") as f:
        f.write(cmd + "\n" + r.stdout + r.stderr)
    return r


def build(name, top, srcs, params="", defs=""):
    mdir = os.path.join(OUT, "sim", name)
    exe = os.path.join(mdir, "V" + top)
    r = sh(f"verilator --binary -j 8 --timing -O3 -Wno-fatal -Wno-lint -Wno-style {params} {defs} {TECHDEF} +incdir+tb/mem "
           f"--top-module {top} -Mdir {mdir} {srcs} tb/mem/tb_dram.sv {HW}", mdir + ".build.log")
    if r.returncode or not os.path.exists(exe):
        raise RuntimeError(f"verilator build failed: {mdir}.build.log")
    return exe


def stats(txt, mkid=None):
    m = re.search(r"(?:STATS cycles \d+|DRAMSTATS) rd_beats (\d+) wr_beats (\d+) ar (\d+) aw (\d+) max_rd_out (\d+) max_wr_out (\d+) "
                  r"first (-?\d+) last (\d+)", txt)
    if not m:
        return {}
    rb, wb, ar, aw, mr, mw, f, l = map(int, m.groups())
    ids = re.search(r"^DRAMID ((?:\d+ ?){16})$", txt, re.M)
    ids = list(map(int, ids.group(1).split())) if ids else [0] * 16
    mk = ids[mkid] if mkid is not None else 0           # keep-bit stand-in beats (ruling R7: reported, not counted)
    span = max(1, l - f + 1)
    return dict(rb=rb, wb=wb, ar=ar, aw=aw, mr=mr, mw=mw, mk=mk, eff=(rb - mk + wb) / span, eff_all=(rb + wb) / span)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tech", choices=sorted(TECHS), default="generic")
    a = ap.parse_args()
    global OUT, TECHDEF
    if a.tech != "generic":
        OUT, TECHDEF = OUT + "-" + a.tech, f"+define+MM_TECH={TECHS[a.tech]}"
    if a.tech == "tsmc65":      # thousands of latch words: unoptimised C++ (an -Os build of gm_top takes many hours)
        TECHDEF += " -MAKEFLAGS OPT_FAST=-O0 -MAKEFLAGS OPT_SLOW=-O0"
    for d in ("vec", "sim", "integ"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    rows, bad = [], 0
    vec = os.path.join(OUT, "vec")
    for c in UNIT:
        r = sh(f"cd tb/mem && {PY} gen_mm.py {c} {vec}", os.path.join(OUT, f"gen_{c}.log"))
        if r.returncode:
            raise RuntimeError(f"gen_mm {c} failed")
    exes = {}
    with ThreadPoolExecutor(4) as ex:
        fut = {k: ex.submit(build, f"unit_{k}", "tb_mm_unit", "tb/mem/tb_mm_unit.sv", p) for k, p in
               (("a8", "-GAXL=3"), ("a4", "-GAXL=2"), ("a2", "-GAXL=1"), ("a8r4", "-GAXL=3 -GMAXR=4 -GMAXW=2"))}
        exes = {k: f.result() for k, f in fut.items()}
    jobs = [(k, c, kn) for k in ("a8", "a4", "a2") for c in UNIT for kn in (NOM, STRESS)]
    jobs += [("a8", "outst", "+lat=200 +bw=100000"), ("a8r4", "outst", "+lat=200 +bw=100000"), ("a8r4", "order", STRESS)]

    def run_unit(j):
        k, c, kn = j
        log = os.path.join(OUT, f"unit_{k}_{c}_{abs(hash(kn)) % 10000}.log")
        t = time.time()
        r = sh(f"{exes[k]} +cmd={vec}/{c}.cmd +exp={vec}/{c}.exp +reg={vec}/{c}.reg +img={vec}/{c}.img {kn}", log)
        m = re.search(r"^(PASS|FAIL) mm_unit (.*)$", r.stdout, re.M)
        res = m.group(0) if m else "FAIL (no result line)"
        st = stats(r.stdout)
        if c == "outst" and "lat=200" in kn:            # every outstanding slot must be used, never more
            want = 4 if k == "a8r4" else 8
            if st.get("mr") != want:
                res = f"FAIL max outstanding {st.get('mr')} != MAXR {want}"
        return (f"mm_unit AXI {32 * {'a8': 8, 'a4': 4, 'a2': 2, 'a8r4': 8}[k]}b{' MAXR4' if k == 'a8r4' else ''}", c, kn, res,
                f"max_out {st.get('mr')}/{st.get('mw')} eff {st.get('eff', 0):.3f}", time.time() - t)
    with ThreadPoolExecutor(8) as ex:
        for row in ex.map(run_unit, jobs):
            rows.append(row); bad += not row[3].startswith("PASS")

    # ---------------------------------------------------------------- integration
    integ = os.path.join(OUT, "integ")
    todo = [x for x in INTEG if not (a.quick and x[2])]
    with ThreadPoolExecutor(4) as ex:
        gens = [ex.submit(sh, f"cd tb/mem && {PY} gen_integ.py {e} {c} {integ}", os.path.join(OUT, f"gen_{e}_{c}.log")) for e, c, _ in todo
                if e != "gmg"]
        sims = {"gm": ex.submit(build, "gm", "tb_gm_top", "tb/gemm/tb_gm_top.sv", "-GMEMW=4 -GMASKW=4", "+define+MM_DRAM +define+MM_DMEMW=24"),
                "vu": ex.submit(build, "vu", "tb_vu_top", "tb/vec/tb_vu_top.sv", "-GMEMW=4 -GMASKW=4", "+define+MM_DRAM +define+MM_DMEMW=24"),
                "gmg": ex.submit(build, "gmg", "tb_gm_top", "tb/gemm/tb_gm_top.sv", "-GMEMW=4 -GMASKW=4",
                                 "+define+MM_DRAM +define+MM_DMEMW=24 +define+MM_MKGEN +define+MM_MKNC=3"),
                "lw": ex.submit(build, "lw", "tb_lw_top", "tb/lowa/tb_lw_top.sv", "-GMEMW=4", "+define+MM_DRAM +define+MM_DMEMW=24")}
        for g in gens:
            if g.result().returncode:
                raise RuntimeError("gen_integ failed: " + g.result().args)
        sims = {k: f.result() for k, f in sims.items()}
    ijobs = [(e, c, NOM) for e, c, _ in todo] + [(e, c, STRESS.replace(" +rstall=30", "")) for e, c, real in todo if not real]
    ijobs += [(e, c, "+lat=24 +bw=100000") for e, c, real in todo if e == "gm" and real]
    # the generator-sourced case: no keep-bit traffic may reach the AXI port (no ARID NR beats)
    # sensitivity: the same bandwidth with strict read priority in the DRAM (writes only on a spare token)
    ijobs += [(e, c, "+lat=24 +wlat=8 +bw=3200 +wfair=0") for e, c, real in todo if real and e in ("gm", "vu")]

    def run_int(j):
        e, c, kn = j
        log = os.path.join(integ, f"{e}_{c}_{abs(hash(kn)) % 10000}.log")
        t = time.time()
        # watchdog: every case gets a cycle limit (a hang fails with the tb's watchdog line, never waits forever)
        r = sh(f"{sims[e]} +desc={integ}/{c}.desc +exp={integ}/{c}.exp.hex +mem=none +reg={integ}/{c}.reg +img={integ}/{c}.img "
               f"+maxcyc={MAXCYC[c]} {kn} {MKCFG.get(e, '')}", log)
        m = re.search(r"^(PASS|FAIL) (gm|vu|lw)_top (.*)$", r.stdout, re.M)
        res = m.group(0) if m else "FAIL (no result line)"
        if "FAIL mm_dram" in r.stdout:
            res = "FAIL mm_dram protocol/response errors"
        cy = re.search(r"CYCLES (\d+)", r.stdout)
        st = stats(r.stdout, MKID.get(e))
        if e == "gmg" and st.get("mk", 1) != 0:
            res = f"FAIL keep-bit beats on the AXI port with the generator: {st.get('mk')}"
        return ("gm_top + mm_top + rn_mkgen" if e == "gmg" else f"{e}_top + mm_top", c, kn, res, f"cycles {int(cy.group(1)) if cy else -1:,} eff {st.get('eff', 0):.3f} "
                f"rd {st.get('rb', 0):,} wr {st.get('wb', 0):,} beats, stand-in mk {st.get('mk', 0):,}", time.time() - t)
    with ThreadPoolExecutor(6) as ex:
        for row in ex.map(run_int, ijobs):
            rows.append(row); bad += not row[3].startswith("PASS")

    lines = ["| unit | case | DRAM knobs | result | notes | time |", "|---|---|---|---|---|---|"]
    lines += [f"| {u} | {c} | {k} | {r} | {n} | {t:.0f}s |" for u, c, k, r, n, t in rows]
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("TEST-MEM", f"[{a.tech}]", "PASS" if bad == 0 else f"FAIL ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
