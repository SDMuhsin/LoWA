#!/usr/bin/env python3
"""`make synth-gemm`: Yosys + OpenSTA of the GEMM engine units (flow/synth/synth.sh for gm_pe,
flow/synth/synth_gemm.sh with the leaf hierarchy kept for the arrays and the engine), in
parallel.  Writes build/synth/gemm_table.md: unit, parameters, cells, area (um^2, tt lib, std
cells only: SRAM macros are blackboxes), worst setup slack at ss_125C_3v00 / tt_025C_3v30 (40 ns,
pre-layout), hold at ss, Yosys + STA runtime.  Exit 1 if any production configuration misses
timing at ss or a run fails.   run_gemm.py [--no-16] [--no-top]
"""
import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "build", "synth")


def run(script, unit, tag, params):
    d = os.path.join(OUT, f"{unit}_{tag}")
    args = " ".join(f"{k}={v}" for k, v in params.items())
    t = time.time()
    r = subprocess.run(f"flow/synth/{script} {unit} {d} {args}", shell=True, cwd=ROOT, capture_output=True, text=True)
    dt = time.time() - t
    sp = os.path.join(d, "summary.txt")
    txt = open(sp).read() if os.path.exists(sp) else ""

    def g(pat):
        m = re.search(pat, txt)
        return float(m.group(1)) if m else float("nan")
    return dict(unit=unit, tag=tag, params=params, rc=r.returncode, dt=dt,
                ss=g(r"STA ss_125C_3v00 wns_setup (-?[\d.]+)"), ssh=g(r"STA ss_125C_3v00 wns_setup -?[\d.]+ wns_hold (-?[\d.]+)"),
                tt=g(r"STA tt_025C_3v30 wns_setup (-?[\d.]+)"), area=g(r"AREA ([\d.]+)"),
                rep=g(r"STA ss_125C_3v00 wns_setup (-?[\d.]+) wns_hold -?[\d.]+ \(after repair"),
                cells=g(r"CELLS (\d+)"), flops=g(r"FLOPS (\d+)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-16", action="store_true")
    ap.add_argument("--no-top", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = [("synth.sh", "gm_pe", f"STAGES{s}", {"FMA_STAGES": s}) for s in (2, 3, 4)]
    jobs += [("synth_gemm.sh", "gm_array", f"{s}x{s}", {"ROWS": s, "COLS": s}) for s in ((4, 8) if a.no_16 else (4, 8, 16))]
    if not a.no_top:
        jobs.append(("synth_gemm.sh", "gm_top", "16x16", {"ROWS": 16, "COLS": 16, "NSFU": 16}))
    with cf.ThreadPoolExecutor(len(jobs)) as ex:
        res = list(ex.map(lambda j: run(*j), jobs))
    lines = ["| unit | parameters | cells | area tt (um^2) | slack ss raw (ns) | slack ss repaired (ns) | slack tt raw (ns) | hold ss (ns) | runtime |",
             "|---|---|---|---|---|---|---|---|---|"]
    bad = 0
    for r in res:
        best = r["rep"] if r["rep"] == r["rep"] else r["ss"]
        ok = r["rc"] == 0 and r["area"] == r["area"] and best >= 0
        prod = not (r["unit"] == "gm_pe" and r["params"].get("FMA_STAGES") != 2)
        bad += (not ok) and prod
        p = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        cells = r["cells"] if r["cells"] == r["cells"] else r["flops"]
        repc = f"{r['rep']:+.2f}" if r["rep"] == r["rep"] else "-"
        lines.append(f"| {r['unit']} | {p} | {cells:,.0f} | {r['area']:,.0f} | {r['ss']:+.2f} | {repc} | {r['tt']:+.2f} | "
                     f"{r['ssh']:+.2f} | {r['dt'] / 60:.1f} min |" + ("" if ok else " **FAIL**"))
    lines.append("")
    lines.append("Yosys 0.62 `synth -flatten -booth` (gm_pe / fp_sfu / SRAM banks kept as hierarchy), abc -D 40000 on "
                 "gf180mcu_as_sc_mcu7t3v3 tt; OpenSTA 2.7.0, 40 ns clock, flow/sta/unit.sdc, inputs registered by "
                 "flow/synth/wrap_<unit>.v; area = standard cells only (SRAM macros are blackboxes); for gm_pe the cells "
                 "column is the flop count.  'repaired' = after OpenROAD repair_design (buffer trees on the high-fanout "
                 "nets that Yosys leaves unbuffered, flow/sta/repair_sta.tcl); the pass criterion uses it where present.")
    lines.append("SYNTH-GEMM " + ("PASS" if bad == 0 else "FAIL"))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "gemm_table.md"), "w") as f:
        f.write(txt + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
