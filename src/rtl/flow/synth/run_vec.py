#!/usr/bin/env python3
"""`make synth-vec`: Yosys + OpenSTA of the vector unit (flow/synth/synth_vec.sh: fp_sfu, vu_lane,
mm_sram_bank / SRAM macro kept as hierarchy): one lane (vu_lane) and the V = 16 unit (vu_top).
Writes build/synth/vec_table.md: unit, parameters, cells, area (um^2, tt lib, std cells only: SRAM
macros are blackboxes), worst setup slack at ss_125C_3v00 raw and after repair_design, tt, hold,
runtime.  Exit 1 if vu_top misses timing at ss after repair or a run fails.   run_vec.py [--no-top]
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


def run(unit, tag, params):
    d = os.path.join(OUT, f"{unit}_{tag}")
    args = " ".join(f"{k}={v}" for k, v in params.items())
    t = time.time()
    r = subprocess.run(f"flow/synth/synth_vec.sh {unit} {d} {args}", shell=True, cwd=ROOT, capture_output=True, text=True)
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
                cells=g(r"CELLS (\d+)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-top", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = [("vu_lane", "NS8", {"NS": 8, "SW": 3})]
    if not a.no_top:
        jobs.append(("vu_top", "V16", {"V": 16, "VW": 4}))
    with cf.ThreadPoolExecutor(len(jobs)) as ex:
        res = list(ex.map(lambda j: run(*j), jobs))
    lines = ["| unit | parameters | cells | area tt (um^2) | slack ss raw (ns) | slack ss repaired (ns) | slack tt raw (ns) | hold ss (ns) | runtime |",
             "|---|---|---|---|---|---|---|---|---|"]
    bad = 0
    for r in res:
        best = r["rep"] if r["rep"] == r["rep"] else r["ss"]
        ok = r["rc"] == 0 and r["area"] == r["area"] and best >= 0
        bad += not ok
        p = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        repc = f"{r['rep']:+.2f}" if r["rep"] == r["rep"] else "-"
        lines.append(f"| {r['unit']} | {p} | {r['cells']:,.0f} | {r['area']:,.0f} | {r['ss']:+.2f} | {repc} | {r['tt']:+.2f} | "
                     f"{r['ssh']:+.2f} | {r['dt'] / 60:.1f} min |" + ("" if ok else " **FAIL**"))
    lines.append("")
    lines.append("Yosys 0.62 `synth -flatten -booth` (vu_lane / fp_sfu / SRAM banks kept as hierarchy), abc -D 40000 on "
                 "gf180mcu_as_sc_mcu7t3v3 tt; OpenSTA, 40 ns clock, flow/sta/unit.sdc, inputs registered by "
                 "flow/synth/wrap_<unit>.v; area = standard cells only (SRAM macros are blackboxes: 2 x 4 per lane + "
                 "6 x 4 column-vector macros in vu_top). 'repaired' = after OpenROAD repair_design (flow/sta/repair_sta.tcl); "
                 "the pass criterion uses it where present.")
    lines.append("SYNTH-VEC " + ("PASS" if bad == 0 else "FAIL"))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "vec_table.md"), "w") as f:
        f.write(txt + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
