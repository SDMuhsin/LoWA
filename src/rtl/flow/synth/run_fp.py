#!/usr/bin/env python3
"""`make synth-fp`: Yosys + OpenSTA of every hw/fp unit (flow/synth/synth.sh), production
configurations and stage-count sweeps, in parallel.  Writes build/synth/fp_table.md:
unit, parameters, flops, area (um^2, tt lib), worst setup slack at ss_125C_3v00 and tt_025C_3v30
(40 ns, pre-layout, ideal clock), worst hold slack at ss.  Exit 1 if a production config misses
timing at ss or a run fails.  The production netlists (build/synth/fp_<unit>_prod/gls.v) feed the
gate-level runs of `run_tests.py --gls`.
"""
import concurrent.futures as cf
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "build", "synth")
PROD = [("fp_fma", {"STAGES": 2}), ("fp_add", {"STAGES": 2}), ("fp_mul", {"STAGES": 1}),
        ("fp_cmp", {"STAGES": 1}), ("fp_i2f", {"STAGES": 1}), ("fp_sfu", {"LF": 2, "NTHR": 4})]
SWEEP = ([("fp_fma", {"STAGES": s}) for s in (1, 3, 4)] + [("fp_add", {"STAGES": s}) for s in (1, 3, 4)]
         + [("fp_mul", {"STAGES": 2}), ("fp_sfu", {"LF": 2, "NTHR": 1}), ("fp_sfu", {"LF": 3, "NTHR": 5})])


def run(unit, params, prod):
    tag = "prod" if prod else "_".join(f"{k}{v}" for k, v in params.items())
    d = os.path.join(OUT, f"{unit}_{tag}")
    args = " ".join(f"{k}={v}" for k, v in params.items())
    r = subprocess.run(f"flow/synth/synth.sh {unit} {d} {args}", shell=True, cwd=ROOT,
                       capture_output=True, text=True)
    txt = open(os.path.join(d, "summary.txt")).read() if os.path.exists(os.path.join(d, "summary.txt")) else ""
    def g(pat):
        m = re.search(pat, txt)
        return float(m.group(1)) if m else float("nan")
    return dict(unit=unit, params=params, prod=prod, rc=r.returncode,
                ss=g(r"STA ss_125C_3v00 wns_setup (-?[\d.]+)"), ssh=g(r"STA ss_125C_3v00 wns_setup -?[\d.]+ wns_hold (-?[\d.]+)"),
                tt=g(r"STA tt_025C_3v30 wns_setup (-?[\d.]+)"), area=g(r"AREA ([\d.]+)"), flops=g(r"FLOPS (\d+)"))


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(u, p, True) for u, p in PROD] + [(u, p, False) for u, p in SWEEP]
    with cf.ThreadPoolExecutor(len(jobs)) as ex:
        res = list(ex.map(lambda j: run(*j), jobs))
    lines = ["| unit | parameters | config | flops | area tt (um^2) | slack ss (ns) | slack tt (ns) | hold ss (ns) |",
             "|---|---|---|---|---|---|---|---|"]
    bad = 0
    for r in res:
        ok = r["rc"] == 0 and r["area"] == r["area"] and (not r["prod"] or r["ss"] >= 0)
        bad += not ok
        p = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        lines.append(f"| {r['unit']} | {p} | {'production' if r['prod'] else 'sweep'} | {r['flops']:.0f} | "
                     f"{r['area']:,.0f} | {r['ss']:+.2f} | {r['tt']:+.2f} | {r['ssh']:+.2f} |"
                     + ("" if ok else " **FAIL**"))
    lines.append("")
    lines.append("Yosys 0.62 `synth -flatten -booth`, abc -D 40000 on gf180mcu_as_sc_mcu7t3v3 tt; OpenSTA 2.7.0, "
                 "40 ns clock, flow/sta/unit.sdc, inputs registered by flow/synth/wrap_<unit>.v; area = the unit only.")
    lines.append("SYNTH-FP " + ("PASS" if bad == 0 else "FAIL"))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "fp_table.md"), "w") as f:
        f.write(txt + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
