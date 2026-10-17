#!/usr/bin/env python3
"""`make synth-lowa`: Yosys + OpenSTA of the LoWA unit (lw_top, NL 4, NSQ 2, NRC 1) through flow/synth/synth_lowa.sh
(synth_vec.sh with lw_sg / lw_adamw / lw_merge / lw_scal, fp_sfu, mm_sram_bank and cm_rstbuf kept as hierarchy), then
flow/sta/lowa_blocks.tcl for the per-block slack after repair_design.  Writes build/synth/lowa_table.md; exit 1 if
lw_top misses timing at ss_125C_3v00 after repair_design."""
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "build", "synth")
BLOCKS = {"u.u_sg": "lw_sg (sampled gradient)", "u.u_ad": "lw_adamw (AdamW)", "u.u_mg": "lw_merge (merge-on-load)",
          "u.u_sc": "lw_scal (LOWA_SCAL)"}


def g(txt, pat):
    m = re.search(pat, txt)
    return float(m.group(1)) if m else float("nan")


def main():
    os.makedirs(OUT, exist_ok=True)
    d = os.path.join(OUT, "lw_top_NL4")
    t = time.time()
    r = subprocess.run(f"flow/synth/synth_lowa.sh lw_top {d} NL=4 NSQ=2 NRC=1", shell=True, cwd=ROOT, capture_output=True, text=True)
    dt = time.time() - t
    txt = open(os.path.join(d, "summary.txt")).read() if os.path.exists(os.path.join(d, "summary.txt")) else ""
    ss, rep, tt = g(txt, r"STA ss_125C_3v00 wns_setup (-?[\d.]+)"), g(txt, r"wns_setup (-?[\d.]+) wns_hold -?[\d.]+ \(after repair"), g(txt, r"STA tt_025C_3v30 wns_setup (-?[\d.]+)")
    ssh, area, cells = g(txt, r"STA ss_125C_3v00 wns_setup -?[\d.]+ wns_hold (-?[\d.]+)"), g(txt, r"AREA ([\d.]+)"), g(txt, r"CELLS (\d+)")
    # per-block cells / area from the Yosys hierarchy stat
    stat = open(os.path.join(d, "stat.txt")).read() if os.path.exists(os.path.join(d, "stat.txt")) else ""
    blk = {}
    for m in re.finditer(r"=== (\S+) ===\n(.*?)(?==== |\Z)", stat, re.S):
        name, body = m.group(1), m.group(2)
        base = re.sub(r"^\$paramod\$[0-9a-f]+\\", "", name).split("\\")[-1]
        base = re.sub(r"\$paramod\\", "", base)
        cm, am = re.search(r"^\s+(\d+)\s+[\d.E+]+\s+cells", body, re.M), re.search(r"Chip area for module '[^']*': ([\d.]+)", body)
        if cm and am:
            blk[base] = (int(cm.group(1)), float(am.group(1)))
    # per-block slack after repair
    env = dict(os.environ, NETLIST=os.path.join(d, "nl.v"), TOP="wrap_lw_top", CORNER="ss_125C_3v00", PERIOD="40",
               SDC=os.path.join(ROOT, "flow", "sta", "unit.sdc"))
    rb = subprocess.run("openroad -exit flow/sta/lowa_blocks.tcl", shell=True, cwd=ROOT, capture_output=True, text=True, env=env)
    open(os.path.join(d, "blocks_ss.rpt"), "w").write(rb.stdout + rb.stderr)
    bslack = dict(re.findall(r"^BLOCK (\S+) wns_setup (\S+)", rb.stdout, re.M))
    ok = r.returncode == 0 and area == area and (rep if rep == rep else ss) >= 0
    repc = f"{rep:+.2f}" if rep == rep else "-"
    lines = ["| unit | parameters | cells | area tt (um^2) | slack ss raw (ns) | slack ss repaired (ns) | slack tt raw (ns) | hold ss (ns) | runtime |",
             "|---|---|---|---|---|---|---|---|---|",
             f"| lw_top | NL=4, NSQ=2, NRC=1 | {cells:,.0f} | {area:,.0f} | {ss:+.2f} | {repc} | {tt:+.2f} | {ssh:+.2f} | {dt / 60:.1f} min |"
             + ("" if ok else " **FAIL**"), "",
             "| block | cells | area tt (um^2, std cells) | slack ss repaired (ns) |", "|---|---|---|---|"]
    for inst, label in BLOCKS.items():
        mod = label.split()[0]
        c, a = blk.get(mod, (float("nan"), float("nan")))
        lines.append(f"| {label} | {c:,.0f} | {a:,.0f} | {bslack.get(inst, '-')} |")
    lines.append("")
    open(os.path.join(OUT, "lowa_table.md"), "w").write("\n".join(lines))
    print("\n".join(lines))
    print("SYNTH-LOWA", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
