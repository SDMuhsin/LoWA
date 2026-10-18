#!/usr/bin/env python3
"""P8e: per-macro power numbers from the generated liberty files (summary numbers only, safe to copy off the server).
  python3 flow/tsmc65/sram/mc_power.py <T65_MC_OUT> <corner_name> <period_ns> <inst> ...
For each macro: leakage (largest leakage_power value of the cell with the retention pin active, i.e. not the
'!RETN' state), and the clock-pin internal energy of a read and of a write access at margin setting 000 (rise + fall
energy of the CLK internal_power group whose condition is chip-enabled read / write; first table value, these tables
are flat).  Energy unit = voltage_unit x capacitive_load_unit (V x pF = pJ).  Then the P8d vectorless activity
(static probability 0.5 on the chip and write enables: an access in 50 % of cycles, half reads, half writes)
gives dynamic power at the period.  The same from the compiler datatable (<inst>.dat: read / write AC currents of
core + periphery at margin setting 000, mA/MHz, times the corner voltage = nJ per access), because one compiler's
liberty carries zero clock-pin energy.  Prints one MCPOWER line per macro."""
import re
import sys


def blocks(text, key):
    """yield the bodies of `key(...) { ... }` groups (brace matched)"""
    for m in re.finditer(r"\b" + key + r"\s*\([^)]*\)\s*\{", text):
        i, depth = m.end(), 1
        while depth:
            c = text[i]
            depth += (c == "{") - (c == "}")
            i += 1
        yield text[m.end():i - 1]


def main():
    out, corner, period, insts = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4:]
    for inst in insts:
        t = open(f"{out}/{inst}/{inst}_{corner}_syn.lib").read().replace("\\\n", " ")
        leak = [float(v) for b in blocks(t, "leakage_power") if "!RETN" not in b
                for v in re.findall(r"value\s*:\s*([0-9.eE+-]+)", b)]
        clk = next(b for b in blocks(t, "pin") if re.search(r"clock\s*:\s*true", b))
        e = {}
        for b in blocks(clk, "internal_power"):
            w = re.search(r'when\s*:\s*"([^"]*)"', b)
            if not w:
                continue
            cond = re.sub(r"\s+", "", w.group(1))
            if "!EMA[2]" not in cond or "!EMA[1]" not in cond or "!EMA[0]" not in cond or "!CEN" not in cond:
                continue
            kind = "w" if "!(WEN)" in cond or "!WEN" in cond else "r"
            vals = [float(re.search(r'values\s*\(\s*"\s*([0-9.eE+-]+)', g).group(1))
                    for g in list(blocks(b, "rise_power")) + list(blocks(b, "fall_power"))]
            e[kind] = max(e.get(kind, 0.0), sum(vals))
        er, ew = e.get("r", float("nan")), e.get("w", float("nan"))
        rows = {l.split()[0]: l.split()[1:] for l in open(f"{out}/{inst}/{inst}.dat") if l.strip() and not l.startswith("#")}
        col = rows["name"].index(corner)
        g = lambda k: float(rows[k][col])
        dr = (g("icc_c_r0") + g("icc_p_r0")) * g("volt") * 1000.0
        dw = (g("icc_c_w0") + g("icc_p_w0")) * g("volt") * 1000.0
        f_mhz = 1000.0 / period
        dyn_mw = 0.5 * 0.5 * (er + ew) * f_mhz * 1e-3          # pJ x MHz = uW
        print(f"MCPOWER {inst} corner {corner} leak_mW {max(leak) if leak else float('nan'):.5f} "
              f"E_read_pJ {er:.3f} E_write_pJ {ew:.3f} dyn_vectorless_mW@{period}ns {dyn_mw:.4f} | datatable "
              f"E_read_pJ {dr:.3f} E_write_pJ {dw:.3f} dyn_vectorless_mW {0.25 * (dr + dw) * f_mhz * 1e-3:.4f}")


if __name__ == "__main__":
    main()
