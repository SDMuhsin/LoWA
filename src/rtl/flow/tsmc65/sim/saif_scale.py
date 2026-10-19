#!/usr/bin/env python3
"""Rescale an RTL SAIF from the testbench clock to the chip clock (P8d power runs).

    saif_scale.py <in.saif> <out.saif> <factor>

The unit testbenches run a 40 ns clock; one testbench cycle stands for one chip cycle, so every duration
(DURATION, T0, T1, TX, TZ) is multiplied by factor = chip_period / 40 and the time unit is set to 1 ns
(PrimeTime accepts only 1 / 10 / 100 x a power-of-ten unit).  Toggle counts (TC, IG) are unchanged."""
import re
import sys

src, dst, f = sys.argv[1], sys.argv[2], float(sys.argv[3])
pat = re.compile(r"\((DURATION|T0|T1|TX|TZ) ([0-9.eE+-]+)\)")
ts = re.compile(r"\(TIMESCALE [^)]*\)")


def scale(m):
    return "(%s %.4f)" % (m.group(1), float(m.group(2)) * f)


seen = False
with open(src) as fi, open(dst, "w") as fo:
    for line in fi:
        if not seen and "TIMESCALE" in line:
            line = ts.sub("(TIMESCALE 1 ns)", line)
            seen = True
        fo.write(pat.sub(scale, line))
