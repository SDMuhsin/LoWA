#!/usr/bin/env python3
"""P9a: drain points of an e2e run (op log): unit ops (static and dynamic program order) at whose start every
earlier unit op of the step has already retired, i.e. the chip is idle apart from the sequencer.  A program chunk
that starts at a drain point and ends at the next one sees exactly the in-situ machine state at its start (all
engines idle, nothing in flight), so its start-to-start duration measured in isolation equals the in-situ one.
  python3 tb/top/drain.py build/e2e/l1_b2_s32_n2 [--step 1]
Prints one line per dynamic unit op: index, static index, segment, name, start, retire, drain flag."""
import argparse
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seg as SG  # noqa: E402


def timeline(d, step=1):
    """[(dyn idx, static idx, segment, name, unit, start, retire)] for the unit ops of one step"""
    words = SG.load_prog(d)
    ops = SG.walk_seg(words)
    stat = static_index(words)
    ev = defaultdict(lambda: defaultdict(list))
    for ln in open(os.path.join(d, "oplog.txt")):
        k, u, c = ln.split()
        ev[int(u)][k].append(int(c))
    n = defaultdict(int)
    for u, _, _, _ in ops_with_pc(words):
        n[u] += 1
    seen = defaultdict(int)
    out = []
    for i, (u, s, name, pc) in enumerate(ops_with_pc(words)):
        j = (step - 1) * n[u] + seen[u]
        seen[u] += 1
        S = ev[u]["S"][j] if j < len(ev[u]["S"]) else None
        R = ev[u]["R"][j] if j < len(ev[u]["R"]) else None
        out.append((i, stat[pc], s, name, u, S, R))
    return out


def static_index(words):
    import isa
    pc, k, m = 0, -1, {}
    while True:
        w0 = int(words[pc // 4])
        nsl = ((w0 >> 21) & 3) + 1
        h = isa.header(w0)
        if h["unit"] != isa.UNIT_SEQ:
            k += 1
            m[pc] = k
        if (w0 & 0xFF) == isa.OPS["END"][0]:
            return m
        pc += 64 * nsl


def ops_with_pc(words):
    """walk_seg, plus the descriptor pc of every unit op"""
    import isa
    term, strd = [0] * 16, [0] * 16
    pc, loop, nloops, seg, out = 0, None, 0, "pre", []
    while True:
        w0 = int(words[pc // 4])
        nsl = ((w0 >> 21) & 3) + 1
        h, name, f, pay = isa.decode_words(words[pc // 4:pc // 4 + 16 * nsl], term)
        nxt = pc + 64 * nsl
        if name == "LOOP":
            nloops += 1
            loop = dict(start=nxt, end=nxt + f["body"], left=f["count"], dir=f["dir"])
            term[isa.T_LIDX] = f["start"]
            term[1:13] = [0] * 12
        elif name == "SETCSR":
            if 1 <= f["csr"] <= 12:
                strd[f["csr"]] = f["value"]
        elif name == "END":
            return out
        elif name not in ("NOP", "FENCE"):
            if seg == "tail":
                pass
            elif loop:
                seg = ("B%d" if loop["dir"] else "F%d") % term[isa.T_LIDX]
            elif nloops == 1:
                seg = "head"
            elif nloops >= 2:
                seg = "B0"
            if name == "GNORM" and seg in ("B0", "head") and nloops >= 1:
                seg = "tail"
            out.append((SG.UIDX[h["unit"]], seg, name, pc))
        pc = nxt
        if loop and pc == loop["end"]:
            loop["left"] -= 1
            if loop["left"] > 0:
                s = -1 if loop["dir"] else 1
                term[isa.T_LIDX] += s
                for k in range(1, 13):
                    term[k] = (term[k] + s * strd[k]) & 0xFFFFFFFF
                pc = loop["start"]
            else:
                loop = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--step", type=int, default=1)
    a = ap.parse_args()
    tl = timeline(a.dir, a.step)
    last_r = -1
    for i, st, s, name, u, S, R in tl:
        drain = S is not None and S >= last_r
        print(f"{i:4d} {st:4d} {s:6s} {name:10s} u{u} S {S} R {R} {'DRAIN' if drain else ''} dur {R - S if R and S else ''}")
        if R is not None:
            last_r = max(last_r, R)


if __name__ == "__main__":
    main()
