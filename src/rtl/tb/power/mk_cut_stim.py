#!/usr/bin/env python3
"""P9b stimulus for gate-level power windows: the compiled e2e program (tb/top/run_e2e.py, one step) started at a
later unit op, so that a window deep in the step (LoWA gradient, AdamW) is reached without simulating the ops
before it at gate level.

  python3 tb/power/mk_cut_stim.py --layers 1 --batch 2 --seq 32 --cut <k> --out <dir>

<k> = index of the first unit op that runs (unit ops counted in execution order from 0, as run_e2e --stop-after).
The interpreter (sw/interp.py) executes every descriptor before op k on the initial image, so the DRAM image
written here holds exactly the data those ops produce (activations, gradients, scalars); the program keeps its
SETCSR descriptors and replaces every other descriptor before op k (unit ops, LOOP, FENCE) by NOPs, slot for
slot.  An event flag set by a skipped op and still pending at the cut is dropped from its first consumer's WAIT
(that op issues as soon as its unit is free, as it would once the producer had retired).  The cut must lie
after the layer loops (every kept descriptor is checked to decode to the same fields with the reset term values; e.g.
layer 0's backward at --layers 2) or inside the one-iteration loop of --layers 1.  Writes dram.reg, dram.bin,
host.txt (one step: CSRs, doorbell, IRQ wait, ack) and ops.txt (op index, pc, name of the kept unit ops)."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "sw"))
sys.path.insert(0, os.path.join(ROOT, "tb", "top"))
import compile as CP  # noqa: E402
import interp  # noqa: E402
import isa  # noqa: E402
import run_e2e as E  # noqa: E402

U32 = np.uint32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--cut", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = CP.Cfg(a.layers, a.batch, a.seq)
    r = CP.compile_all(c)
    inp, s0, img, cs = r["inp"], int(r["inp"]["step_s"]), r["img"], r["csr"]
    # DRAM regions exactly as run_e2e.py (image + every write range)
    items = CP.program(c, CP.EPOCHS * int(inp["n_batches"]))
    flat = CP.flatten(items, c, None, s0)
    wr_iv = []
    for name, f, _, _ in flat:
        if isa.OPS[name][1] != isa.UNIT_SEQ:
            wr_iv += CP.ranges(name, f, c)[1]
    img_iv = [(ad, ad + 4 * arr.size) for ad, arr in img.reg]
    img_iv.append((CP.REG["PROG"], CP.REG["PROG"] + 4 * r["prog"].size + 256))
    regs = E.merge_regions(img_iv + wr_iv, 120)
    m = interp.Machine(CP.IMG_END, cs["SEED"], s0, cs["SLOT_DELTA"], 0)
    m.load(img)
    prog0 = m.mem[CP.REG["PROG"] // 4:CP.REG["PROG"] // 4 + r["prog"].size].copy()
    # execute descriptors up to unit op k (interp.Machine.run, stopped at the cut)
    pc, loop, k, pend, nsk = CP.REG["PROG"], None, 0, set(), 0
    nop = isa.OPS["NOP"][0] | (0xF << 12) | (0xF << 16)
    newp = prog0.copy()
    while True:
        w0 = int(m.mem[pc // 4])
        h = isa.header(w0)
        nsl = h["nsl"]
        hh, name, f, _ = isa.decode_words(m.mem[pc // 4:pc // 4 + 16 * nsl], m.term)
        unit_op = isa.OPS[name][1] != isa.UNIT_SEQ
        if unit_op and k == a.cut:
            break
        assert name != "END", "cut beyond the program"
        nxt = pc + 64 * nsl
        if name == "LOOP":
            loop = dict(start=nxt, end=nxt + f["body"], left=f["count"], dir=f["dir"])
            m.term[isa.T_LIDX] = f["start"]
            for q in range(1, 13):
                m.term[q] = 0
        elif name == "SETCSR":
            if 1 <= f["csr"] <= 12:
                m.str[f["csr"]] = f["value"]
        elif name in ("GEMM", "GEMM_ACC2"):
            m.gemm(f)
        elif name in ("VEC", "LOSS", "GNORM"):
            m.vecop(f)
        elif name in ("LOWA_SG", "LOWA_SCAL", "ADAMW"):
            m.lowa(f)
        elif name == "DMA_COPY":
            m.dma(f)
        if unit_op:
            if h["wait"] != 0xF and h["clr"]:
                pend.discard(h["wait"])
            if h["sig"] != 0xF:
                pend.add(h["sig"])
            k += 1
        if name != "SETCSR":
            o = (pc - CP.REG["PROG"]) // 4
            newp[o:o + 16 * nsl] = 0
            for s in range(nsl):
                newp[o + 16 * s] = nop
            nsk += 1
        pc = nxt
        if loop is not None and pc == loop["end"]:
            loop["left"] -= 1
            if loop["left"] > 0:
                sgn = -1 if loop["dir"] else 1
                m.term[isa.T_LIDX] += sgn
                for q in range(1, 13):
                    m.term[q] = (m.term[q] + sgn * m.str[q]) & 0xFFFFFFFF
                pc = loop["start"]
            else:
                loop = None
    # inside a loop: only the one-iteration layer loop of --layers 1 (the chip, its LOOP descriptor replaced by a NOP,
    # runs the rest of the body once, straight through, with the terms at their reset values, as the loop would)
    assert loop is None or (loop["left"] == 1 and all(t == 0 for t in m.term[:14])), "cut inside a multi-iteration loop"
    # after the loops the chip's terms are at their reset values (0; STEP / SLOT from the CSRs) while the
    # interpreter's need not be: every kept descriptor must decode to the same fields with both
    t_rst = [0] * 16
    t_rst[isa.T_STEP], t_rst[isa.T_SLOT] = m.term[isa.T_STEP], m.term[isa.T_SLOT]
    p3 = pc
    while True:
        h3 = isa.header(int(m.mem[p3 // 4]))
        w3 = m.mem[p3 // 4:p3 // 4 + 16 * h3["nsl"]]
        d_int, d_rst = isa.decode_words(w3, m.term), isa.decode_words(w3, t_rst)
        assert str(d_int[2]) == str(d_rst[2]) and str(d_int[3]) == str(d_rst[3]), f"descriptor at {p3:#x} uses loop terms"
        if isa.OPNAME[h3["op"]] == "END":
            break
        p3 += 64 * h3["nsl"]
    # kept ops: drop pending flags from their first consumer
    ops, kk, p2 = [], a.cut, pc
    while True:
        o = (p2 - CP.REG["PROG"]) // 4
        h = isa.header(int(newp[o]))
        name = isa.OPNAME[h["op"]]
        if name == "END":
            break
        if isa.OPS[name][1] != isa.UNIT_SEQ:
            if h["wait"] in pend:
                pend.discard(h["wait"])
                newp[o] = U32((int(newp[o]) & ~((0xF << 12) | (1 << 20))) | (0xF << 12))
                print(f"op {kk} {name}: wait on flag {h['wait']} (producer skipped) dropped")
            ops.append(f"{kk} {p2:#x} {name}")
            kk += 1
        elif name == "FENCE" or name == "LOOP":
            ops.append(f"- {p2:#x} {name}")
        p2 += 64 * h["nsl"]
    m.mem[CP.REG["PROG"] // 4:CP.REG["PROG"] // 4 + newp.size] = newp
    os.makedirs(a.out, exist_ok=True)
    pk = np.concatenate([m.mem[lo // 4:hi // 4] for lo, hi in regs])
    with open(os.path.join(a.out, "dram.reg"), "w") as fr:
        fr.write("".join(f"{lo // 4:08x} {(hi - lo) // 4:08x}\n" for lo, hi in regs))
    pk.astype(">u4").tofile(os.path.join(a.out, "dram.bin"))
    seed = cs["SEED"]
    hs = [(1, 0x020, cs["PROG_BASE"]), (1, 0x024, cs["SLOT_DELTA"]), (1, 0x018, cs["STEP_T"]),
          (1, 0x060, seed & 0xFFFFFFFF), (1, 0x064, seed >> 32), (1, 0x008, 0x11), (6, 0, 0),
          (1, 0x010, 0x80000000), (4, 0, 0), (1, 0x014, 0), (7, 0, 0)]
    with open(os.path.join(a.out, "host.txt"), "w") as fh:
        fh.write("".join(f"{op:x} {x:x} {y:x}\n" for op, x, y in hs))
    with open(os.path.join(a.out, "ops.txt"), "w") as fo:
        fo.write("\n".join(ops) + "\n")
    print(f"cut at unit op {a.cut} (pc {pc:#x}): {nsk} descriptors replaced by NOPs, {len(regs)} regions, "
          f"{pk.size * 4 / 2**20:.1f} MiB; kept {sum(1 for x in ops if not x.startswith('-'))} unit ops")


if __name__ == "__main__":
    main()
