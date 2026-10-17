#!/usr/bin/env python3
"""P6a unit vectors for tb_mm_unit: command list + expected beats / keep bits / final memory, computed by an
independent python reference (sequential semantics in command order) on a memory_map-placed DRAM image.
    gen_mm.py <case> <outdir> [--seed n]
Cases: burst, pattern, write, order, outst, stress (see docs/p6a_report.md section 3)."""
import argparse
import os

import numpy as np

from mm_image import REGION, Image, layer_addr

D = 768


class Case:
    def __init__(self, seed):
        self.r = np.random.default_rng(seed)
        self.img = Image()
        self.cmd, self.exp = [], []
        self.mask_bits = None

    # ---------------------------------------------------------------- reference semantics
    def rd(self, p, addr, n, g=0):
        assert 1 <= n <= 64 and addr % 4 == 0
        self.cmd.append(f"0 {p:x} {addr:08x} {n:x}")
        self._beats(p, addr, n, g)

    def _beats(self, p, addr, n, g=0):
        w = addr // 4
        for b in range((n + 7) // 8):
            nv = min(8, n - 8 * b)
            ws = [self.img.rd(w + 8 * b + j) for j in range(nv)] + [0] * (8 - nv)
            self.exp.append(f"R {p:x} {g:x} {nv:x} " + " ".join(f"{x:08x}" for x in ws))

    def wr(self, p, addr, mask, data):
        assert addr % 4 == 0
        self.cmd.append(f"1 {p:x} {addr:08x} {mask:02x} " + " ".join(f"{int(x):08x}" for x in data))
        for j in range(8):
            if mask >> j & 1:
                self.img.wr(addr // 4 + j, int(data[j]))

    def mk(self, idx):
        self.cmd.append(f"2 {idx:08x}")
        k = 0
        for j in range(16):
            k |= int(self.mask_bits[idx + j]) << j
        self.exp.append(f"K {k:04x}")

    def gather(self, idaddr, stride, base, n):
        self.cmd.append(f"3 {idaddr:08x} {stride:08x} {base:08x} {n:x}")
        i = self.img.rd(idaddr // 4)
        self.exp.append(f"R 1 1 1 {i:08x} " + " ".join(["00000000"] * 7))
        self._beats(0, base + i * stride, n)

    def rand_words(self, n):
        return self.r.integers(0, 2 ** 32, n, dtype=np.uint64).astype(np.uint32)

    def write(self, out, name, check=()):
        """check: list of (byte_addr, nwords) whose final contents are compared."""
        os.makedirs(out, exist_ok=True)
        self.img_init.write(os.path.join(out, name + ".reg"), os.path.join(out, name + ".img"))
        with open(os.path.join(out, name + ".cmd"), "w") as f:
            f.write("\n".join(self.cmd) + "\n")
        with open(os.path.join(out, name + ".exp"), "w") as f:
            f.write("\n".join(self.exp) + "\n")
            for a, n in check:
                for w in range(a // 4, a // 4 + n):
                    f.write(f"M {w:08x} {self.img.rd(w):08x}\n")

    def freeze(self):
        """snapshot of the initial image (the DRAM model loads this; self.img evolves with the writes)."""
        self.img_init = Image()
        for b, d in self.img.reg.items():
            self.img_init.reg[b] = d.copy()


def c_burst(c, n=400):
    base = REGION["SCR_ACT"] + 0x1000 - 256           # 3 page boundaries inside
    nw = 3 * 1024 + 512
    c.img.put(base, c.rand_words(nw)); c.freeze()
    # directed: every start offset 0..63 words below a 4 KiB boundary, len 64 and the remainder
    pb = REGION["SCR_ACT"] + 0x2000
    for k in range(1, 65):
        c.rd(k % 2, pb - 4 * k, 64)
        c.rd(k % 2, pb - 4 * k, k)
    for _ in range(n):
        ln = int(c.r.integers(1, 65))
        a = base + 4 * int(c.r.integers(0, nw - ln))
        c.rd(int(c.r.integers(0, 2)), a, ln)
    return []


def c_pattern(c):
    r = c.r
    # 2-D strided ROW tile: W1 of layer 0 rows 0..63, k chunk 2 (words 128..191), row stride 768 words
    w1 = layer_addr(0, "W1")
    c.img.put(w1, c.rand_words(64 * D))
    # TRN tile of a head slice (V of layer 0 in STASH: rows k = 0..63, words h*64 .. h*64+63), and an
    # unaligned row-major tile (50-word rows packed back to back, vector-unit style) in SCR_ACT
    x0 = REGION["STASH_X0"]
    c.img.put(x0, c.rand_words(64 * D))
    sa = REGION["SCR_ACT"]
    c.img.put(sa, c.rand_words(64 * 50 + 64))
    # gather: 32 ids (int32 words) in STEP_IN, rows of EMB_WORD (only the gathered rows are in the image)
    ids = r.choice(50265, 32, replace=False)
    c.img.put(REGION["STEP_IN"], ids.astype(np.uint32))
    for i in ids:
        c.img.put(REGION["EMB_WORD"] + int(i) * D * 4, c.rand_words(D))
    # keep-bit stand-in
    nb = 1 << 14
    c.mask_bits = r.integers(0, 2, nb + 64).astype(np.uint8)
    c.img.put(REGION["MASK_STANDIN"], np.packbits(c.mask_bits[:nb], bitorder="little").view(np.uint32))
    c.freeze()
    for rr in range(64):
        c.rd(0, w1 + 4 * (rr * D + 128), 64)                      # ROW
    for h in (0, 5, 11):
        for k in range(64):
            c.rd(0, x0 + 4 * (k * D + h * 64), 64)                # TRN source rows of a head slice
    for rr in range(64):
        c.rd(1, sa + 4 * (rr * 50), 50)                            # unaligned rows
        c.rd(1, sa + 4 * (rr * 50 + 3), 13)
    for t in range(32):
        for ch in (0, 5, 11):
            c.gather(REGION["STEP_IN"] + 4 * t, D * 4, REGION["EMB_WORD"] + 4 * 64 * ch, 64)
    for i in range(0, 4096, 16):
        c.mk(i)                                                    # attention-row stream (aligned)
    for rr in range(200):
        c.mk(rr * 3 + 7)                                           # head-like rows of C = 3, straddling
    for _ in range(300):
        c.mk(int(r.integers(0, nb - 16)))                          # random
    return []


def c_write(c):
    r = c.r
    base = REGION["SCR_ATT"] + 0x1000 - 512
    nw = 2048 + 1024
    c.img.put(base, c.rand_words(nw)); c.freeze()
    # contiguous aligned streams (coalesce to 16-beat bursts, cross 4 KiB), both channels
    a = base
    for i in range(70):
        c.wr(i % 2 if i >= 40 else 0, a + 32 * i, 0xFF, c.rand_words(8))
    # unaligned contiguous stream, full masks
    a2 = base + 4 * (600 + 3)
    for i in range(40):
        c.wr(1, a2 + 32 * i, 0xFF, c.rand_words(8))
    # head-like rows: 3 words at a 12-byte row stride
    a3 = base + 4 * 1100
    for i in range(100):
        c.wr(0, a3 + 12 * i, 0x07, c.rand_words(8))
    # random beats, random masks (incl. 0)
    for _ in range(300):
        c.wr(int(r.integers(0, 2)), base + 4 * int(r.integers(0, nw - 8)), int(r.integers(0, 256)), c.rand_words(8))
    # read everything back through the read channels
    for o in range(0, nw - 64, 61):
        c.rd(int(r.integers(0, 2)), base + 4 * o, 64)
    return [(base, nw)]


def c_order(c, n=1500):
    r = c.r
    base = REGION["SCR_CLS"]
    nw = 192
    c.img.put(base, c.rand_words(nw)); c.freeze()
    for _ in range(n):
        op = r.integers(0, 3)
        if op == 0:
            ln = int(r.integers(1, 65)); c.rd(int(r.integers(0, 2)), base + 4 * int(r.integers(0, nw - ln)), ln)
        else:
            c.wr(int(r.integers(0, 2)), base + 4 * int(r.integers(0, nw - 8)), int(r.integers(1, 256)), c.rand_words(8))
    return [(base, nw)]


def c_outst(c, n=600):
    base = REGION["SCR_ACT"]
    c.img.put(base, c.rand_words(n * 8 + 64)); c.freeze()
    for i in range(n):
        c.rd(i % 2, base + 32 * i, 8)
    return []


CASES = {"burst": c_burst, "pattern": c_pattern, "write": c_write, "order": c_order, "outst": c_outst}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    c = Case(a.seed)
    chk = CASES[a.case](c)
    c.write(a.out, a.case, chk)
    print(f"{a.case}: {len(c.cmd)} commands, {len(c.exp)} expected lines, image {c.img.words()} words")


if __name__ == "__main__":
    main()
