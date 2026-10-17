#!/usr/bin/env python3
"""P6a integration vectors: an existing engine case (tb/gemm/gen_gemm.py, tb/vec/gen_vec.py, tb/lowa/gen_lowa.py; goldens
from L1 exactly as in P3-P5) with its tensors placed at docs/memory_map.md addresses instead of the flat testbench memory,
written as a tb_dram image (.reg / .img).  The engine generator's allocator is redirected: the n-th alloc() of the case
takes the n-th address of TARGETS[case] (real tensors -> their memory_map region), further allocs go to a bump pointer
in SCR_ACT slot 3.  Keep bits go to the MASK_STANDIN region (bit i at byte i/8, bit i%8; test stand-in, ruling R2).
    gen_integ.py <engine> <case> <outdir>        engine = gm | vu | lw"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("gemm", "vec", "lowa"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
from mm_image import REGION, Image, layer_addr  # noqa: E402

N, D, F = 4096, 768, 3072
SLOT = N * F * 4                                        # one SCR_ACT ping-pong slot (bytes)
STASH_L0 = REGION["STASH"]                              # layer 0 stash: z1, z2, Q, K, V [N x D], u [N x F]
TARGETS = {
    # u = h1 W1^T + b1, g = GELU(u):  h1 (SCR_ACT slot 0), W1 / b1 (LAYERS, layer 0), g (slot 1), u (stash u, layer 0)
    ("gm", "real_ffn1_256"): [REGION["SCR_ACT"], layer_addr(0, "W1"), layer_addr(0, "b1"), REGION["SCR_ACT"] + SLOT,
                              STASH_L0 + 5 * N * D * 4],
    # O = P' V: P' (SCR_ACT slot 0, all sequences), V (stash V, layer 0), O (slot 1)
    ("gm", "real_pv"): [REGION["SCR_ACT"], STASH_L0 + 4 * N * D * 4, REGION["SCR_ACT"] + SLOT],
    # LN1 of layer 0: x (stash z1), gamma / beta (LAYERS LN1), out (slot 0), stats (stash LN stats)
    ("vu", "real_ln_fwd"): [STASH_L0, layer_addr(0, "LN1"), layer_addr(0, "LN1") + D * 4, REGION["SCR_ACT"],
                            STASH_L0 + (5 * N * D + N * F) * 4, STASH_L0 + (5 * N * D + N * F + N) * 4],
    # AdamW of the head (in place): p, g, m, v at the head offset of PARAM / GRAD / ADAM_M / ADAM_V, scalars OPT_SCAL
    ("lw", "adamw_real_head"): [REGION[r] + 6144 * 4 for r in ("PARAM", "GRAD", "ADAM_M", "ADAM_V")] + [REGION["OPT_SCAL"]],
    # sampled gradient of q_0: centres (LOWA_CTR module 0), h taps (CONST + 4), grad (GRAD module 0), dQ (SCR_ACT slot 0),
    # X0 (STASH_X0)
    ("lw", "sg_real_l0q"): [REGION["LOWA_CTR"], REGION["CONST"] + 4, REGION["GRAD"], REGION["SCR_ACT"], REGION["STASH_X0"]],
    # embeddings: word table (EMB_WORD, compacted rows, see gen_vec), positions (EMB_POS), tt0 / LN gamma / beta
    # (EMB_TT_LN), ids (STEP_IN slot 0), out (STASH_X0)
    ("vu", "real_emb"): [REGION["EMB_WORD"], REGION["EMB_POS"], REGION["EMB_TT_LN"], REGION["EMB_TT_LN"] + D * 4,
                         REGION["EMB_TT_LN"] + 2 * D * 4, REGION["STEP_IN"], REGION["STASH_X0"]],
}


class MapAlloc:
    def __init__(self, targets):
        self.t = list(targets)
        self.bump = (REGION["SCR_ACT"] + 3 * SLOT) // 4
        self.allocs = []                                # (word, nwords)

    def __call__(self, obj, nwords):
        if self.t:
            a = self.t.pop(0) // 4
        else:
            a = self.bump
            self.bump = (self.bump + nwords + 7) // 8 * 8
        self.allocs.append((a, nwords))
        return a


def image(allocs, chunks, keep):
    img = Image()
    data = {a: arr for a, arr in chunks}
    # merge allocations into 32-byte-line padded regions (the DMA reads whole AXI lines)
    # (+64 words: the engines' last row / tile requests may read up to one 64-word request past a tensor)
    spans = sorted((a // 8 * 8, (a + n + 64 + 7) // 8 * 8) for a, n in allocs)
    merged = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    for lo, hi in merged:
        buf = np.zeros(hi - lo, np.uint32)
        for a, arr in data.items():
            if lo <= a < hi:
                buf[a - lo:a - lo + arr.size] = arr
        img.reg[lo] = buf
    if keep is not None and len(keep):
        # zero padding: the engines request keep bits past the tensor (the P3-P5 testbench answered 0 there)
        bits = np.zeros((len(keep) + 65536 + 255) // 256 * 256, np.uint8)
        bits[:len(keep)] = np.asarray(keep, np.uint8)
        img.reg[REGION["MASK_STANDIN"] // 4] = np.packbits(bits, bitorder="little").view(np.uint32)
    return img


def main():
    eng, case, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    ma = MapAlloc(TARGETS.get((eng, case), []))
    if eng == "gm":
        import gen_gemm as G
        G.Mem.alloc = lambda self, n: ma(self, n)
        G.Mem.write = lambda self, path: None
        c = G.build(case, 16, 16, 1)
        c.write(out)
        chunks, keep = c.mem.chunks, (None if c.mask is None else np.asarray(c.mask).reshape(-1))
    elif eng == "vu":
        import gen_vec as V
        V.Case.alloc = lambda self, n: ma(self, n)
        c = V.build(case, out, 16)
        chunks, keep = c.chunks, c.keep
    else:
        import gen_gemm as G
        G.Mem.alloc = lambda self, n: ma(self, n)
        G.Mem.write = lambda self, path: None
        import gen_lowa as LW
        c = LW.build(case, out)
        c.write(out)
        chunks, keep = c.mem.chunks, None
    img = image(ma.allocs, chunks, keep)
    img.write(os.path.join(out, case + ".reg"), os.path.join(out, case + ".img"))
    print(f"{eng} {case}: image {img.words()} words in {len(img.reg)} regions, {len(ma.t)} unused targets")


if __name__ == "__main__":
    main()
