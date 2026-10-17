#!/usr/bin/env python3
"""LoWA table images for the chip (P5): per module the merge hit list, its per-block directory and the
centre list, built from the L1 model dump (`lowa.<i>.eflat`, `lowa.<i>.hval` of model/runs/model_fp32.npz,
which are SGWALinear's own buffers: centres randperm(m*n, seed 777)[:k], 25 diamond taps in
torch.nonzero(H) order, reflection fold at the edges).  Nothing is computed from the seed on chip.

Per module (m = n = 768, k = 256, s = 25):
  hits    6400 x 32 bit, sorted by (block a = row // 64, block b = col // 64, entry e = 25 j + tap ascending):
          [5:0] row mod 64, [11:6] col mod 64, [16:12] tap, [24:17] atom j, [31:25] 0
  dir     256 x 32 bit, block (a, b) at a * 16 + b: [15:0] first hit index, [23:16] hit count, [31:24] 0
  centres 256 x 32 bit, (ci << 16) | cj in atom order (memory_map.md LOWA_CTR)
The same list serves ROW (block (nb, kc) = (a, b)) and TRN (block (nb, kc) = (b, a)) W-tile loads: the
cell set of a block is the same, only the tile visiting order differs (docs/p5_report.md).
TAPS = the 25 (di, dj) offsets in nonzero(H) order (row-major over the 7 x 7 box, |di| + |dj| <= 3).
"""
import numpy as np

D = 768
K_ATOM, S_TAP, KH = 256, 25, 3
BW = 64              # W block side (BC = BR = 64)
NBW = 16             # block grid side in the directory
TAPS = [(di, dj) for di in range(-KH, KH + 1) for dj in range(-KH, KH + 1) if abs(di) + abs(dj) <= KH]
CENTRE_TAP = TAPS.index((0, 0))


def reflect(p, n):
    p = np.where(p < 0, -1 - p, p)
    return np.where(p >= n, 2 * n - 1 - p, p)


def build(eflat, m=D, n=D):
    """eflat: [k*s] flat cells (atom-major, tap order).  Returns dict of uint32 arrays hits, dir, centres,
    plus rows/cols [k, s] int64 (the L1 lowa_sg inputs)."""
    e = np.asarray(eflat, np.int64).reshape(-1, S_TAP)
    k_at = e.shape[0]
    rows, cols = e // n, e % n
    ci, cj = rows[:, CENTRE_TAP], cols[:, CENTRE_TAP]
    off = np.array([t[0] for t in TAPS]), np.array([t[1] for t in TAPS])
    assert np.array_equal(rows, reflect(ci[:, None] + off[0][None, :], m)), "tap rows are not the reflected diamond"
    assert np.array_equal(cols, reflect(cj[:, None] + off[1][None, :], n)), "tap cols are not the reflected diamond"
    j = np.repeat(np.arange(k_at), S_TAP)
    tap = np.tile(np.arange(S_TAP), k_at)
    r, c = rows.reshape(-1), cols.reshape(-1)
    a, b = r // BW, c // BW
    key = (a * NBW + b) * (k_at * S_TAP) + np.arange(k_at * S_TAP)     # block, then e ascending
    order = np.argsort(key, kind="stable")
    hits = ((r[order] % BW) | ((c[order] % BW) << 6) | (tap[order] << 12) | (j[order] << 17)).astype(np.uint32)
    blk = (a * NBW + b)[order]
    dirw = np.zeros(NBW * NBW, np.uint32)
    cnt = np.bincount(blk, minlength=NBW * NBW)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    assert cnt.max() < 256
    dirw[:] = (start & 0xFFFF) | (cnt << 16)
    centres = ((ci << 16) | cj).astype(np.uint32)
    return dict(hits=hits, dir=dirw, centres=centres, rows=rows, cols=cols, ci=ci, cj=cj, cnt=cnt)


if __name__ == "__main__":
    import sys
    z = np.load(sys.argv[1] if len(sys.argv) > 1 else "model/runs/model_fp32.npz")
    mx = []
    for i in range(24):
        t = build(z[f"lowa.{i}.eflat"])
        mx.append(int(t["cnt"].max()))
        nz = t["cnt"][t["cnt"] > 0]
        print(f"module {i:2d}: blocks with hits {nz.size}/144, hits per block min {nz.min()} mean {nz.mean():.1f} max {nz.max()}, "
              f"folded taps {(np.abs(t['rows'] - t['ci'][:, None]) > 3).sum() + (np.abs(t['cols'] - t['cj'][:, None]) > 3).sum()}, "
              f"colliding cells {t['rows'].size - np.unique(t['rows'] * D + t['cols']).size}")
    print("max hits per block over 24 modules:", max(mx))
