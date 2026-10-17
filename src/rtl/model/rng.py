"""On-chip dropout RNG (spec D1, dataflow.md section 7): Threefry-2x32-13, counter based.

The ONE implementation shared by L0 (mask injection) and L1 (hwmodel).  A mask is a pure
function of (seed, step, site, layer, element index); it is regenerated, never stored.

  counter (c0, c1) = (element >> 1, step << 8 | site << 4 | layer), key = (SEED_LO, SEED_HI)
  word = x[element & 1];  keep = (word >> 8) >= DROP_THRESH   (P(drop) = 1677722 / 2^24)
"""
import numpy as np

SITE = {"emb": 0, "attn": 1, "attn_out": 2, "ffn_out": 3, "cls1": 4, "cls2": 5}
DROP_THRESH = 1677722
DROP_SCALE = np.uint32(0x3F8E38E4).view(np.float32)          # fp32(1/0.9)
_ROT = (13, 15, 26, 6, 17, 29, 16, 24)
_PARITY = np.uint32(0x1BD11BDA)


def threefry2x32_13(c0, c1, k0, k1):
    """Random123 threefry2x32 with 13 rounds (key injection after every 4th round)."""
    c0 = np.asarray(c0, np.uint32).copy()
    c1 = np.asarray(c1, np.uint32).copy()
    ks = [np.uint32(k0), np.uint32(k1), np.uint32(_PARITY ^ np.uint32(k0) ^ np.uint32(k1))]
    with np.errstate(over="ignore"):
        x0 = c0 + ks[0]
        x1 = c1 + ks[1]
        for r in range(13):
            x0 += x1
            s = _ROT[r % 8]
            x1 = (x1 << np.uint32(s)) | (x1 >> np.uint32(32 - s))
            x1 ^= x0
            if r % 4 == 3:
                j = r // 4 + 1
                x0 += ks[j % 3]
                x1 += ks[(j + 1) % 3] + np.uint32(j)
    return x0, x1


def keep_mask(shape, seed, step, site, layer, chunk=1 << 24):
    """uint8 keep-mask (1 = keep) for a tensor of `shape`, element index = row-major index."""
    n = int(np.prod(shape))
    c1 = np.uint32((step << 8) | (SITE[site] << 4) | layer)
    k0, k1 = np.uint32(seed & 0xFFFFFFFF), np.uint32(seed >> 32)
    out = np.empty(n, np.uint8)
    for s in range(0, n, chunk):                       # chunk is even, so pairs never split
        e = np.arange(s, min(n, s + chunk), dtype=np.uint64)
        lo = e[::2] >> np.uint64(1)
        x0, x1 = threefry2x32_13(lo.astype(np.uint32), np.full(lo.shape, c1, np.uint32), k0, k1)
        w = np.empty(e.shape, np.uint32)
        w[0::2] = x0
        w[1::2] = x1[: len(w[1::2])]
        out[s:s + len(e)] = (w >> np.uint32(8)) >= DROP_THRESH
    return out.reshape(shape)


# Random123 kat_vectors, threefry2x32 13 rounds
KAT = [((0x00000000, 0x00000000), (0x00000000, 0x00000000), (0x9d1c5ec6, 0x8bd50731)),
       ((0xffffffff, 0xffffffff), (0xffffffff, 0xffffffff), (0xfd36d048, 0x2d17272c)),
       ((0x243f6a88, 0x85a308d3), (0x13198a2e, 0x03707344), (0xba3e4725, 0xf27d669e))]


def selftest():
    for ctr, key, want in KAT:
        x0, x1 = threefry2x32_13([ctr[0]], [ctr[1]], *key)
        got = (int(x0[0]), int(x1[0]))
        assert got == want, (ctr, key, [hex(g) for g in got], [hex(w) for w in want])
    m = keep_mask((64, 1000), 42, 0, "attn", 3)
    assert abs(1 - m.mean() - 0.1) < 0.005, m.mean()
    return True


if __name__ == "__main__":
    print("threefry KAT + drop rate:", selftest())
