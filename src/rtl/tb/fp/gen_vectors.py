#!/usr/bin/env python3
"""Test vectors for the hw/fp units.  Expected values come ONLY from the L1 entry points
(model/hwmodel, `import hwmodel as H`, docs/arith.md section 5); nothing here computes a result.

    gen_vectors.py UNIT KIND N SEED OUT
      UNIT  fma | add | mul | cmp | i2f | sfu
      KIND  directed (N ignored) | random (N vectors)
Line format (hex, one vector per line):
      fma: a b c y      add: a b sub y      mul: a b y      cmp: a b flags mx mn
      i2f: x y          sfu: fn x y   (fn: 0 exp 1 recip 2 rsqrt 3 sqrt 4 tanh 5 erf 6 ln)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model"))
import hwmodel as H  # noqa: E402

U32 = np.uint32
FNS = ["exp", "recip", "rsqrt", "sqrt", "tanh", "erf", "log"]
SFU_FN = {n: i for i, n in enumerate(FNS)}


def f2u(x):
    return np.asarray(x, np.float32).view(U32)


def u2f(u):
    return np.asarray(u, U32).view(np.float32)


def bits(s, e, f):
    return ((np.asarray(s, np.uint64) << 31) | (np.asarray(e, np.uint64) << 23)
            | np.asarray(f, np.uint64)).astype(U32)


def write_hex(path, cols, widths=None):
    """cols: list of uint arrays (same length).  Fast vectorised hex formatting."""
    n = len(cols[0])
    widths = widths or [8] * len(cols)
    parts = []
    lut = np.frombuffer(b"0123456789abcdef", np.uint8)
    for k, (c, w) in enumerate(zip(cols, widths)):
        c = np.asarray(c, np.uint64)
        sh = np.arange(w - 1, -1, -1, dtype=np.uint64) * 4
        nib = ((c[:, None] >> sh[None, :]) & 15).astype(np.uint8)
        parts.append(lut[nib])
        sep = np.full((n, 1), ord(" ") if k < len(cols) - 1 else ord("\n"), np.uint8)
        parts.append(sep)
    np.concatenate(parts, axis=1).tofile(path)


# --------------------------------------------------------------------------- operand pools
SPECIAL = [0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001,
           0xFFFFFFFF, 0x7FBFFFFF, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00400000,
           0x00800000, 0x80800000, 0x00800001, 0x00FFFFFF, 0x01000000, 0x3F800000, 0xBF800000,
           0x3F800001, 0x3F7FFFFF, 0x40000000, 0x3F000000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F000000,
           0x1F800000, 0x5F800000, 0x0C800000, 0x33800000, 0x34000000, 0x4B000000, 0x4B400000,
           0x40490FDB, 0xC02DF854, 0x3EAAAAAB, 0x00000100, 0x7E800000, 0x01800000, 0x20000000]


def rand_normal(r, n, elo=1, ehi=254):
    return bits(r.integers(0, 2, n), r.integers(elo, ehi + 1, n), r.integers(0, 1 << 23, n))


def mix_operands(r, n):
    """n x 3 operand words drawn from several families (shared by fma/add/mul/cmp)."""
    fam = r.integers(0, 10, n)
    out = np.empty((n, 3), U32)
    any_bits = r.integers(0, 1 << 32, (n, 3), dtype=np.uint64).astype(U32)
    nrm = np.stack([rand_normal(r, n) for _ in range(3)], 1)
    mid = np.stack([rand_normal(r, n, 100, 154) for _ in range(3)], 1)          # |x| ~ 2^-27..2^27
    near = np.stack([rand_normal(r, n, 120, 134) for _ in range(3)], 1)
    sp = np.asarray(SPECIAL, U32)[r.integers(0, len(SPECIAL), (n, 3))]
    sub = (r.integers(0, 1 << 23, (n, 3)) | (r.integers(0, 2, (n, 3)) << 31)).astype(U32)
    out[:] = any_bits
    for f, src in ((1, nrm), (2, mid), (3, near), (4, near), (5, mid)):
        out[fam == f] = src[fam == f]
    # family 6: one operand special / subnormal, others normal
    m6 = fam == 6
    pick = r.integers(0, 3, n)
    rep = np.where(r.integers(0, 2, n) == 0, sp[:, 0], sub[:, 0])
    out[m6] = mid[m6]
    out[m6, pick[m6]] = rep[m6]
    # family 7: few-bit significands (exact ties for add and fma)
    m7 = fam == 7
    fb = (r.integers(0, 1 << 6, (n, 3)) << 17).astype(np.uint64)
    tie = bits(r.integers(0, 2, (n, 3)), r.integers(110, 145, (n, 3)), fb)
    out[m7] = tie[m7]
    # family 8: tiny exponents (results around 2^-126 and FTZ)
    m8 = fam == 8
    tiny = np.stack([rand_normal(r, n, 1, 70), rand_normal(r, n, 1, 70), rand_normal(r, n, 1, 6)], 1)
    out[m8] = tiny[m8]
    # family 9: huge exponents (overflow)
    m9 = fam == 9
    big = np.stack([rand_normal(r, n, 150, 254), rand_normal(r, n, 150, 254), rand_normal(r, n, 200, 254)], 1)
    out[m9] = big[m9]
    return out


def cancel_c(r, a, b):
    """c = -(a*b) (L1 mul) nudged by a few ulps: massive cancellation for fma."""
    p = f2u(H.mul(u2f(a), u2f(b))) ^ U32(0x80000000)
    return (p.astype(np.int64) + r.integers(-3, 4, len(a))).astype(U32)


# --------------------------------------------------------------------------- per unit
def fma_cases(kind, n, r):
    if kind == "directed":
        sp = np.asarray(SPECIAL, U32)
        g = np.array(np.meshgrid(sp, sp, sp, indexing="ij")).reshape(3, -1).T
        k = np.arange(1, 4096, dtype=U32)
        extra = [
            g,
            np.stack([np.full_like(k, 0x00800000), U32(0x3F800000) - k, np.zeros_like(k)], 1),
            np.stack([U32(0x00800000) + k, np.full_like(k, 0x3F7FFFFF), np.zeros_like(k)], 1),
            np.stack([U32(0x00800000) + k, np.full_like(k, 0x3F7FFFFF), np.full_like(k, 0x80800000)], 1),
            np.stack([np.full_like(k, 0x3F800000), np.full_like(k, 0x3F800000), U32(0xBF800000) - k], 1),
            np.stack([np.full_like(k, 0x3F800000) + k, np.full_like(k, 0x3F800000) + k,
                      np.full_like(k, 0xBF800000) + 2 * k], 1),
        ]
        # every product exponent vs addend exponent offset in [-80, 80] (alignment shifter)
        off = np.arange(-80, 81)
        ea = np.repeat(127 + (off // 2), 64)
        ec = np.repeat(127 + off, 64) + 0
        ec = np.clip(ec + 0, 1, 254)
        m = len(ea)
        extra.append(np.stack([bits(r.integers(0, 2, m), ea, r.integers(0, 1 << 23, m)),
                               bits(r.integers(0, 2, m), np.full(m, 127), r.integers(0, 1 << 23, m)),
                               bits(r.integers(0, 2, m), ec, r.integers(0, 1 << 23, m))], 1))
        a = rand_normal(r, 20000, 64, 190)
        b = rand_normal(r, 20000, 64, 190)
        extra.append(np.stack([a, b, cancel_c(r, a, b)], 1))
        cases = np.concatenate(extra).astype(U32)
    else:
        cases = mix_operands(r, n)
        m = r.integers(0, 6, n) == 0                    # 1/6: cancellation
        cases[m, 2] = cancel_c(r, cases[m, 0], cases[m, 1])
    a, b, c = (u2f(cases[:, i]) for i in range(3))
    return [cases[:, 0], cases[:, 1], cases[:, 2], f2u(H.fma(a, b, c))]


def add_cases(kind, n, r):
    if kind == "directed":
        sp = np.asarray(SPECIAL, U32)
        g = np.array(np.meshgrid(sp, sp, indexing="ij")).reshape(2, -1).T
        k = np.arange(1, 4096, dtype=U32)
        parts = [g,
                 np.stack([np.full_like(k, 0x3F800000) + k, np.full_like(k, 0xBF800000) + k - 1], 1),
                 np.stack([np.full_like(k, 0x00800000) + k, np.full_like(k, 0x80800000) + 2 * k], 1),
                 np.stack([np.full_like(k, 0x01000000) + k, np.full_like(k, 0x80FFFFFF) - k], 1),
                 np.stack([np.full_like(k, 0x3F800000), U32(0x33000000) + k], 1),      # 1 + tiny (ties)
                 np.stack([np.full_like(k, 0x7F7FFFFF), U32(0x73000000) + k], 1)]      # overflow by rounding
        d = np.arange(0, 60)                              # every exponent difference 0..59
        m = 200 * len(d)
        eb = np.repeat(140 - d, 200)
        parts.append(np.stack([bits(r.integers(0, 2, m), np.full(m, 140), r.integers(0, 1 << 23, m)),
                               bits(r.integers(0, 2, m), eb, r.integers(0, 1 << 23, m))], 1))
        ab = np.concatenate(parts).astype(U32)
        sub = np.zeros(len(ab), U32)
        ab = np.concatenate([ab, ab])
        sub = np.concatenate([sub, np.ones_like(sub)])
    else:
        ops = mix_operands(r, n)
        ab = ops[:, :2].copy()
        m = r.integers(0, 4, n) == 0                    # 1/4: b ~ -a (cancellation)
        ab[m, 1] = (ab[m, 0].astype(np.int64) ^ 0x80000000) + r.integers(-3, 4, int(m.sum()))
        ab = ab.astype(U32)
        sub = r.integers(0, 2, n).astype(U32)
        flip = sub == 1                                  # keep cancellation meaningful for sub
        ab[m & flip, 1] ^= U32(0x80000000)
    a, b = u2f(ab[:, 0]), u2f(ab[:, 1])
    y = np.where(sub == 1, f2u(H.sub(a, b)), f2u(H.add(a, b))).astype(U32)
    return [ab[:, 0], ab[:, 1], sub, y], [8, 8, 1, 8]


def mul_cases(kind, n, r):
    if kind == "directed":
        sp = np.asarray(SPECIAL, U32)
        g = np.array(np.meshgrid(sp, sp, indexing="ij")).reshape(2, -1).T
        k = np.arange(1, 4096, dtype=U32)
        parts = [g,
                 np.stack([np.full_like(k, 0x00800000), U32(0x3F800000) - k], 1),
                 np.stack([U32(0x00800000) + k, np.full_like(k, 0x3F7FFFFF)], 1),
                 np.stack([U32(0x1F800000) + k, U32(0x1F800000) - k], 1),
                 np.stack([U32(0x5F800000) + k, U32(0x5F7FFFFF) - k], 1)]
        e = np.arange(1, 255)
        m = len(e) * 40
        parts.append(np.stack([bits(r.integers(0, 2, m), np.repeat(e, 40), r.integers(0, 1 << 23, m)),
                               bits(r.integers(0, 2, m), 254 - np.repeat(e, 40) + r.integers(-2, 3, m),
                                    r.integers(0, 1 << 23, m))], 1))
        ab = np.concatenate(parts).astype(U32)
    else:
        ab = mix_operands(r, n)[:, :2].copy()
    a, b = u2f(ab[:, 0]), u2f(ab[:, 1])
    return [ab[:, 0], ab[:, 1], f2u(H.mul(a, b))]


def cmp_cases(kind, n, r):
    if kind == "directed":
        sp = np.asarray(SPECIAL, U32)
        ab = np.array(np.meshgrid(sp, sp, indexing="ij")).reshape(2, -1).T
        k = np.arange(0, 2000, dtype=U32)
        ab = np.concatenate([ab, np.stack([U32(0x3F800000) + k, U32(0x3F800000) + k // 2], 1),
                             np.stack([U32(0xBF800000) + k, U32(0xBF800000) + k // 2], 1)])
    else:
        ab = mix_operands(r, n)[:, :2].copy()
        m = r.integers(0, 4, n) == 0
        ab[m, 1] = ab[m, 0] + r.integers(-1, 2, int(m.sum())).astype(U32)
    a, b = u2f(ab[:, 0]), u2f(ab[:, 1])
    return [ab[:, 0], ab[:, 1], H.cmp(a, b).astype(U32), f2u(H.fmax(a, b)), f2u(H.fmin(a, b))], \
        [8, 8, 1, 8, 8]


def i2f_cases(kind, n, r):
    if kind == "directed":
        k = np.arange(0, 32)
        p = (np.uint64(1) << k.astype(np.uint64)).astype(np.int64)
        v = np.concatenate([p, p - 1, p + 1, -p, -p + 1, -p - 1, [0, 1, -1, 2**31 - 1, -2**31],
                            (1 << 24) + np.arange(-8, 9), -(1 << 25) + np.arange(-16, 17),
                            np.arange(-300, 301), (1 << 30) + np.arange(-64, 65) * 32])
        x = (v & 0xFFFFFFFF).astype(np.uint64).astype(U32)
    else:
        x = r.integers(0, 1 << 32, n, dtype=np.uint64).astype(U32)
        sh = r.integers(0, 32, n).astype(U32)
        x = np.where(r.integers(0, 2, n) == 0, x, x >> sh).astype(U32)
    return [x, f2u(H.i2f(x.view(np.int32)))]


# ------------------------------------------------------------------------------- SFU
THRESH = {  # inputs at and around every branch / range threshold of arith.md section 3
    "exp": [89.0, -104.0, 88.72283935546875, -87.33654022216797, -103.97, 0.0, 1.0, -1.0,
            0.34657359, -0.34657359, 1e-8, -1e-8, 64.0, -75.0, -71.04119],
    "recip": [1.0, 2.0, 1.0023887, 1.0023887 * 2 ** 40, 1.0023887 * 2 ** -40, 2.0 ** 126, 2.0 ** -126,
              2.0 ** 127, 1.7014118e38, 3.4028235e38, 1.5],
    "rsqrt": [1.0, 2.0, 4.0, 1.0490096e-22, 2.0 ** -126, 2.0 ** -125, 3.4028235e38, 0.5, 3.0],
    "sqrt": [1.0, 2.0, 4.0, 2.0 ** -126, 2.0 ** -125, 3.4028235e38, 0.5, 3.0, 1e-30, 1e30],
    "tanh": [9.0, 1.0, -0.9122643, 0.9122643, 8.999999, 0.999999, 1e-20, 0.5, 2.0, 4.5],
    "erf": [3.92, 0.921875, -1.0600996, 1.0600996, 0.9218749, 3.9199998, 1e-20, 0.5, 2.0, 3.0],
    "log": [1.571041, 1.0, 2.0, 0.5, 1.4142135, 1.4142137, 0.70710677, 2.0 ** -126, 3.4028235e38,
            0.9999999, 1.0000001],
}


def sfu_directed(fn, r):
    xs = [np.asarray(SPECIAL, U32)]
    t = f2u(np.float32(THRESH[fn]))
    for d in range(-4, 5):                              # +-4 ulps around each threshold, both signs
        xs += [(t.astype(np.int64) + d).astype(U32), (t.astype(np.int64) + d).astype(U32) ^ U32(0x80000000)]
    e = np.arange(0, 256, dtype=np.uint64)              # every binade boundary, both signs
    for f in (0, 1, 0x7FFFFF, 0x7FFFFE, 0x400000, 0x3504F3, 0x3504F4):
        xs += [bits(0, e, f), bits(1, e, f)]
    # the worst point of every binade (exhaustive L1 ulp scan, positive and negative)
    worst = []
    for s in (0, 1):
        for ex in range(1, 255):
            lo, hi = int(bits(s, ex, 0)), int(bits(s, ex, 0x7FFFFF))
            w = H.ulp_scan(fn, lo, hi)
            if w["n"] > 0:
                worst.append(f2u(np.float32(w["arg"])))
    xs.append(np.asarray(worst, U32))
    return np.concatenate(xs).astype(U32)


def sfu_random(fn, n, r):
    k = r.integers(0, 4, n)
    any_bits = r.integers(0, 1 << 32, n, dtype=np.uint64).astype(U32)
    lo, hi = {"exp": (-110, 95), "recip": (-1e3, 1e3), "rsqrt": (0, 1e3), "sqrt": (0, 1e3),
              "tanh": (-10, 10), "erf": (-4.5, 4.5), "log": (0, 1e3)}[fn]
    uni = f2u(np.float32(r.uniform(lo, hi, n)))
    logu = rand_normal(r, n, 1, 254)
    if fn in ("rsqrt", "sqrt", "log"):
        logu &= U32(0x7FFFFFFF)
    near = f2u(np.float32(r.uniform(-1.2, 1.2, n) * (4.0 if fn in ("tanh", "erf") else 2.0)))
    return np.select([k == 0, k == 1, k == 2], [any_bits, uni, logu], near).astype(U32)


def sfu_cases(kind, n, r, fns):
    xs, fs = [], []
    for fn in fns:
        x = sfu_directed(fn, r) if kind == "directed" else sfu_random(fn, n, r)
        xs.append(x)
        fs.append(np.full(len(x), SFU_FN[fn], U32))
    x, f = np.concatenate(xs), np.concatenate(fs)
    if kind == "mixed":                                  # interleave functions (thread isolation)
        p = r.permutation(len(x))
        x, f = x[p], f[p]
    y = np.empty_like(x)
    for fn in FNS:
        m = f == SFU_FN[fn]
        if m.any():
            y[m] = f2u(getattr(H, fn)(u2f(x[m])))
    return [f, x, y], [1, 8, 8]


def main():
    unit, kind, n, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    r = np.random.default_rng(seed)
    if unit == "fma":
        cols, w = fma_cases(kind, n, r), None
    elif unit == "add":
        cols, w = add_cases(kind, n, r)
    elif unit == "mul":
        cols, w = mul_cases(kind, n, r), None
    elif unit == "cmp":
        cols, w = cmp_cases(kind, n, r)
    elif unit == "i2f":
        cols, w = i2f_cases(kind, n, r), None
    elif unit.startswith("sfu"):                         # sfu (all fns) or sfu:exp,tanh
        fns = unit.split(":")[1].split(",") if ":" in unit else FNS
        k = kind if kind != "mixed" else "random"
        cols, w = sfu_cases(k, n, r, fns)
        if kind == "mixed":
            p = r.permutation(len(cols[0]))
            cols = [c[p] for c in cols]
    else:
        raise SystemExit(f"unknown unit {unit}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write_hex(out, cols, w)
    print(f"{out}: {len(cols[0])} vectors")


if __name__ == "__main__":
    main()
