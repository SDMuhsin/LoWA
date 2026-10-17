"""Proof of the L1 arithmetic (docs/arith.md section 1, 2).

1. add / mul / fma of csrc/lwta.c vs an EXACT integer reference (this file, no floats): inputs
   decoded to integer significand x 2^exp, the exact result formed in integers, rounded once to
   24 bits RNE with unbounded exponent, then FTZ (|r| < 2^-126 -> +-0, sign of the exact value),
   overflow -> inf, NaN canonical 0x7FC00000.  Random bit patterns + directed sets (cancellation,
   ties, results around 2^-126, subnormal inputs, specials).  Bitwise equality required.
2. SFU: exhaustive ulp scan over every fp32 input in the stated domain (C, vs double libm), the
   worst point re-checked in mpmath at 50 digits, and the special-value table.
   python3 test_arith.py [--quick]
"""
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hwmodel as H  # noqa: E402

QNAN = 0x7FC00000
RNG = np.random.default_rng(20261014)


def dec(u):
    """-> ('nan'|'inf'|'num', sign, M, E) with value (-1)^s M 2^E; subnormals read as zero."""
    s, e, f = u >> 31, (u >> 23) & 0xFF, u & 0x7FFFFF
    if e == 0xFF:
        return ("nan" if f else "inf", s, 0, 0)
    if e == 0:
        return ("num", s, 0, 0)
    return ("num", s, f | 0x800000, e - 150)


def pack(s, M, E):
    """round (-1)^s M 2^E (M > 0) to fp32 RNE, unbounded exponent, then FTZ / overflow."""
    sh = M.bit_length() - 24
    if sh > 0:
        q, rem = M >> sh, M & ((1 << sh) - 1)
        half = 1 << (sh - 1)
        if rem > half or (rem == half and q & 1):
            q += 1
        if q == 1 << 24:
            q >>= 1
            sh += 1
    else:
        q = M << -sh
    e = E + sh + 23                      # unbiased exponent of q * 2^(E + sh)
    if e > 127:
        return (s << 31) | 0x7F800000
    if e < -126:
        return s << 31
    return (s << 31) | ((e + 127) << 23) | (q - (1 << 23))


def ref_fma(ua, ub, uc, kind):
    A, Bv, C = dec(ua), dec(ub), dec(uc)
    if kind == "add":                    # a + c  (b ignored)
        Bv = ("num", 0, 1, 0)
    if kind == "mul":                    # a * b + (-0): exact product, -0 keeps x*y = +-0 signs
        C = ("num", 1, 0, 0)
    if "nan" in (A[0], Bv[0], C[0]):
        return QNAN
    ps = A[1] ^ Bv[1]
    if A[0] == "inf" or Bv[0] == "inf":
        if (A[0] == "num" and A[2] == 0) or (Bv[0] == "num" and Bv[2] == 0):
            return QNAN
        if C[0] == "inf" and C[1] != ps:
            return QNAN
        return (ps << 31) | 0x7F800000
    if C[0] == "inf":
        return (C[1] << 31) | 0x7F800000
    PM, PE = A[2] * Bv[2], A[3] + Bv[3]
    if PM == 0 and C[2] == 0:
        return ((ps & C[1]) << 31)       # +-0 + +-0: -0 only if both negative
    if PM == 0:
        return pack(C[1], C[2], C[3])
    if C[2] == 0:
        return pack(ps, PM, PE)
    E = min(PE, C[3])
    S = (-1) ** ps * PM * 2 ** (PE - E) + (-1) ** C[1] * C[2] * 2 ** (C[3] - E)
    if S == 0:
        return 0
    return pack(1 if S < 0 else 0, abs(S), E)


def f2u(x):
    return np.asarray(x, np.float32).view(np.uint32)


def u2f(u):
    return np.asarray(u, np.uint32).view(np.float32)


def bits_of(sign, exp, frac):
    return (np.uint32(sign) << 31) | (np.uint32(exp) << 23) | np.uint32(frac)


def gen_cases(n):
    """(a, b, c) uint32 triples: random patterns plus directed families."""
    r = RNG
    out = []
    out.append(r.integers(0, 2 ** 32, (n, 3), dtype=np.uint64).astype(np.uint32))       # any bits
    e = r.integers(1, 255, (n, 3))
    out.append(bits_of(r.integers(0, 2, (n, 3)), e, r.integers(0, 2 ** 23, (n, 3))))   # normals
    # cancellation: c = -(a*b) nudged by a few ulps
    a = u2f(bits_of(r.integers(0, 2, n), r.integers(64, 190, n), r.integers(0, 2 ** 23, n)))
    b = u2f(bits_of(r.integers(0, 2, n), r.integers(64, 190, n), r.integers(0, 2 ** 23, n)))
    with np.errstate(all="ignore"):
        c = -(a * b)
    c = u2f(f2u(c) + r.integers(-3, 4, n).astype(np.uint32))
    out.append(np.stack([f2u(a), f2u(b), f2u(c)], 1))
    # ties: few-bit significands so products land exactly on half-ulp boundaries
    fa = (r.integers(0, 2 ** 12, n) << 11).astype(np.uint32)
    fb = (r.integers(0, 2 ** 12, n) << 11).astype(np.uint32)
    ea, eb = r.integers(100, 150, n), r.integers(100, 150, n)
    ec = ea + eb - 127 + r.integers(-26, 3, n)
    t = np.stack([bits_of(r.integers(0, 2, n), ea, fa), bits_of(r.integers(0, 2, n), eb, fb),
                  bits_of(r.integers(0, 2, n), np.clip(ec, 1, 254), r.integers(0, 2 ** 23, n) & ~0xFFF)], 1)
    out.append(t.astype(np.uint32))
    # results around 2^-126: product exponent near -126, c tiny or cancelling
    ea = r.integers(1, 126, n)
    eb = np.clip(127 - 126 + 127 - ea + r.integers(-3, 3, n) - 1, 1, 254)
    t = np.stack([bits_of(r.integers(0, 2, n), ea, r.integers(0, 2 ** 23, n)),
                  bits_of(r.integers(0, 2, n), eb, r.integers(0, 2 ** 23, n)),
                  bits_of(r.integers(0, 2, n), r.integers(0, 4, n), r.integers(0, 2 ** 23, n))], 1)
    out.append(t.astype(np.uint32))
    # exactly representable tiny results and the 2^-126 rounding boundary
    sp = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0x00000001, 0x807FFFFF,
          0x00800000, 0x80800000, 0x3F800000, 0x7F7FFFFF, 0x0C800000, 0x33800000, 0x00FFFFFF]
    g = np.array(np.meshgrid(sp, sp, sp)).reshape(3, -1).T.astype(np.uint32)
    out.append(g)
    k = np.arange(1, 4096, dtype=np.uint32)
    out.append(np.stack([np.full_like(k, 0x00800000), np.full_like(k, 0x3F800000) - k,
                         np.zeros_like(k)], 1))                 # 2^-126 * (1 - k ulp) -> flush
    out.append(np.stack([np.full_like(k, 0x00800000) + k, np.full_like(k, 0x3F7FFFFF),
                         np.zeros_like(k)], 1))                 # just above / at the boundary
    return np.concatenate(out).astype(np.uint32)


def check_basic(n):
    cases = gen_cases(n)
    a, b, c = (u2f(cases[:, i]) for i in range(3))
    got = {"fma": f2u(H.fma(a, b, c)), "add": f2u(H.add(a, c)), "mul": f2u(H.mul(a, b))}
    res = {}
    for kind, g in got.items():
        bad = 0
        for i in range(len(cases)):
            want = ref_fma(int(cases[i, 0]), int(cases[i, 1]), int(cases[i, 2]), kind)
            if int(g[i]) != want:
                if bad < 5:
                    print(f"  MISMATCH {kind} {[hex(int(v)) for v in cases[i]]} got {hex(int(g[i]))} want {hex(want)}")
                bad += 1
        res[kind] = (len(cases), bad)
    return res


# SFU domains: (name, [(lo_bits, hi_bits), ...]) -- every fp32 input in the ranges
DOMAINS = {
    "exp": [(0x00000000, 0x42B17217), (0x80000000, 0xC2AEAC4F)],
    "recip": [(0x00800000, 0x7E800000), (0x80800000, 0xFE800000)],
    "rsqrt": [(0x00800000, 0x7F7FFFFF)],
    "sqrt": [(0x00800000, 0x7F7FFFFF)],
    "tanh": [(0x00000000, 0x7F7FFFFF), (0x80000000, 0xFF7FFFFF)],
    "erf": [(0x00000000, 0x7F7FFFFF), (0x80000000, 0xFF7FFFFF)],
    "log": [(0x00800000, 0x7F7FFFFF)],
}
MPF = {"exp": mp.exp, "recip": lambda x: 1 / x, "rsqrt": lambda x: 1 / mp.sqrt(x), "sqrt": mp.sqrt,
       "tanh": mp.tanh, "erf": mp.erf, "log": mp.log}
SPECIALS = [0.0, -0.0, np.inf, -np.inf, np.nan, 1e-40, -1e-40, 1.0, -1.0, 88.72, -87.4, 100.0, -200.0]


def ulp_mp(fn, x, y):
    mp.mp.dps = 50
    r = MPF[fn](mp.mpf(float(x)))
    e = mp.floor(mp.log(abs(r), 2))
    return float(abs(mp.mpf(float(y)) - r) / mp.mpf(2) ** (e - 23))


def check_sfu():
    rows = {}
    for fn, rngs in DOMAINS.items():
        tot = dict(max_ulp=0.0, n=0, n_ge1ulp=0, n_not_cr=0, arg=None)
        for lo, hi in rngs:
            r = H.ulp_scan(fn, lo, hi)
            tot["n"] += r["n"]
            tot["n_ge1ulp"] += r["n_ge1ulp"]
            tot["n_not_cr"] += r["n_not_cr"]
            if r["max_ulp"] >= tot["max_ulp"]:
                tot["max_ulp"], tot["arg"] = r["max_ulp"], r["arg"]
        y = getattr(H, fn)(np.array([tot["arg"]], np.float32))[0]
        tot["mp_ulp_at_arg"] = ulp_mp(fn, tot["arg"], y)
        sp = getattr(H, fn)(np.array(SPECIALS, np.float32))
        tot["specials"] = " ".join(f"{v:g}" for v in sp)
        rows[fn] = tot
    return rows


if __name__ == "__main__":
    n = 2000 if "--quick" in sys.argv else 60000
    ok = True
    print("== add / mul / fma vs exact integer reference ==")
    for kind, (tot, bad) in check_basic(n).items():
        print(f"  {kind:4s}: {tot} cases, {bad} mismatches")
        ok &= bad == 0
    print("== SFU exhaustive ulp scans ==")
    print(f"  {'fn':6s} {'inputs':>11s} {'max ulp':>8s} {'(mpmath)':>9s} {'at x':>14s} {'>=1ulp':>8s} {'not CR %':>9s}")
    sfu = check_sfu()
    for fn, r in sfu.items():
        print(f"  {fn:6s} {r['n']:11d} {r['max_ulp']:8.4f} {r['mp_ulp_at_arg']:9.4f} {r['arg']:14.8g} "
              f"{r['n_ge1ulp']:8d} {100 * r['n_not_cr'] / r['n']:9.4f}")
        ok &= abs(r["max_ulp"] - r["mp_ulp_at_arg"]) < 1e-3
    print("  specials (inputs: " + " ".join(f"{v:g}" for v in SPECIALS) + ")")
    for fn, r in sfu.items():
        print(f"  {fn:6s} {r['specials']}")
    print("ARITH", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
