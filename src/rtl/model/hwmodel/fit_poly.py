"""Derives the fp32 polynomial constants of docs/arith.md (run once; the hex goes into csrc/lwta.c).

Weighted minimax by Lawson iterations on a dense grid (targets in mpmath, 40 digits), then the
coefficients are rounded to fp32 from the highest degree down, re-fitting the lower ones after each
rounding so they absorb its error.  The ulp numbers that matter are measured on the C code
(test_arith.py), not here.
"""
import mpmath as mp
import numpy as np

mp.mp.dps = 40


def lawson(Phi, f, w, iters=300):
    u = np.ones(len(f)) / len(f)
    for _ in range(iters):
        sw = np.sqrt(u) * w
        c, *_ = np.linalg.lstsq(Phi * sw[:, None], f * sw, rcond=None)
        e = np.abs((Phi @ c - f) * w)
        u = u * e
        u /= u.sum()
    return c, np.max(np.abs((Phi @ c - f) * w))


def fit(xs, f, w, deg, lo=0):
    """f ~ sum_{k=lo..deg} c_k x^k, coefficients rounded to fp32 high to low with re-fit."""
    ks = list(range(lo, deg + 1))
    fixed = {}
    for top in reversed(ks):
        free = [k for k in ks if k not in fixed]
        Phi = np.stack([xs ** k for k in free], 1)
        rhs = f - sum(c * xs ** k for k, c in fixed.items())
        c, _ = lawson(Phi, rhs, w)
        fixed[top] = float(np.float32(c[free.index(top)]))
    cs = [fixed[k] for k in ks]
    err = np.max(np.abs((sum(c * xs ** k for k, c in zip(ks, cs)) - f) * w))
    return cs, err


def hexf(v):
    return "0x%08X" % np.float32(v).view(np.uint32)


def grid(a, b, n=6000):
    t = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    return np.sort(0.5 * (a + b) + 0.5 * (b - a) * t)


def show(name, cs, err, lo=0):
    print(f"{name}: max weighted err {err:.3e}")
    for k, c in enumerate(cs):
        print(f"   c{k + lo} = {hexf(c)}  /* {np.float32(c):.9e} */")


if __name__ == "__main__":
    # exp: e^r = 1 + r * G(r), |r| <= ln2/2
    r = grid(-0.3466, 0.3466)
    G = np.array([float((mp.exp(x) - 1) / x) if x != 0 else 1.0 for x in r])
    w = np.abs(r) / np.exp(r)
    cs, e = fit(r, G, w, 6)
    show("exp G (Horner in r, then fma(G, r, 1))", cs, e)
    # tanh small: tanh(x) = x + x z P(z), z = x^2, |x| < 1
    T = 1.0
    x = grid(1e-4, T)
    z = x * x
    P = np.array([float((mp.tanh(v) - v) / v ** 3) for v in x])
    w = x ** 3 / np.tanh(x)
    cs, e = fit(z, P, w, 8)
    show("tanh P(z)", cs, e)
    # erf small: erf(x) = x + x Q(z), |x| < 0.921875
    x = grid(1e-4, 0.921875)
    z = x * x
    Q = np.array([float(mp.erf(v) / v - 1) for v in x])
    w = x / np.array([float(mp.erf(v)) for v in x])
    cs, e = fit(z, Q, w, 6)
    show("erf small Q(z)", cs, e)
    # erf large: erf(a) = 1 - exp(-(a + a R(a))), 0.921875 <= a < 3.92
    a = grid(0.921875, 3.92)
    R = np.array([float((-mp.log(mp.erfc(v)) - v) / v) for v in a])
    w = np.array([float(mp.erfc(v) * v) for v in a]) / 0.84
    cs, e = fit(a, R, w, 9)
    show("erf large R(a)", cs, e)
    # log: log1p(f) = f + f^2 Q(f), f in [sqrt(.5)-1, sqrt2-1]
    f = grid(np.sqrt(0.5) - 1, np.sqrt(2) - 1)
    f = f[np.abs(f) > 1e-6]
    Q = np.array([float((mp.log1p(v) - v) / v ** 2) for v in f])
    w = f ** 2 / np.abs(np.log1p(f))
    cs, e = fit(f, Q, w, 8)
    show("log Q(f)", cs, e)
