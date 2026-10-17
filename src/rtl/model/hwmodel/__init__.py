"""L1 hwmodel: the chip's arithmetic (csrc/lwta.c) behind numpy-level entry points.

Every function takes / returns float32 numpy arrays and performs exactly the rounded ops the
hardware performs (docs/arith.md, docs/dataflow.md).  These are also the golden-vector entry
points for the P2-P5 unit testbenches.
"""
import ctypes as ct
import os
import subprocess

import numpy as np

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_SO = os.path.join(_DIR, "liblwta.so")
if (not os.path.exists(_SO)
        or os.path.getmtime(_SO) < os.path.getmtime(os.path.join(_DIR, "lwta.c"))):
    subprocess.run(["make", "-s", "-C", _DIR], check=True)
_L = ct.CDLL(_SO)

f32 = np.float32
_P = ct.c_void_p
_l, _i, _f = ct.c_long, ct.c_int, ct.c_float


def _p(a):
    return a.ctypes.data_as(_P) if a is not None else None


def _c(a, dt=np.float32):
    return np.ascontiguousarray(a, dtype=dt)


def hexf(x):
    return int(np.float32(x).view(np.uint32))


def fbits(u):
    return np.uint32(u).view(np.float32)


# ------------------------------------------------------------------------------ element-wise
def _unary(name):
    fn = getattr(_L, name)
    fn.argtypes = [_l, _P, _P]

    def op(x):
        x = _c(np.atleast_1d(x))
        y = np.empty_like(x)
        fn(x.size, _p(x), _p(y))
        return y
    op.__name__ = name[3:]
    return op


def _binary(name):
    fn = getattr(_L, name)
    fn.argtypes = [_l, _P, _P, _P]

    def op(a, b):
        a, b = np.broadcast_arrays(_c(np.atleast_1d(a)), _c(np.atleast_1d(b)))
        a, b = _c(a), _c(b)
        y = np.empty_like(a)
        fn(a.size, _p(a), _p(b), _p(y))
        return y
    op.__name__ = name[3:]
    return op


exp, recip, rsqrt, sqrt, tanh, erf, log, gelu = (
    _unary("lw_" + n) for n in ("exp", "recip", "rsqrt", "sqrt", "tanh", "erf", "log", "gelu"))
add, sub, mul, fmax, fmin = (_binary("lw_" + n) for n in ("add", "sub", "mul", "max", "min"))
gelu_bwd = _binary("lw_gelu_bwd")      # (dg, u) -> du
tanh_bwd = _binary("lw_tanh_bwd")      # (d_out, out) -> d_in

_L.lw_fma.argtypes = [_l, _P, _P, _P, _P]
_L.lw_cmp.argtypes = [_l, _P, _P, _P]
_L.lw_i2f.argtypes = [_l, _P, _P]


def cmp(a, b):
    """compare flags on DAZ'd operands: bit0 a<b, bit1 a==b, bit2 a>b, bit3 unordered (uint8)."""
    a, b = np.broadcast_arrays(_c(np.atleast_1d(a)), _c(np.atleast_1d(b)))
    a, b = _c(a), _c(b)
    y = np.empty(a.shape, np.uint8)
    _L.lw_cmp(a.size, _p(a), _p(b), _p(y))
    return y


def i2f(x):
    """signed int32 -> fp32, RNE."""
    x = _c(np.atleast_1d(x), np.int32)
    y = np.empty(x.shape, np.float32)
    _L.lw_i2f(x.size, _p(x), _p(y))
    return y


def fma(a, b, c):
    a, b, c = (_c(t) for t in np.broadcast_arrays(*(_c(np.atleast_1d(t)) for t in (a, b, c))))
    y = np.empty_like(a)
    _L.lw_fma(a.size, _p(a), _p(b), _p(c), _p(y))
    return y


_L.lw_scale.argtypes = [_l, _P, _f, _P]
_L.lw_drop.argtypes = [_l, _P, _P, _f, _P]
_L.lw_add_bias.argtypes = [_l, _l, _P, _P, _P]
_L.lw_colsum.argtypes = [_l, _l, _P, _P]


def scale(x, s):
    x = _c(x)
    y = np.empty_like(x)
    _L.lw_scale(x.size, _p(x), f32(s), _p(y))
    return y


def drop(x, keep, ds):
    """dropout forward and backward (the same op): keep ? x * ds : +0."""
    x, keep = _c(x), _c(keep, np.uint8)
    assert keep.shape == x.shape
    y = np.empty_like(x)
    _L.lw_drop(x.size, _p(x), _p(keep), f32(ds), _p(y))
    return y


def add_bias(x, bias):
    x, bias = _c(x), _c(bias)
    y = np.empty_like(x)
    _L.lw_add_bias(x.size // bias.size, bias.size, _p(x), _p(bias), _p(y))
    return y


def colsum(x):
    x = _c(x)
    out = np.empty(x.shape[-1], f32)
    _L.lw_colsum(x.size // x.shape[-1], x.shape[-1], _p(x), _p(out))
    return out


# ------------------------------------------------------------------------------------- GEMM
_L.lw_gemm.argtypes = [_i] * 5 + [_P] + [_l] * 4 + [_P] + [_l] * 4 + [_P] + [_l] * 4


_L.lw_set_kc.argtypes = [_i]


def set_kc(kc):
    """GEMM chunk length (contract: 64).  0 = one FMA chain over all of K (P0's D4, for comparison)."""
    _L.lw_set_kc(kc)


def gemm_raw(nb1, nb2, M, N, K, A, a_str, Bm, b_str, C, c_str, a_off=0, b_off=0, c_off=0):
    """C(i,j) = sum_k A(i,k) B(j,k) by the contract (acc = +0, k ascending, one FMA each).
    *_str = (batch1, batch2, row, col) element strides; *_off element offsets into the arrays."""
    for arr in (A, Bm, C):
        assert arr.dtype == np.float32 and arr.flags.c_contiguous
    pa = ct.c_void_p(A.ctypes.data + 4 * a_off)
    pb = ct.c_void_p(Bm.ctypes.data + 4 * b_off)
    pc = ct.c_void_p(C.ctypes.data + 4 * c_off)
    _L.lw_gemm(nb1, nb2, M, N, K, pa, *a_str, pb, *b_str, pc, *c_str)


def gemm(A, Bm, bias=None):
    """C = A @ Bm.T for 2-D A [M x K], Bm [N x K] (torch weight layout), + bias (add after)."""
    A, Bm = _c(A), _c(Bm)
    M, K = A.shape
    N = Bm.shape[0]
    assert Bm.shape[1] == K
    C = np.empty((M, N), f32)
    gemm_raw(1, 1, M, N, K, A, (0, 0, K, 1), Bm, (0, 0, K, 1), C, (0, 0, N, 1))
    return add_bias(C, bias) if bias is not None else C


def gemm_nn(A, Bkn):
    """C = A @ Bkn for A [M x K], Bkn [K x N] (the 'TRN' B side)."""
    A, Bkn = _c(A), _c(Bkn)
    M, K = A.shape
    N = Bkn.shape[1]
    C = np.empty((M, N), f32)
    gemm_raw(1, 1, M, N, K, A, (0, 0, K, 1), Bkn, (0, 0, 1, N), C, (0, 0, N, 1))
    return C


def gemm_tn(Akm, Bkn):
    """C = Akm.T @ Bkn for Akm [K x M], Bkn [K x N] (both 'TRN', e.g. head dW = dZ^T X)."""
    Akm, Bkn = _c(Akm), _c(Bkn)
    K, M = Akm.shape
    N = Bkn.shape[1]
    C = np.empty((M, N), f32)
    gemm_raw(1, 1, M, N, K, Akm, (0, 0, 1, M), Bkn, (0, 0, 1, N), C, (0, 0, N, 1))
    return C


# ----------------------------------------------------------------------------- vector unit
_L.lw_ln_fwd.argtypes = [_l, _i, _P, _P, _P, _f, _f, _P, _P, _P]
_L.lw_ln_bwd.argtypes = [_l, _i, _P, _P, _P, _P, _P, _f, _P]
LN_EPS = f32(1e-5)


def ln_fwd(x, g, b, eps=LN_EPS):
    x = _c(x)
    d = x.shape[-1]
    rows = x.size // d
    y = np.empty_like(x)
    mu, rs = np.empty(rows, f32), np.empty(rows, f32)
    _L.lw_ln_fwd(rows, d, _p(x), _p(_c(g)), _p(_c(b)), f32(eps), f32(1.0 / d), _p(y), _p(mu), _p(rs))
    return y, mu, rs


def ln_bwd(z, mu, rstd, g, dy):
    z, dy = _c(z), _c(dy)
    d = z.shape[-1]
    dx = np.empty_like(z)
    _L.lw_ln_bwd(z.size // d, d, _p(z), _p(_c(mu)), _p(_c(rstd)), _p(_c(g)), _p(dy), f32(1.0 / d), _p(dx))
    return dx


_L.lw_smax_fwd.argtypes = [_i, _i, _i, _P, _P, _P, _f, _P, _P, _P, _P]
_L.lw_smax_bwd.argtypes = [_l, _i, _P, _P, _P, _f, _f, _P]


def smax_fwd(S, kmask, keep, ds):
    """S [B, H, L, L], kmask [B, L] (1 = real key), keep [B, H, L, L] or None -> P, P', m, r."""
    S = _c(S)
    Bn, H, L, _ = S.shape
    P, Pd = np.empty_like(S), np.empty_like(S)
    m, r = np.empty(S.shape[:3], f32), np.empty(S.shape[:3], f32)
    keep = None if keep is None else _c(keep, np.uint8)
    _L.lw_smax_fwd(Bn, H, L, _p(S), _p(_c(kmask, np.uint8)), _p(keep), f32(ds), _p(P), _p(Pd), _p(m), _p(r))
    return P, Pd, m, r


def smax_bwd(P, dPd, keep, ds, scale_=0.125):
    P, dPd = _c(P), _c(dPd)
    L = P.shape[-1]
    dS = np.empty_like(P)
    keep = None if keep is None else _c(keep, np.uint8)
    _L.lw_smax_bwd(P.size // L, L, _p(P), _p(dPd), _p(keep), f32(ds), f32(scale_), _p(dS))
    return dS


# ------------------------------------------------------------------------------------- LoWA
_L.lw_lowa_merge.argtypes = [_l, _i, _P, _P, _P, _f, _P]
_L.lw_lowa_sg.argtypes = [_l, _i, _i, _i, _i, _P, _P, _P, _P, _P, _f, _P]


def lowa_merge(W0, eflat, coef, h, gamma):
    W = _c(W0).copy()
    eflat = _c(eflat, np.int64)
    _L.lw_lowa_merge(eflat.size, h.size, _p(eflat), _p(_c(coef)), _p(_c(h)), f32(gamma), _p(W))
    return W


def lowa_sg(dY, X, rows, cols, h, gamma):
    """rows/cols: [k, s] cell coordinates of each atom's taps (tap order = torch.nonzero(H))."""
    dY, X = _c(dY), _c(X)
    rows, cols = _c(rows, np.int64), _c(cols, np.int64)
    k, s = rows.shape
    g = np.empty(k, f32)
    _L.lw_lowa_sg(dY.shape[0], dY.shape[1], X.shape[1], k, s, _p(dY), _p(X), _p(rows), _p(cols),
                  _p(_c(h)), f32(gamma), _p(g))
    return g


# ------------------------------------------------------------------------------- loss, optim
_L.lw_ce.argtypes = [_i, _i, _P, _P, _P, _f, _P, _P]


def ce(z, y, w, inv_b):
    z = _c(z)
    Bn, Cn = z.shape
    dz = np.empty_like(z)
    loss = np.zeros(1, f32)
    _L.lw_ce(Bn, Cn, _p(z), _p(_c(y, np.int64)), _p(_c(w)), f32(inv_b), _p(loss), _p(dz))
    return loss[0], dz


_L.lw_mse.argtypes = [_i, _P, _P, _P, _f, _f, _P, _P]


def mse(z, y, w, inv_b):
    """MSE loss (C = 1): loss = mul(sum_b add-chain of mul(mul(d, d), w_b), inv_b), d = sub(z, y);
    dz = mul(mul(d, inv2b), w_b) with inv2b = mul(2, inv_b) (exact).  P4 entry point."""
    z, y = _c(z).reshape(-1), _c(y).reshape(-1)
    dz = np.empty_like(z)
    loss = np.zeros(1, f32)
    inv2b = mul(f32(2.0), f32(inv_b))[0]
    _L.lw_mse(z.size, _p(z), _p(y), _p(_c(w)), f32(inv_b), f32(inv2b), _p(loss), _p(dz))
    return loss[0], dz.reshape(-1, 1)


_L.lw_gnorm_clip.argtypes = [_i, _P, _P, _P, _f, _P, _P, _P]


def gnorm_clip(gflat, offsets, rowlen, eps=1e-6):
    """in place on gflat (tensors at offsets, each made of rows of rowlen[i]);
    returns (per-tensor norms, total norm G, clip coefficient c)."""
    off = _c(offsets, np.int64)
    nt = off.size - 1
    norms = np.empty(nt, f32)
    G, c = np.zeros(1, f32), np.zeros(1, f32)
    _L.lw_gnorm_clip(nt, _p(off), _p(_c(rowlen, np.int64)), _p(gflat), f32(eps), _p(norms), _p(G), _p(c))
    return norms, G[0], c[0]


_L.lw_adamw.argtypes = [_l, _P, _P, _P, _P, _P]


def adamw(p, g, m, v, sc):
    """in place on p, m, v.  sc = [decay, step, recip(bc2s), 1-b1, b2, 1-b2, eps] (fp32)."""
    for a in (p, g, m, v):
        assert a.dtype == np.float32 and a.flags.c_contiguous
    _L.lw_adamw(p.size, _p(p), _p(g), _p(m), _p(v), _p(_c(sc)))


_L.lw_ulp_scan.argtypes = [_i, ct.c_uint32, ct.c_uint32, _P]
SFU_IDS = {"exp": 0, "recip": 1, "rsqrt": 2, "sqrt": 3, "tanh": 4, "erf": 5, "log": 6}


def ulp_scan(fn, lo, hi):
    out = np.zeros(6)
    _L.lw_ulp_scan(SFU_IDS[fn], lo, hi, _p(out))
    return dict(max_ulp=out[0], arg=fbits(int(out[1])), n=int(out[2]), skipped=int(out[3]),
                n_ge1ulp=int(out[4]), n_not_cr=int(out[5]))


# ------------------------------------------------------------------ step scalars (P5 entry point)
def step_scalars(s, warmup, ttot, inv_w, inv_tw, lr0, wd, b1, b2, b1t_prev, b2t_prev):
    """LOWA_SCAL: dataflow.md section 6.2 exactly as hwmodel/step.py evaluates it, on the existing
    primitives (no new numerics).  s = completed steps (int), warmup / ttot ints, inv_w = fp32(1/w),
    inv_tw = fp32(1/(T-w)) host constants, lr0 = per-group base rates, b1t_prev / b2t_prev = the
    running products before this step (1.0 before the first).  Returns a dict of fp32 scalars:
    fr, b1t, b2t, bc1, rbc2 and per group lr, step, decay."""
    sf = i2f(np.array([s], np.int32))[0]
    if s < warmup:
        fr = mul(sf, f32(inv_w))[0]
    else:
        fr = fmax(mul(i2f(np.array([ttot - s], np.int32))[0], f32(inv_tw)), f32(0.0))[0]
    b1t = mul(f32(b1t_prev), f32(b1))[0]
    b2t = mul(f32(b2t_prev), f32(b2))[0]
    bc1 = sub(f32(1.0), b1t)[0]
    rbc2 = recip(sqrt(sub(f32(1.0), b2t)))[0]
    out = dict(fr=fr, b1t=b1t, b2t=b2t, bc1=bc1, rbc2=rbc2, lr=[], step=[], decay=[])
    for g in range(len(lr0)):
        lr = mul(f32(lr0[g]), fr)[0]
        out["lr"].append(lr)
        out["step"].append(mul(lr, recip(bc1))[0])
        out["decay"].append(sub(f32(1.0), mul(lr, f32(wd)))[0])
    return out
