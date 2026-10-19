"""GPU port of the L1 hwmodel (P9d): the entry points of hwmodel/__init__.py on torch cuda tensors.

Same names, same arguments, same results bit for bit (proven by gpuemu/check_gpu.py, not assumed).
Kernels: gpuemu/lwta_gpu.cu, a line-by-line port of hwmodel/csrc/lwta.c.  All launches go to the
legacy default stream, the stream torch uses unless told otherwise, so ordering with torch ops holds.
"""
import ctypes as ct
import os
import subprocess

import numpy as np
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_DIR, "liblwta_gpu.so")
if not os.path.exists(_SO) or os.path.getmtime(_SO) < os.path.getmtime(os.path.join(_DIR, "lwta_gpu.cu")):
    subprocess.run(["make", "-s", "-C", _DIR], check=True)
_L = ct.CDLL(_SO)
f32 = np.float32
_P, _l, _i, _f, _u = ct.c_void_p, ct.c_long, ct.c_int, ct.c_float, ct.c_uint32
DEV = torch.device("cuda")
F32 = torch.float32

_L.gw_elem.argtypes = [_i, _l, _P, _P, _P, _P]
_L.gw_scale.argtypes = [_l, _P, _f, _P]
_L.gw_drop.argtypes = [_l, _P, _P, _f, _P]
_L.gw_add_bias.argtypes = [_l, _l, _P, _P, _P]
_L.gw_colsum.argtypes = [_l, _l, _P, _P]
_L.gw_gemm.argtypes = [_i] * 5 + [_P] + [_l] * 4 + [_P] + [_l] * 4 + [_P] + [_l] * 4
_L.gw_ln_fwd.argtypes = [_l, _i, _P, _P, _P, _f, _f, _P, _P, _P]
_L.gw_ln_bwd.argtypes = [_l, _i, _P, _P, _P, _P, _P, _f, _P]
_L.gw_smax_fwd.argtypes = [_i, _i, _i, _P, _P, _P, _f, _P, _P, _P, _P]
_L.gw_smax_bwd.argtypes = [_l, _i, _P, _P, _P, _f, _f, _P]
_L.gw_lowa_merge.argtypes = [_i, _P, _P, _i, _P, _P, _P, _f, _P]
_L.gw_lowa_sg.argtypes = [_l, _i, _i, _i, _i, _P, _P, _P, _P, _P, _f, _P, _P]
_L.gw_ce.argtypes = [_i, _i, _P, _P, _P, _f, _P, _P]
_L.gw_gnorm_clip.argtypes = [_i, _P, _P, _l, _P, _f, _P, _P]
_L.gw_gnorm_clip2.argtypes = [_i, _P, _P, _P, _l, _P, _l, _P, _f, _P, _P]
_L.gw_adamw.argtypes = [_l, _P, _P, _P, _P, _P]
_L.gw_keep.argtypes = [_l, _u, _u, _u, _u, _P]

OPS = {n: i for i, n in enumerate(["add", "sub", "mul", "max", "min", "exp", "recip", "rsqrt", "sqrt", "tanh",
                                   "erf", "log", "gelu", "gelu_bwd", "tanh_bwd", "fma"])}


def _p(t):
    return None if t is None else ct.c_void_p(t.data_ptr())


def _c(t, dt=F32):
    """cuda, contiguous, dtype; numpy and scalars accepted (copied to the device)."""
    if not torch.is_tensor(t):
        t = torch.as_tensor(np.asarray(t))
    return t.to(DEV, dt).contiguous()


def check():
    torch.cuda.synchronize()
    e = _L.gw_last_error()
    assert e == 0, f"CUDA error {e}"


def _unary(name):
    def op(x):
        x = _c(x)
        y = torch.empty_like(x)
        _L.gw_elem(OPS[name], x.numel(), _p(x), None, None, _p(y))
        return y
    op.__name__ = name
    return op


def _binary(name):
    def op(a, b):
        a, b = _c(a), _c(b)
        a, b = torch.broadcast_tensors(a, b)
        a, b = a.contiguous(), b.contiguous()
        y = torch.empty_like(a)
        _L.gw_elem(OPS[name], a.numel(), _p(a), _p(b), None, _p(y))
        return y
    op.__name__ = name
    return op


exp, recip, rsqrt, sqrt, tanh, erf, log, gelu = (_unary(n) for n in
                                                 ("exp", "recip", "rsqrt", "sqrt", "tanh", "erf", "log", "gelu"))
add, sub, mul, fmax, fmin, gelu_bwd, tanh_bwd = (_binary(n) for n in
                                                 ("add", "sub", "mul", "max", "min", "gelu_bwd", "tanh_bwd"))


def fma(a, b, c):
    a, b, c = (t.contiguous() for t in torch.broadcast_tensors(_c(a), _c(b), _c(c)))
    y = torch.empty_like(a)
    _L.gw_elem(OPS["fma"], a.numel(), _p(a), _p(b), _p(c), _p(y))
    return y


def scale(x, s):
    x = _c(x)
    y = torch.empty_like(x)
    _L.gw_scale(x.numel(), _p(x), f32(s), _p(y))
    return y


def drop(x, keep, ds):
    x, keep = _c(x), _c(keep, torch.uint8)
    assert keep.numel() == x.numel()
    y = torch.empty_like(x)
    _L.gw_drop(x.numel(), _p(x), _p(keep), f32(ds), _p(y))
    return y


def add_bias(x, bias):
    x, bias = _c(x), _c(bias)
    y = torch.empty_like(x)
    _L.gw_add_bias(x.numel() // bias.numel(), bias.numel(), _p(x), _p(bias), _p(y))
    return y


def colsum(x):
    x = _c(x)
    out = torch.empty(x.shape[-1], dtype=F32, device=DEV)
    _L.gw_colsum(x.numel() // x.shape[-1], x.shape[-1], _p(x), _p(out))
    return out


def gemm_raw(nb1, nb2, M, N, K, A, a_str, Bm, b_str, C, c_str):
    for t in (A, Bm, C):
        assert t.dtype == F32 and t.is_contiguous() and t.is_cuda
    _L.gw_gemm(nb1, nb2, M, N, K, _p(A), *a_str, _p(Bm), *b_str, _p(C), *c_str)


def gemm(A, Bm, bias=None):
    A, Bm = _c(A), _c(Bm)
    M, K = A.shape
    N = Bm.shape[0]
    assert Bm.shape[1] == K
    C = torch.empty((M, N), dtype=F32, device=DEV)
    gemm_raw(1, 1, M, N, K, A, (0, 0, K, 1), Bm, (0, 0, K, 1), C, (0, 0, N, 1))
    return add_bias(C, bias) if bias is not None else C


def gemm_nn(A, Bkn):
    A, Bkn = _c(A), _c(Bkn)
    M, K = A.shape
    N = Bkn.shape[1]
    C = torch.empty((M, N), dtype=F32, device=DEV)
    gemm_raw(1, 1, M, N, K, A, (0, 0, K, 1), Bkn, (0, 0, 1, N), C, (0, 0, N, 1))
    return C


def gemm_tn(Akm, Bkn):
    Akm, Bkn = _c(Akm), _c(Bkn)
    K, M = Akm.shape
    N = Bkn.shape[1]
    C = torch.empty((M, N), dtype=F32, device=DEV)
    gemm_raw(1, 1, M, N, K, Akm, (0, 0, 1, M), Bkn, (0, 0, 1, N), C, (0, 0, N, 1))
    return C


LN_EPS = f32(1e-5)


def ln_fwd(x, g, b, eps=LN_EPS):
    x = _c(x)
    d = x.shape[-1]
    rows = x.numel() // d
    y = torch.empty_like(x)
    mu, rs = (torch.empty(rows, dtype=F32, device=DEV) for _ in range(2))
    g, b = _c(g), _c(b)
    _L.gw_ln_fwd(rows, d, _p(x), _p(g), _p(b), f32(eps), f32(1.0 / d), _p(y), _p(mu), _p(rs))
    return y, mu, rs


def ln_bwd(z, mu, rstd, g, dy):
    z, dy = _c(z), _c(dy)
    d = z.shape[-1]
    dx = torch.empty_like(z)
    mu, rstd, g = _c(mu), _c(rstd), _c(g)
    _L.gw_ln_bwd(z.numel() // d, d, _p(z), _p(mu), _p(rstd), _p(g), _p(dy), f32(1.0 / d), _p(dx))
    return dx


def smax_fwd(S, kmask, keep, ds):
    S = _c(S)
    Bn, H, L, _ = S.shape
    P, Pd = torch.empty_like(S), torch.empty_like(S)
    m, r = (torch.empty(S.shape[:3], dtype=F32, device=DEV) for _ in range(2))
    keep = None if keep is None else _c(keep, torch.uint8)
    kmask = _c(kmask, torch.uint8)
    _L.gw_smax_fwd(Bn, H, L, _p(S), _p(kmask),_p(keep), f32(ds), _p(P), _p(Pd), _p(m), _p(r))
    return P, Pd, m, r


def smax_bwd(P, dPd, keep, ds, scale_=0.125):
    P, dPd = _c(P), _c(dPd)
    L = P.shape[-1]
    dS = torch.empty_like(P)
    keep = None if keep is None else _c(keep, torch.uint8)
    _L.gw_smax_bwd(P.numel() // L, L, _p(P), _p(dPd), _p(keep), f32(ds), f32(scale_), _p(dS))
    return dS


class MergePlan:
    """per-cell hit lists of one LoWA module (eflat), ascending e inside a cell."""

    def __init__(self, eflat, s):
        e = np.asarray(eflat, np.int64)
        order = np.argsort(e, kind="stable")                    # by cell, ascending e within a cell
        cells, start = np.unique(e[order], return_index=True)
        self.ncell = len(cells)
        self.start = torch.as_tensor(np.append(start, len(e)).astype(np.int64), device=DEV)
        self.order = torch.as_tensor(order.astype(np.int64), device=DEV)
        self.eflat = torch.as_tensor(e, device=DEV)
        self.s = s


def lowa_merge(W0, plan, coef, h, gamma):
    W = _c(W0).clone()
    coef, h = _c(coef), _c(h)
    _L.gw_lowa_merge(plan.ncell, _p(plan.start), _p(plan.order), plan.s, _p(plan.eflat), _p(coef),
                     _p(h), f32(gamma), _p(W))
    return W


def lowa_sg(dY, X, rows, cols, h, gamma):
    dY, X = _c(dY), _c(X)
    rows, cols, h = _c(rows, torch.int64), _c(cols, torch.int64), _c(h)
    k, s = rows.shape
    g = torch.empty(k, dtype=F32, device=DEV)
    need = dY.shape[0] * s * k
    if _WORK.get("sg") is None or _WORK["sg"].numel() < need:
        _WORK["sg"] = torch.empty(need, dtype=F32, device=DEV)
    _L.gw_lowa_sg(dY.shape[0], dY.shape[1], X.shape[1], k, s, _p(dY), _p(X), _p(rows), _p(cols),
                  _p(h), f32(gamma), _p(g), _p(_WORK["sg"]))
    return g


_WORK = {}


def ce(z, y, w, inv_b):
    z = _c(z)
    Bn, Cn = z.shape
    dz = torch.empty_like(z)
    loss = torch.zeros(1, dtype=F32, device=DEV)
    y, w = _c(y, torch.int64), _c(w)
    _L.gw_ce(Bn, Cn, _p(z), _p(y), _p(w),f32(inv_b), _p(loss), _p(dz))
    return loss[0], dz


def gnorm_clip(gflat, offsets, rowlen, eps=1e-6):
    """in place on gflat (cuda); returns (norms, G, c) as cuda tensors."""
    off = _c(offsets, torch.int64)
    nt = off.numel() - 1
    norms = torch.empty(nt, dtype=F32, device=DEV)
    Gc = torch.zeros(2, dtype=F32, device=DEV)
    offn, rl = np.asarray(offsets, np.int64), np.asarray(rowlen, np.int64)
    nrows = (offn[1:] - offn[:-1]) // rl
    rowbase = _c(np.concatenate([[0], np.cumsum(nrows)]).astype(np.int64), torch.int64)
    rs = torch.empty(int(nrows.sum()), dtype=F32, device=DEV)
    rowlen = _c(rl, torch.int64)
    _L.gw_gnorm_clip2(nt, _p(off), _p(rowlen), _p(rowbase), rs.numel(), _p(rs), gflat.numel(), _p(gflat),
                      f32(eps), _p(norms), _p(Gc))
    return norms, Gc[0], Gc[1]


def adamw(p, g, m, v, sc):
    for t in (p, g, m, v):
        assert t.dtype == F32 and t.is_contiguous() and t.is_cuda
    sc = np.ascontiguousarray(sc, np.float32)
    _L.gw_adamw(p.numel(), _p(p), _p(g), _p(m), _p(v), sc.ctypes.data_as(_P))


SITE = {"emb": 0, "attn": 1, "attn_out": 2, "ffn_out": 3, "cls1": 4, "cls2": 5}
DROP_THRESH = 1677722


def keep_mask(shape, seed, step, site, layer):
    """model/rng.py keep_mask on the GPU (uint8 cuda tensor of `shape`)."""
    n = int(np.prod(shape))
    c1 = ((step << 8) | (SITE[site] << 4) | layer) & 0xFFFFFFFF
    out = torch.empty(n, dtype=torch.uint8, device=DEV)
    _L.gw_keep(n, c1, seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF, DROP_THRESH, _p(out))
    return out.reshape(shape)
