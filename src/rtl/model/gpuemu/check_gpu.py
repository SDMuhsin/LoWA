"""P9d: prove the GPU port (gpuemu) bit-exact against L1 (hwmodel), kernel by kernel.

  python gpuemu/check_gpu.py [--sfu-n N]      (run from src/rtl/model, CUDA_VISIBLE_DEVICES=0)

1. add / mul / fma on test_arith.gen_cases (random bits, normals, cancellation, ties, results
   around 2^-126 and the 2^-126 rounding boundary, subnormal inputs, all specials).
2. every SFU function and composite element-wise op on N random bit patterns + all specials +
   dense sweeps of the ranges a training step uses.
3. GEMM (all stride layouts used by the step, odd sizes, K not a multiple of 16 or 64), LN fwd/bwd,
   softmax fwd/bwd, LoWA merge / sampled gradient, CE, GNORM/clip, AdamW, colsum, dropout masks.
Bitwise equality is required everywhere; whole-step equality is checked by check_step.py.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hwmodel"))
import hwmodel as H  # noqa: E402
import gpuemu as G  # noqa: E402
import rng  # noqa: E402
from hwmodel.test_arith import gen_cases  # noqa: E402

R = np.random.default_rng(20261019)
FAIL = []


def u(x):
    return np.asarray(x, np.float32).view(np.uint32)


def same(name, l1, gp):
    a = u(np.asarray(l1, np.float32))
    b = u(gp.detach().cpu().numpy() if torch.is_tensor(gp) else np.asarray(gp, np.float32))
    bad = int((a.reshape(-1) != b.reshape(-1)).sum())
    print(f"  {name:28s} {a.size:>12d} values  {bad} mismatches", flush=True)
    if bad:
        FAIL.append(name)
        i = np.nonzero(a.reshape(-1) != b.reshape(-1))[0][:3]
        print("     first:", [(int(k), hex(int(a.reshape(-1)[k])), hex(int(b.reshape(-1)[k]))) for k in i])


def f(a):
    return np.asarray(a, np.float32)


def prims():
    print("== add / mul / fma on test_arith.gen_cases ==")
    c = gen_cases(60000)
    a, b, cc = (c[:, i].view(np.float32).copy() for i in range(3))
    same("add", H.add(a, cc), G.add(a, cc))
    same("sub", H.sub(a, cc), G.sub(a, cc))
    same("mul", H.mul(a, b), G.mul(a, b))
    same("fma", H.fma(a, b, cc), G.fma(a, b, cc))
    same("max", H.fmax(a, b), G.fmax(a, b))
    same("min", H.fmin(a, b), G.fmin(a, b))


def sfu(n):
    print(f"== SFU and composites: {n} random bit patterns + sweeps ==")
    x = R.integers(0, 2 ** 32, n, dtype=np.uint64).astype(np.uint32).view(np.float32)
    sp = f([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-40, -1e-40, 1.0, -1.0, 88.72, -87.4, 100.0, -200.0,
            89.0, -104.0, 9.0, -9.0, 3.92, -3.92, 0.921875, 1.0, 2.0 ** -126, 2.0 ** -127])
    sw = np.concatenate([np.linspace(-110, 110, 2_000_001, dtype=np.float32),
                         np.linspace(-10, 10, 2_000_001, dtype=np.float32),
                         np.linspace(-1e-3, 1e-3, 200_001, dtype=np.float32),
                         np.geomspace(1e-38, 1e38, 2_000_001).astype(np.float32)])
    xs = np.concatenate([x, sp, sw]).astype(np.float32)
    for fn in ("exp", "recip", "rsqrt", "sqrt", "tanh", "erf", "log", "gelu"):
        same(fn, getattr(H, fn)(xs), getattr(G, fn)(xs))
    y = np.roll(xs, 7)
    same("gelu_bwd", H.gelu_bwd(y, xs), G.gelu_bwd(y, xs))
    same("tanh_bwd", H.tanh_bwd(y, xs), G.tanh_bwd(y, xs))
    same("scale", H.scale(xs, 0.125), G.scale(xs, 0.125))
    k = (R.random(xs.size) > 0.1).astype(np.uint8)
    same("drop", H.drop(xs, k, rng.DROP_SCALE), G.drop(xs, k, rng.DROP_SCALE))


def rn(*shape, s=1.0):
    return (R.standard_normal(shape) * s).astype(np.float32)


def kernels():
    print("== GEMM ==")
    for (M, N, K) in [(37, 53, 1), (64, 64, 64), (100, 70, 65), (129, 33, 200), (300, 768, 768), (32, 2, 768),
                      (768, 768, 32), (257, 130, 3072), (96, 50, 17), (200, 140, 65), (4096, 768, 3), (130, 300, 9)]:
        A, Bm = rn(M, K), rn(N, K)
        A[R.random(A.shape) < 0.05] = 0.0
        A[R.random(A.shape) < 0.02] = -0.0
        same(f"gemm {M}x{N}x{K}", H.gemm(A, Bm), G.gemm(A, Bm))
        Bkn = rn(K, N)
        same(f"gemm_nn {M}x{N}x{K}", H.gemm_nn(A, Bkn), G.gemm_nn(A, Bkn))
        Akm = rn(K, M)
        same(f"gemm_tn {M}x{N}x{K}", H.gemm_tn(Akm, Bkn), G.gemm_tn(Akm, Bkn))
    bias = rn(53)
    A, Bm = rn(37, 20), rn(53, 20)
    same("gemm + bias", H.gemm(A, Bm, bias), G.gemm(A, Bm, bias))
    # the attention layouts of step.py (B, 12 heads, L, 64)
    import hwmodel.step as S
    Bn, L = 3, 48
    D = 768
    rowmaj, tokcol, sq, sqT = S.attn_strides(Bn, L)
    Q, K_, V = rn(Bn * L, D), rn(Bn * L, D), rn(Bn * L, D)
    S1 = np.empty((Bn, 12, L, L), np.float32)
    H.gemm_raw(Bn, 12, L, L, 64, Q, rowmaj, K_, rowmaj, S1, sq)
    S2 = torch.empty((Bn, 12, L, L), dtype=torch.float32, device="cuda")
    G.gemm_raw(Bn, 12, L, L, 64, G._c(Q), rowmaj, G._c(K_), rowmaj, S2, sq)
    same("gemm_raw QK^T", S1, S2)
    P = rn(Bn, 12, L, L)
    for nm, (pa, ba) in {"PV": (sq, tokcol), "P^T dO": (sqT, tokcol)}.items():
        O1 = np.empty((Bn * L, D), np.float32)
        H.gemm_raw(Bn, 12, L, 64, L, P, pa, V, ba, O1, rowmaj)
        O2 = torch.empty((Bn * L, D), dtype=torch.float32, device="cuda")
        G.gemm_raw(Bn, 12, L, 64, L, G._c(P), pa, G._c(V), ba, O2, rowmaj)
        same("gemm_raw " + nm, O1, O2)

    print("== vector-unit kernels ==")
    x, g, b = rn(77, 768, s=3.0), rn(768), rn(768)
    y1, m1, r1 = H.ln_fwd(x, g, b)
    y2, m2, r2 = G.ln_fwd(x, g, b)
    same("ln_fwd y", y1, y2); same("ln_fwd mu", m1, m2); same("ln_fwd rstd", r1, r2)
    dy = rn(77, 768)
    same("ln_bwd", H.ln_bwd(x, m1, r1, g, dy), G.ln_bwd(x, m1, r1, g, dy))
    Bn, L = 4, 40
    Sx = rn(Bn, 12, L, L, s=4.0)
    km = np.ones((Bn, L), np.uint8)
    km[1, 25:] = 0
    km[3, 2:] = 0
    keep = (R.random(Sx.shape) > 0.1).astype(np.uint8)
    for kk in (keep, None):
        o1 = H.smax_fwd(Sx, km, kk, rng.DROP_SCALE)
        o2 = G.smax_fwd(Sx, km, kk, rng.DROP_SCALE)
        for nm, a1, a2 in zip(("P", "Pd", "m", "r"), o1, o2):
            same(f"smax_fwd {nm} keep={kk is not None}", a1, a2)
        dPd = rn(*Sx.shape)
        same(f"smax_bwd keep={kk is not None}", H.smax_bwd(o1[0], dPd, kk, rng.DROP_SCALE, 0.125),
             G.smax_bwd(o1[0], dPd, kk, rng.DROP_SCALE, 0.125))
    z = rn(33, 77)
    same("colsum", H.colsum(z), G.colsum(z))

    print("== LoWA, loss, optimizer ==")
    W = np.load(os.path.join(os.path.dirname(HERE), "runs", "model_fp32.npz"))
    e, h = W["lowa.3.eflat"].astype(np.int64), W["lowa.3.hval"]
    s = h.size
    W0 = rn(768, 768, s=0.05)
    coef = rn(256, s=0.3)
    plan = G.MergePlan(e, s)
    same("lowa_merge", H.lowa_merge(W0, e, coef, h, 0.552427), G.lowa_merge(W0, plan, coef, h, 0.552427))
    dY, X = rn(500, 768), rn(500, 768)
    rows, cols = (e // 768).reshape(-1, s), (e % 768).reshape(-1, s)
    same("lowa_sg", H.lowa_sg(dY, X, rows, cols, h, 0.552427), G.lowa_sg(dY, X, rows, cols, h, 0.552427))
    zz = rn(32, 2, s=3.0)
    lab = R.integers(0, 2, 32)
    w = np.ones(32, np.float32)
    w[20:] = 0
    l1, d1 = H.ce(zz, lab, w, H.recip(np.array([20], np.float32))[0])
    l2, d2 = G.ce(zz, lab, w, H.recip(np.array([20], np.float32))[0])
    same("ce loss", np.array([l1]), l2.reshape(1)); same("ce dz", d1, d2)
    shapes = [(256,)] * 24 + [(768, 768), (768,), (2, 768), (2,)]
    grads = [rn(*sh, s=1e-3) for sh in shapes]
    off = np.cumsum([0] + [gg.size for gg in grads])
    flat = np.concatenate([gg.reshape(-1) for gg in grads]).astype(np.float32)
    fg = G._c(flat.copy())
    n1, G1, c1 = H.gnorm_clip(flat, off, [gg.shape[-1] for gg in grads])
    n2, G2, c2 = G.gnorm_clip(fg, off, [gg.shape[-1] for gg in grads])
    same("gnorm norms", n1, n2); same("gnorm G", np.array([G1]), G2.reshape(1))
    same("gnorm c", np.array([c1]), c2.reshape(1)); same("gnorm g*c", flat, fg)
    for big in (1.0, 1e3):
        grads = [rn(*sh, s=big) for sh in shapes]
        flat = np.concatenate([gg.reshape(-1) for gg in grads]).astype(np.float32)
        fg = G._c(flat.copy())
        n1, G1, c1 = H.gnorm_clip(flat, off, [gg.shape[-1] for gg in grads])
        n2, G2, c2 = G.gnorm_clip(fg, off, [gg.shape[-1] for gg in grads])
        same(f"gnorm clip(scale {big:g})", flat, fg)
    p, gr, m, v = rn(5000), rn(5000, s=1e-2), rn(5000, s=1e-3), np.abs(rn(5000, s=1e-5))
    sc = np.array([0.999, 1e-3, 1.2, 0.1, 0.999, 0.001, 1e-8], np.float32)
    tp, tg, tm, tv = (G._c(t.copy()) for t in (p, gr, m, v))
    H.adamw(p, gr, m, v, sc)
    G.adamw(tp, tg, tm, tv, sc)
    same("adamw p", p, tp); same("adamw m", m, tm); same("adamw v", v, tv)

    print("== Threefry masks ==")
    for shp, seed, step, site, layer in [((32, 128, 768), 42, 0, "emb", 0), ((32, 12, 128, 128), 46, 3449, "attn", 11),
                                         ((7, 33), 43, 17, "cls2", 0), ((32, 768), 44, 206, "cls1", 0)]:
        a = rng.keep_mask(shp, seed, step, site, layer)
        b = G.keep_mask(shp, seed, step, site, layer)
        bad = int((a != b.cpu().numpy()).sum())
        print(f"  keep {str(shp):22s} seed {seed} step {step} {site}: {a.size} bits, {bad} mismatches")
        if bad:
            FAIL.append("keep")


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--sfu-n") + 1]) if "--sfu-n" in sys.argv else 1 << 24
    print("L1 threads:", H._L.lw_omp_threads())
    prims()
    sfu(n)
    kernels()
    G.check()
    print("GPU-vs-L1", "PASS" if not FAIL else f"FAIL {FAIL}")
    sys.exit(1 if FAIL else 0)
