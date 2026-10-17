#!/usr/bin/env python
"""Gates for `src/sgwa_adapter.py`.  Prints the canonical `N passed, M failed`.

  env/bin/python src/verify_sgwa.py
"""
import math
import os
import sys
import warnings

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sgwa_adapter as S

RES = [0, 0]


def ck(cond, msg):
    RES[0 if cond else 1] += 1
    print(("PASS " if cond else "FAIL ") + msg)


def path_laplacian(n):
    A = torch.zeros(n, n, dtype=torch.float64)
    i = torch.arange(n - 1)
    A[i, i + 1] = 1
    A[i + 1, i] = 1
    return torch.diag(A.sum(1)) - A


def main():
    K = 3
    # -- G1 the kernel -----------------------------------------------------
    H = S.wavelet_kernel(K)
    dd = torch.arange(-K, K + 1).abs()
    hop = dd[:, None] + dd[None, :]
    ck(torch.all(H[hop > K] == 0).item(), "G1a kernel is exactly 0 outside the K-hop diamond")
    ck(int((H.abs() > 0).sum()) == 2 * K * K + 2 * K + 1, "G1b kernel fills the diamond (25 at K=3)")
    ck(abs(H.norm().item() - 1) < 1e-12, "G1c kernel has unit Frobenius norm")
    ck(torch.allclose(H, H.T) and torch.allclose(H, H.flip(0)), "G1d kernel has the grid's symmetry")

    # -- G2 exactness against p(L) on the finite Neumann grid ---------------
    m, n = 13, 11
    L = torch.kron(path_laplacian(m), torch.eye(n, dtype=torch.float64)) + \
        torch.kron(torch.eye(m, dtype=torch.float64), path_laplacian(n))
    ck(torch.linalg.eigvalsh(L).max().item() < S.LAMBDA_MAX, "G2a lambda_max bound 8 holds")
    t = S.default_scale()
    c = S.cheby_coeffs(lambda lam: S.mother_wavelet(t * lam), K)
    PL = S.apply_cheby(lambda X: L @ X, torch.eye(m * n, dtype=torch.float64), c)
    # the infinite-grid normaliser, read off an interior atom
    base = nn.Linear(n, m, bias=False).double()
    lay = S.SGWALinear(base, k=m * n, K=K, scaling=1.0, seed=0, materialise=True).double()
    centers = S.sample_centers(m, n, m * n, 0)
    # ONE global constant (the infinite-grid kernel norm), read off an interior atom
    norm = None
    for a in range(m * n):
        i, j = divmod(int(centers[a]), n)
        if K <= i < m - K and K <= j < n - K:
            norm = PL[:, centers[a]].norm().item()
            break
    errs, corner_err = [], None
    for a in range(m * n):
        with torch.no_grad():
            lay.coef.zero_()
            lay.coef[a] = 1.0
        psi = lay.get_delta_weight().reshape(-1)
        ref = PL[:, centers[a]] / norm
        e = (psi - ref).abs().max().item()
        errs.append(e)
        if int(centers[a]) == 0:
            corner_err = e
    ck(max(errs) < 1e-12, f"G2b every atom (all {m*n}, incl. folded boundary ones) == p(L) delta_a "
                          f"[max err {max(errs):.2e}]")
    ck(corner_err is not None and corner_err < 1e-12, "G2c the corner atom (double fold) is exact")

    # -- G3..G6 on a real-shaped module -------------------------------------
    torch.manual_seed(0)
    d = 96
    base = nn.Linear(d, d).double()
    mats = S.SGWALinear(base, k=64, K=K, seed=777, materialise=True).double()
    fact = S.SGWALinear(base, k=64, K=K, seed=777, materialise=False).double()
    x = torch.randn(5, 7, d, dtype=torch.float64)
    ck(torch.equal(mats(x), base(x)) and torch.allclose(fact(x), base(x), atol=0, rtol=0),
       "G3 zero init: both paths reproduce the frozen layer")
    with torch.no_grad():
        coef = torch.randn(64, dtype=torch.float64)
        mats.coef.copy_(coef)
        fact.coef.copy_(coef)
    ym, yf = mats(x), fact(x)
    ck((ym - yf).abs().max().item() < 1e-12, f"G4 forward: merged == factored "
                                             f"[{(ym - yf).abs().max().item():.1e}]")
    xa = x.clone().requires_grad_(True)
    xb = x.clone().requires_grad_(True)
    g = torch.randn_like(ym)
    (mats(xa) * g).sum().backward()
    (fact(xb) * g).sum().backward()
    ck((mats.coef.grad - fact.coef.grad).abs().max().item() < 1e-10
       and (xa.grad - xb.grad).abs().max().item() < 1e-10,
       "G5 gradient (coef and input): merged == factored")
    ok = True
    for a in range(64):
        with torch.no_grad():
            mats.coef.zero_()
            mats.coef[a] = 1.0
        i, j = divmod(int(S.sample_centers(d, d, 64, 777)[a]), d)
        if K <= i < d - K and K <= j < d - K:
            ok &= abs(mats.get_delta_weight().norm().item() - mats.scaling) < 1e-12
    ck(ok, "G6 interior atom norm == scaling exactly ([R.281] atom definition)")
    ck(abs(S.SGWALinear(nn.Linear(768, 768)).scaling - 150 / math.sqrt(2 * 768 * 768)) < 1e-15
       and abs(150 / math.sqrt(2 * 768 * 768) - 0.138107) < 1e-6,
       "G7 derived scale = FourierFT's published atom 0.138107 at 768x768")
    big = S.SGWALinear(nn.Linear(768, 768), k=256)
    ck(big.n_params() == 256 and big.eflat.numel() == 256 * 25,
       "G8 256 trainable / module (x24 = 6,144) and 25 nnz per atom")

    # -- G10 opt-in sampled (SDDMM) backward == autograd's dense G^T X path ----
    errs = []
    for (mm, nn_, kk, sd) in ((96, 96, 64, 777), (16, 16, 200, 777), (24, 40, 128, 5)):
        torch.manual_seed(sd)
        b0 = nn.Linear(nn_, mm).double()
        for p_ in b0.parameters():
            p_.requires_grad_(False)
        A = S.SGWALinear(b0, k=kk, K=K, seed=sd).double()
        B = S.SGWALinear(b0, k=kk, K=K, seed=sd, sampled_grad=True).double()
        with torch.no_grad():
            cc = torch.randn(kk, dtype=torch.float64)
            A.coef.copy_(cc)
            B.coef.copy_(cc)
        xx = torch.randn(3, 9, nn_, dtype=torch.float64)
        gg = torch.randn(3, 9, mm, dtype=torch.float64)
        xa, xb = xx.clone().requires_grad_(True), xx.clone().requires_grad_(True)
        ya, yb = A(xa), B(xb)
        (ya * gg).sum().backward()
        (yb * gg).sum().backward()
        errs.append(max((ya - yb).abs().max().item(), (A.coef.grad - B.coef.grad).abs().max().item(),
                        (xa.grad - xb.grad).abs().max().item()))
    ck(max(errs) < 1e-10, f"G10 sampled backward (union-of-support SDDMM) == dense autograd: y, grad_c, "
                          f"grad_x; interior, folded and non-square grids [max err {max(errs):.1e}]")

    # -- G9 no device sync in either path ------------------------------------
    if torch.cuda.is_available():
        for mat, smp in ((True, False), (False, False), (True, True)):
            lay = S.SGWALinear(nn.Linear(768, 768).cuda(), k=256, materialise=mat,
                               sampled_grad=smp).cuda()
            xc = torch.randn(4096, 768, device="cuda", requires_grad=True)
            for _ in range(3):
                lay(xc).sum().backward()
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("warn")
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    lay(xc).square().mean().backward()
                    nsync = sum(1 for r in w if "sync" in str(r.message).lower())
            finally:
                torch.cuda.set_sync_debug_mode("default")
            ck(nsync == 0, f"G9 0 device syncs fwd+bwd, materialise={mat}"
                           f"{', sampled_grad' if smp else ''} [{nsync}]")
    else:
        ck(False, "G9 needs CUDA")

    print(f"{RES[0]} passed, {RES[1]} failed")
    sys.exit(1 if RES[1] else 0)


if __name__ == "__main__":
    main()
