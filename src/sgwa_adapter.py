"""SGWA -- Spectral Graph Wavelet Adapter.

The weight update of an m x n linear layer is a signal on the 2-D grid graph
G = P_m [] P_n over its index set, and is synthesised from k spectral graph
wavelets (Hammond, Vandergheynst & Gribonval, ACHA 2011):

    dW = gamma * sum_a c_a psi_a ,     psi_a = p(L) delta_a ,

where L = L_m (x) I + I (x) L_n is the combinatorial Laplacian of the grid (the
path Laplacians carry degree-1 endpoints, i.e. Neumann boundaries), p is the
order-K shifted-Chebyshev approximation of the SGWT mother wavelet g(t*lambda)
on [0, lambda_max], and a ranges over k seeded random grid vertices.  Only the
k coefficients c are trained (zero-initialised).  The origin of this design is
`llmdocs/archive/gasa/RESEARCH_LOG.md` Phase A, verdict `ideate-gsp`.

THE KERNEL (fixed, derived, never searched)
  g      SGWT mother wavelet, alpha = beta = 2, x1 = 1, x2 = 2 (the toolbox's).
  K      Chebyshev order = locality radius in hops.  K = 3, Phase A's design
         point: the atom is supported on the 25-vertex 3-hop diamond.
  t      ONE scale, placed so g's passband peak x* = (12 - sqrt 12)/6 = 1.4226
         sits at the spectral midpoint lambda_max / 2.  ⛔ A multi-scale frame
         is NOT used: at order 3 the toolbox's coarse bands are not
         representable (relative spectral error 0.58-0.92, measured), so they
         would be mis-approximated atoms rather than wavelets.
  lambda_max = 8, the grid-Laplacian bound the SGWT toolbox would estimate.
  The kernel is normalised to unit Frobenius norm on the infinite grid, so the
  ATOM (the Frobenius norm of the dW one unit coefficient step makes, [R.281])
  is exactly `gamma` for every interior atom.

WHY THE APPLY IS EXACT AND CHEAP
  p(L) is a polynomial of the 5-point stencil, so p(L) delta_a on the INFINITE
  grid is a fixed (2K+1)^2 kernel H (zero outside the K-hop diamond).  The
  Neumann path Laplacian is the infinite-line Laplacian acting on the
  half-sample-symmetric extension of a signal, so on the finite grid
      psi_a[ refl(i_a + di), refl(j_a + dj) ] += H[di, dj]
  with refl(p) = -1-p (p < 0), 2N-1-p (p >= N): boundary atoms FOLD, they are
  not truncated.  Exact for K < min(m, n) / 2; `verify_sgwa.py` checks it
  against p(L) built from the explicit grid Laplacian.

  Two execution paths, the same object (verify_sgwa G4/G5):
    materialise=True   W0 + dW built by ONE index_add of the k*|diamond|
                       values into a copy of W0, then one GEMM.  The frozen
                       GEMM is reused, so the adapter adds O(k K^2 + mn)
                       elementwise work per forward, amortised over the batch.
    materialise=False  no m x n tensor: gather the 2K+1 input columns of each
                       atom (N x k x (2K+1)), mix by the fixed H, scale by c,
                       index_add into the 2K+1 output rows.  O(N k (2K+1)^2).
  Neither path forces a device sync (all indices are precomputed buffers).

  SAMPLED BACKWARD (opt-in, sampled_grad=True, merged path only; default off)
    Autograd on the merged path forms the dense dL/dW = G^T X (b m n MACs) and
    then gathers its k*s atom cells.  `_SGWASampledLinear` computes only the
    cells it needs (an SDDMM): for each cell (i,l) in the UNION U of the atom
    supports, S[i,l] = sum_t G[t,i] X[t,l]  (|U| b MACs, |U| <= k s), then
    grad_c = gamma * sum_e h_e S[cell(a,e)].  dX = G (W0 + dW) as before.  The
    forward is the merged forward, unchanged.  Same gradient to fp32 rounding
    (verify_sgwa G10).  The b x |U| gathers are irregular: see
    results/sgwa/sampled_grad_timing.txt for what that costs on a GPU.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LAMBDA_MAX = 8.0
X_PEAK = (12.0 - math.sqrt(12.0)) / 6.0        # argmax of g on [x1, x2]


def mother_wavelet(x):
    """SGWT kernel g (alpha = beta = 2, x1 = 1, x2 = 2), float64 numpy-free."""
    x = torch.as_tensor(x, dtype=torch.float64)
    lo = x ** 2
    mid = -5.0 + 11.0 * x - 6.0 * x ** 2 + x ** 3
    hi = 4.0 * x.clamp_min(1e-300) ** -2
    return torch.where(x < 1.0, lo, torch.where(x <= 2.0, mid, hi))


def default_scale(lmax: float = LAMBDA_MAX) -> float:
    return X_PEAK / (lmax / 2.0)


def cheby_coeffs(f, order: int, lmax: float = LAMBDA_MAX, n: int = 2000):
    """Shifted-Chebyshev coefficients of f on [0, lmax], by n-point
    Gauss-Chebyshev quadrature (n = 2000: the projection itself, not the SGWT
    toolbox's default n = order + 1, which moves the K=3 kernel by 2.3%).
    ⚠ At order 3 and the default scale the atom keeps 87% of its energy in its
    centre entry and has mean p(0) = -0.167 != g(0) = 0 [independent review,
    2026-10-06]: it is close to a single-entry update with a local halo."""
    a = lmax / 2.0
    th = math.pi * (torch.arange(1, n + 1, dtype=torch.float64) - 0.5) / n
    fx = f(a * torch.cos(th) + a)
    return torch.stack([2.0 / n * (fx * torch.cos(k * th)).sum()
                        for k in range(order + 1)])


def _lap_inf(X):
    """Combinatorial Laplacian of Z^2 on a zero-padded patch."""
    P = F.pad(X, (1, 1, 1, 1))
    return 4 * X - P[:-2, 1:-1] - P[2:, 1:-1] - P[1:-1, :-2] - P[1:-1, 2:]


def apply_cheby(apply_L, x, coeffs, lmax: float = LAMBDA_MAX):
    """p(L) x by the Chebyshev three-term recurrence (sgwt_cheby_op)."""
    a = lmax / 2.0
    Lt = lambda v: (apply_L(v) - a * v) / a
    T_prev, T_cur = x, Lt(x)
    out = 0.5 * coeffs[0] * T_prev + coeffs[1] * T_cur
    for k in range(2, len(coeffs)):
        T_prev, T_cur = T_cur, 2 * Lt(T_cur) - T_prev
        out = out + coeffs[k] * T_cur
    return out


def wavelet_kernel(K: int = 3, scale: Optional[float] = None,
                   lmax: float = LAMBDA_MAX) -> torch.Tensor:
    """(2K+1, 2K+1) float64 kernel H = p(L) delta on the infinite grid,
    unit Frobenius norm.  Exactly zero outside the K-hop diamond.
    [A.1] K = 0 is the UN-FILTERED ablation rung: H = [[1]], so every atom is a
    single unit entry at its seeded vertex (a plain sparse update with the same
    centres, the same unit atom norm, hence the same P = lr * gamma).  K = 0 used
    to raise (no order-1 Chebyshev term), so every earlier result is unchanged."""
    if K == 0:
        return torch.ones(1, 1, dtype=torch.float64)
    t = default_scale(lmax) if scale is None else scale
    c = cheby_coeffs(lambda lam: mother_wavelet(t * lam), K, lmax)
    N = 2 * K + 3                              # 1 guard ring: nothing reaches it
    d = torch.zeros(N, N, dtype=torch.float64)
    d[N // 2, N // 2] = 1.0
    H = apply_cheby(_lap_inf, d, c, lmax)[1:-1, 1:-1]
    return H / H.norm()


def _reflect(p: torch.Tensor, n: int) -> torch.Tensor:
    """Half-sample symmetric index fold (the Neumann path boundary)."""
    p = torch.where(p < 0, -1 - p, p)
    return torch.where(p >= n, 2 * n - 1 - p, p)


def sample_centers(m: int, n: int, k: int, seed: int) -> torch.Tensor:
    """k distinct grid vertices, one seeded randperm (FourierFT's own draw)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(m * n, generator=g)[:k]


class _SGWASampledLinear(torch.autograd.Function):
    """y = x (W0 + dW)^T + bias, with a sampled (SDDMM) coefficient gradient."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, coef, W0, bias, eflat, hval, urow, ucol, uinv, scaling):
        m, n = W0.shape
        vals = (scaling * coef).unsqueeze(1) * hval.to(W0.dtype)
        W = W0.reshape(-1).index_add(0, eflat, vals.reshape(-1).to(W0.dtype)).view(m, n)
        ctx.save_for_backward(x, W, hval, urow, ucol, uinv)
        ctx.scaling, ctx.k, ctx.has_bias = scaling, coef.shape[0], bias is not None
        return F.linear(x, W, bias)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gy):
        x, W, hval, urow, ucol, uinv = ctx.saved_tensors
        m, n = W.shape
        G2, X2 = gy.reshape(-1, m), x.reshape(-1, n)
        need = ctx.needs_input_grad
        dx = dcoef = dW0 = dbias = None
        if need[0]:
            dx = gy.matmul(W.to(gy.dtype))
        if need[1]:
            dt = torch.promote_types(gy.dtype, x.dtype)
            # SDDMM over the union of supports: S[u] = sum_t G[t, i_u] X[t, l_u]
            Su = G2.index_select(1, urow).to(dt).mul_(X2.index_select(1, ucol).to(dt)).sum(0)
            gv = Su.index_select(0, uinv).view(ctx.k, -1)                 # (k, s)
            dcoef = ((gv * hval.to(dt)).sum(1) * ctx.scaling).to(W.dtype)
        if need[2]:                                       # frozen in practice; kept exact
            dW0 = G2.t().matmul(X2.to(G2.dtype)).to(W.dtype)
        if ctx.has_bias and need[3]:
            dbias = G2.sum(0)
        return dx, dcoef, dW0, dbias, None, None, None, None, None, None


class SGWALinear(nn.Module):
    sampled_grad = False      # class default: subclasses that skip __init__ (TP shard) stay dense
    def __init__(self, base_layer: nn.Linear, k: int = 256, K: int = 3,
                 scaling: Optional[float] = None, seed: int = 777,
                 materialise: bool = True, scale: Optional[float] = None,
                 sampled_grad: bool = False):
        super().__init__()
        self.base_layer = base_layer
        self.m, self.n = base_layer.out_features, base_layer.in_features
        if 2 * K >= min(self.m, self.n):
            raise ValueError("the boundary fold is exact only for K < min(m, n) / 2")
        self.k, self.K, self.seed = k, K, seed
        self.materialise = materialise
        if sampled_grad and not materialise:
            raise ValueError("sampled_grad is a backward for the merged path (materialise=True)")
        self.sampled_grad = sampled_grad
        # [R.305] a-priori matched atom: FourierFT's published-default atom
        # 150 / sqrt(2mn), the rule SCoRA ships.  DERIVED, never swept.
        self.scaling = (150.0 / math.sqrt(2.0 * self.m * self.n)
                        if scaling is None else float(scaling))
        dev, dt = base_layer.weight.device, base_layer.weight.dtype
        H = wavelet_kernel(K, scale)
        w = 2 * K + 1
        flat = sample_centers(self.m, self.n, k, seed)
        ci, cj = flat // self.n, flat % self.n
        off = torch.arange(-K, K + 1)
        rows = _reflect(ci[:, None] + off[None, :], self.m)          # (k, w)
        cols = _reflect(cj[:, None] + off[None, :], self.n)          # (k, w)
        # factored path: the kernel box, shared by every atom
        self.register_buffer("rows", rows.reshape(-1).to(dev), persistent=False)
        self.register_buffer("cols", cols.reshape(-1).to(dev), persistent=False)
        self.register_buffer("H_t", H.T.contiguous().to(dev, dt), persistent=False)
        # merged path: the diamond only (box zeros are never written)
        di, dj = torch.nonzero(H.abs() > 0, as_tuple=True)
        self.register_buffer("hval", H[di, dj].to(dev, dt), persistent=False)
        e = rows[:, di] * self.n + cols[:, dj]                        # (k, |diamond|)
        self.register_buffer("eflat", e.reshape(-1).to(dev), persistent=False)
        # sampled backward: the union of the atom supports (atoms may overlap,
        # boundary folds may repeat a cell) and each atom entry's slot in it
        uf, uinv = torch.unique(e.reshape(-1), sorted=True, return_inverse=True)
        self.register_buffer("urow", (uf // self.n).to(dev), persistent=False)
        self.register_buffer("ucol", (uf % self.n).to(dev), persistent=False)
        self.register_buffer("uinv", uinv.to(dev), persistent=False)
        self.coef = nn.Parameter(torch.zeros(k, device=dev,
                                             dtype=base_layer.weight.dtype))
        self.w = w

    # -- the delta --------------------------------------------------------- #
    def get_delta_weight(self) -> torch.Tensor:
        vals = (self.scaling * self.coef).unsqueeze(1) * self.hval.to(self.coef.dtype)
        dW = torch.zeros(self.m * self.n, device=self.coef.device, dtype=self.coef.dtype)
        return dW.index_add(0, self.eflat, vals.reshape(-1)).view(self.m, self.n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W0 = self.base_layer.weight
        if self.sampled_grad:
            return _SGWASampledLinear.apply(x, self.coef, W0, self.base_layer.bias, self.eflat,
                                            self.hval, self.urow, self.ucol, self.uinv,
                                            self.scaling)
        if self.materialise:
            vals = (self.scaling * self.coef).unsqueeze(1) * self.hval.to(W0.dtype)
            W = W0.reshape(-1).index_add(0, self.eflat, vals.reshape(-1).to(W0.dtype))
            return F.linear(x, W.view(self.m, self.n), self.base_layer.bias)
        out = F.linear(x, W0, self.base_layer.bias)
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.n)
        xw = x2.index_select(1, self.cols).view(-1, self.k, self.w)  # (N, k, w)
        z = (xw @ self.H_t.to(x.dtype)) * (self.scaling * self.coef).to(x.dtype)[:, None]
        out2 = out.reshape(-1, self.m).index_add(1, self.rows, z.reshape(z.shape[0], -1))
        return out2.view(*lead, self.m)

    def n_params(self) -> int:
        return self.coef.numel()

    def extra_repr(self) -> str:
        return (f"m={self.m}, n={self.n}, k={self.k}, K={self.K}, "
                f"nnz/atom={self.hval.numel()}, scaling={self.scaling:.8g}, "
                f"seed={self.seed}, materialise={self.materialise}, "
                f"sampled_grad={self.sampled_grad}")


class SGWAAdapterModel(nn.Module):
    """Mirrors `SLRAdapterModel` exactly."""

    def __init__(self, model: nn.Module, target_modules, k: int = 256, K: int = 3,
                 scaling: Optional[float] = None, seed: int = 777,
                 materialise: bool = True, freeze_classifier_dense: bool = False,
                 sampled_grad: bool = False):
        super().__init__()
        self.model = model
        self.target_modules = list(target_modules)
        self.adapted_modules = []
        for p in model.parameters():
            p.requires_grad = False
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(tm in name for tm in self.target_modules):
                continue
            parts = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parts[0]] if len(parts) == 2 else model
            adapted = SGWALinear(module, k=k, K=K, scaling=scaling, seed=seed,
                                 materialise=materialise, sampled_grad=sampled_grad)
            setattr(parent, parts[-1], adapted)
            self.adapted_modules.append(name)
        for name, p in model.named_parameters():
            if "classifier" in name or "score" in name:
                if freeze_classifier_dense and "classifier.dense" in name:
                    continue
                p.requires_grad = True

    def gradient_checkpointing_enable(self, **kw):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kw)

    def forward(self, **kw):
        return self.model(**kw)

    def print_trainable_parameters(self):
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {tr:,} || all params: {tot:,} || "
              f"trainable%: {tr / tot * 100:.4f}")
        return tr

    def get_adapter_params(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters()
                   if p.requires_grad and n.endswith(".coef"))


def get_sgwa_model(model: nn.Module, target_modules, **kw):
    return SGWAAdapterModel(model, target_modules, **kw)
