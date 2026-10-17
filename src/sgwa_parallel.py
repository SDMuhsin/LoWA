"""[S.4] SGWA-Pop -- multi-adapter and multi-GPU execution of SGWA, in the transform domain.

Design: llmdocs/S4_sgwa_parallel_design.md.  Everything here is the SAME object as
`SGWALinear` (verify_sgwa_parallel.py gates it); only the execution differs.

  S4-A  SGWALinearPop     R adapters of one frozen layer in one batched forward.
                          mode "merged":  (R, m, n) W_eff by one index_add, one bmm.
                          mode "stencil": one GEMM on W0 + local correction, with a
                          custom backward that recomputes the gathers and never
                          forms the dense weight gradient.
  S4-B  PopAdamW          AdamW over (R, ...) tensors with a per-run lr vector,
                          decoupled wd, and PER-RUN global-norm clipping.
  S4-C  shard_centres /   the partition-aligned graph: one Neumann grid per TP
        SGWALinearTPShard shard, so no atom straddles a shard and the adapter adds
                          no collective to a tensor-parallel step.
  S4-D  merge_coefs       exact linear merge of adapters (soups / local SGD / BTM).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from sgwa_adapter import SGWALinear, _reflect, sample_centers, wavelet_kernel


# ============================================================================ #
#  S4-A  the population layer
# ============================================================================ #
def _buffers(m, n, k, K, seed, dev, dt, centres: Optional[torch.Tensor] = None,
             scale=None):
    """The fixed basis, shared by every adapter of the population (P1)."""
    H = wavelet_kernel(K, scale)
    w = 2 * K + 1
    flat = sample_centers(m, n, k, seed) if centres is None else centres
    ci, cj = flat // n, flat % n
    off = torch.arange(-K, K + 1)
    rows = _reflect(ci[:, None] + off[None, :], m)
    cols = _reflect(cj[:, None] + off[None, :], n)
    di, dj = torch.nonzero(H.abs() > 0, as_tuple=True)
    e = rows[:, di] * n + cols[:, dj]
    return dict(rows=rows.reshape(-1).to(dev), cols=cols.reshape(-1).to(dev),
                H_t=H.T.contiguous().to(dev, dt), hval=H[di, dj].to(dev, dt),
                eflat=e.reshape(-1).to(dev), w=w, centres=flat)


class _StencilPop(torch.autograd.Function):
    """out = x W0^T + b + sum_a C[rid, a] * (x[:, cols_a] H) e_{rows_a}.

    Saves only x, C and rid.  Backward recomputes the two gathers, so the saved
    set equals the frozen GEMM's and the (N, k, w) mix is never stored; the
    coefficient gradient is k windowed inner products -- no m x n x N GEMM."""

    @staticmethod
    def forward(ctx, x2, W0, bias, C, rid, rows, cols, H_t):
        N = x2.shape[0]
        w = H_t.shape[1]
        out = F.linear(x2, W0, bias)
        xw = x2.index_select(1, cols).view(N, -1, w)              # (N, k, w)
        mix = xw @ H_t                                             # (N, k, w)
        z = mix * C.index_select(0, rid).unsqueeze(-1)             # (N, k, w)
        out = out.index_add(1, rows, z.reshape(N, -1))
        ctx.save_for_backward(x2, W0, C, rid, rows, cols, H_t)
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, g):
        x2, W0, C, rid, rows, cols, H_t = ctx.saved_tensors
        N = x2.shape[0]
        w = H_t.shape[1]
        Cr = C.index_select(0, rid)                                # (N, k)
        dz = g.index_select(1, rows).view(N, -1, w)                # (N, k, w)
        gx = g @ W0                                                # frozen GEMM's own backward
        # d mix = dz * C ;  d xw = d mix @ H_t^T ;  d x2[:, cols] += d xw
        dxw = (dz * Cr.unsqueeze(-1)) @ H_t.transpose(0, 1)
        gx = gx.index_add(1, cols, dxw.reshape(N, -1))
        gC = None
        if ctx.needs_input_grad[3]:
            xw = x2.index_select(1, cols).view(N, -1, w)           # recomputed
            mix = xw @ H_t
            per_tok = (dz * mix).sum(-1)                           # (N, k)
            gC = torch.zeros_like(C).index_add(0, rid, per_tok)
        gb = g.sum(0) if (ctx.has_bias and ctx.needs_input_grad[2]) else None
        return gx, None, gb, gC, None, None, None, None


class SGWALinearPop(nn.Module):
    """R SGWA adapters on one frozen nn.Linear, one batched forward.

    coef: (R, k).  scaling: (R,) per-run gamma.  The row -> run map is set per
    batch with `set_rid(rid_seq)` (one id per SEQUENCE, run-major); in "merged"
    mode the sub-batches must be equal-sized (bmm), in "stencil" mode any map."""

    def __init__(self, base_layer: nn.Linear, R: int, k: int = 256, K: int = 3,
                 scaling=None, seed: int = 777, mode: str = "merged"):
        super().__init__()
        assert mode in ("merged", "stencil")
        self.base_layer, self.R, self.k, self.K, self.seed, self.mode = base_layer, R, k, K, seed, mode
        self.m, self.n = base_layer.out_features, base_layer.in_features
        if 2 * K >= min(self.m, self.n):
            raise ValueError("the boundary fold is exact only for K < min(m, n) / 2")
        dev, dt = base_layer.weight.device, base_layer.weight.dtype
        b = _buffers(self.m, self.n, k, K, seed, dev, dt)
        for name in ("rows", "cols", "H_t", "hval", "eflat"):
            self.register_buffer(name, b[name], persistent=False)
        self.w = b["w"]
        if scaling is None:
            scaling = 150.0 / math.sqrt(2.0 * self.m * self.n)
        sc = torch.as_tensor(scaling, dtype=dt).reshape(-1)
        if sc.numel() == 1:
            sc = sc.expand(R).clone()
        assert sc.numel() == R
        self.register_buffer("scaling", sc.to(dev), persistent=False)
        self.coef = nn.Parameter(torch.zeros(R, k, device=dev, dtype=dt))
        self.rid_seq: Optional[torch.Tensor] = None
        self.runs: Optional[torch.Tensor] = None

    def set_rid(self, rid_seq: torch.Tensor, runs: Optional[torch.Tensor] = None):
        """rid_seq: one run id per sequence of the coming batch (run-major).  runs:
        the ids of the runs present, in order (default: all R), so a CHUNK of the
        population can be executed without touching the other runs' W_eff."""
        self.rid_seq = rid_seq.to(self.coef.device)
        self.runs = None if runs is None else runs.to(self.coef.device)

    def effective(self):
        return self.scaling.unsqueeze(1) * self.coef                   # (R, k)

    def get_delta_weight(self, r: int) -> torch.Tensor:
        vals = self.effective()[r].unsqueeze(1) * self.hval.to(self.coef.dtype)
        dW = torch.zeros(self.m * self.n, device=self.coef.device, dtype=self.coef.dtype)
        return dW.index_add(0, self.eflat, vals.reshape(-1)).view(self.m, self.n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        T = x.shape[1] if x.dim() == 3 else 1
        lead = x.shape[:-1]
        rid_seq = self.rid_seq
        if rid_seq is None:                   # a plain single-run call: run 0
            rid_seq = torch.zeros(B, dtype=torch.long, device=x.device)
        W0, bias = self.base_layer.weight, self.base_layer.bias
        C = self.effective().to(W0.dtype)
        x2 = x.reshape(-1, self.n)
        if self.mode == "stencil":
            rid = rid_seq.unsqueeze(1).expand(B, T).reshape(-1)
            out = _StencilPop.apply(x2, W0, bias, C, rid, self.rows, self.cols, self.H_t)
            return out.view(*lead, self.m)
        # merged: run-major, equal-sized sub-batches of the runs present
        runs = self.runs if self.runs is not None else torch.arange(self.R, device=x.device)
        R = runs.numel()
        if self.runs is not None:
            C = C.index_select(0, runs)
        assert B % R == 0, "merged mode needs run-major equal sub-batches"
        vals = C.unsqueeze(2) * self.hval.to(W0.dtype)                  # (R, k, nnz)
        mn = self.m * self.n
        idx = (self.eflat.unsqueeze(0) + torch.arange(R, device=x.device).unsqueeze(1) * mn)
        W = W0.reshape(1, mn).expand(R, mn).reshape(-1).index_add(0, idx.reshape(-1), vals.reshape(-1))
        W = W.view(R, self.m, self.n)
        xr = x2.view(R, -1, self.n)
        out = torch.bmm(xr, W.transpose(1, 2))
        if bias is not None:
            out = out + bias
        return out.reshape(*lead, self.m)

    def extra_repr(self):
        return (f"R={self.R}, m={self.m}, n={self.n}, k={self.k}, K={self.K}, "
                f"mode={self.mode}, seed={self.seed}")


# ============================================================================ #
#  S4-B  the population optimiser
# ============================================================================ #
class PopAdamW:
    """AdamW (torch semantics: decoupled wd, bias-corrected) over tensors whose
    leading dim indexes the run, with a per-run lr vector and per-run clipping.

    groups: list of dict(params=[(R, ...) tensors], lr=(R,) tensor, weight_decay=float)
    step(lam, clip): lam = scalar schedule multiplier (HF LambdaLR semantics), clip =
    max global norm PER RUN over every param of every group (0 disables)."""

    def __init__(self, groups, betas=(0.9, 0.999), eps=1e-8):
        self.groups = []
        for g in groups:
            ps = list(g["params"])
            R = ps[0].shape[0]
            lr = torch.as_tensor(g["lr"], dtype=torch.float32, device=ps[0].device).reshape(-1)
            lr = lr.expand(R).clone() if lr.numel() == 1 else lr
            self.groups.append(dict(params=ps, lr=lr, weight_decay=float(g.get("weight_decay", 0.0)),
                                    m=[torch.zeros_like(p) for p in ps],
                                    v=[torch.zeros_like(p) for p in ps]))
        self.b1, self.b2, self.eps, self.t = betas[0], betas[1], eps, 0

    def zero_grad(self):
        for g in self.groups:
            for p in g["params"]:
                p.grad = None

    @torch.no_grad()
    def per_run_grad_norm(self):
        sq = None
        for g in self.groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                s = p.grad.reshape(p.shape[0], -1).pow(2).sum(1)
                sq = s if sq is None else sq + s
        return sq.sqrt()

    @torch.no_grad()
    def step(self, lam: float = 1.0, clip: float = 0.0):
        self.t += 1
        scale = None
        if clip > 0:
            norm = self.per_run_grad_norm()
            scale = (clip / (norm + 1e-6)).clamp(max=1.0)           # torch.nn.utils.clip_grad_norm_
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for g in self.groups:
            lr = g["lr"] * lam                                          # (R,)
            for p, m, v in zip(g["params"], g["m"], g["v"]):
                if p.grad is None:
                    continue
                shp = (-1,) + (1,) * (p.dim() - 1)
                grad = p.grad if scale is None else p.grad * scale.view(shp).to(p.dtype)
                lr_ = lr.view(shp).to(p.dtype)
                if g["weight_decay"]:
                    p.mul_(1 - lr_ * g["weight_decay"])
                m.mul_(self.b1).add_(grad, alpha=1 - self.b1)
                v.mul_(self.b2).addcmul_(grad, grad, value=1 - self.b2)
                denom = (v.sqrt() / math.sqrt(bc2)).add_(self.eps)     # torch: sqrt(v)/sqrt(bc2) + eps
                p.sub_((lr_ / bc1) * m / denom)                        # torch: addcdiv(m, denom, -lr/bc1)


# ============================================================================ #
#  S4-C  partition-aligned graph for tensor parallelism
# ============================================================================ #
def shard_centres(m: int, n: int, k: int, seed: int, G: int) -> List[torch.Tensor]:
    """The single global draw (seed) split by owning row-shard.  Returned centres
    are LOCAL flat indices on shard g's own (m/G, n) grid."""
    assert m % G == 0
    ms = m // G
    flat = sample_centers(m, n, k, seed)
    ci, cj = flat // n, flat % n
    out = []
    for g in range(G):
        sel = (ci // ms) == g
        out.append(((ci[sel] - g * ms) * n + cj[sel]))
    return out


def seam_atoms(m: int, n: int, k: int, K: int, seed: int, G: int) -> int:
    """How many of the k atoms lie within K rows of a shard seam (where the
    union graph folds and the single grid spills)."""
    ms = m // G
    ci = sample_centers(m, n, k, seed) // n
    loc = ci % ms
    inner = [g for g in range(1, G)]           # seams at g*ms
    if not inner:
        return 0
    return int(((loc < K) & (ci >= ms)).sum() + ((loc >= ms - K) & (ci < m - ms)).sum())


class SGWALinearTPShard(SGWALinear):
    """SGWA on ONE row-shard of a column-parallel Linear (Megatron layout: this
    rank owns out-features [g*m/G, (g+1)*m/G)).  Its graph is the shard's OWN
    Neumann grid, so every atom here is complete on this rank: synthesis and the
    coefficient gradient need no collective.  Built like SGWALinear but with the
    shard's subset of the global centre draw."""

    def __init__(self, base_shard: nn.Linear, centres_local: torch.Tensor, K: int = 3,
                 scaling: Optional[float] = None, seed: int = 777, scale: Optional[float] = None):
        nn.Module.__init__(self)
        self.base_layer = base_shard
        self.m, self.n = base_shard.out_features, base_shard.in_features
        if 2 * K >= min(self.m, self.n):
            raise ValueError("the boundary fold is exact only for K < min(m, n) / 2")
        self.k, self.K, self.seed, self.materialise = int(centres_local.numel()), K, seed, True
        self.scaling = (150.0 / math.sqrt(2.0 * self.m * self.n) if scaling is None else float(scaling))
        dev, dt = base_shard.weight.device, base_shard.weight.dtype
        b = _buffers(self.m, self.n, self.k, K, seed, dev, dt, centres=centres_local, scale=scale)
        for name in ("rows", "cols", "H_t", "hval", "eflat"):
            self.register_buffer(name, b[name], persistent=False)
        self.w = b["w"]
        self.coef = nn.Parameter(torch.zeros(self.k, device=dev, dtype=dt))


def union_graph_layer(base_layer: nn.Linear, G: int, k=256, K=3, scaling=None, seed=777):
    """Reference: the SAME adapter as the G shards, as one module on the disjoint
    union graph (for the gate).  Returns (layer, shard_list)."""
    m, n = base_layer.out_features, base_layer.in_features
    ms = m // G
    cents = shard_centres(m, n, k, seed, G)
    shards = []
    for g in range(G):
        lin = nn.Linear(n, ms, bias=base_layer.bias is not None).to(base_layer.weight.device, base_layer.weight.dtype)
        with torch.no_grad():
            lin.weight.copy_(base_layer.weight[g * ms:(g + 1) * ms])
            if base_layer.bias is not None:
                lin.bias.copy_(base_layer.bias[g * ms:(g + 1) * ms])
        lin.weight.requires_grad_(False)
        if lin.bias is not None:
            lin.bias.requires_grad_(False)
        shards.append(SGWALinearTPShard(lin, cents[g], K=K, scaling=scaling, seed=seed))
    return shards


# ============================================================================ #
#  S4-D  exact merge
# ============================================================================ #
def merge_coefs(C: torch.Tensor, weights: Optional[Sequence[float]] = None) -> torch.Tensor:
    """(R, k) -> (k,): sum_r w_r c_r.  By linearity of the synthesis this IS the
    adapter whose dW is sum_r w_r dW_r (gate G6).  weights None = mean."""
    R = C.shape[0]
    w = torch.full((R,), 1.0 / R, dtype=C.dtype, device=C.device) if weights is None \
        else torch.as_tensor(list(weights), dtype=C.dtype, device=C.device)
    return (w.unsqueeze(1) * C).sum(0)


# ============================================================================ #
#  the population model (roberta sequence classification)
# ============================================================================ #
class PopClassificationHead(nn.Module):
    """R RobertaClassificationHeads: dropout -> dense -> tanh -> dropout -> out_proj,
    each run's head seeded by its own seed (normal(0, 0.02), zero bias)."""

    def __init__(self, d: int, num_labels: int, seeds: Sequence[int], p_drop: float,
                 device, dtype, init_std: float = 0.02):
        super().__init__()
        R = len(seeds)
        Wd = torch.empty(R, d, d, dtype=dtype)
        Wo = torch.empty(R, d, num_labels, dtype=dtype)
        for r, s in enumerate(seeds):
            g = torch.Generator().manual_seed(int(s))
            Wd[r].normal_(0.0, init_std, generator=g)
            Wo[r].normal_(0.0, init_std, generator=g)
        self.Wd = nn.Parameter(Wd.to(device))
        self.bd = nn.Parameter(torch.zeros(R, 1, d, dtype=dtype, device=device))
        self.Wo = nn.Parameter(Wo.to(device))
        self.bo = nn.Parameter(torch.zeros(R, 1, num_labels, dtype=dtype, device=device))
        self.drop = nn.Dropout(p_drop)
        self.R = R

    def forward(self, h_cls: torch.Tensor) -> torch.Tensor:
        """h_cls: (R*B_r, d) run-major -> logits (R, B_r, C)."""
        h = h_cls.view(self.R, -1, h_cls.shape[-1])
        h = self.drop(h)
        h = torch.tanh(torch.bmm(h, self.Wd) + self.bd)
        h = self.drop(h)
        return torch.bmm(h, self.Wo) + self.bo

    def params(self):
        return [self.Wd, self.bd, self.Wo, self.bo]


class SGWAPopModel(nn.Module):
    """roberta backbone (frozen, shared) + SGWALinearPop on the targets + R heads."""

    def __init__(self, hf_model, target_modules, R, k=256, K=3, scaling=None, seed=777,
                 mode="merged", num_labels=2, head_seeds=None):
        super().__init__()
        self.backbone = hf_model.roberta if hasattr(hf_model, "roberta") else hf_model.base_model
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.pops: List[SGWALinearPop] = []
        mods = dict(self.backbone.named_modules())
        for name, module in list(self.backbone.named_modules()):
            if not isinstance(module, nn.Linear) or not any(t in name for t in target_modules):
                continue
            parent_name, attr = name.rsplit(".", 1)
            pop = SGWALinearPop(module, R, k=k, K=K, scaling=scaling, seed=seed, mode=mode)
            setattr(mods[parent_name], attr, pop)
            self.pops.append(pop)
        cfg = hf_model.config
        self.head = PopClassificationHead(cfg.hidden_size, num_labels,
                                          head_seeds if head_seeds is not None else list(range(R)),
                                          cfg.hidden_dropout_prob,
                                          next(self.backbone.parameters()).device,
                                          next(self.backbone.parameters()).dtype,
                                          cfg.initializer_range)
        self.R = R

    def set_rid(self, rid_seq):
        for p in self.pops:
            p.set_rid(rid_seq)

    def forward(self, input_ids, attention_mask, rid_seq):
        self.set_rid(rid_seq)
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(h[:, 0, :])                                  # (R, B_r, C)

    def coef_params(self):
        return [p.coef for p in self.pops]
