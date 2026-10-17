#!/usr/bin/env python3
"""[S.4] gates for src/sgwa_parallel.py -- the population / sharded executions
are the SAME object as SGWALinear.  env/bin/python src/verify_sgwa_parallel.py

  G1  pop-merged forward  == per-run SGWALinear forward        (fp64, 1e-10)
  G2  pop-stencil forward == per-run SGWALinear forward        (fp64, 1e-10)
  G3  d/dx and d/dc of both pop modes == per-run autograd      (fp64, 1e-10)
  G4  stencil saves no (N, k, w) tensor: saved bytes == frozen GEMM's (+C, rid)
  G5  TP shards: concat(shards) == union-graph layer, exactly; per-shard coef
      grads == the union layer's restricted; 0 atoms straddle; seam count
  G6  merge linearity: dW(sum w_r c_r) == sum w_r dW(c_r); LoRA counterexample
  G7  PopAdamW(R=1) + per-run clip == torch AdamW + clip_grad_norm_, 30 steps
  G8  0 device syncs in a pop forward+backward (either mode)
"""
import math, os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sgwa_adapter import SGWALinear
from sgwa_parallel import (PopAdamW, SGWALinearPop, SGWALinearTPShard, _StencilPop,
                           merge_coefs, seam_atoms, shard_centres, union_graph_layer)

OK = FAIL = 0


def ck(cond, msg):
    global OK, FAIL
    if cond:
        OK += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


def per_run_ref(lin, R, k, K, seed, scal, coefs):
    refs = []
    for r in range(R):
        s = SGWALinear(lin, k=k, K=K, scaling=float(scal[r]), seed=seed, materialise=True)
        with torch.no_grad():
            s.coef.copy_(coefs[r])
        refs.append(s)
    return refs


def main(dev):
    torch.manual_seed(0)
    dt = torch.float64
    m, n, k, K, R, Br, T = 64, 48, 20, 3, 4, 3, 5
    lin = nn.Linear(n, m).to(dev, dt)
    lin.weight.requires_grad_(False); lin.bias.requires_grad_(False)
    scal = torch.tensor([0.1, 0.2, 0.4, 0.8], dtype=dt)
    coefs = torch.randn(R, k, dtype=dt, device=dev)
    refs = per_run_ref(lin, R, k, K, 777, scal, coefs)
    x = torch.randn(R * Br, T, n, dtype=dt, device=dev)
    rid = torch.arange(R, device=dev).repeat_interleave(Br)
    ref_out = torch.cat([refs[r](x[r * Br:(r + 1) * Br]) for r in range(R)], 0)

    pops = {}
    for mode in ("merged", "stencil"):
        p = SGWALinearPop(lin, R, k=k, K=K, scaling=scal, seed=777, mode=mode)
        with torch.no_grad():
            p.coef.copy_(coefs)
        p.set_rid(rid)
        pops[mode] = p
    print("G1/G2 population forward == per-run SGWALinear")
    for mode, p in pops.items():
        err = (p(x) - ref_out).abs().max().item()
        ck(err < 1e-10, f"{mode}: max |diff| = {err:.2e}")

    print("G3 gradients (d/dx, d/dc) == per-run autograd")
    g = torch.randn_like(ref_out)
    xr = x.clone().requires_grad_(True)
    outs = torch.cat([refs[r](xr[r * Br:(r + 1) * Br]) for r in range(R)], 0)
    outs.backward(g)
    gx_ref = xr.grad.clone()
    gc_ref = torch.stack([s.coef.grad for s in refs])
    for mode, p in pops.items():
        xp = x.clone().requires_grad_(True)
        p.coef.grad = None
        p(xp).backward(g)
        ex = (xp.grad - gx_ref).abs().max().item()
        ec = (p.coef.grad - gc_ref).abs().max().item()
        ck(ex < 1e-10 and ec < 1e-10, f"{mode}: d/dx {ex:.2e}  d/dc {ec:.2e}")

    print("G4 stencil backward saves the frozen GEMM's tensors only")
    p = pops["stencil"]
    x2 = x.reshape(-1, n).clone().requires_grad_(True)
    rid_tok = rid.unsqueeze(1).expand(R * Br, T).reshape(-1)
    out = _StencilPop.apply(x2, lin.weight, lin.bias, p.effective(), rid_tok, p.rows, p.cols, p.H_t)
    saved = out.grad_fn.saved_tensors
    nbytes = sum(t.numel() * t.element_size() for t in saved)
    gemm_like = x2.numel() * 8 + lin.weight.numel() * 8
    extra = nbytes - gemm_like
    mixbytes = x2.shape[0] * k * (2 * K + 1) * 8
    ck(extra < mixbytes / 4, f"saved {nbytes} B = GEMM {gemm_like} B + {extra} B "
                             f"(indices, C, rid); the (N,k,w) mix would be {mixbytes} B")

    print("G5 tensor-parallel shards on the partition-aligned graph")
    G = 2
    m2, n2, k2 = 96, 48, 40
    lin2 = nn.Linear(n2, m2).to(dev, dt)
    lin2.weight.requires_grad_(False); lin2.bias.requires_grad_(False)
    shards = union_graph_layer(lin2, G, k=k2, K=K, scaling=0.3, seed=777)
    cents = shard_centres(m2, n2, k2, 777, G)
    ck(sum(c.numel() for c in cents) == k2, f"the global draw splits into {[c.numel() for c in cents]} atoms")
    # straddle check: every atom's rows lie in its shard (by construction)
    strad = 0
    for sh in shards:
        strad += int(((sh.rows < 0) | (sh.rows >= sh.m)).sum())
    ck(strad == 0, "0 atoms straddle a shard (each shard's rows are local)")
    # the same adapter as ONE layer on the disjoint-union graph: build by hand
    xx = torch.randn(7, n2, dtype=dt, device=dev)
    for sh in shards:
        with torch.no_grad():
            sh.coef.normal_()
    out_tp = torch.cat([sh(xx) for sh in shards], 1)
    # reference: dense dW = blockdiag of the shards' deltas, applied to the full W0
    dW = torch.cat([sh.get_delta_weight() for sh in shards], 0)
    out_ref = F.linear(xx, lin2.weight + dW, lin2.bias)
    e = (out_tp - out_ref).abs().max().item()
    ck(e < 1e-10, f"concat(shard outputs) == full layer with the union-graph dW: {e:.2e}")
    # per-shard gradient == restricted gradient of the union layer (no exchange needed)
    gg = torch.randn_like(out_ref)
    out_tp.backward(gg)
    ms = m2 // G
    for gi, sh in enumerate(shards):
        gW = gg[:, gi * ms:(gi + 1) * ms].T @ xx                # this rank's g_out slice only
        # coefficient gradient = <gW, atom_a> = scaling * sum over atom support
        vals = sh.hval.view(1, -1).expand(sh.k, -1)
        gref = sh.scaling * (gW.reshape(-1)[sh.eflat].view(sh.k, -1) * vals).sum(1)
        ec = (sh.coef.grad - gref).abs().max().item()
        ck(ec < 1e-10, f"shard {gi}: coef grad from its own slice of g_out: {ec:.2e}")
    # seams vs the single grid: differences only at atoms within K of the seam
    single = SGWALinear(lin2, k=k2, K=K, scaling=0.3, seed=777)
    ns = seam_atoms(m2, n2, k2, K, 777, G)
    # count atoms of the single grid whose support crosses the seam row ms
    rows_single = single.rows.view(k2, -1)
    cross = int(((rows_single.min(1).values < ms) & (rows_single.max(1).values >= ms)).sum())
    ck(cross == ns, f"atoms whose single-grid support crosses the seam: {cross} (seam_atoms says {ns})")
    print(f"       768x768, k=256, K=3, G=2 -> seam atoms = {seam_atoms(768, 768, 256, 3, 777, 2)} "
          f"(expected ~ 256*6/768 = 2);  G=8 -> {seam_atoms(768, 768, 256, 3, 777, 8)}")

    print("G6 exact linear merge")
    p = pops["merged"]
    w = [0.1, 0.2, 0.3, 0.4]
    cm = merge_coefs(p.effective(), w)                       # merge the EFFECTIVE coefs
    dW_m = torch.zeros(m * n, dtype=dt, device=dev).index_add(
        0, p.eflat, (cm.unsqueeze(1) * p.hval).reshape(-1)).view(m, n)
    dW_sum = sum(w[r] * p.get_delta_weight(r) for r in range(R))
    e = (dW_m - dW_sum).abs().max().item()
    ck(e < 1e-12, f"dW(sum w_r c_r) == sum w_r dW(c_r): {e:.2e}")
    A = torch.randn(R, 4, n, dtype=dt); B = torch.randn(R, m, 4, dtype=dt)
    lora_avg = (B.mean(0) @ A.mean(0)); avg_lora = torch.einsum("rma,ran->mn", B, A) / R
    e2 = (lora_avg - avg_lora).norm().item() / avg_lora.norm().item()
    ck(e2 > 0.1, f"LoRA counterexample: ||mean(B)mean(A) - mean(BA)|| / ||mean(BA)|| = {e2:.2f}")

    print("G7 PopAdamW(R=1) + per-run clip == torch AdamW + clip_grad_norm_")
    torch.manual_seed(1)
    p_ref = nn.Parameter(torch.randn(50, dtype=torch.float32, device=dev))
    q_ref = nn.Parameter(torch.randn(7, 3, dtype=torch.float32, device=dev))
    p_pop = nn.Parameter(p_ref.detach().clone().unsqueeze(0))
    q_pop = nn.Parameter(q_ref.detach().clone().unsqueeze(0))
    opt_ref = torch.optim.AdamW([{"params": [p_ref], "lr": 0.05}, {"params": [q_ref], "lr": 5e-3}],
                                weight_decay=0.01)
    opt_pop = PopAdamW([{"params": [p_pop], "lr": [0.05], "weight_decay": 0.01},
                        {"params": [q_pop], "lr": [5e-3], "weight_decay": 0.01}])
    tgt_p = torch.randn(50, device=dev); tgt_q = torch.randn(7, 3, device=dev)
    worst = 0.0
    for t in range(30):
        lam = min(1.0, (t + 1) / 5) * (1 - t / 40)
        for grp in opt_ref.param_groups:
            grp["lr"] = (0.05 if grp["params"][0] is p_ref else 5e-3) * lam
        loss = ((p_ref - tgt_p) ** 2).sum() * 30 + ((q_ref - tgt_q) ** 2).sum() * 30
        opt_ref.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p_ref, q_ref], 1.0)
        opt_ref.step()
        loss2 = ((p_pop[0] - tgt_p) ** 2).sum() * 30 + ((q_pop[0] - tgt_q) ** 2).sum() * 30
        opt_pop.zero_grad(); loss2.backward()
        opt_pop.step(lam=lam, clip=1.0)
        worst = max(worst, (p_pop[0] - p_ref).abs().max().item(), (q_pop[0] - q_ref).abs().max().item())
    ck(worst < 1e-5, f"30 steps with warmup/decay, wd and clipping: max |diff| = {worst:.2e}")
    # per-run independence: two runs in one PopAdamW == two separate ones
    R2 = 2
    pp = nn.Parameter(torch.randn(R2, 30, device=dev))
    seps = [nn.Parameter(pp[r].detach().clone()) for r in range(R2)]
    lrs = [0.1, 0.01]
    o_pop = PopAdamW([{"params": [pp], "lr": lrs, "weight_decay": 0.01}])
    o_sep = [torch.optim.AdamW([s], lr=lrs[r], weight_decay=0.01) for r, s in enumerate(seps)]
    tg = torch.randn(R2, 30, device=dev)
    for t in range(20):
        o_pop.zero_grad(); (((pp - tg) ** 2).sum(1) * torch.tensor([1., 50.], device=dev)).sum().backward()
        o_pop.step(clip=1.0)
        for r in range(R2):
            o_sep[r].zero_grad(); (((seps[r] - tg[r]) ** 2).sum() * [1., 50.][r]).backward()
            torch.nn.utils.clip_grad_norm_([seps[r]], 1.0); o_sep[r].step()
    e = max((pp[r] - seps[r]).abs().max().item() for r in range(R2))
    ck(e < 1e-5, f"two runs with different lr and different clip factors stay independent: {e:.2e}")

    print("G8 device syncs")
    if dev.type == "cuda":
        lin32 = nn.Linear(768, 768).to(dev); lin32.weight.requires_grad_(False); lin32.bias.requires_grad_(False)
        xs = torch.randn(10 * 8, 16, 768, device=dev)
        rid8 = torch.arange(10, device=dev).repeat_interleave(8)
        for mode in ("merged", "stencil"):
            pm = SGWALinearPop(lin32, 10, k=256, K=3, seed=777, mode=mode)
            pm.set_rid(rid8)
            pm(xs).sum().backward(); torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                pm(xs).sum().backward()
                ck(True, f"{mode}: forward+backward issue no device sync")
            except RuntimeError as ex:
                ck(False, f"{mode}: sync -> {str(ex)[:80]}")
            finally:
                torch.cuda.set_sync_debug_mode("default")
            torch.cuda.synchronize()
    else:
        print("  skip (cpu)")
    print(f"\n[verify_sgwa_parallel] {OK} passed, {FAIL} failed")
    return FAIL == 0


if __name__ == "__main__":
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sys.exit(0 if main(dev) else 1)
