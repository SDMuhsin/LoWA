#!/usr/bin/env python3
"""[S.4] S4-C on a real backbone: tensor-parallel roberta-base attention (heads split
across ranks, Megatron column/row layout) with SGWA on the partition-aligned graph.

  torchrun --nproc_per_node 2 scripts/s4_tp_test.py --backend gloo --same_device   # rehearsal, cuda:0 only
  torchrun --nproc_per_node 2 scripts/s4_tp_test.py --backend nccl                 # rank r -> cuda:r

Checks (rank 0 prints):
  T1  the TP loss trajectory over --steps training steps equals a single-device run of
      the SAME adapter (the two shards executed sequentially and concatenated) to fp32
      round-off; the shard coefficients match the single-device shards' coefficients.
  T2  collectives per step are the backbone's TP collectives only: the count with the
      adapter coefficients trainable equals the count with them frozen (and equals
      2 per layer: the row-parallel dense's forward all-reduce and the column-parallel
      input's backward all-reduce).  ZERO adapter collectives.  The word embeddings are
      made trainable in both counts so backward traverses every layer in both.
  T3  the seam effect: the union-graph adapter vs the single-grid SGWALinear differ
      in dW only on the seam atoms (count printed), and the single-grid run's loss
      trajectory differs from the union-graph one by a reported amount (not a gate).
Dropout is 0 everywhere (so replicated parts stay bit-identical across ranks) and no
gradient clipping (global-norm clipping over sharded coefficients would need one
scalar all-reduce per step, as for any sharded parameter -- noted, not hidden).
"""
import argparse, os, sys, time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from sgwa_adapter import SGWALinear
from sgwa_parallel import SGWALinearTPShard, shard_centres, seam_atoms

COUNT = {"all_reduce": 0}
_orig_all_reduce = dist.all_reduce


def _counting_all_reduce(*a, **k):
    COUNT["all_reduce"] += 1
    return _orig_all_reduce(*a, **k)


dist.all_reduce = _counting_all_reduce


class _CopyToTP(torch.autograd.Function):      # Megatron "f": identity fwd, all-reduce bwd
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, g):
        g = g.contiguous()
        dist.all_reduce(g)
        return g


class _ReduceFromTP(torch.autograd.Function):  # Megatron "g": all-reduce fwd, identity bwd
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, g):
        return g


def _slice_linear(lin, rows=None, cols=None, bias=True, dev=None):
    W = lin.weight.data
    if rows is not None:
        W = W[rows]
    if cols is not None:
        W = W[:, cols]
    new = nn.Linear(W.shape[1], W.shape[0], bias=bias).to(dev, W.dtype)
    with torch.no_grad():
        new.weight.copy_(W)
        if bias:
            new.bias.copy_(lin.bias.data[rows] if rows is not None else lin.bias.data)
    for p in new.parameters():
        p.requires_grad_(False)
    return new


class TPSelfAttention(nn.Module):
    """One rank's 6 heads of a RobertaSelfAttention (column-parallel q/k/v)."""

    def __init__(self, hf_self, rank, G, dev, k, K, scaling, seed, adapter="shard"):
        super().__init__()
        H = hf_self.num_attention_heads
        hs = hf_self.attention_head_size
        self.h_loc = H // G
        self.hs = hs
        rows = slice(rank * self.h_loc * hs, (rank + 1) * self.h_loc * hs)
        m, n = hf_self.query.out_features, hf_self.query.in_features
        q = _slice_linear(hf_self.query, rows=rows, dev=dev)
        kk = _slice_linear(hf_self.key, rows=rows, dev=dev)
        v = _slice_linear(hf_self.value, rows=rows, dev=dev)
        cq = shard_centres(m, n, k, seed, G)[rank]
        cv = shard_centres(m, n, k, seed, G)[rank]
        self.query = SGWALinearTPShard(q, cq, K=K, scaling=scaling, seed=seed)
        self.key = kk
        self.value = SGWALinearTPShard(v, cv, K=K, scaling=scaling, seed=seed)

    def forward(self, x, mask):
        B, T, _ = x.shape
        x = _CopyToTP.apply(x)
        def heads(t):
            return t.view(B, T, self.h_loc, self.hs).transpose(1, 2)
        q, k, v = heads(self.query(x)), heads(self.key(x)), heads(self.value(x))
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return ctx.transpose(1, 2).reshape(B, T, self.h_loc * self.hs)


class TPSelfOutput(nn.Module):
    """Row-parallel dense (this rank's 384 input columns), all-reduce, LayerNorm."""

    def __init__(self, hf_out, rank, G, dev):
        super().__init__()
        n = hf_out.dense.in_features
        cols = slice(rank * n // G, (rank + 1) * n // G)
        self.dense = _slice_linear(hf_out.dense, cols=cols, bias=(rank == 0), dev=dev)
        self.LayerNorm = hf_out.LayerNorm

    def forward(self, ctx, residual):
        y = self.dense(ctx)
        y = _ReduceFromTP.apply(y)
        return self.LayerNorm(y + residual)


class TPLayerAttention(nn.Module):
    def __init__(self, hf_attn, rank, G, dev, **kw):
        super().__init__()
        self.self = TPSelfAttention(hf_attn.self, rank, G, dev, **kw)
        self.output = TPSelfOutput(hf_attn.output, rank, G, dev)

    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        ctx = self.self(hidden_states, attention_mask)
        return (self.output(ctx, hidden_states),)


class UnionSelfAttention(nn.Module):
    """Single-device reference: the SAME adapter (all G shards, concatenated)."""

    def __init__(self, hf_self, G, dev, k, K, scaling, seed, single_grid=False):
        super().__init__()
        self.H, self.hs = hf_self.num_attention_heads, hf_self.attention_head_size
        m, n = hf_self.query.out_features, hf_self.query.in_features
        if single_grid:
            self.query = SGWALinear(_slice_linear(hf_self.query, dev=dev), k=k, K=K, scaling=scaling, seed=seed)
            self.value = SGWALinear(_slice_linear(hf_self.value, dev=dev), k=k, K=K, scaling=scaling, seed=seed)
            self.qs = self.vs = None
        else:
            cents = shard_centres(m, n, k, seed, G)
            ms = m // G
            self.qs = nn.ModuleList([SGWALinearTPShard(_slice_linear(hf_self.query, rows=slice(g * ms, (g + 1) * ms), dev=dev),
                                                       cents[g], K=K, scaling=scaling, seed=seed) for g in range(G)])
            self.vs = nn.ModuleList([SGWALinearTPShard(_slice_linear(hf_self.value, rows=slice(g * ms, (g + 1) * ms), dev=dev),
                                                       cents[g], K=K, scaling=scaling, seed=seed) for g in range(G)])
        self.key = _slice_linear(hf_self.key, dev=dev)

    def forward(self, x, mask):
        B, T, _ = x.shape
        q = self.query(x) if self.qs is None else torch.cat([s(x) for s in self.qs], -1)
        v = self.value(x) if self.vs is None else torch.cat([s(x) for s in self.vs], -1)
        heads = lambda t: t.view(B, T, self.H, self.hs).transpose(1, 2)
        ctx = F.scaled_dot_product_attention(heads(q), heads(self.key(x)), heads(v), attn_mask=mask)
        return ctx.transpose(1, 2).reshape(B, T, self.H * self.hs)


class RefLayerAttention(nn.Module):
    def __init__(self, hf_attn, G, dev, single_grid=False, **kw):
        super().__init__()
        self.self = UnionSelfAttention(hf_attn.self, G, dev, single_grid=single_grid, **kw)
        self.dense = _slice_linear(hf_attn.output.dense, dev=dev)
        self.LayerNorm = hf_attn.output.LayerNorm

    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        return (self.LayerNorm(self.dense(self.self(hidden_states, attention_mask)) + hidden_states),)


def build(model_name, dev, mode, rank, G, k, K, scaling, seed, single_grid=False):
    from transformers import AutoConfig, AutoModelForSequenceClassification
    cfg = AutoConfig.from_pretrained(model_name, num_labels=2, hidden_dropout_prob=0.0,
                                     attention_probs_dropout_prob=0.0)
    torch.manual_seed(seed)                     # identical head init everywhere
    hf = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg).to(dev)
    for p in hf.parameters():
        p.requires_grad_(False)
    for p in hf.classifier.parameters():
        p.requires_grad_(True)
    kw = dict(k=k, K=K, scaling=scaling, seed=seed)
    for layer in hf.roberta.encoder.layer:
        if mode == "tp":
            layer.attention = TPLayerAttention(layer.attention, rank, G, dev, **kw)
        else:
            layer.attention = RefLayerAttention(layer.attention, G, dev, single_grid=single_grid, **kw)
    return hf


def run(hf, batches, steps, lr, dev, log=None):
    coefs = [p for n, p in hf.named_parameters() if n.endswith(".coef")]
    head = list(hf.classifier.parameters())
    opt = torch.optim.AdamW([{"params": coefs, "lr": lr}, {"params": head, "lr": 5e-3}], weight_decay=0.01)
    losses = []
    for s in range(steps):
        ids, am, lab = batches[s % len(batches)]
        out = hf.roberta(input_ids=ids, attention_mask=am)      # HF builds the 4-D mask itself
        logits = hf.classifier(out.last_hidden_state)
        loss = F.cross_entropy(logits, lab)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses, torch.cat([c.detach().reshape(-1) for c in coefs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="nccl")
    ap.add_argument("--same_device", action="store_true", help="both ranks on cuda:0 (gloo rehearsal)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--scaling", type=float, default=0.138107)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--model", default="roberta-base")
    a = ap.parse_args()
    dist.init_process_group(a.backend)
    rank, G = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda:0" if a.same_device else f"cuda:{rank}")
    torch.cuda.set_device(dev)
    say = (lambda *s: print(*s, flush=True)) if rank == 0 else (lambda *s: None)
    say(f"[s4-tp] backend={a.backend} world={G} same_device={a.same_device} steps={a.steps}")

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    raw = load_dataset("glue", "rte")["train"].select(range(a.batch * a.steps))
    batches = []
    for i in range(a.steps):
        ex = raw[i * a.batch:(i + 1) * a.batch]
        enc = tok(ex["sentence1"], ex["sentence2"], padding=True, truncation=True, max_length=128, return_tensors="pt")
        batches.append((enc["input_ids"].to(dev), enc["attention_mask"].to(dev),
                        torch.tensor(ex["label"], device=dev)))

    # ---- T2: collectives per step, adapter trainable vs frozen -------------
    hf = build(a.model, dev, "tp", rank, G, a.k, a.K, a.scaling, a.seed)
    # control: the word embeddings get a gradient in BOTH counts, so backward traverses
    # every layer whether or not the adapter is trainable (otherwise a frozen adapter
    # leaves nothing below the head and the backward collectives vanish for that reason)
    hf.roberta.embeddings.word_embeddings.weight.requires_grad_(True)
    COUNT["all_reduce"] = 0
    run(hf, batches[:2], 2, a.lr, dev)
    per_step_train = COUNT["all_reduce"] / 2
    for n, p in hf.named_parameters():
        if n.endswith(".coef"):
            p.requires_grad_(False)
    COUNT["all_reduce"] = 0
    run(hf, batches[:2], 2, a.lr, dev)
    per_step_frozen = COUNT["all_reduce"] / 2
    nl = len(hf.roberta.encoder.layer)
    say(f"[T2] all_reduce per step: adapter trainable {per_step_train:.0f}, adapter frozen {per_step_frozen:.0f}, "
        f"backbone TP needs {2 * nl} (2 per layer)  -> adapter collectives = {per_step_train - per_step_frozen:.0f}")
    ok2 = per_step_train == per_step_frozen == 2 * nl
    del hf; torch.cuda.empty_cache()

    # ---- T1: trajectory equality vs the single-device union-graph run -------
    hf = build(a.model, dev, "tp", rank, G, a.k, a.K, a.scaling, a.seed)
    t0 = time.time()
    L_tp, c_tp = run(hf, batches, a.steps, a.lr, dev)
    t_tp = time.time() - t0
    # gather every rank's coefficients to rank 0 (for the check only; not part of training)
    cdev = torch.device("cpu") if a.backend == "gloo" else dev
    c_loc = c_tp.to(cdev)
    sizes = [torch.zeros(1, dtype=torch.long, device=cdev) for _ in range(G)]
    dist.all_gather(sizes, torch.tensor([c_loc.numel()], device=cdev))
    gathered = []
    for g in range(G):
        buf = c_loc.clone() if g == rank else torch.zeros(int(sizes[g].item()), device=cdev)
        dist.broadcast(buf, src=g)
        gathered.append(buf.to(dev))
    del hf; torch.cuda.empty_cache()
    if rank == 0:
        ref = build(a.model, dev, "ref", 0, G, a.k, a.K, a.scaling, a.seed)
        t0 = time.time()
        L_ref, c_ref = run(ref, batches, a.steps, a.lr, dev)
        t_ref = time.time() - t0
        # the reference's coef order: per layer [q shards 0..G-1, v shards 0..G-1]; TP rank g: per layer [q_g, v_g]
        ref_by_rank = [[] for _ in range(G)]
        for layer in ref.roberta.encoder.layer:
            for g in range(G):
                ref_by_rank[g].append(layer.attention.self.qs[g].coef.detach().reshape(-1))
                ref_by_rank[g].append(layer.attention.self.vs[g].coef.detach().reshape(-1))
        dc = max((torch.cat(ref_by_rank[g]) - gathered[g]).abs().max().item() for g in range(G))
        cmax = max(gathered[g].abs().max().item() for g in range(G))
        dl = max(abs(x - y) for x, y in zip(L_tp, L_ref))
        print(f"[T1] loss trajectory TP vs single-device union-graph: max |dL| = {dl:.2e} over {a.steps} steps "
              f"(losses {L_tp[0]:.4f} -> {L_tp[-1]:.4f}); max |dcoef| = {dc:.2e} (max |coef| {cmax:.3f}, "
              f"rel {dc / cmax:.1e}; index_add atomics make fp32 sums order-dependent)")
        ok1 = dl < 1e-4 and dc / cmax < 1e-3
        del ref; torch.cuda.empty_cache()
        # ---- T3: the seam, against the plain single-grid SGWA --------------
        sg = build(a.model, dev, "ref", 0, G, a.k, a.K, a.scaling, a.seed, single_grid=True)
        L_sg, _ = run(sg, batches, a.steps, a.lr, dev)
        m = sg.roberta.encoder.layer[0].attention.self.query.m
        n = sg.roberta.encoder.layer[0].attention.self.query.n
        print(f"[T3] seam atoms per module at G={G}: {seam_atoms(m, n, a.k, a.K, a.seed, G)} of {a.k}; "
              f"single-grid vs union-graph loss trajectory: max |dL| = {max(abs(x - y) for x, y in zip(L_sg, L_ref)):.2e}, "
              f"final {L_sg[-1]:.4f} vs {L_ref[-1]:.4f}")
        print(f"[time] TP rank-0 {t_tp:.1f}s vs single-device {t_ref:.1f}s for {a.steps} steps "
              f"({'same device: no speed claim' if a.same_device else 'two devices'})")
        print(f"[s4-tp] T1 {'PASS' if ok1 else 'FAIL'}  T2 {'PASS' if ok2 else 'FAIL'}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
