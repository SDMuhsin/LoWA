"""L1 training step (hwmodel/step.py) on the GPU, op for op, with gpuemu kernels (P9d).

The body of step() is hwmodel/step.py's, line by line, with H -> G (same entry points, bit-exact,
check_gpu.py) and numpy -> torch.  Additions, all from spec.md D3 and no new numerics:
  * loss weights w_b and B_ACT (a short last batch is padded with dummy rows of weight 0, the loss
    divides by B_ACT), exactly as the chip's step descriptor;
  * warmup / total steps as parameters (step.py hard-codes the RTE run: 140 / 30 x 78);
  * the running products beta1^t, beta2^t are carried in the state (the chip holds them on chip);
    step.py recomputes the same sequence of H.mul from 1.0, the bits are identical.
Step scalars (lr, bias corrections) use L1's own scalar primitives (hwmodel, CPU), as step.py does.
forward_eval() is the forward half with dropout off (CSR DROP_EN = 0): logits only.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import gpuemu as G  # noqa: E402
import hwmodel as H  # noqa: E402
import rng  # noqa: E402
from hwmodel.step import (BETA1, BETA1C, BETA2, BETA2C, EPS, GAMMA, HEAD, LR0, WD,  # noqa: E402
                          attn_strides)

f32 = np.float32
D, NH, DH = 768, 12, 64
DEV = G.DEV


class GWeights:
    """frozen backbone + LoWA tables on the GPU (from model_fp32.npz), DAZ applied once on load
    (no-op: the checkpoint has no subnormals, checked)."""

    def __init__(self, path):
        z = np.load(path)
        self.z = {}
        for k in z.files:
            a = z[k]
            if a.dtype == np.float32 and not k.startswith("lowa.") and not k.startswith("classifier."):
                assert not ((np.abs(a) > 0) & (np.abs(a) < 2.0 ** -126)).any(), k
                self.z[k] = torch.as_tensor(a, device=DEV).contiguous()
        self.lw = []
        for i in range(24):
            e, h = z[f"lowa.{i}.eflat"].astype(np.int64), z[f"lowa.{i}.hval"].astype(np.float32)
            assert float(z[f"lowa.{i}.gamma"]) == float(GAMMA)
            s = h.size
            self.lw.append(dict(plan=G.MergePlan(e, s), rows=torch.as_tensor((e // D).reshape(-1, s), device=DEV),
                                cols=torch.as_tensor((e % D).reshape(-1, s), device=DEV),
                                h=torch.as_tensor(h, device=DEV)))
        self.qkv_cat = {}

    def __getitem__(self, k):
        return self.z[k]

    def layer(self, l, name):
        return self.z[f"roberta.encoder.layer.{l}.{name}"]


class State:
    """trainable state on the GPU: coef [24, 256], head, AdamW m / v (two flat groups), beta^t."""

    def __init__(self, coef, head, m_coef=None, m_head=None, v_coef=None, v_head=None, b1t=1.0, b2t=1.0):
        self.coef = G._c(coef).reshape(24, 256).clone()
        self.head = {k: G._c(head[k]).clone() for k in HEAD}
        nh = sum(self.head[k].numel() for k in HEAD)
        z = lambda n: torch.zeros(n, dtype=torch.float32, device=DEV)  # noqa: E731
        self.m = [G._c(m_coef).reshape(-1).clone() if m_coef is not None else z(6144),
                  G._c(m_head).reshape(-1).clone() if m_head is not None else z(nh)]
        self.v = [G._c(v_coef).reshape(-1).clone() if v_coef is not None else z(6144),
                  G._c(v_head).reshape(-1).clone() if v_head is not None else z(nh)]
        self.b1t, self.b2t = f32(b1t), f32(b2t)

    def to_np(self):
        d = {"coef": self.coef.cpu().numpy()}
        d.update({k: self.head[k].cpu().numpy() for k in HEAD})
        for tag, arrs in (("m_", self.m), ("v_", self.v)):
            d[tag + "coef"] = arrs[0].reshape(24, 256).cpu().numpy()
            o = 0
            for k in HEAD:
                n = self.head[k].numel()
                d[tag + k] = arrs[1][o:o + n].reshape(self.head[k].shape).cpu().numpy()
                o += n
        d["b1t"], d["b2t"] = np.array(self.b1t), np.array(self.b2t)
        return d


def embed(W, ids):
    Bn, L = ids.shape
    nz = (ids != 1).to(torch.int64)
    pos = torch.cumsum(nz, 1) * nz + 1
    e = G.add(G.add(W["roberta.embeddings.word_embeddings.weight"][ids.reshape(-1)],
                    W["roberta.embeddings.token_type_embeddings.weight"][0]),
              W["roberta.embeddings.position_embeddings.weight"][pos.reshape(-1)])
    x, _, _ = G.ln_fwd(e, W["roberta.embeddings.LayerNorm.weight"], W["roberta.embeddings.LayerNorm.bias"])
    return x


def step(W, st, ids, am, labels, lw, b_act, s, seed, warmup, t_total):
    """one training step in place on `st`; returns a dict of host scalars / small arrays."""
    ids = ids.to(DEV, torch.int64).contiguous()
    am = am.to(DEV, torch.uint8).contiguous()
    Bn, L = ids.shape
    N = Bn * L
    DS = rng.DROP_SCALE
    coef = st.coef
    head = st.head
    masks = {}

    def keep(site, layer, shape):
        key = (site, layer)
        if key not in masks:
            masks[key] = G.keep_mask(shape, seed, s, site, layer)
        return masks[key]

    rowmaj, tokcol, sq, sqT = attn_strides(Bn, L)
    lws = W.lw

    # ---------------------------------------------------------------------------- EMB
    x = embed(W, ids)
    x = G.drop(x, keep("emb", 0, (Bn, L, D)).reshape(N, D), DS)

    # ------------------------------------------------------------------------ forward
    sts = []
    for l in range(12):
        Wq = G.lowa_merge(W.layer(l, "attention.self.query.base_layer.weight"), lws[2 * l]["plan"], coef[2 * l], lws[2 * l]["h"], GAMMA)
        Wv = G.lowa_merge(W.layer(l, "attention.self.value.base_layer.weight"), lws[2 * l + 1]["plan"], coef[2 * l + 1], lws[2 * l + 1]["h"], GAMMA)
        Wk = W.layer(l, "attention.self.key.weight")
        Q = G.gemm(x, Wq, W.layer(l, "attention.self.query.base_layer.bias"))
        K = G.gemm(x, Wk, W.layer(l, "attention.self.key.bias"))
        V = G.gemm(x, Wv, W.layer(l, "attention.self.value.base_layer.bias"))
        S = torch.empty((Bn, NH, L, L), dtype=torch.float32, device=DEV)
        G.gemm_raw(Bn, NH, L, L, DH, Q, rowmaj, K, rowmaj, S, sq)
        S = G.scale(S, 0.125)
        ka = keep("attn", l, (Bn, NH, L, L))
        P, Pd, _, _ = G.smax_fwd(S, am, ka, DS)
        del S
        O = torch.empty((N, D), dtype=torch.float32, device=DEV)
        G.gemm_raw(Bn, NH, L, DH, L, Pd, sq, V, tokcol, O, rowmaj)
        y = G.gemm(O, W.layer(l, "attention.output.dense.weight"), W.layer(l, "attention.output.dense.bias"))
        z1 = G.add(G.drop(y, keep("attn_out", l, (Bn, L, D)).reshape(N, D), DS), x)
        h1, mu1, rs1 = G.ln_fwd(z1, W.layer(l, "attention.output.LayerNorm.weight"), W.layer(l, "attention.output.LayerNorm.bias"))
        u = G.gemm(h1, W.layer(l, "intermediate.dense.weight"), W.layer(l, "intermediate.dense.bias"))
        g = G.gelu(u)
        y2 = G.gemm(g, W.layer(l, "output.dense.weight"), W.layer(l, "output.dense.bias"))
        del g
        z2 = G.add(G.drop(y2, keep("ffn_out", l, (Bn, L, D)).reshape(N, D), DS), h1)
        xn, mu2, rs2 = G.ln_fwd(z2, W.layer(l, "output.LayerNorm.weight"), W.layer(l, "output.LayerNorm.bias"))
        sts.append(dict(x=x, Wq=Wq, Wk=Wk, Wv=Wv, Q=Q, K=K, V=V, P=P, Pd=Pd, z1=z1, mu1=mu1, rs1=rs1,
                        u=u, z2=z2, mu2=mu2, rs2=rs2))
        x = xn

    # ------------------------------------------------------------------- head + loss
    k1, k2 = keep("cls1", 0, (Bn, D)), keep("cls2", 0, (Bn, D))
    xc = G.drop(x.reshape(Bn, L, D)[:, 0].contiguous(), k1, DS)
    a = G.tanh(G.gemm(xc, head["dw"], head["db"]))
    a2 = G.drop(a, k2, DS)
    z = G.gemm(a2, head["ow"], head["ob"])
    inv_b = H.recip(np.array([b_act], f32))[0]
    loss, dz = G.ce(z, labels, lw, inv_b)
    gh = {"ow": G.gemm_tn(dz, a2), "ob": G.colsum(dz)}
    dh = G.tanh_bwd(G.drop(G.gemm_nn(dz, head["ow"]), k2, DS), a)
    gh["dw"], gh["db"] = G.gemm_tn(dh, xc), G.colsum(dh)
    dout = torch.zeros((N, D), dtype=torch.float32, device=DEV)
    dout.reshape(Bn, L, D)[:, 0] = G.drop(G.gemm_nn(dh, head["dw"]), k1, DS)

    # ------------------------------------------------------------------------ backward
    gcoef = torch.zeros((24, 256), dtype=torch.float32, device=DEV)
    for l in reversed(range(12)):
        c = sts[l]
        dz2 = G.ln_bwd(c["z2"], c["mu2"], c["rs2"], W.layer(l, "output.LayerNorm.weight"), dout)
        dy2 = G.drop(dz2, keep("ffn_out", l, None).reshape(N, D), DS)
        du = G.gelu_bwd(G.gemm_nn(dy2, W.layer(l, "output.dense.weight")), c["u"])
        dh1 = G.add(G.gemm_nn(du, W.layer(l, "intermediate.dense.weight")), dz2)
        del du
        dz1 = G.ln_bwd(c["z1"], c["mu1"], c["rs1"], W.layer(l, "attention.output.LayerNorm.weight"), dh1)
        dyo = G.drop(dz1, keep("attn_out", l, None).reshape(N, D), DS)
        dO = G.gemm_nn(dyo, W.layer(l, "attention.output.dense.weight"))
        dPd = torch.empty((Bn, NH, L, L), dtype=torch.float32, device=DEV)
        G.gemm_raw(Bn, NH, L, L, DH, dO, rowmaj, c["V"], rowmaj, dPd, sq)
        dS = G.smax_bwd(c["P"], dPd, keep("attn", l, None), DS, 0.125)
        del dPd
        dQ, dK, dV = (torch.empty((N, D), dtype=torch.float32, device=DEV) for _ in range(3))
        G.gemm_raw(Bn, NH, L, DH, L, c["Pd"], sqT, dO, tokcol, dV, rowmaj)
        G.gemm_raw(Bn, NH, L, DH, L, dS, sq, c["K"], tokcol, dQ, rowmaj)
        G.gemm_raw(Bn, NH, L, DH, L, dS, sqT, c["Q"], tokcol, dK, rowmaj)
        del dS
        for i, dY in ((2 * l, dQ), (2 * l + 1, dV)):
            gcoef[i] = G.lowa_sg(dY, c["x"], lws[i]["rows"], lws[i]["cols"], lws[i]["h"], GAMMA)
        if l > 0:
            dout = G.add(G.gemm_nn(torch.cat([dQ, dK, dV], 1), torch.cat([c["Wq"], c["Wk"], c["Wv"]], 0)), dz1)
        sts[l] = None

    # ---------------------------------------------------------------- GNORM + AdamW
    grads = [gcoef[i] for i in range(24)] + [gh[k] for k in HEAD]
    off = np.cumsum([0] + [gg.numel() for gg in grads])
    flat = torch.cat([gg.reshape(-1) for gg in grads]).contiguous()
    norms, Gn, clip = G.gnorm_clip(flat, off, [gg.shape[-1] for gg in grads])
    sf = f32(s)
    if s < warmup:
        fr = H.mul(sf, f32(1.0 / warmup))[0]
    else:
        fr = H.fmax(H.mul(f32(t_total - s), f32(1.0 / (t_total - warmup))), f32(0.0))[0]
    st.b1t, st.b2t = H.mul(st.b1t, BETA1)[0], H.mul(st.b2t, BETA2)[0]
    bc1 = H.sub(f32(1.0), st.b1t)[0]
    rbc2 = H.recip(H.sqrt(H.sub(f32(1.0), st.b2t)))[0]
    lrs = []
    params = [coef.reshape(-1).clone(), torch.cat([head[k].reshape(-1) for k in HEAD])]
    gsplit = [flat[:off[24]].contiguous(), flat[off[24]:].contiguous()]
    for grp in range(2):
        lr = H.mul(LR0[grp], fr)[0]
        lrs.append(lr)
        stp = H.mul(lr, H.recip(bc1))[0]
        decay = H.sub(f32(1.0), H.mul(lr, WD))[0]
        G.adamw(params[grp], gsplit[grp], st.m[grp], st.v[grp],
                np.array([decay, stp, rbc2, BETA1C, BETA2, BETA2C, EPS], f32))
    st.coef = params[0].reshape(24, 256)
    o = 0
    for k in HEAD:
        n = head[k].numel()
        st.head[k] = params[1][o:o + n].reshape(head[k].shape).clone()
        o += n
    return dict(loss=loss, logits=z, g_coef=gcoef, gnorm=Gn, clip=clip, lr=lrs,
                **{"g_" + k: gh[k] for k in HEAD})


@torch.no_grad()
def forward_eval(W, st, ids, am):
    """forward with the chip's arithmetic, dropout off: logits [B, C]."""
    ids = ids.to(DEV, torch.int64).contiguous()
    am = am.to(DEV, torch.uint8).contiguous()
    Bn, L = ids.shape
    N = Bn * L
    rowmaj, tokcol, sq, sqT = attn_strides(Bn, L)
    x = embed(W, ids)
    for l in range(12):
        lq, lv = W.lw[2 * l], W.lw[2 * l + 1]
        Wq = G.lowa_merge(W.layer(l, "attention.self.query.base_layer.weight"), lq["plan"], st.coef[2 * l], lq["h"], GAMMA)
        Wv = G.lowa_merge(W.layer(l, "attention.self.value.base_layer.weight"), lv["plan"], st.coef[2 * l + 1], lv["h"], GAMMA)
        Q = G.gemm(x, Wq, W.layer(l, "attention.self.query.base_layer.bias"))
        K = G.gemm(x, W.layer(l, "attention.self.key.weight"), W.layer(l, "attention.self.key.bias"))
        V = G.gemm(x, Wv, W.layer(l, "attention.self.value.base_layer.bias"))
        S = torch.empty((Bn, NH, L, L), dtype=torch.float32, device=DEV)
        G.gemm_raw(Bn, NH, L, L, DH, Q, rowmaj, K, rowmaj, S, sq)
        P, _, _, _ = G.smax_fwd(G.scale(S, 0.125), am, None, rng.DROP_SCALE)
        O = torch.empty((N, D), dtype=torch.float32, device=DEV)
        G.gemm_raw(Bn, NH, L, DH, L, P, sq, V, tokcol, O, rowmaj)
        z1 = G.add(G.gemm(O, W.layer(l, "attention.output.dense.weight"), W.layer(l, "attention.output.dense.bias")), x)
        h1, _, _ = G.ln_fwd(z1, W.layer(l, "attention.output.LayerNorm.weight"), W.layer(l, "attention.output.LayerNorm.bias"))
        u = G.gemm(h1, W.layer(l, "intermediate.dense.weight"), W.layer(l, "intermediate.dense.bias"))
        z2 = G.add(G.gemm(G.gelu(u), W.layer(l, "output.dense.weight"), W.layer(l, "output.dense.bias")), h1)
        x, _, _ = G.ln_fwd(z2, W.layer(l, "output.LayerNorm.weight"), W.layer(l, "output.LayerNorm.bias"))
    xc = x.reshape(Bn, L, D)[:, 0].contiguous()
    a = G.tanh(G.gemm(xc, st.head["dw"], st.head["db"]))
    return G.gemm(a, st.head["ow"], st.head["ob"])
