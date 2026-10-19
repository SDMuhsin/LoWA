"""L1: one full LoWA training step of roberta-base with the chip's arithmetic and schedule.

Follows spec.md section 4 / isa.md section 4 op by op, every rounding as in dataflow.md:
  EMB -> 12 x (merge-on-load Q/V, QKV, S, softmax, P'V, O-proj + drop + residual, LN1, FFN1 + GELU,
  FFN2 + drop + residual, LN2) -> head + CE -> head bwd -> 12 x layer bwd with the sampled
  coefficient gradient -> GNORM/clip -> step scalars -> AdamW (coef group, head group).
Layer 11 is computed for all rows: the chip's CLS pruning (D6) only skips rows whose values never
reach the head and whose gradients are exact zeros, so every non-zero result is identical.

  python -m hwmodel.step RUN_DIR A|B [KC]  reads model_fp32.npz + step{X}_l0f32.npz (inputs, state)
                                           writes step{X}_l1.npz with the L0 dump's keys
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hwmodel as H  # noqa: E402
import rng  # noqa: E402

f32 = np.float32
D, NH, DH = 768, 12, 64
HEAD = ("dw", "db", "ow", "ob")
# host-side fp32 constants (CONST region): optimizer + schedule
BETA1C, BETA1, BETA2, BETA2C, EPS, WD = f32(0.1), f32(0.9), f32(0.999), f32(0.001), f32(1e-8), f32(0.01)
LR0 = (f32(0.166667), f32(5e-3))
GAMMA = f32(0.552427)
WARMUP, EPOCHS = 140, 30


class Weights:
    def __init__(self, path):
        z = np.load(path)
        self.z = {k: z[k] for k in z.files}

    def __getitem__(self, k):
        return self.z[k]

    def layer(self, l, name):
        return self.z[f"roberta.encoder.layer.{l}.{name}"]

    def lowa(self, i):
        """(eflat [k*s], rows [k, s], cols [k, s], h [s]) of module i (q_l = 2l, v_l = 2l + 1)."""
        e, h = self.z[f"lowa.{i}.eflat"].astype(np.int64), self.z[f"lowa.{i}.hval"]
        assert float(self.z[f"lowa.{i}.gamma"]) == float(GAMMA)
        s = h.size
        return e, (e // D).reshape(-1, s), (e % D).reshape(-1, s), h


def attn_strides(Bn, L):
    rowmaj = (L * D, DH, D, 1)            # Q/K/V/O/dO as [B, L, 12, 64]: (b, h, row=token, col=d)
    tokcol = (L * D, DH, 1, D)            # same tensors with row = d, col = token
    sq = (NH * L * L, L * L, L, 1)        # S/P [B, 12, L, L] row = query
    sqT = (NH * L * L, L * L, 1, L)       # transposed view, row = key
    return rowmaj, tokcol, sq, sqT


def step(W, inp, verbose=True):
    t0 = time.time()
    T = {}
    ids, am, labels = inp["input_ids"], inp["attention_mask"].astype(np.uint8), inp["labels"]
    Bn, L = ids.shape
    N = Bn * L
    s = int(inp["step_s"])
    seed = int(inp["seed"])
    DS = rng.DROP_SCALE
    coef = inp["pre_coef"].astype(f32).copy()            # [24, 256]
    head = {k: inp["pre_" + k].astype(f32).copy() for k in HEAD}
    masks = {}

    def keep(site, layer, shape):
        key = (site, layer)
        if key not in masks:
            masks[key] = rng.keep_mask(shape, seed, s, site, layer)
        return masks[key]

    rowmaj, tokcol, sq, sqT = attn_strides(Bn, L)
    lw = [W.lowa(i) for i in range(24)]

    # ---------------------------------------------------------------------------- EMB
    nz = (ids != 1).astype(np.int64)
    pos = np.cumsum(nz, 1) * nz + 1
    e = H.add(H.add(W["roberta.embeddings.word_embeddings.weight"][ids.reshape(-1)],
                    W["roberta.embeddings.token_type_embeddings.weight"][0]),
              W["roberta.embeddings.position_embeddings.weight"][pos.reshape(-1)])
    x, _, _ = H.ln_fwd(e, W["roberta.embeddings.LayerNorm.weight"], W["roberta.embeddings.LayerNorm.bias"])
    x = H.drop(x, keep("emb", 0, (Bn, L, D)).reshape(N, D), DS)
    T["emb"] = time.time() - t0

    # ------------------------------------------------------------------------ forward
    st = []
    for l in range(12):
        Wq = H.lowa_merge(W.layer(l, "attention.self.query.base_layer.weight"), lw[2 * l][0], coef[2 * l], lw[2 * l][3], GAMMA)
        Wv = H.lowa_merge(W.layer(l, "attention.self.value.base_layer.weight"), lw[2 * l + 1][0], coef[2 * l + 1], lw[2 * l + 1][3], GAMMA)
        Wk = W.layer(l, "attention.self.key.weight")
        Q = H.gemm(x, Wq, W.layer(l, "attention.self.query.base_layer.bias"))
        K = H.gemm(x, Wk, W.layer(l, "attention.self.key.bias"))
        V = H.gemm(x, Wv, W.layer(l, "attention.self.value.base_layer.bias"))
        S = np.empty((Bn, NH, L, L), f32)
        H.gemm_raw(Bn, NH, L, L, DH, Q, rowmaj, K, rowmaj, S, sq)
        S = H.scale(S, 0.125)
        ka = keep("attn", l, (Bn, NH, L, L))
        P, Pd, _, _ = H.smax_fwd(S, am, ka, DS)
        O = np.empty((N, D), f32)
        H.gemm_raw(Bn, NH, L, DH, L, Pd, sq, V, tokcol, O, rowmaj)
        y = H.gemm(O, W.layer(l, "attention.output.dense.weight"), W.layer(l, "attention.output.dense.bias"))
        z1 = H.add(H.drop(y, keep("attn_out", l, (Bn, L, D)).reshape(N, D), DS), x)
        h1, mu1, rs1 = H.ln_fwd(z1, W.layer(l, "attention.output.LayerNorm.weight"), W.layer(l, "attention.output.LayerNorm.bias"))
        u = H.gemm(h1, W.layer(l, "intermediate.dense.weight"), W.layer(l, "intermediate.dense.bias"))
        g = H.gelu(u)
        y2 = H.gemm(g, W.layer(l, "output.dense.weight"), W.layer(l, "output.dense.bias"))
        z2 = H.add(H.drop(y2, keep("ffn_out", l, (Bn, L, D)).reshape(N, D), DS), h1)
        xn, mu2, rs2 = H.ln_fwd(z2, W.layer(l, "output.LayerNorm.weight"), W.layer(l, "output.LayerNorm.bias"))
        st.append(dict(x=x, Wq=Wq, Wk=Wk, Wv=Wv, Q=Q, K=K, V=V, P=P, Pd=Pd, z1=z1, mu1=mu1, rs1=rs1,
                       u=u, z2=z2, mu2=mu2, rs2=rs2))
        x = xn
    T["fwd"] = time.time() - t0 - T["emb"]

    # ------------------------------------------------------------------- head + loss
    k1, k2 = keep("cls1", 0, (Bn, D)), keep("cls2", 0, (Bn, D))
    xc = H.drop(x.reshape(Bn, L, D)[:, 0], k1, DS)
    a = H.tanh(H.gemm(xc, head["dw"], head["db"]))
    a2 = H.drop(a, k2, DS)
    z = H.gemm(a2, head["ow"], head["ob"])
    # spec.md D3: a short batch is padded with dummy rows of loss weight 0 and the loss divides by
    # B_ACT (optional inputs loss_w / b_act; the P1 dumps have neither: all rows real)
    b_act = int(inp["b_act"]) if "b_act" in inp else Bn
    lossw = inp["loss_w"].astype(f32) if "loss_w" in inp else np.ones(Bn, f32)
    inv_b = H.recip(np.array([b_act], f32))[0]
    loss, dz = H.ce(z, labels, lossw, inv_b)
    gh = {"ow": H.gemm_tn(dz, a2), "ob": H.colsum(dz)}
    dh = H.tanh_bwd(H.drop(H.gemm_nn(dz, head["ow"]), k2, DS), a)
    gh["dw"], gh["db"] = H.gemm_tn(dh, xc), H.colsum(dh)
    dout = np.zeros((N, D), f32)
    dout.reshape(Bn, L, D)[:, 0] = H.drop(H.gemm_nn(dh, head["dw"]), k1, DS)

    # ------------------------------------------------------------------------ backward
    t1 = time.time()
    gcoef = np.zeros((24, 256), f32)
    for l in reversed(range(12)):
        c = st[l]
        dz2 = H.ln_bwd(c["z2"], c["mu2"], c["rs2"], W.layer(l, "output.LayerNorm.weight"), dout)
        dy2 = H.drop(dz2, keep("ffn_out", l, None).reshape(N, D), DS)
        du = H.gelu_bwd(H.gemm_nn(dy2, W.layer(l, "output.dense.weight")), c["u"])
        dh1 = H.add(H.gemm_nn(du, W.layer(l, "intermediate.dense.weight")), dz2)
        dz1 = H.ln_bwd(c["z1"], c["mu1"], c["rs1"], W.layer(l, "attention.output.LayerNorm.weight"), dh1)
        dyo = H.drop(dz1, keep("attn_out", l, None).reshape(N, D), DS)
        dO = H.gemm_nn(dyo, W.layer(l, "attention.output.dense.weight"))
        dPd = np.empty((Bn, NH, L, L), f32)
        H.gemm_raw(Bn, NH, L, L, DH, dO, rowmaj, c["V"], rowmaj, dPd, sq)
        dS = H.smax_bwd(c["P"], dPd, keep("attn", l, None), DS, 0.125)
        dQ, dK, dV = (np.empty((N, D), f32) for _ in range(3))
        H.gemm_raw(Bn, NH, L, DH, L, c["Pd"], sqT, dO, tokcol, dV, rowmaj)
        H.gemm_raw(Bn, NH, L, DH, L, dS, sq, c["K"], tokcol, dQ, rowmaj)
        H.gemm_raw(Bn, NH, L, DH, L, dS, sqT, c["Q"], tokcol, dK, rowmaj)
        for i, dY in ((2 * l, dQ), (2 * l + 1, dV)):
            _, rows, cols, h = lw[i]
            gcoef[i] = H.lowa_sg(dY, c["x"], rows, cols, h, GAMMA)
        if l > 0:
            dout = H.add(H.gemm_nn(np.concatenate([dQ, dK, dV], 1),
                                   np.concatenate([c["Wq"], c["Wk"], c["Wv"]], 0)), dz1)
        st[l] = None
    T["bwd"] = time.time() - t1

    # ---------------------------------------------------------------- GNORM + AdamW
    t2 = time.time()
    grads = [gcoef[i] for i in range(24)] + [gh[k] for k in HEAD]
    off = np.cumsum([0] + [g.size for g in grads])
    flat = np.concatenate([g.reshape(-1) for g in grads]).astype(f32)
    norms, G, clip = H.gnorm_clip(flat, off, [g.shape[-1] for g in grads])
    nb = int(inp["n_batches"])
    Ttot = int(inp["t_total"]) if "t_total" in inp else EPOCHS * nb      # optional: other tasks' runs
    warmup = int(inp["warmup"]) if "warmup" in inp else WARMUP
    sf = f32(s)
    if s < warmup:
        fr = H.mul(sf, f32(1.0 / warmup))[0]
    else:
        fr = H.fmax(H.mul(f32(Ttot - s), f32(1.0 / (Ttot - warmup))), f32(0.0))[0]
    b1t, b2t = f32(1.0), f32(1.0)
    for _ in range(s + 1):                               # running products held on chip
        b1t, b2t = H.mul(b1t, BETA1)[0], H.mul(b2t, BETA2)[0]
    bc1 = H.sub(f32(1.0), b1t)[0]
    rbc2 = H.recip(H.sqrt(H.sub(f32(1.0), b2t)))[0]
    lrs, out_p = [], {}
    params = [coef.reshape(-1).copy(), np.concatenate([head[k].reshape(-1) for k in HEAD])]
    ms = [inp["pre_m_coef"].reshape(-1).astype(f32).copy(),
          np.concatenate([inp["pre_m_" + k].reshape(-1) for k in HEAD]).astype(f32)]
    vs = [inp["pre_v_coef"].reshape(-1).astype(f32).copy(),
          np.concatenate([inp["pre_v_" + k].reshape(-1) for k in HEAD]).astype(f32)]
    gsplit = [flat[:off[24]], flat[off[24]:]]
    for grp in range(2):
        lr = H.mul(LR0[grp], fr)[0]
        lrs.append(lr)
        stp = H.mul(lr, H.recip(bc1))[0]
        decay = H.sub(f32(1.0), H.mul(lr, WD))[0]
        H.adamw(params[grp], np.ascontiguousarray(gsplit[grp]), ms[grp], vs[grp],
                np.array([decay, stp, rbc2, BETA1C, BETA2, BETA2C, EPS], f32))
    T["opt"] = time.time() - t2

    def unflat(tag, arrs):
        out_p[tag + "coef"] = arrs[0].reshape(24, 256)
        o = 0
        for k in HEAD:
            n = head[k].size
            out_p[tag + k] = arrs[1][o:o + n].reshape(head[k].shape)
            o += n
    unflat("post_", params)
    unflat("post_m_", ms)
    unflat("post_v_", vs)
    T["total"] = time.time() - t0
    d = {"loss": np.array(loss), "logits": z, "g_coef": gcoef, "gnorm": np.array(G), "clip": np.array(clip),
         "lr": np.array(lrs), **{"g_" + k: gh[k] for k in HEAD}, **out_p,
         "time": np.array([T[k] for k in ("emb", "fwd", "bwd", "opt", "total")])}
    if verbose:
        print(f"[L1] s={s} B={Bn} L={L} loss={float(loss):.7f} G={float(G):.6f} clip={float(clip):.7f} "
              f"lr={lrs} time " + " ".join(f"{k}={v:.1f}s" for k, v in T.items()), flush=True)
    return d


def main():
    """python -m hwmodel.step RUN_DIR A|B [KC]   (KC = reduction chunk, contract 64; 0 = P0's single
    chains, written to step{X}_l1_kc0.npz for the comparison in p1_report.md)"""
    run_dir, name = sys.argv[1], sys.argv[2]
    kc = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    H.set_kc(kc)
    W = Weights(os.path.join(run_dir, "model_fp32.npz"))
    inp = dict(np.load(os.path.join(run_dir, f"step{name}_l0f32.npz")))
    d = step(W, inp)
    np.savez(os.path.join(run_dir, f"step{name}_l1{'' if kc == 64 else f'_kc{kc}'}.npz"), **d)


if __name__ == "__main__":
    main()
