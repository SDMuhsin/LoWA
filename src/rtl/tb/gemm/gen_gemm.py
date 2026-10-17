#!/usr/bin/env python3
"""GEMM engine test cases for tb/gemm/tb_gm_top.sv.  Every expected value comes ONLY from the L1
entry points (model/hwmodel: gemm_raw, add_bias, scale, gelu, gelu_bwd, drop, add, lowa_merge,
smax_fwd, ln_fwd, ...); nothing here computes a result.  Real data: model/runs (roberta-base step B
inputs + weights, docs/p1_report.md), run through the L1 ops of model/hwmodel/step.py.

    gen_gemm.py CASE OUTDIR [--rows R --cols C]      (R x C only sets the small-config shapes)
Writes OUTDIR/<case>.{desc,mem.hex,exp.hex,mask.hex}:
  desc     "key hex" lines: the gm_top descriptor fields (addresses are byte addresses) + exp_n, mask_n
  mem.hex  $readmemh word image (@addr markers), one fp32 word per line
  exp.hex  "wordaddr value" lines: every C (and u) word the engine must write
  mask.hex one keep bit per line, indexed by the C element index (dropout chains only)
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model"))
import hwmodel as H  # noqa: E402
import rng  # noqa: E402

f32, U32 = np.float32, np.uint32
D, NH, DH, FF = 768, 12, 64, 3072
RUNS = os.path.join(HERE, "..", "..", "model", "runs")


def f2u(x):
    return np.ascontiguousarray(x, f32).view(U32)


class Mem:
    """Word-addressed memory image; tensors are placed at 32-byte aligned offsets."""

    def __init__(self):
        self.chunks = []        # (word addr, uint32 array)
        self.top = 64           # words; leave page 0 empty

    def alloc(self, nwords):
        a = self.top
        self.top = (self.top + nwords + 7) // 8 * 8
        return a

    def place(self, arr):
        arr = f2u(np.ascontiguousarray(arr, f32)).reshape(-1)
        a = self.alloc(arr.size)
        self.chunks.append((a, arr))
        return a

    def write(self, path):
        with open(path, "w") as f:
            for a, arr in self.chunks:
                f.write(f"@{a:x}\n")
                np.savetxt(f, arr, fmt="%08x")


def rand_mat(r, shape, lo=118, hi=136):
    """random normals with exponents in [lo, hi] (values ~2^-9 .. 2^9), a few zeros / tiny values."""
    n = int(np.prod(shape))
    s = r.integers(0, 2, n).astype(np.uint64) << 31
    e = r.integers(lo, hi + 1, n).astype(np.uint64) << 23
    m = r.integers(0, 1 << 23, n).astype(np.uint64)
    u = (s | e | m).astype(U32)
    z = r.integers(0, 50, n) == 0
    u[z] = r.choice(np.array([0x00000000, 0x80000000, 0x00400000, 0x0C800000], U32), int(z.sum()))
    return u.view(f32).reshape(shape)


DESC_KEYS = ["m", "n", "k0", "k1", "k2", "nb1", "nb2",
             "a_base0", "a_base1", "a_base2", "a_rs", "a_bs1", "a_bs2", "a_trn",
             "b_base0", "b_base1", "b_base2", "b_rs", "b_bs1", "b_bs2", "b_trn",
             "c_base", "c_rs", "c_bs1", "c_bs2", "bias_base", "aux_base", "o2_base",
             "ep_bias", "ep_scale", "ep_gelu", "ep_gelu_bwd", "ep_drop", "ep_resid", "ep_sexp", "ep_ds"]


class Case:
    def __init__(self, name):
        self.name = name
        self.mem = Mem()
        self.d = {k: 0 for k in DESC_KEYS}
        self.d.update(nb1=1, nb2=1, ep_ds=0x3F8E38E4)
        self.exp = []           # (word addr, uint32 array) expected writes
        self.mask = None
        self.macs = 0
        self.model = (0, 0, 0, 1)   # (M, N, K, batches) for the budget model

    def expect(self, base_words, arr, rs_words, bs1_words=0, bs2_words=0):
        """arr: [.., rows, cols] float32 at base + b1*bs1 + b2*bs2 + r*rs + c (word addresses)."""
        a = f2u(arr)
        if a.ndim == 2:
            a = a[None, None]
        if a.ndim == 3:
            a = a[:, None]
        nb1, nb2, rows, cols = a.shape
        for b1 in range(nb1):
            for b2 in range(nb2):
                for r in range(rows):
                    self.exp.append((base_words + b1 * bs1_words + b2 * bs2_words + r * rs_words, a[b1, b2, r]))

    def write(self, outdir):
        p = os.path.join(outdir, self.name)
        self.mem.write(p + ".mem.hex")
        n = 0
        with open(p + ".exp.hex", "w") as f:
            for a, row in self.exp:
                for i, v in enumerate(row):
                    f.write(f"{a + i:x} {int(v):08x}\n")
                n += len(row)
        self.d["exp_n"] = n
        self.d["mask_n"] = 0
        if self.mask is not None:
            with open(p + ".mask.hex", "w") as f:
                for v in self.mask.reshape(-1):
                    f.write(f"{int(v):x}\n")
            self.d["mask_n"] = int(self.mask.size)
        with open(p + ".desc", "w") as f:
            for k in DESC_KEYS + ["exp_n", "mask_n"]:
                f.write(f"{k} {int(self.d[k]) & 0xFFFFFFFF:x}\n")
            M, N, K, nb = self.model
            f.write(f"model_m {M:x}\nmodel_n {N:x}\nmodel_k {K:x}\nmodel_nb {nb:x}\nmacs {self.macs:x}\n")


# ----------------------------------------------------------------------------- plain GEMMs
def plain_gemm(c, r, M, N, K, a_trn=False, b_trn=False, nb1=1, nb2=1, ep=None, segs=None, seed_mask=5):
    """C[b1, b2] = A[b1, b2] @ B[b1, b2]^T (+ epilogue), A stored [M x K] (ROW) or [K x M] (TRN),
    B stored [N x K] (ROW) or [K x N] (TRN).  segs: list of K lengths (concatenated along K)."""
    segs = segs or [K]
    assert sum(segs) == K and len(segs) <= 3
    ep = ep or {}
    A = rand_mat(r, (nb1, nb2, M, K))
    B = rand_mat(r, (nb1, nb2, N, K))
    # golden: one GEMM over the full K (the contract's chunk sequence is the same for the
    # concatenation when every segment but the last is a multiple of 64)
    C = np.empty((nb1, nb2, M, N), f32)
    H.gemm_raw(nb1, nb2, M, N, K, np.ascontiguousarray(A), (nb2 * M * K, M * K, K, 1),
               np.ascontiguousarray(B), (nb2 * N * K, N * K, K, 1), C, (nb2 * M * N, M * N, N, 1))
    c.macs = nb1 * nb2 * M * N * K
    c.model = (M, N, K, nb1 * nb2)
    # operands in memory; K segments (GEMM_ACC2) are separate base addresses with the one row
    # stride of the descriptor, as the real dQ | dK | dV and Wq ; Wk ; Wv operands are: here they
    # are column (ROW mode) / row (TRN mode) ranges of one array
    Am = np.transpose(A, (0, 1, 3, 2)) if a_trn else A       # [K x M] or [M x K]
    Bm = np.transpose(B, (0, 1, 3, 2)) if b_trn else B
    a0 = c.mem.place(np.ascontiguousarray(Am))
    b0 = c.mem.place(np.ascontiguousarray(Bm))
    k0 = 0
    for si, ks in enumerate(segs):
        c.d[f"a_base{si}"] = 4 * (a0 + (k0 * M if a_trn else k0))
        c.d[f"b_base{si}"] = 4 * (b0 + (k0 * N if b_trn else k0))
        c.d[f"k{si}"] = ks
        k0 += ks
    c.d.update(m=M, n=N, nb1=nb1, nb2=nb2, a_trn=int(a_trn), b_trn=int(b_trn))
    c.d["a_rs"] = 4 * (M if a_trn else K)
    c.d["b_rs"] = 4 * (N if b_trn else K)
    c.d["a_bs1"], c.d["a_bs2"] = 4 * nb2 * M * K, 4 * M * K
    c.d["b_bs1"], c.d["b_bs2"] = 4 * nb2 * N * K, 4 * N * K
    cb = c.mem.alloc(nb1 * nb2 * M * N)
    c.d.update(c_base=4 * cb, c_rs=4 * N, c_bs1=4 * nb2 * M * N, c_bs2=4 * M * N)
    # epilogue chain (dataflow.md order: bias, scale, activation, dropout, residual)
    Y = C
    if ep.get("bias"):
        bias = rand_mat(r, (N,))
        c.d["bias_base"] = 4 * c.mem.place(bias)
        c.d["ep_bias"] = 1
        Y = H.add_bias(Y, bias)
    if ep.get("scale") is not None:
        e = ep["scale"]
        c.d["ep_scale"] = 1
        c.d["ep_sexp"] = e & 0xFF
        Y = H.scale(Y, f32(2.0 ** e))
    if ep.get("gelu"):
        c.d["ep_gelu"] = 1
        o2 = c.mem.alloc(nb1 * nb2 * M * N)
        c.d["o2_base"] = 4 * o2
        c.expect(o2, Y, N, nb2 * M * N, M * N)
        Y = H.gelu(Y)
    if ep.get("gelu_bwd"):
        c.d["ep_gelu_bwd"] = 1
        U = rand_mat(r, (nb1, nb2, M, N), 120, 130)
        c.d["aux_base"] = 4 * c.mem.place(U)
        Y = H.gelu_bwd(Y, U)
    if ep.get("drop"):
        c.d["ep_drop"] = 1
        keep = rng.keep_mask((nb1, nb2, M, N), 777, 3, "attn_out", seed_mask)
        c.mask = keep
        Y = H.drop(Y, keep, rng.DROP_SCALE)
    if ep.get("resid"):
        c.d["ep_resid"] = 1
        R = rand_mat(r, (nb1, nb2, M, N))
        c.d["aux_base"] = 4 * c.mem.place(R)
        Y = H.add(Y, R)
    c.expect(cb, Y, N, nb2 * M * N, M * N)
    return c


# ----------------------------------------------------------------------------- real data
class Real:
    """Layer-0 forward of the roberta-base step B batch, through L1 (model/hwmodel/step.py)."""

    def __init__(self):
        self.W = np.load(os.path.join(RUNS, "model_fp32.npz"))
        inp = dict(np.load(os.path.join(RUNS, "stepB_l0f32.npz")))
        self.inp = inp
        ids, am = inp["input_ids"], inp["attention_mask"].astype(np.uint8)
        self.Bn, self.L = ids.shape
        self.N = self.Bn * self.L
        self.s, self.seed = int(inp["step_s"]), int(inp["seed"])
        self.am = am
        W = self.W
        nz = (ids != 1).astype(np.int64)
        pos = np.cumsum(nz, 1) * nz + 1
        e = H.add(H.add(W["roberta.embeddings.word_embeddings.weight"][ids.reshape(-1)],
                        W["roberta.embeddings.token_type_embeddings.weight"][0]),
                  W["roberta.embeddings.position_embeddings.weight"][pos.reshape(-1)])
        x, _, _ = H.ln_fwd(e, W["roberta.embeddings.LayerNorm.weight"], W["roberta.embeddings.LayerNorm.bias"])
        self.x = H.drop(x, self.keep("emb", 0, (self.Bn, self.L, D)).reshape(self.N, D), rng.DROP_SCALE)
        self._cache = {}

    def keep(self, site, layer, shape):
        return rng.keep_mask(shape, self.seed, self.s, site, layer)

    def layer(self, name):
        return self.W[f"roberta.encoder.layer.0.{name}"]

    def merged(self, which):
        """LoWA-merged weight of layer-0 query / value, from L1 lowa_merge with the step-B coefs."""
        i = 0 if which == "query" else 1
        coef = self.inp["pre_coef"][i].astype(f32)
        eflat, h = self.W[f"lowa.{i}.eflat"].astype(np.int64), self.W[f"lowa.{i}.hval"]
        gamma = f32(self.W[f"lowa.{i}.gamma"])
        return H.lowa_merge(self.layer(f"attention.self.{which}.base_layer.weight"), eflat, coef, h, gamma)

    def attn(self):
        """Q, K, V, S, P', O of layer 0 (all L1)."""
        if "O" in self._cache:
            return self._cache
        Bn, L, N = self.Bn, self.L, self.N
        rowmaj = (L * D, DH, D, 1)
        tokcol = (L * D, DH, 1, D)
        sq = (NH * L * L, L * L, L, 1)
        Q = H.gemm(self.x, self.merged("query"), self.layer("attention.self.query.base_layer.bias"))
        K = H.gemm(self.x, self.layer("attention.self.key.weight"), self.layer("attention.self.key.bias"))
        V = H.gemm(self.x, self.merged("value"), self.layer("attention.self.value.base_layer.bias"))
        S = np.empty((Bn, NH, L, L), f32)
        H.gemm_raw(Bn, NH, L, L, DH, Q, rowmaj, K, rowmaj, S, sq)
        Ss = H.scale(S, 0.125)
        ka = self.keep("attn", 0, (Bn, NH, L, L))
        P, Pd, _, _ = H.smax_fwd(Ss, self.am, ka, rng.DROP_SCALE)
        O = np.empty((N, D), f32)
        H.gemm_raw(Bn, NH, L, DH, L, Pd, sq, V, tokcol, O, rowmaj)
        self._cache.update(Q=Q, K=K, V=V, S=S, Ss=Ss, P=P, Pd=Pd, O=O)
        return self._cache


def real_qproj(c, R):
    """Q = X Wq_merged^T + bq, 4096 x 768 x 768 (ROW x ROW, bias)."""
    Wq = R.merged("query")
    bq = R.layer("attention.self.query.base_layer.bias")
    Q = H.gemm(R.x, Wq, bq)
    M, N, K = R.N, D, D
    c.d.update(m=M, n=N, k0=K, a_base0=4 * c.mem.place(R.x), a_rs=4 * K,
               b_base0=4 * c.mem.place(Wq), b_rs=4 * K, bias_base=4 * c.mem.place(bq), ep_bias=1)
    cb = c.mem.alloc(M * N)
    c.d.update(c_base=4 * cb, c_rs=4 * N)
    c.expect(cb, Q, N)
    c.macs, c.model = M * N * K, (M, N, K, 1)
    return c


def real_scores(c, R):
    """S = (Q K^T) * 0.125 per (b, head): batched 128 x 128 x 64, ROW x ROW, scale epilogue."""
    a = R.attn()
    Bn, L = R.Bn, R.L
    c.d.update(m=L, n=L, k0=DH, nb1=Bn, nb2=NH,
               a_base0=4 * c.mem.place(a["Q"]), a_rs=4 * D, a_bs1=4 * L * D, a_bs2=4 * DH,
               b_base0=4 * c.mem.place(a["K"]), b_rs=4 * D, b_bs1=4 * L * D, b_bs2=4 * DH,
               ep_scale=1, ep_sexp=(-3) & 0xFF)
    cb = c.mem.alloc(Bn * NH * L * L)
    c.d.update(c_base=4 * cb, c_rs=4 * L, c_bs1=4 * NH * L * L, c_bs2=4 * L * L)
    c.expect(cb, a["Ss"], L, NH * L * L, L * L)
    c.macs, c.model = Bn * NH * L * L * DH, (L, L, DH, Bn * NH)
    return c


def real_pv(c, R):
    """O = P' V per head: 128 x 64 x 128, A = P' ROW, B = V TRN (the forward transposed-B mode)."""
    a = R.attn()
    Bn, L = R.Bn, R.L
    c.d.update(m=L, n=DH, k0=L, nb1=Bn, nb2=NH,
               a_base0=4 * c.mem.place(a["Pd"]), a_rs=4 * L, a_bs1=4 * NH * L * L, a_bs2=4 * L * L,
               b_base0=4 * c.mem.place(a["V"]), b_rs=4 * D, b_bs1=4 * L * D, b_bs2=4 * DH, b_trn=1)
    cb = c.mem.alloc(R.N * D)
    c.d.update(c_base=4 * cb, c_rs=4 * D, c_bs1=4 * L * D, c_bs2=4 * DH)
    O = a["O"].reshape(Bn, L, NH, DH).transpose(0, 2, 1, 3)       # [b, h, token, dh]
    c.expect(cb, np.ascontiguousarray(O), D, L * D, DH)
    c.macs, c.model = Bn * NH * L * DH * L, (L, DH, L, Bn * NH)
    return c


def real_attnout(c, R):
    """z1 = x + drop(O Wo^T + bo): 4096 x 768 x 768 with bias + dropout mask + residual."""
    a = R.attn()
    Wo, bo = R.layer("attention.output.dense.weight"), R.layer("attention.output.dense.bias")
    y = H.gemm(a["O"], Wo, bo)
    keep = R.keep("attn_out", 0, (R.Bn, R.L, D)).reshape(R.N, D)
    z1 = H.add(H.drop(y, keep, rng.DROP_SCALE), R.x)
    M, N, K = R.N, D, D
    c.d.update(m=M, n=N, k0=K, a_base0=4 * c.mem.place(a["O"]), a_rs=4 * K,
               b_base0=4 * c.mem.place(Wo), b_rs=4 * K, bias_base=4 * c.mem.place(bo),
               aux_base=4 * c.mem.place(R.x), ep_bias=1, ep_drop=1, ep_resid=1)
    c.mask = keep
    cb = c.mem.alloc(M * N)
    c.d.update(c_base=4 * cb, c_rs=4 * N)
    c.expect(cb, z1, N)
    c.macs, c.model = M * N * K, (M, N, K, 1)
    return c


def real_ffn1(c, R, rows=None):
    """u = h1 W1^T + b1, g = GELU(u): M x 3072 x 768 with bias + GELU (u stored to out2)."""
    a = R.attn()
    Wo, bo = R.layer("attention.output.dense.weight"), R.layer("attention.output.dense.bias")
    y = H.gemm(a["O"], Wo, bo)
    keep = R.keep("attn_out", 0, (R.Bn, R.L, D)).reshape(R.N, D)
    z1 = H.add(H.drop(y, keep, rng.DROP_SCALE), R.x)
    h1, _, _ = H.ln_fwd(z1, R.layer("attention.output.LayerNorm.weight"), R.layer("attention.output.LayerNorm.bias"))
    if rows:
        h1 = h1[:rows]
    W1, b1 = R.layer("intermediate.dense.weight"), R.layer("intermediate.dense.bias")
    u = H.gemm(h1, W1, b1)
    g = H.gelu(u)
    M, N, K = h1.shape[0], FF, D
    c.d.update(m=M, n=N, k0=K, a_base0=4 * c.mem.place(h1), a_rs=4 * K,
               b_base0=4 * c.mem.place(W1), b_rs=4 * K, bias_base=4 * c.mem.place(b1), ep_bias=1, ep_gelu=1)
    cb = c.mem.alloc(M * N)
    o2 = c.mem.alloc(M * N)
    c.d.update(c_base=4 * cb, c_rs=4 * N, o2_base=4 * o2)
    c.expect(cb, g, N)
    c.expect(o2, u, N)
    c.macs, c.model = M * N * K, (M, N, K, 1)
    return c


def real_dv(c, R):
    """dV = P'^T dO per head: 128 x 64 x 128 with A = P' TRN and B = dO TRN (the backward
    transposed GEMM of dataflow.md).  dO is the real attention context O (same layout as dO)."""
    a = R.attn()
    Bn, L, N = R.Bn, R.L, R.N
    dO = a["O"]
    sqT = (NH * L * L, L * L, 1, L)
    tokcol = (L * D, DH, 1, D)
    rowmaj = (L * D, DH, D, 1)
    dV = np.empty((N, D), f32)
    H.gemm_raw(Bn, NH, L, DH, L, a["Pd"], sqT, dO, tokcol, dV, rowmaj)
    c.d.update(m=L, n=DH, k0=L, nb1=Bn, nb2=NH,
               a_base0=4 * c.mem.place(a["Pd"]), a_rs=4 * L, a_bs1=4 * NH * L * L, a_bs2=4 * L * L, a_trn=1,
               b_base0=4 * c.mem.place(dO), b_rs=4 * D, b_bs1=4 * L * D, b_bs2=4 * DH, b_trn=1)
    cb = c.mem.alloc(N * D)
    c.d.update(c_base=4 * cb, c_rs=4 * D, c_bs1=4 * L * D, c_bs2=4 * DH)
    dVh = dV.reshape(Bn, L, NH, DH).transpose(0, 2, 1, 3)
    c.expect(cb, np.ascontiguousarray(dVh), D, L * D, DH)
    c.macs, c.model = Bn * NH * L * DH * L, (L, DH, L, Bn * NH)
    return c


def real_dgelu(c, R, rows=256):
    """du = (dy2 W2) * GELU'(u): rows x 3072 x 768, B = W2 TRN (dX GEMM), gelu-bwd epilogue with
    the real u of FFN1; dy2 = drop(z2-like) uses the real h1 rows as the gradient stand-in."""
    a = R.attn()
    Wo, bo = R.layer("attention.output.dense.weight"), R.layer("attention.output.dense.bias")
    y = H.gemm(a["O"], Wo, bo)
    keep = R.keep("attn_out", 0, (R.Bn, R.L, D)).reshape(R.N, D)
    z1 = H.add(H.drop(y, keep, rng.DROP_SCALE), R.x)
    h1, _, _ = H.ln_fwd(z1, R.layer("attention.output.LayerNorm.weight"), R.layer("attention.output.LayerNorm.bias"))
    h1 = h1[:rows]
    u = H.gemm(h1, R.layer("intermediate.dense.weight"), R.layer("intermediate.dense.bias"))
    W2 = R.layer("output.dense.weight")                     # [768 x 3072]
    keep2 = R.keep("ffn_out", 0, (R.Bn, R.L, D)).reshape(R.N, D)[:rows]
    dy2 = H.drop(h1, keep2, rng.DROP_SCALE)
    du = H.gelu_bwd(H.gemm_nn(dy2, W2), u)
    M, N, K = rows, FF, D
    c.d.update(m=M, n=N, k0=K, a_base0=4 * c.mem.place(dy2), a_rs=4 * K,
               b_base0=4 * c.mem.place(W2), b_rs=4 * N, b_trn=1,
               aux_base=4 * c.mem.place(u), ep_gelu_bwd=1)
    cb = c.mem.alloc(M * N)
    c.d.update(c_base=4 * cb, c_rs=4 * N)
    c.expect(cb, du, N)
    c.macs, c.model = M * N * K, (M, N, K, 1)
    return c


# ----------------------------------------------------------------------------- case table
def small_shapes(rows, cols):
    """(M, N, K) for the unit cases, scaled to the array so that partial tiles occur."""
    return dict(m1=rows, n1=cols * 4, m2=2 * rows + 5, n2=4 * cols + 7)


def build(case, rows, cols, seed=1):
    r = np.random.default_rng(seed)
    c = Case(case)
    sh = small_shapes(rows, cols)
    M1, N1, M2, N2 = sh["m1"], sh["n1"], sh["m2"], sh["n2"]
    if case.startswith("custom_"):
        # custom_M_N_K[_nb1_nb2_atrn_btrn]
        v = [int(t) for t in case.split("_")[1:]] + [1, 1, 0, 0]
        return plain_gemm(c, r, v[0], v[1], v[2], a_trn=bool(v[5]), b_trn=bool(v[6]), nb1=v[3], nb2=v[4])
    if case.startswith("chunk_k"):
        K = int(case[7:])
        return plain_gemm(c, r, M1, N1, K)
    if case == "partial":
        return plain_gemm(c, r, M2, N2, 100)
    if case == "partial_small":
        return plain_gemm(c, r, 3, 5, 70)
    if case == "batched":
        return plain_gemm(c, r, M2, N2, 70, nb1=2, nb2=3)
    if case == "mode_at":
        return plain_gemm(c, r, M2, N2, 130, a_trn=True)
    if case == "mode_bt":
        return plain_gemm(c, r, M2, N2, 130, b_trn=True)
    if case == "mode_abt":
        return plain_gemm(c, r, M2, N2, 130, a_trn=True, b_trn=True, nb1=2)
    if case == "segs":
        return plain_gemm(c, r, M2, N2, 64 + 128 + 70, segs=[64, 128, 70])
    if case == "segs2":
        return plain_gemm(c, r, M2, N2, 128 + 64, segs=[128, 64], b_trn=True)
    eps = {"ep_bias": dict(bias=True), "ep_scale": dict(scale=-3), "ep_bias_scale": dict(bias=True, scale=-3),
           "ep_gelu": dict(bias=True, gelu=True), "ep_gelu_nobias": dict(gelu=True),
           "ep_drop": dict(drop=True), "ep_resid": dict(resid=True),
           "ep_bias_drop_resid": dict(bias=True, drop=True, resid=True),
           "ep_gelu_bwd": dict(gelu_bwd=True), "ep_all": dict(bias=True, scale=1, drop=True, resid=True)}
    if case in eps:
        return plain_gemm(c, r, M2, N2, 130, ep=eps[case])
    if case == "ep_gelu_k64":
        return plain_gemm(c, r, 4 * rows, 2 * cols * 4, 64, ep=dict(bias=True, gelu=True))
    if case == "ep_gelu_bwd_k64":
        return plain_gemm(c, r, 4 * rows, 2 * cols * 4, 64, ep=dict(gelu_bwd=True))
    R = Real()
    if case == "real_qproj":
        return real_qproj(c, R)
    if case == "real_scores":
        return real_scores(c, R)
    if case == "real_pv":
        return real_pv(c, R)
    if case == "real_attnout":
        return real_attnout(c, R)
    if case == "real_ffn1":
        return real_ffn1(c, R)
    if case == "real_ffn1_256":
        return real_ffn1(c, R, rows=256)
    if case == "real_dv":
        return real_dv(c, R)
    if case == "real_dgelu":
        return real_dgelu(c, R)
    raise SystemExit(f"unknown case {case}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("outdir")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--cols", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    c = build(a.case, a.rows, a.cols, a.seed)
    c.write(a.outdir)
    print(f"{a.case}: {c.d['exp_n']} expected words, mem top {c.mem.top} words, macs {c.macs}")


if __name__ == "__main__":
    main()
