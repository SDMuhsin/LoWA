#!/usr/bin/env python3
"""Vector-unit test cases (P4): descriptor sequences, memory images and L1 expectations.
    gen_vec.py <case> <outdir> [--v V]
Writes OUTDIR/<case>.{desc,mem.hex,exp.hex,mask.hex}:
  desc     "key hex" lines (vu_top descriptor fields, byte addresses); a "run" line starts the unit;
           exp_n / mask_n / elements / model are bookkeeping for the testbench and the driver
  mem.hex  $readmemh word image, exp.hex "wordaddr value" lines, mask.hex one keep bit per line
Every expected value comes from an L1 entry point (model/hwmodel: ln_fwd, ln_bwd, smax_fwd,
smax_bwd, ce, mse, gnorm_clip, tanh, tanh_bwd, drop, add, mul, scale).  The real cases replay
model/hwmodel/step.py on the step-B batch with the entry points hooked, so that their inputs and
outputs are exactly what the L1 step computes (docs/p4_report.md section 3).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "flow", "gen"))
import hwmodel as H  # noqa: E402
import rng  # noqa: E402
from vu_ucode import VID  # noqa: E402

f32, U32 = np.float32, np.uint32
D = 768
RUNS = os.path.join(ROOT, "model", "runs")
DS = rng.DROP_SCALE
KEYS = ("vop rows cols rows_per_b emb_l st_en x_reset g_reset a_base a_rs b_base b_rs c_base c_rs cv0_base cv1_base "
        "cv2_base km_rs st0_base st1_base o2_base ids_base mk_base mk_rs c0 c1 c2 c3 ds").split()


def f2u(x):
    return np.ascontiguousarray(x, f32).view(U32)


def fb(x):
    """fp32 bit pattern of a python float / numpy scalar"""
    return int(np.array(x, f32).view(U32))


class Case:
    def __init__(self, name):
        self.name = name
        self.chunks = []            # (word addr, uint32 array)
        self.top = 64               # words
        self.exp = []               # (word addr, uint32 array)
        self.keep = []              # keep bits
        self.lines = []
        self.d = {k: 0 for k in KEYS}
        self.d.update(ds=0x3F8E38E4, rows_per_b=0xFFF0, emb_l=16)
        self.elements = 0
        self.model = 0

    def alloc(self, nwords):
        a = self.top
        self.top = (self.top + nwords + 7) // 8 * 8
        return a

    def place(self, arr, dtype=f32):
        arr = np.ascontiguousarray(arr, dtype).view(U32).reshape(-1)
        a = self.alloc(arr.size)
        self.chunks.append((a, arr))
        return a * 4                # byte address

    def expect(self, byte_addr, arr):
        self.exp.append((byte_addr // 4, np.ascontiguousarray(arr, f32).view(U32).reshape(-1)))

    def mask_alloc(self, keep):
        keep = np.ascontiguousarray(keep, np.uint8).reshape(-1)
        base = len(self.keep)
        self.keep.extend(int(v) for v in keep)
        return base

    def run(self, **fields):
        self.d.update(fields)
        for k in KEYS:
            self.lines.append(f"{k} {int(self.d[k]) & 0xFFFFFFFF:x}")
        self.lines.append("run 0")

    def write(self, outdir):
        p = os.path.join(outdir, self.name)
        with open(p + ".mem.hex", "w") as f:
            for a, arr in self.chunks:
                f.write(f"@{a:x}\n")
                np.savetxt(f, arr, fmt="%08x")
        n = 0
        with open(p + ".exp.hex", "w") as f:
            for a, row in self.exp:
                for i, v in enumerate(row):
                    f.write(f"{a + i:x} {int(v):08x}\n")
                n += len(row)
        with open(p + ".mask.hex", "w") as f:
            for v in self.keep:
                f.write(f"{v:x}\n")
        with open(p + ".desc", "w") as f:
            f.write(f"exp_n {n:x}\nmask_n {len(self.keep):x}\nelements {self.elements:x}\nmodel {self.model:x}\n")
            f.write("\n".join(self.lines) + "\n")


# ----------------------------------------------------------------------------- op emitters
def kmask_words(kmask):
    """[B, L] key mask -> [B, ceil(L/32)] uint32 bit rows (bit j of word j >> 5)"""
    Bn, L = kmask.shape
    nw = (L + 31) // 32
    w = np.zeros((Bn, nw), U32)
    for b in range(Bn):
        for j in range(L):
            if kmask[b, j]:
                w[b, j >> 5] |= U32(1) << U32(j & 31)
    return w


def op_ln_fwd(c, x, g, b, st=True):
    rows, d = x.shape
    y, mu, rs = H.ln_fwd(x, g, b)
    xa, ga, ba = c.place(x), c.place(g), c.place(b)
    ya = c.alloc(rows * d) * 4
    m0, m1 = c.alloc(rows) * 4, c.alloc(rows) * 4
    c.run(vop=VID["LN_FWD"], rows=rows, cols=d, a_base=xa, a_rs=d * 4, c_base=ya, c_rs=d * 4, cv1_base=ga, cv2_base=ba,
          st0_base=m0, st1_base=m1, st_en=int(st), c0=fb(1.0 / d), c1=fb(1e-5))
    c.expect(ya, y)
    if st:
        c.expect(m0, mu)
        c.expect(m1, rs)
    c.elements += 3 * rows * d
    return y, mu, rs


def op_ln_bwd(c, z, mu, rs, g, dy):
    rows, d = z.shape
    dx = H.ln_bwd(z, mu, rs, g, dy)
    za, dya, ga, m0, m1 = c.place(z), c.place(dy), c.place(g), c.place(mu), c.place(rs)
    oa = c.alloc(rows * d) * 4
    c.run(vop=VID["LN_BWD"], rows=rows, cols=d, a_base=za, a_rs=d * 4, b_base=dya, b_rs=d * 4, c_base=oa, c_rs=d * 4,
          cv0_base=ga, st0_base=m0, st1_base=m1, st_en=0, c0=fb(1.0 / d))
    c.expect(oa, dx)
    c.elements += 2 * rows * d
    return dx


def op_smax_fwd(c, S, kmask, keep, st=True):
    Bn, Hn, L, _ = S.shape
    P, Pd, m, r = H.smax_fwd(S, kmask, keep, DS)
    rows = Bn * Hn * L
    sa = c.place(S)
    kw = kmask_words(kmask)
    ka = c.place(kw, U32)
    mb = c.mask_alloc(keep)
    oa = c.alloc(rows * L) * 4
    m0, m1 = c.alloc(rows) * 4, c.alloc(rows) * 4
    c.run(vop=VID["SMAX_FWD"], rows=rows, cols=L, rows_per_b=Hn * L, a_base=sa, a_rs=L * 4, c_base=oa, c_rs=L * 4,
          cv0_base=ka, km_rs=kw.shape[1] * 4, st0_base=m0, st1_base=m1, st_en=int(st), mk_base=mb, mk_rs=L)
    c.expect(oa, Pd)
    if st:
        c.expect(m0, m)
        c.expect(m1, r)
    c.elements += 3 * rows * L
    return P, Pd, m, r


def op_smax_rec(c, S, kmask, m, r, P):
    Bn, Hn, L, _ = S.shape
    rows = Bn * Hn * L
    sa = c.place(S)
    kw = kmask_words(kmask)
    ka = c.place(kw, U32)
    m0, m1 = c.place(m), c.place(r)
    oa = c.alloc(rows * L) * 4
    c.run(vop=VID["SMAX_REC"], rows=rows, cols=L, rows_per_b=Hn * L, a_base=sa, a_rs=L * 4, c_base=oa, c_rs=L * 4,
          cv0_base=ka, km_rs=kw.shape[1] * 4, st0_base=m0, st1_base=m1, st_en=0)
    c.expect(oa, P)
    c.elements += 2 * rows * L


def op_smax_bwd(c, P, dPd, keep):
    Bn, Hn, L, _ = P.shape
    rows = Bn * Hn * L
    dS = H.smax_bwd(P, dPd, keep, DS, 0.125)
    pa, da = c.place(P), c.place(dPd)
    mb = c.mask_alloc(keep)
    oa = c.alloc(rows * L) * 4
    c.run(vop=VID["SMAX_BWD"], rows=rows, cols=L, a_base=pa, a_rs=L * 4, b_base=da, b_rs=L * 4, c_base=oa, c_rs=L * 4,
          mk_base=mb, mk_rs=L, c0=fb(0.125), st_en=0)
    c.expect(oa, dS)
    c.elements += 2 * rows * L
    return dS


def op_emb(c, ids, word, pos_tab, tt0, g, b, keep, L):
    """ids [B, L] int; word [vocab, d]; pos_tab [npos, d]; keep [B*L, d]"""
    Bn, Lq = ids.shape
    assert Lq == L
    d = word.shape[1]
    N = Bn * L
    nz = (ids != 1).astype(np.int64)
    pos = np.cumsum(nz, 1) * nz + 1
    e = H.add(H.add(word[ids.reshape(-1)], tt0), pos_tab[pos.reshape(-1)])
    y, _, _ = H.ln_fwd(e, g, b)
    y = H.drop(y, keep, DS)
    wa, pa, ta, ga, ba = c.place(word), c.place(pos_tab), c.place(tt0), c.place(g), c.place(b)
    ia = c.place(ids.reshape(-1).astype(np.int32), np.int32)
    mb = c.mask_alloc(keep)
    oa = c.alloc(N * d) * 4
    c.run(vop=VID["EMB"], rows=N, cols=d, emb_l=L, a_base=wa, a_rs=d * 4, b_base=pa, b_rs=d * 4, c_base=oa, c_rs=d * 4,
          cv0_base=ta, cv1_base=ga, cv2_base=ba, ids_base=ia, mk_base=mb, mk_rs=d, c0=fb(1.0 / d), c1=fb(1e-5), st_en=0)
    c.expect(oa, y)
    c.elements += 3 * N * d
    return y


def op_ce(c, z, y, w, b_act):
    Bn, C = z.shape
    inv_b = H.recip(np.array([b_act], f32))[0]
    loss, dz = H.ce(z, y, w, inv_b)
    za = c.place(z)
    la = c.place(np.asarray(y, np.int32), np.int32)
    wa = c.place(w)
    oa = c.alloc(Bn * C) * 4
    o2 = c.alloc(8) * 4
    c.run(vop=VID["CE"], rows=Bn, cols=C, a_base=za, a_rs=C * 4, c_base=oa, c_rs=C * 4, st0_base=la, st1_base=wa,
          o2_base=o2, c0=fb(inv_b), x_reset=1, g_reset=0, st_en=0)
    c.expect(oa, dz)
    c.expect(o2, np.array([loss], f32))
    c.elements += 3 * Bn * C + Bn
    return loss, dz


def op_mse(c, z, yt, w, b_act):
    Bn = z.shape[0]
    inv_b = H.recip(np.array([b_act], f32))[0]
    inv2b = H.mul(f32(2.0), inv_b)[0]
    loss, dz = H.mse(z, yt, w, inv_b)
    za, ya, wa = c.place(z.reshape(-1, 1)), c.place(yt.reshape(-1)), c.place(w)
    oa = c.alloc(Bn) * 4
    o2 = c.alloc(8) * 4
    c.run(vop=VID["MSE"], rows=Bn, cols=1, a_base=za, a_rs=4, c_base=oa, c_rs=4, st0_base=ya, st1_base=wa, o2_base=o2,
          c0=fb(inv_b), c1=fb(inv2b), x_reset=1, st_en=0)
    c.expect(oa, dz)
    c.expect(o2, np.array([loss], f32))
    c.elements += Bn * 2
    return loss, dz


def op_gnorm(c, grads):
    """grads: list of 2-D arrays (rows x rowlen).  GN_ROWS per tensor, GN_FIN, E_SCALER per tensor."""
    off = np.cumsum([0] + [g.size for g in grads])
    flat = np.concatenate([g.reshape(-1) for g in grads]).astype(f32)
    norms, G, cl = H.gnorm_clip(flat, off, [g.shape[-1] for g in grads])
    addrs = [c.place(g) for g in grads]
    na = c.alloc(8 * len(grads)) * 4
    for i, g in enumerate(grads):
        c.run(vop=VID["GN_ROWS"], rows=g.shape[0], cols=g.shape[1], a_base=addrs[i], a_rs=g.shape[1] * 4, o2_base=na + 32 * i,
              x_reset=1, g_reset=int(i == 0), st_en=0)
        c.expect(na + 32 * i, np.array([norms[i]], f32))
        c.elements += g.size
    o2 = c.alloc(8) * 4
    c.run(vop=VID["GN_FIN"], rows=0, cols=0, o2_base=o2, c0=fb(1e-6), x_reset=0, g_reset=0)
    c.expect(o2, np.array([G, cl], f32))
    for i, g in enumerate(grads):
        oa = c.alloc(g.size) * 4
        c.run(vop=VID["E_SCALER"], rows=g.shape[0], cols=g.shape[1], a_base=addrs[i], a_rs=g.shape[1] * 4, c_base=oa,
              c_rs=g.shape[1] * 4, x_reset=0, g_reset=0)
        c.expect(oa, flat[off[i]:off[i + 1]].reshape(g.shape))
        c.elements += g.size
    return norms, G, cl


def op_elem(c, sub, a, b=None, keep=None, scalar=None):
    rows, cols = a.shape
    if sub == "E_ADD":
        y = H.add(a, b)
    elif sub == "E_MUL":
        y = H.mul(a, b)
    elif sub == "E_SCALE":
        y = H.mul(a, f32(scalar))
    elif sub == "E_DROP":
        y = H.drop(a, keep, DS)
    elif sub == "E_TANH":
        y = H.tanh(a)
    elif sub == "E_TANHB":
        y = H.tanh_bwd(b, a)
    else:
        y = a.copy()
    aa = c.place(a)
    ba = c.place(b) if b is not None else 0
    mb = c.mask_alloc(keep) if keep is not None else 0
    oa = c.alloc(rows * cols) * 4
    c.run(vop=VID[sub], rows=rows, cols=cols, a_base=aa, a_rs=cols * 4, b_base=ba, b_rs=cols * 4, c_base=oa, c_rs=cols * 4,
          mk_base=mb, mk_rs=cols, c0=fb(scalar if scalar is not None else 0.0), st_en=0)
    c.expect(oa, y)
    c.elements += rows * cols
    return y


# ----------------------------------------------------------------------------- random data
def rnd(r, *shape, lo=-2.0, hi=2.0):
    return r.uniform(lo, hi, shape).astype(f32)


def corner_rows(r, rows, cols):
    """random rows plus directed ones: identical values, subnormal / FTZ, max magnitude, zeros, +-"""
    x = rnd(r, rows, cols)
    if rows > 1:
        x[1, :] = f32(0.75)
    if rows > 2:
        x[2, :] = np.uint32(0x00400001).view(f32) * np.where(r.uniform(size=cols) > 0.5, 1, -1)   # subnormal -> DAZ
    if rows > 3:
        x[3, :] = rnd(r, cols, lo=-3e38, hi=3e38)
    if rows > 4:
        x[4, :] = 0.0
        x[4, ::2] = -0.0
    if rows > 5:
        x[5, :] = rnd(r, cols, lo=-1e-20, hi=1e-20)
    return x


# ----------------------------------------------------------------------------- real data
class Real:
    """One L1 step on the step-B batch with the per-op entry points hooked (inputs / outputs recorded)."""

    def __init__(self):
        cache = os.path.join(ROOT, "build", "vec", "real_cache.npz")
        if os.path.exists(cache):
            self.z = dict(np.load(cache, allow_pickle=True))
            return
        import hwmodel.step as S
        rec = {}

        def hook(name, nmax=3):
            orig = getattr(H, name)

            def w(*a, **k):
                a_rec = tuple(np.array(x, copy=True) if isinstance(x, np.ndarray) else x for x in a)   # gnorm_clip is in place
                out = orig(*a, **k)
                n = sum(1 for key in rec if key.startswith(name + "."))
                if n < nmax:
                    rec[f"{name}.{n}"] = (a_rec, out)
                return out
            setattr(H, name, w)
            return orig
        origs = {n: hook(n) for n in ("ln_fwd", "ln_bwd", "smax_fwd", "smax_bwd", "ce", "gnorm_clip", "tanh", "tanh_bwd", "drop")}
        W = S.Weights(os.path.join(RUNS, "model_fp32.npz"))
        inp = dict(np.load(os.path.join(RUNS, "stepB_l0f32.npz")))
        S.step(W, inp, verbose=False)
        for n, o in origs.items():
            setattr(H, n, o)
        z = {}
        for key, (a, out) in rec.items():
            for i, v in enumerate(a):
                if v is not None:
                    z[f"{key}.in{i}"] = np.asarray(v)
            outs = out if isinstance(out, tuple) else (out,)
            for i, v in enumerate(outs):
                z[f"{key}.out{i}"] = np.asarray(v)
        ids, am = inp["input_ids"], inp["attention_mask"].astype(np.uint8)
        z["ids"], z["am"] = ids, am
        z["seed"], z["step_s"] = np.array(int(inp["seed"])), np.array(int(inp["step_s"]))
        z["labels"] = inp["labels"]
        for k in ("roberta.embeddings.word_embeddings.weight", "roberta.embeddings.position_embeddings.weight",
                  "roberta.embeddings.token_type_embeddings.weight", "roberta.embeddings.LayerNorm.weight",
                  "roberta.embeddings.LayerNorm.bias"):
            z[k] = W[k]
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, **z)
        self.z = z

    def keep(self, site, layer, shape):
        return rng.keep_mask(shape, int(self.z["seed"]), int(self.z["step_s"]), site, layer)


# ----------------------------------------------------------------------------- cases
def build(case, outdir, v):
    r = np.random.default_rng(abs(hash(case)) % (2 ** 31))
    c = Case(case)
    if case == "ln_fwd":
        for rows, cols in ((37, 200), (16, 64), (20, 65), (7, 1), (3, 4096), (17, 768), (1, 128)):
            op_ln_fwd(c, corner_rows(r, rows, cols), rnd(r, cols), rnd(r, cols), st=(rows != 16))
    elif case == "ln_bwd":
        for rows, cols in ((37, 200), (16, 64), (20, 65), (7, 1), (3, 4096), (17, 768)):
            x = corner_rows(r, rows, cols)
            g, b = rnd(r, cols), rnd(r, cols)
            _, mu, rs = H.ln_fwd(x, g, b)
            op_ln_bwd(c, x, mu, rs, g, corner_rows(r, rows, cols))
    elif case == "smax":
        for Bn, Hn, L in ((2, 3, 48), (1, 1, 128), (1, 2, 1), (3, 1, 16), (1, 1, 4), (1, 1, 256)):
            S = rnd(r, Bn, Hn, L, L, lo=-30, hi=30)
            if Hn * L >= 3:
                S.reshape(-1, L)[1, :] = 0.5                                 # identical values
                S.reshape(-1, L)[2, :] = rnd(r, L, lo=-3e38, hi=3e38)         # max-magnitude logits
            kmask = (r.uniform(size=(Bn, L)) > 0.2).astype(np.uint8)
            if Bn > 1:
                kmask[1, :] = 0                                              # fully masked rows
            kmask[0, 0] = 1
            keep = (r.uniform(size=(Bn, Hn, L, L)) > 0.1).astype(np.uint8)
            P, Pd, m, rr = op_smax_fwd(c, S, kmask, keep, st=True)
            op_smax_rec(c, S, kmask, m, rr, P)
            op_smax_bwd(c, P, rnd(r, Bn, Hn, L, L), keep)
    elif case == "emb":
        Bn, L, d, vocab, npos = 3, 16, 200, 40, 20
        ids = r.integers(2, vocab, (Bn, L))
        ids[0, 10:] = 1
        ids[2, 3:] = 1
        keep = (r.uniform(size=(Bn * L, d)) > 0.1).astype(np.uint8)
        op_emb(c, ids, rnd(r, vocab, d), rnd(r, npos, d), rnd(r, d), rnd(r, d), rnd(r, d), keep, L)
        Bn, L, d = 2, 32, 768
        ids = r.integers(2, vocab, (Bn, L))
        ids[1, 20:] = 1
        keep = (r.uniform(size=(Bn * L, d)) > 0.1).astype(np.uint8)
        op_emb(c, ids, rnd(r, vocab, d), rnd(r, L + 2, d), rnd(r, d), rnd(r, d), rnd(r, d), keep, L)
    elif case == "loss":
        for Bn, C in ((32, 3), (32, 2), (5, 16), (1, 2), (17, 3)):
            z = rnd(r, Bn, C, lo=-5, hi=5)
            if Bn > 2:
                z[1, :] = 0.3
                z[2, :] = rnd(r, C, lo=-3e38, hi=3e38)
            y = r.integers(0, C, Bn)
            w = np.ones(Bn, f32)
            b_act = Bn
            if Bn > 4:
                w[-2:] = 0
                b_act = Bn - 2
            op_ce(c, z, y, w, b_act)
        for Bn in (32, 5, 1):
            z = rnd(r, Bn, 1, lo=-3, hi=3)
            w = np.ones(Bn, f32)
            op_mse(c, z, rnd(r, Bn), w, Bn)
    elif case == "gnorm":
        shapes = [(1, 256), (1, 256), (1, 256), (70, 200), (1, 77), (3, 77), (1, 3), (2, 4096), (65, 64)]
        grads = [rnd(r, *s, lo=-0.1, hi=0.1) for s in shapes]
        grads[4][0, :] = 0.0
        op_gnorm(c, grads)
    elif case == "elem":
        a, b = corner_rows(r, 20, 50), rnd(r, 20, 50)
        keep = (r.uniform(size=(20, 50)) > 0.1).astype(np.uint8)
        op_elem(c, "E_ADD", a, b)
        op_elem(c, "E_MUL", a, b)
        op_elem(c, "E_SCALE", a, scalar=0.3)
        op_elem(c, "E_DROP", a, keep=keep)
        th = op_elem(c, "E_TANH", a)
        op_elem(c, "E_TANHB", th, b)
        op_elem(c, "E_COPY", a)
        op_elem(c, "E_ADD", rnd(r, 1, 4096), rnd(r, 1, 4096))
        op_elem(c, "E_TANH", rnd(r, 7, 1, lo=-12, hi=12))
        op_elem(c, "E_DROP", rnd(r, 3, 777), keep=(r.uniform(size=(3, 777)) > 0.5).astype(np.uint8))
    elif case == "elem_drop":
        a = rnd(r, 4, 20)
        op_elem(c, "E_DROP", a, keep=(r.uniform(size=(4, 20)) > 0.3).astype(np.uint8))
    elif case == "elem_drop2":
        a = corner_rows(r, 20, 50)
        op_elem(c, "E_SCALE", a, scalar=0.3)
        op_elem(c, "E_DROP", a, keep=(r.uniform(size=(20, 50)) > 0.1).astype(np.uint8))
    elif case == "ce_mid":
        z = rnd(r, 32, 3, lo=-5, hi=5)
        op_ce(c, z, r.integers(0, 3, 32), np.ones(32, f32), 32)
    elif case in ("smax_small", "ln_small"):
        if case == "smax_small":
            Bn, Hn, L = 1, 1, 48
            S = rnd(r, Bn, Hn, L, L, lo=-30, hi=30)
            kmask = (r.uniform(size=(Bn, L)) > 0.2).astype(np.uint8)
            keep = (r.uniform(size=(Bn, Hn, L, L)) > 0.1).astype(np.uint8)
            P, Pd, m, rr = op_smax_fwd(c, S, kmask, keep, st=True)
            op_smax_bwd(c, P, rnd(r, Bn, Hn, L, L), keep)
        else:
            x = corner_rows(r, 6, 70)
            g, b = rnd(r, 70), rnd(r, 70)
            _, mu, rs = op_ln_fwd(c, x, g, b, st=True)
            op_ln_bwd(c, x, mu, rs, g, rnd(r, 6, 70))
    elif case == "ce_corner":
        z = rnd(r, 5, 3, lo=-5, hi=5)
        z[1, :] = 0.3
        z[2, :] = rnd(r, 3, lo=-3e38, hi=3e38)
        op_ce(c, z, r.integers(0, 3, 5), np.ones(5, f32), 5)
    elif case == "ce_w":
        z = rnd(r, 5, 3, lo=-5, hi=5)
        w = np.ones(5, f32)
        w[-2:] = 0
        op_ce(c, z, r.integers(0, 3, 5), w, 3)
    elif case == "smax_tiny":
        S = rnd(r, 1, 1, 4, 4, lo=-3, hi=3)
        kmask = np.array([[1, 1, 0, 1]], np.uint8)
        keep = np.ones((1, 1, 4, 4), np.uint8)
        keep[0, 0, 1, 2] = 0
        P, Pd, m, rr = op_smax_fwd(c, S, kmask, keep, st=True)
    elif case.startswith("smax_l"):
        L = int(case[6:])
        S = rnd(r, 1, 1, L, L, lo=-30, hi=30)
        kmask = (r.uniform(size=(1, L)) > 0.2).astype(np.uint8)
        kmask[0, 0] = 1
        keep = (r.uniform(size=(1, 1, L, L)) > 0.1).astype(np.uint8)
        op_smax_fwd(c, S, kmask, keep, st=True)
    elif case.startswith("ce_rows"):
        Bn = int(case[7:])
        z = rnd(r, Bn, 3, lo=-5, hi=5)
        z[1, :] = 0.3
        z[2, :] = rnd(r, 3, lo=-3e38, hi=3e38)
        op_ce(c, z, r.integers(0, 3, Bn), np.ones(Bn, f32), Bn)
    elif case == "elem_big":
        op_elem(c, "E_ADD", rnd(r, 1024, 768), rnd(r, 1024, 768))
        op_elem(c, "E_SCALE", rnd(r, 1024, 768), scalar=0.5)
    elif case == "ce_small":
        z = rnd(r, 4, 3, lo=-2, hi=2)
        op_ce(c, z, np.array([0, 2, 1, 1]), np.ones(4, f32), 4)
    elif case.startswith("real_"):
        R = Real()
        z = R.z
        if case == "real_smax":
            S, km, keep, ds = z["smax_fwd.0.in0"], z["smax_fwd.0.in1"], z["smax_fwd.0.in2"], z["smax_fwd.0.in3"]
            assert float(ds) == float(DS)
            P, Pd, m, rr = op_smax_fwd(c, S, km, keep, st=True)
            assert np.array_equal(f2u(Pd), f2u(z["smax_fwd.0.out1"]))
        elif case == "real_smax_bwd":
            P, dPd, keep = z["smax_bwd.0.in0"], z["smax_bwd.0.in1"], z["smax_bwd.0.in2"]
            dS = op_smax_bwd(c, P, dPd, keep)
            assert np.array_equal(f2u(dS), f2u(z["smax_bwd.0.out0"]))
        elif case == "real_ln_fwd":
            x, g, b = z["ln_fwd.1.in0"], z["ln_fwd.1.in1"], z["ln_fwd.1.in2"]          # LN1 of layer 0
            y, _, _ = op_ln_fwd(c, x, g, b, st=True)
            assert np.array_equal(f2u(y), f2u(z["ln_fwd.1.out0"]))
        elif case == "real_ln_bwd":
            zz, mu, rs, g, dy = (z[f"ln_bwd.0.in{i}"] for i in range(5))                 # LN2 of layer 11
            dx = op_ln_bwd(c, zz, mu, rs, g, dy)
            assert np.array_equal(f2u(dx), f2u(z["ln_bwd.0.out0"]))
        elif case == "real_emb":
            ids = z["ids"]
            Bn, L = ids.shape
            uniq, inv = np.unique(ids.reshape(-1), return_inverse=True)
            if 1 not in uniq:                                     # keep the pad id at index 1
                uniq = np.sort(np.append(uniq, 1))
                inv = np.searchsorted(uniq, ids.reshape(-1))
            # compact vocabulary: row i = word table row uniq[i]; the pad id must map to id 1
            order = list(uniq)
            pad_i = order.index(1)
            order[1], order[pad_i] = order[pad_i], order[1]
            remap = {int(w): i for i, w in enumerate(order)}
            cids = np.vectorize(lambda t: remap[int(t)])(ids).astype(np.int64)
            word = z["roberta.embeddings.word_embeddings.weight"][np.array(order)]
            keep = R.keep("emb", 0, (Bn, L, D)).reshape(Bn * L, D)
            y = op_emb(c, cids, word, z["roberta.embeddings.position_embeddings.weight"],
                       z["roberta.embeddings.token_type_embeddings.weight"][0], z["roberta.embeddings.LayerNorm.weight"],
                       z["roberta.embeddings.LayerNorm.bias"], keep, L)
            assert np.array_equal(f2u(y), f2u(z["drop.0.out0"]))
        elif case == "real_loss":
            zl, y, w, inv_b = z["ce.0.in0"], z["ce.0.in1"], z["ce.0.in2"], z["ce.0.in3"]
            loss, dz = op_ce(c, zl, y, w, zl.shape[0])
            assert float(inv_b) == float(H.recip(np.array([zl.shape[0]], f32))[0])
            assert np.array_equal(f2u(dz), f2u(z["ce.0.out1"])) and fb(loss) == fb(z["ce.0.out0"])
            # head tanh and its backward on the real pre-activation
            a = z["tanh.0.in0"]
            th = op_elem(c, "E_TANH", a)
            assert np.array_equal(f2u(th), f2u(z["tanh.0.out0"]))
            dh = op_elem(c, "E_TANHB", z["tanh_bwd.0.in1"], z["tanh_bwd.0.in0"])
            assert np.array_equal(f2u(dh), f2u(z["tanh_bwd.0.out0"]))
        elif case == "real_gnorm":
            flat, off, rowlen = z["gnorm_clip.0.in0"], z["gnorm_clip.0.in1"], z["gnorm_clip.0.in2"]
            grads = [flat[off[i]:off[i + 1]].reshape(-1, int(rowlen[i])) for i in range(len(rowlen))]
            norms, G, cl = op_gnorm(c, grads)
            assert np.array_equal(f2u(norms), f2u(z["gnorm_clip.0.out0"])) and fb(cl) == fb(z["gnorm_clip.0.out2"])
        else:
            raise SystemExit(f"unknown case {case}")
    else:
        raise SystemExit(f"unknown case {case}")
    c.write(outdir)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("outdir")
    ap.add_argument("--v", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    c = build(a.case, a.outdir, a.v)
    print(f"{a.case}: {len(c.exp)} expected blocks, {c.elements} elements, {c.top} words")


if __name__ == "__main__":
    main()
