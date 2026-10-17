#!/usr/bin/env python3
"""LoWA-TA compiler (P6c): model config -> (a) the descriptor program of one full training step (docs/isa.md),
(b) the DRAM memory image (docs/memory_map.md).

  python3 sw/compile.py [--layers 12] [--batch 32] [--seq 128] [--task ce|mse] [--out build/compile]

Config: roberta-base (d 768, 12 heads, FFN 3072), LoWA on query/value (24 modules, k 256, s 25), PI ruling R6 HPs
(gamma 0.552427, lr 0.166667 / classifier 5e-3, wd 0.01, warmup 140, 30 epochs, sampled grad).  Data / state:
model/runs/model_fp32.npz (the real checkpoint as dumped by model/l0_ref.py's loader) and stepB_l0f32.npz (batch,
pre-step coefficients / head / Adam state, s, seed).  --layers / --batch / --seq < full give the reduced,
structurally complete program used for the RTL run (first B sequences, first L tokens, layers 0..n-1).

Program (one descriptor list, every op's operands placed by this file):
  SETCSR STR1..4 ; EMB ; DMA_COPY X0 -> STASH_X0 ; LOWA_SCAL ; LOOP up { 11 forward ops } ; head fwd ; LOSS ;
  head bwd ; LOOP down (l = n-1 .. 1) { 19 backward ops } ; layer-0 backward ; GNORM x 29 ; DMA_COPY clip ;
  ADAMW coef ; ADAMW head ; END
Cross-unit ordering: FENCE descriptors placed by `place_fences` from the read / write byte ranges of every op
(dependency on an unfenced earlier op of another unit -> FENCE that unit first); flags (WAIT / SIG) are not
needed by this schedule.  The LoWA sampled gradient of layer l therefore runs beside the dX GEMM of layer l and
the next layer's vector / GEMM ops until a consumer of its output (GNORM) or a writer of its inputs appears.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "docs", "calc"))
sys.path.insert(0, os.path.join(ROOT, "flow", "gen"))
import isa  # noqa: E402
import lowa_tables as LT  # noqa: E402
import memmap  # noqa: E402

f32, U32 = np.float32, np.uint32
D, NH, DH, FF = 768, 12, 64, 3072
GAMMA = f32(0.552427)
LR0 = (f32(0.166667), f32(5e-3))
WD, B1, B2, B1C, B2C, EPS = f32(0.01), f32(0.9), f32(0.999), f32(0.1), f32(0.001), f32(1e-8)
WARMUP, EPOCHS = 140, 30
DS_BITS = 0x3F8E38E4
RUNS = os.path.join(ROOT, "model", "runs")

REG = {nm: base for nm, base, _, _, _ in memmap.layout(3)}
REG["STEP_IN1"] = 0x7C9C_A000          # P6c: second STEP_IN slot (int32 ids do not fit 2 slots in STEP_IN)
REG["LOWA_DIR"] = 0x7C9D_0000          # P6c: merge-hit block directories, 24 x 1 KiB
SLOT_BYTES = 24576
IMG_END = REG["LOWA_DIR"] + 24 * 1024
LAYER_OFF, _o = {}, 0
for _nm, _n in (("Wq", D * D), ("bq", D), ("Wk", D * D), ("bk", D), ("Wv", D * D), ("bv", D), ("Wo", D * D), ("bo", D),
                ("LN1", 2 * D), ("W1", FF * D), ("b1", FF), ("W2", D * FF), ("b2", D), ("LN2", 2 * D)):
    LAYER_OFF[_nm] = _o
    _o += 4 * _n
LAYER_STRIDE = _o                       # 28,351,488
HF_NAME = {"Wq": "attention.self.query.base_layer.weight", "bq": "attention.self.query.base_layer.bias",
           "Wk": "attention.self.key.weight", "bk": "attention.self.key.bias",
           "Wv": "attention.self.value.base_layer.weight", "bv": "attention.self.value.base_layer.bias",
           "Wo": "attention.output.dense.weight", "bo": "attention.output.dense.bias",
           "W1": "intermediate.dense.weight", "b1": "intermediate.dense.bias",
           "W2": "output.dense.weight", "b2": "output.dense.bias"}
LN_NAME = {"LN1": "attention.output.LayerNorm", "LN2": "output.LayerNorm"}
# CONST word offsets
C_GAMMA, C_H, C_SCAL, C_ONES, C_ZERO = 0, 1, 32, 64, 128
# OPT_SCAL word offsets
O_GFIN, O_NORMS = 32, 64
# term selectors
S_LAYER, S_STASH, S_MOD, S_HITS = 1, 2, 3, 4
HITS_BYTES, MOD_BYTES = 6400 * 4, 1024
HEAD_KEYS = ("dw", "db", "ow", "ob")


def fb(x):
    return int(np.array([x], f32).view(U32)[0])


class Cfg:
    def __init__(self, layers=12, B=32, L=128, task="ce", C=2):
        self.layers, self.B, self.L, self.task, self.C = layers, B, L, task, C
        self.N = B * L
        N = self.N
        # stash per layer (bytes): z1 z2 Q K V [N x D], u [N x FF], mu1 rs1 mu2 rs2 [N], m r [B*12*L]
        self.st_off, o = {}, 0
        for nm, n in (("z1", N * D), ("z2", N * D), ("Q", N * D), ("K", N * D), ("V", N * D), ("u", N * FF),
                      ("mu1", N), ("rs1", N), ("mu2", N), ("rs2", N), ("sm", B * NH * L), ("sr", B * NH * L)):
            self.st_off[nm] = o
            o += 4 * n
        self.stash_stride = 113_704_960            # memory_map.md: 12 x 113.7 MB (full size)
        assert o <= self.stash_stride
        # SCR_ACT sub-buffers (units of N words)
        att = NH * L                               # attention buffer = B*12*L*L words = N * 12 L
        so = {"X": 0, "H1": 768, "O": 1536, "G": 2304, "A0": 5376, "A1": 5376 + att, "A2": 5376 + 2 * att,
              "S1": 0, "S2": 768, "DZ1": 1536, "DU": 2304}
        top = 5376 + 3 * att
        so.update(DOB=top, DKB=top + 768, DOUT=top + 1536)
        assert top + 2304 <= 4 * FF, "SCR_ACT overflow"
        self.scr = {k: REG["SCR_ACT"] + 4 * N * v for k, v in so.items()}
        cls = {}
        for k, i in (("xc", 0), ("pre", 1), ("a", 2), ("a2", 3), ("dh0", 1), ("dh1", 3), ("dh", 4), ("dc", 5),
                     ("z", 6), ("dz", 7)):   # dh0 / dh1 reuse pre / a2 (dead by then; the fences order it)
            cls[k] = REG["SCR_CLS"] + i * 4 * B * D
        assert 8 * 4 * B * D <= 786432
        self.cls = cls
        # STEP_IN slot layout (bytes): ids int32 [N], kmask words [B x ceil(L/32)], labels [B], weights [B]
        self.kmw = (L + 31) // 32
        self.si = {"ids": 0, "km": 4 * N, "lab": 4 * N + 4 * B * self.kmw}
        self.si["w"] = self.si["lab"] + 4 * B
        assert self.si["w"] + 4 * B <= SLOT_BYTES
        self.head_n = D * D + D + C * D + C

    def lay(self, l, nm):
        return REG["LAYERS"] + l * LAYER_STRIDE + LAYER_OFF[nm]

    def st(self, l, nm):
        return REG["STASH"] + l * self.stash_stride + self.st_off[nm]

    def head(self, k):
        o = {"dw": 0, "db": D * D, "ow": D * D + D, "ob": D * D + D + self.C * D}[k]
        return 4 * (6144 + o)


# ---------------------------------------------------------------------------------------------- program
class T:
    """a field value that is layer- / step- / slot-relative: v + TERM[sel] (sel 1..12 = layer terms)."""
    def __init__(self, v, sel):
        self.v, self.sel = int(v), sel


class Op:
    def __init__(self, name, f):
        self.name, self.f = name, f


class Loop:
    def __init__(self, body_fn, layers, down):
        self.body_fn, self.layers, self.down = body_fn, list(layers), down


def gemm(M, N, K, a, a_rs, b, b_rs, c, c_rs, a_trn=0, b_trn=0, nb1=1, nb2=1, a_bs=(0, 0), b_bs=(0, 0), c_bs=(0, 0),
         bias=0, aux=0, o2=0, scale=None, gelu=0, gelu_bwd=0, drop=None, resid=0, segs=None, merge=None, acc2=False):
    """a / b: base address or a list of segment bases (GEMM_ACC2); segs = K lengths."""
    a = a if isinstance(a, list) else [a]
    b = b if isinstance(b, list) else [b]
    segs = segs or [K]
    f = dict(m=M, n=N, nb1=nb1, nb2=nb2, a_trn=a_trn, b_trn=b_trn, a_rs=a_rs, b_rs=b_rs, c_base=c, c_rs=c_rs,
             a_bs1=a_bs[0], a_bs2=a_bs[1], b_bs1=b_bs[0], b_bs2=b_bs[1], c_bs1=c_bs[0], c_bs2=c_bs[1],
             bias_base=bias, aux_base=aux, o2_base=o2, ep_ds=DS_BITS, ep_bias=int(bool(bias)), ep_gelu=gelu,
             ep_gelu_bwd=gelu_bwd, ep_resid=resid)
    for i, (av, bv, ks) in enumerate(zip(a, b, segs)):
        f[f"a_base{i}"], f[f"b_base{i}"], f[f"k{i}"] = av, bv, ks
    if scale is not None:
        f.update(ep_scale=1, ep_sexp=scale)
    if drop is not None:
        f.update(ep_drop=1, site=drop[0], layer=drop[1])
    if merge is not None:                       # merge = (trn, [(seg, module addr tuple)])
        trn, tabs = merge
        f.update(m_en=1, m_trn=trn, m_nb=N // 64, m_mb=(M + 63) // 64, m_h_base=REG["CONST"] + 4 * C_H, m_gamma=fb(GAMMA))
        for i, ks in enumerate(segs):
            f[f"m_kc{i}"] = ks // 64
        for slot, (seg, (hits, dirw, coef)) in zip("ab", tabs):
            f[f"m_ms{seg}"] = 1 if slot == "a" else 2
            f[f"m_hits_{slot}"], f[f"m_dir_{slot}"], f[f"m_coef_{slot}"] = hits, dirw, coef
    return Op("GEMM_ACC2" if acc2 else "GEMM", f)


def vec(vop, rows, cols, **kw):
    f = dict(vop=isa.VID[vop], rows=rows, cols=cols, ds=DS_BITS, rows_per_b=0xFFF0, emb_l=16)
    drop = kw.pop("drop", None)
    if drop is not None:
        f.update(site=drop[0], layer=drop[1], mk_base=kw.pop("mk_base", 0), mk_rs=kw.pop("mk_rs", cols))
    f.update(kw)
    name = "LOSS" if vop in ("CE", "MSE") else "GNORM" if vop.startswith("GN_") else "VEC"
    return Op(name, f)


def lw(name, **kw):
    f = dict(op=isa.LOWA_OP[name])
    f.update(kw)
    return Op(name, f)


def dma(src, dst, n0, n1=1, s_s1=0, d_s1=0, n2=1, s_s2=0, d_s2=0):
    return Op("DMA_COPY", dict(src=src, dst=dst, n0=n0, n1=n1, s_s1=s_s1, d_s1=d_s1, n2=n2, s_s2=s_s2, d_s2=d_s2))


def mod_tabs(i):
    return (REG["LOWA_HITS"] + i * HITS_BYTES, REG["LOWA_DIR"] + i * MOD_BYTES, REG["PARAM"] + i * MOD_BYTES)


SITE = {"emb": 0, "attn": 1, "attn_out": 2, "ffn_out": 3, "cls1": 4, "cls2": 5}


def attn_geo(c):
    L, B = c.L, c.B
    rm = dict(bs=(4 * L * D, 4 * DH), rs=4 * D)        # Q/K/V/O as (b, h, token, d)
    sq = dict(bs=(4 * NH * L * L, 4 * L * L), rs=4 * L)
    return rm, sq


def fwd_layer(c, l, Lv):
    """forward ops of layer l (Lv = the layer-relative wrapper: Lv(addr, sel))."""
    N, B, L = c.N, c.B, c.L
    s = c.scr
    rm, sq = attn_geo(c)
    W = lambda nm: Lv(c.lay(l, nm), S_LAYER)
    St = lambda nm: Lv(c.st(l, nm), S_STASH)
    q, v = 2 * l, 2 * l + 1
    mt = lambda i: tuple(Lv(x, S_HITS if k == 0 else S_MOD) for k, x in enumerate(mod_tabs(i)))
    lay = Lv(l, isa.T_LIDX)
    ops = [
        gemm(N, D, D, s["X"], 4 * D, W("Wq"), 4 * D, St("Q"), 4 * D, bias=W("bq"), merge=(0, [(0, mt(q))])),
        gemm(N, D, D, s["X"], 4 * D, W("Wk"), 4 * D, St("K"), 4 * D, bias=W("bk")),
        gemm(N, D, D, s["X"], 4 * D, W("Wv"), 4 * D, St("V"), 4 * D, bias=W("bv"), merge=(0, [(0, mt(v))])),
        gemm(L, L, DH, St("Q"), rm["rs"], St("K"), rm["rs"], s["A0"], sq["rs"], nb1=B, nb2=NH, a_bs=rm["bs"],
             b_bs=rm["bs"], c_bs=sq["bs"], scale=-3),
        vec("SMAX_FWD", B * NH * L, L, rows_per_b=NH * L, a_base=s["A0"], a_rs=4 * L, c_base=s["A1"], c_rs=4 * L,
            cv0_base=Lv(REG["STEP_IN"] + c.si["km"], isa.T_SLOT), km_rs=4 * c.kmw, st0_base=St("sm"), st1_base=St("sr"),
            st_en=1, drop=(SITE["attn"], lay), mk_rs=L),
        gemm(L, DH, L, s["A1"], sq["rs"], St("V"), rm["rs"], s["O"], rm["rs"], b_trn=1, nb1=B, nb2=NH, a_bs=sq["bs"],
             b_bs=rm["bs"], c_bs=rm["bs"]),
        gemm(N, D, D, s["O"], 4 * D, W("Wo"), 4 * D, St("z1"), 4 * D, bias=W("bo"), drop=(SITE["attn_out"], lay),
             aux=s["X"], resid=1),
        vec("LN_FWD", N, D, a_base=St("z1"), a_rs=4 * D, c_base=s["H1"], c_rs=4 * D, cv1_base=W("LN1"),
            cv2_base=Lv(c.lay(l, "LN1") + 4 * D, S_LAYER), st0_base=St("mu1"), st1_base=St("rs1"), st_en=1,
            c0=fb(1.0 / D), c1=fb(1e-5)),
        gemm(N, FF, D, s["H1"], 4 * D, W("W1"), 4 * D, s["G"], 4 * FF, bias=W("b1"), gelu=1, o2=St("u")),
        gemm(N, D, FF, s["G"], 4 * FF, W("W2"), 4 * FF, St("z2"), 4 * D, bias=W("b2"), drop=(SITE["ffn_out"], lay),
             aux=s["H1"], resid=1),
        vec("LN_FWD", N, D, a_base=St("z2"), a_rs=4 * D, c_base=s["X"], c_rs=4 * D, cv1_base=W("LN2"),
            cv2_base=Lv(c.lay(l, "LN2") + 4 * D, S_LAYER), st0_base=St("mu2"), st1_base=St("rs2"), st_en=1,
            c0=fb(1.0 / D), c1=fb(1e-5)),
    ]
    return ops


def bwd_layer(c, l, Lv):
    N, B, L = c.N, c.B, c.L
    s = c.scr
    rm, sq = attn_geo(c)
    W = lambda nm: Lv(c.lay(l, nm), S_LAYER)
    St = lambda nm: Lv(c.st(l, nm), S_STASH)
    q, v = 2 * l, 2 * l + 1
    lay = Lv(l, isa.T_LIDX)
    rows_att = B * NH * L
    ops = [
        vec("LN_BWD", N, D, a_base=St("z2"), a_rs=4 * D, b_base=s["DOUT"], b_rs=4 * D, c_base=s["S1"], c_rs=4 * D,
            cv0_base=W("LN2"), st0_base=St("mu2"), st1_base=St("rs2"), c0=fb(1.0 / D)),
        vec("E_DROP", N, D, a_base=s["S1"], a_rs=4 * D, c_base=s["S2"], c_rs=4 * D, drop=(SITE["ffn_out"], lay)),
        gemm(N, FF, D, s["S2"], 4 * D, W("W2"), 4 * FF, s["DU"], 4 * FF, b_trn=1, gelu_bwd=1, aux=St("u")),
        gemm(N, D, FF, s["DU"], 4 * FF, W("W1"), 4 * D, s["S2"], 4 * D, b_trn=1, aux=s["S1"], resid=1),
        vec("LN_BWD", N, D, a_base=St("z1"), a_rs=4 * D, b_base=s["S2"], b_rs=4 * D, c_base=s["DZ1"], c_rs=4 * D,
            cv0_base=W("LN1"), st0_base=St("mu1"), st1_base=St("rs1"), c0=fb(1.0 / D)),
        vec("E_DROP", N, D, a_base=s["DZ1"], a_rs=4 * D, c_base=s["S1"], c_rs=4 * D, drop=(SITE["attn_out"], lay)),
        gemm(N, D, D, s["S1"], 4 * D, W("Wo"), 4 * D, s["DOB"], 4 * D, b_trn=1),
        gemm(L, L, DH, St("Q"), rm["rs"], St("K"), rm["rs"], s["A0"], sq["rs"], nb1=B, nb2=NH, a_bs=rm["bs"],
             b_bs=rm["bs"], c_bs=sq["bs"], scale=-3),
        vec("SMAX_REC", rows_att, L, rows_per_b=NH * L, a_base=s["A0"], a_rs=4 * L, c_base=s["A1"], c_rs=4 * L,
            cv0_base=Lv(REG["STEP_IN"] + c.si["km"], isa.T_SLOT), km_rs=4 * c.kmw, st0_base=St("sm"), st1_base=St("sr")),
        gemm(L, L, DH, s["DOB"], rm["rs"], St("V"), rm["rs"], s["A0"], sq["rs"], nb1=B, nb2=NH, a_bs=rm["bs"],
             b_bs=rm["bs"], c_bs=sq["bs"]),
        vec("SMAX_BWD", rows_att, L, a_base=s["A1"], a_rs=4 * L, b_base=s["A0"], b_rs=4 * L, c_base=s["A2"], c_rs=4 * L,
            c0=fb(0.125), drop=(SITE["attn"], lay), mk_rs=L),
        vec("E_DROP", rows_att, L, a_base=s["A1"], a_rs=4 * L, c_base=s["A0"], c_rs=4 * L, drop=(SITE["attn"], lay)),
        # dV = P'^T dO -> V's stash slot (V is dead after dP')
        gemm(L, DH, L, s["A0"], sq["rs"], s["DOB"], rm["rs"], St("V"), rm["rs"], a_trn=1, b_trn=1, nb1=B, nb2=NH,
             a_bs=sq["bs"], b_bs=rm["bs"], c_bs=rm["bs"]),
    ]
    if l > 0:   # dK = dS^T Q (before dQ overwrites Q)
        ops.append(gemm(L, DH, L, s["A2"], sq["rs"], St("Q"), rm["rs"], s["DKB"], rm["rs"], a_trn=1, b_trn=1, nb1=B,
                        nb2=NH, a_bs=sq["bs"], b_bs=rm["bs"], c_bs=rm["bs"]))
    # dQ = dS K -> Q's slot
    ops.append(gemm(L, DH, L, s["A2"], sq["rs"], St("K"), rm["rs"], St("Q"), rm["rs"], b_trn=1, nb1=B, nb2=NH,
                    a_bs=sq["bs"], b_bs=rm["bs"], c_bs=rm["bs"]))
    if l > 0:   # X_l = LN2_{l-1}(z2_{l-1}) recomputed into K's slot (K is dead after dQ)
        ops.append(vec("LN_FWD", N, D, a_base=Lv(c.st(l - 1, "z2"), S_STASH), a_rs=4 * D, c_base=St("K"), c_rs=4 * D,
                       cv1_base=Lv(c.lay(l - 1, "LN2"), S_LAYER), cv2_base=Lv(c.lay(l - 1, "LN2") + 4 * D, S_LAYER),
                       st_en=0, c0=fb(1.0 / D), c1=fb(1e-5)))
        xa = St("K")
    else:
        xa = REG["STASH_X0"]
    for i, dy in ((q, St("Q")), (v, St("V"))):
        ops.append(lw("LOWA_SG", a0=dy, a1=4 * D, a2=xa, a3=4 * D, a4=Lv(REG["LOWA_CTR"] + i * MOD_BYTES, S_MOD),
                      a5=REG["CONST"] + 4 * C_H, a6=Lv(REG["GRAD"] + i * MOD_BYTES, S_MOD), a7=fb(GAMMA), len=N))
    if l > 0:   # dX = [dQ dK dV] [Wq; Wk; Wv] (merge-on-load q / v, TRN) + dz1
        mt = lambda i: tuple(Lv(x, S_HITS if k == 0 else S_MOD) for k, x in enumerate(mod_tabs(i)))
        ops.append(gemm(N, D, 3 * D, [St("Q"), s["DKB"], St("V")], 4 * D, [W("Wq"), W("Wk"), W("Wv")], 4 * D,
                        s["DOUT"], 4 * D, b_trn=1, segs=[D, D, D], aux=s["DZ1"], resid=1, acc2=True,
                        merge=(1, [(0, mt(q)), (2, mt(v))])))
    return ops


def head_ops(c):
    B, L, C, N = c.B, c.L, c.C, c.N
    h, P, G = c.cls, REG["PARAM"], REG["GRAD"]
    cst = REG["CONST"]
    lab = T(REG["STEP_IN"] + c.si["lab"], isa.T_SLOT)
    wts = T(REG["STEP_IN"] + c.si["w"], isa.T_SLOT)
    import hwmodel as H
    inv_b = fb(H.recip(np.array([B], f32))[0])
    loss_vop = "CE" if c.task == "ce" else "MSE"
    lk = dict(c0=inv_b, x_reset=1)
    if c.task == "mse":
        lk["c1"] = fb(H.mul(f32(2.0), f32(np.frombuffer(np.uint32(inv_b).tobytes(), f32)[0]))[0])
    ops = [
        vec("E_DROP", B, D, a_base=c.scr["X"], a_rs=4 * L * D, c_base=h["xc"], c_rs=4 * D, drop=(SITE["cls1"], 0)),
        gemm(B, D, D, h["xc"], 4 * D, P + c.head("dw"), 4 * D, h["pre"], 4 * D, bias=P + c.head("db")),
        vec("E_TANH", B, D, a_base=h["pre"], a_rs=4 * D, c_base=h["a"], c_rs=4 * D),
        vec("E_DROP", B, D, a_base=h["a"], a_rs=4 * D, c_base=h["a2"], c_rs=4 * D, drop=(SITE["cls2"], 0)),
        gemm(B, C, D, h["a2"], 4 * D, P + c.head("ow"), 4 * D, h["z"], 4 * C, bias=P + c.head("ob")),
        vec(loss_vop, B, C, a_base=h["z"], a_rs=4 * C, c_base=h["dz"], c_rs=4 * C, st0_base=lab, st1_base=wts,
            o2_base=REG["LOSS_OUT"], **lk),
        # head backward
        gemm(C, D, B, h["dz"], 4 * C, h["a2"], 4 * D, G + c.head("ow"), 4 * D, a_trn=1, b_trn=1),
        gemm(1, C, B, cst + 4 * C_ONES, 4 * B, h["dz"], 4 * C, G + c.head("ob"), 4 * C, b_trn=1),
        gemm(B, D, C, h["dz"], 4 * C, P + c.head("ow"), 4 * D, h["dh0"], 4 * D, b_trn=1),
        vec("E_DROP", B, D, a_base=h["dh0"], a_rs=4 * D, c_base=h["dh1"], c_rs=4 * D, drop=(SITE["cls2"], 0)),
        vec("E_TANHB", B, D, a_base=h["a"], a_rs=4 * D, b_base=h["dh1"], b_rs=4 * D, c_base=h["dh"], c_rs=4 * D),
        gemm(D, D, B, h["dh"], 4 * D, h["xc"], 4 * D, G + c.head("dw"), 4 * D, a_trn=1, b_trn=1),
        gemm(1, D, B, cst + 4 * C_ONES, 4 * B, h["dh"], 4 * D, G + c.head("db"), 4 * D, b_trn=1),
        gemm(B, D, D, h["dh"], 4 * D, P + c.head("dw"), 4 * D, h["dc"], 4 * D, b_trn=1),
        dma(cst + 4 * C_ZERO, c.scr["DOUT"], D, n1=N, s_s1=0, d_s1=4 * D),               # dout = 0
        vec("E_DROP", B, D, a_base=h["dc"], a_rs=4 * D, c_base=c.scr["DOUT"], c_rs=4 * L * D, drop=(SITE["cls1"], 0)),
    ]
    assert B <= 64, "bias gradient by a ones-GEMM equals the L1 column sum only within one K chunk (B <= 64)"
    return ops


def opt_ops(c):
    G, O = REG["GRAD"], REG["OPT_SCAL"]
    tens = [(G + i * MOD_BYTES, 1, 256) for i in range(24 if c.layers == 12 else 2 * c.layers)]
    tens += [(G + c.head("dw"), D, D), (G + c.head("db"), 1, D), (G + c.head("ow"), c.C, D), (G + c.head("ob"), 1, c.C)]
    ops = []
    for i, (a, r, cl) in enumerate(tens):
        ops.append(vec("GN_ROWS", r, cl, a_base=a, a_rs=4 * cl, o2_base=O + 4 * (O_NORMS + 8 * i), x_reset=1,
                       g_reset=int(i == 0)))
    ops.append(vec("GN_FIN", 0, 0, o2_base=O + 4 * O_GFIN, c0=fb(1e-6)))
    ops.append(dma(O + 4 * (O_GFIN + 1), O + 4 * 15, 1, n1=2, s_s1=0, d_s1=32))       # clip -> both scalar blocks
    ncoef = 256 * len([t for t in tens if t[2] == 256 and t[1] == 1 and t[0] < G + c.head("dw")])
    ops.append(lw("ADAMW", a0=REG["PARAM"], a1=G, a2=REG["ADAM_M"], a3=REG["ADAM_V"], a4=O + 32, len=ncoef, flag=1))
    hb = c.head("dw")
    ops.append(lw("ADAMW", a0=REG["PARAM"] + hb, a1=G + hb, a2=REG["ADAM_M"] + hb, a3=REG["ADAM_V"] + hb, a4=O + 64,
                  len=c.head_n, flag=1))
    return ops


def program(c, ttot):
    ident = lambda v, sel: v
    pre = [Op("SETCSR", dict(csr=1, value=LAYER_STRIDE)), Op("SETCSR", dict(csr=2, value=c.stash_stride)),
           Op("SETCSR", dict(csr=3, value=2 * MOD_BYTES)), Op("SETCSR", dict(csr=4, value=2 * HITS_BYTES))]
    si = lambda k: T(REG["STEP_IN"] + c.si[k], isa.T_SLOT)
    pre.append(vec("EMB", c.N, D, emb_l=c.L, a_base=REG["EMB_WORD"], a_rs=4 * D, b_base=REG["EMB_POS"], b_rs=4 * D,
                   c_base=c.scr["X"], c_rs=4 * D, cv0_base=REG["EMB_TT_LN"], cv1_base=REG["EMB_TT_LN"] + 4 * D,
                   cv2_base=REG["EMB_TT_LN"] + 8 * D, ids_base=si("ids"), drop=(SITE["emb"], 0), mk_rs=D,
                   c0=fb(1.0 / D), c1=fb(1e-5)))
    pre.append(dma(c.scr["X"], REG["STASH_X0"], c.N * D))
    pre.append(lw("LOWA_SCAL", a0=T(0, isa.T_STEP), a1=WARMUP, a2=ttot, a3=REG["CONST"] + 4 * C_SCAL, a4=REG["OPT_SCAL"]))
    items = pre + [Loop(lambda l, Lv: fwd_layer(c, l, Lv), range(c.layers), False)] + head_ops(c)
    if c.layers > 1:
        items.append(Loop(lambda l, Lv: bwd_layer(c, l, Lv), range(c.layers - 1, 0, -1), True))
    items += bwd_layer(c, 0, ident) + opt_ops(c) + [Op("END", {})]
    return items


# ------------------------------------------------------------------------------- ranges + fences
def _ext(base, rows, rs, cols, bs=(), nb=()):
    hi = base + (rows - 1) * rs + 4 * cols
    for s_, n_ in zip(bs, nb):
        hi += (n_ - 1) * s_
    return (base, hi)


def ranges(name, f, c):
    """(reads, writes) byte ranges [lo, hi) of one resolved op."""
    from collections import defaultdict
    f = defaultdict(int, f)
    R, Wr = [], []
    if name in ("GEMM", "GEMM_ACC2"):
        M, N, nb = f["m"], f["n"], (f["nb1"], f["nb2"])
        for i in range(3):
            k = f[f"k{i}"]
            if not k:
                continue
            ra, ca = (k, M) if f["a_trn"] else (M, k)
            rb, cb = (k, N) if f["b_trn"] else (N, k)
            R.append(_ext(f[f"a_base{i}"], ra, f["a_rs"], ca, (f["a_bs1"], f["a_bs2"]), nb))
            R.append(_ext(f[f"b_base{i}"], rb, f["b_rs"], cb, (f["b_bs1"], f["b_bs2"]), nb))
        cg = lambda b: _ext(b, M, f["c_rs"], N, (f["c_bs1"], f["c_bs2"]), nb)
        Wr.append(cg(f["c_base"]))
        if f["ep_bias"]:
            R.append((f["bias_base"], f["bias_base"] + 4 * N))
        if f["ep_resid"] or f["ep_gelu_bwd"]:
            R.append(cg(f["aux_base"]))
        if f["ep_gelu"]:
            Wr.append(cg(f["o2_base"]))
        if f["m_en"]:
            for s in "ab":
                if f[f"m_hits_{s}"]:
                    R += [(f[f"m_hits_{s}"], f[f"m_hits_{s}"] + HITS_BYTES), (f[f"m_dir_{s}"], f[f"m_dir_{s}"] + MOD_BYTES),
                          (f[f"m_coef_{s}"], f[f"m_coef_{s}"] + MOD_BYTES)]
            R.append((f["m_h_base"], f["m_h_base"] + 100))
    elif name in ("VEC", "LOSS", "GNORM"):
        vop = isa.VOPS[f["vop"]]
        rows, cols = f["rows"], f["cols"]
        mat = lambda b, rs: _ext(b, rows, rs, cols)
        if vop not in ("GN_FIN",):
            if vop == "EMB":
                R += [(f["a_base"], f["a_base"] + 4 * D * 50265), (f["b_base"], f["b_base"] + 4 * D * 514),
                      (f["ids_base"], f["ids_base"] + 4 * rows)]
            else:
                R.append(mat(f["a_base"], f["a_rs"]))
        if vop in ("LN_BWD", "SMAX_BWD", "E_ADD", "E_MUL", "E_TANHB"):
            R.append(mat(f["b_base"], f["b_rs"]))
        if vop not in ("GN_ROWS", "GN_FIN"):
            Wr.append(mat(f["c_base"], f["c_rs"]))
        for k in ("cv0_base", "cv1_base", "cv2_base"):
            if f[k]:
                R.append((f[k], f[k] + 4 * max(cols, 1) * (rows if k == "cv0_base" and vop.startswith("SMAX") else 1)))
        if vop in ("LN_BWD", "SMAX_REC"):
            R += [(f["st0_base"], f["st0_base"] + 4 * rows), (f["st1_base"], f["st1_base"] + 4 * rows)]
        if vop in ("CE", "MSE"):
            R += [(f["st0_base"], f["st0_base"] + 4 * rows), (f["st1_base"], f["st1_base"] + 4 * rows)]
        if f["st_en"]:
            Wr += [(f["st0_base"], f["st0_base"] + 4 * rows), (f["st1_base"], f["st1_base"] + 4 * rows)]
        if vop in ("CE", "MSE", "GN_ROWS", "GN_FIN"):
            Wr.append((f["o2_base"], f["o2_base"] + 32))
        if vop in ("GN_ROWS", "GN_FIN"):        # the norm accumulator is unit state: order GNORM ops on the unit
            pass
    elif name in ("LOWA_SG", "LOWA_SCAL", "ADAMW"):
        if f["op"] == 0:
            R += [_ext(f["a0"], f["len"], f["a1"], D), _ext(f["a2"], f["len"], f["a3"], D), (f["a4"], f["a4"] + 1024),
                  (f["a5"], f["a5"] + 100)]
            Wr.append((f["a6"], f["a6"] + 1024))
        elif f["op"] == 1:
            R.append((f["a3"], f["a3"] + 64))
            Wr.append((f["a4"], f["a4"] + 96))
        else:
            for k in ("a0", "a2", "a3"):
                Wr.append((f[k], f[k] + 4 * f["len"]))
            R += [(f["a1"], f["a1"] + 4 * f["len"]), (f["a4"], f["a4"] + 32)]
    elif name == "DMA_COPY":
        R.append(_ext(f["src"], f["n1"], f["s_s1"], f["n0"], (f["s_s2"],), (f["n2"],)))
        Wr.append(_ext(f["dst"], f["n1"], f["d_s1"], f["n0"], (f["d_s2"],), (f["n2"],)))
    return R, Wr


def _ov(A, Bs):
    return any(a0 < b1 and b0 < a1 for a0, a1 in A for b0, b1 in Bs)


def unit_of(name):
    u = isa.OPS[name][1]
    return u


# ----------------------------------------------------------------------------------------- assembly
def resolve(f, lterm=None):
    """field dict with T / Lv values -> plain ints (lterm: {sel: value} of the current iteration)."""
    out = {}
    for k, v in f.items():
        if isinstance(v, T):
            out[k] = (v.v + (lterm or {}).get(v.sel, 0)) & 0xFFFFFFFF
        else:
            out[k] = int(v)
    return out


def flatten(items, c, strides, step_t=0):
    """unrolled op list [(name, resolved fields, item index, body position)] for the fence analysis."""
    flat = []
    for ii, it in enumerate(items):
        if isinstance(it, Loop):
            for l in it.layers:
                ops = it.body_fn(l, lambda v, sel: v)
                for j, op in enumerate(ops):
                    flat.append((op.name, resolve(op.f, {isa.T_STEP: step_t}), ii, j))
        else:
            flat.append((it.name, resolve(it.f, {isa.T_STEP: step_t}), ii, None))
    return flat


def place_fences(flat, c):
    """{(item, pos): mask} FENCE masks to insert before each op (OR over loop iterations)."""
    pend = {u: [] for u in (isa.UNIT_DMA, isa.UNIT_GEMM, isa.UNIT_VEC, isa.UNIT_LOWA)}
    need = {}
    for name, f, ii, j in flat:
        u = unit_of(name)
        if u == isa.UNIT_SEQ:
            continue
        R, Wr = ranges(name, f, c)
        mask = 0
        for v, lst in pend.items():
            if v == u:
                continue
            for (r2, w2) in lst:
                if _ov(w2, R + Wr) or _ov(r2, Wr):
                    mask |= isa.UNIT_FBIT[v]
                    break
        if mask:
            for v in list(pend):
                if isa.UNIT_FBIT[v] & mask:
                    pend[v] = []
            need[(ii, j)] = need.get((ii, j), 0) | mask
        pend[u].append((R, Wr))
    return need


def assemble(items, need, c):
    words, nops, ndesc = [], {}, 0

    def emit(name, f, rel):
        nonlocal ndesc
        w = isa.encode(name, f, rel)
        words.append(w)
        nops[name] = nops.get(name, 0) + 1
        ndesc += 1
        return w.size * 4

    def emit_op(op, lv_mode, key):
        nb = 0
        if key in need:
            nb += emit("FENCE", dict(mask=need[key]), {})
        f, rel = {}, {}
        for k, v in op.f.items():
            if isinstance(v, T):
                f[k], rel[k] = v.v, v.sel
            else:
                f[k] = v
        return nb + emit(op.name, f, rel)

    for ii, it in enumerate(items):
        if isinstance(it, Loop):
            l0 = it.layers[0]
            ops = it.body_fn(l0, lambda v, sel: T(0 if sel == isa.T_LIDX else v, sel))
            idx = len(words)
            emit("LOOP", dict(body=0, count=len(it.layers), start=l0, dir=int(it.down)), {})
            body = 0
            for j, op in enumerate(ops):
                body += emit_op(op, True, (ii, j))
            words[idx] = isa.encode("LOOP", dict(body=body, count=len(it.layers), start=l0, dir=int(it.down)))
        else:
            emit_op(it, False, (ii, None))
    prog = np.concatenate(words).astype(U32)
    return prog, nops, ndesc


# --------------------------------------------------------------------------------------------- image
class Image:
    def __init__(self):
        self.reg = []                         # (byte addr, uint32 array)

    def put(self, a, arr):
        assert a % 4 == 0
        self.reg.append((a, np.ascontiguousarray(arr).view(U32).reshape(-1).copy()))


def load_inputs(c, run_dir=RUNS):
    W = np.load(os.path.join(run_dir, "model_fp32.npz"))
    inp = dict(np.load(os.path.join(run_dir, "stepB_l0f32.npz")))
    B, L = c.B, c.L
    ids = inp["input_ids"][:B, :L].astype(np.int64)
    am = inp["attention_mask"][:B, :L].astype(np.uint8)
    return W, inp, ids, am, inp["labels"][:B]


def build_image(c, W, inp, ids, am, labels, slot=0):
    img = Image()
    Wf = lambda k: np.asarray(W[k], f32)
    # CONST
    cst = np.zeros(1024, f32)
    h = Wf("lowa.0.hval")
    assert all(np.array_equal(Wf(f"lowa.{i}.hval"), h) for i in range(24))
    cst[C_GAMMA] = GAMMA
    cst[C_H:C_H + 25] = h
    ttot = EPOCHS * int(inp["n_batches"])
    cst[C_SCAL:C_SCAL + 10] = [f32(1.0 / WARMUP), f32(1.0 / (ttot - WARMUP)), LR0[0], LR0[1], WD, B1, B2, B1C, B2C, EPS]
    cst[C_ONES:C_ONES + 64] = 1.0
    img.put(REG["CONST"], cst)
    # embeddings: only the rows the batch uses (sparse image; the whole table is 154 MB)
    word = Wf("roberta.embeddings.word_embeddings.weight")
    for r in np.unique(ids):
        img.put(REG["EMB_WORD"] + 4 * D * int(r), word[r])
    img.put(REG["EMB_POS"], Wf("roberta.embeddings.position_embeddings.weight"))
    img.put(REG["EMB_TT_LN"], np.concatenate([Wf("roberta.embeddings.token_type_embeddings.weight")[0],
                                              Wf("roberta.embeddings.LayerNorm.weight"),
                                              Wf("roberta.embeddings.LayerNorm.bias")]))
    for l in range(c.layers):
        pre = f"roberta.encoder.layer.{l}."
        for nm, hf in HF_NAME.items():
            img.put(c.lay(l, nm), Wf(pre + hf))
        for nm, hf in LN_NAME.items():
            img.put(c.lay(l, nm), np.concatenate([Wf(pre + hf + ".weight"), Wf(pre + hf + ".bias")]))
    nmod = 2 * c.layers
    for i in range(nmod):
        t = LT.build(W[f"lowa.{i}.eflat"])
        assert float(W[f"lowa.{i}.gamma"]) == float(GAMMA)
        img.put(REG["LOWA_CTR"] + i * MOD_BYTES, t["centres"])
        img.put(REG["LOWA_HITS"] + i * HITS_BYTES, t["hits"])
        img.put(REG["LOWA_DIR"] + i * MOD_BYTES, t["dir"])
    # PARAM / ADAM state (pre-step), head layout dw, db, ow, ob
    for reg, tag in (("PARAM", "pre_"), ("ADAM_M", "pre_m_"), ("ADAM_V", "pre_v_")):
        coef = np.asarray(inp[tag + "coef"], f32)[:nmod].reshape(-1)
        img.put(REG[reg], coef)
        img.put(REG[reg] + c.head("dw"), np.concatenate([np.asarray(inp[tag + k], f32).reshape(-1) for k in HEAD_KEYS]))
    # OPT_SCAL: running products beta^s held on chip (the state after s steps)
    import hwmodel as H
    s = int(inp["step_s"])
    b1t, b2t = f32(1.0), f32(1.0)
    for _ in range(s):
        b1t, b2t = H.mul(b1t, B1)[0], H.mul(b2t, B2)[0]
    osc = np.zeros(1024, f32)
    osc[0], osc[1] = b1t, b2t
    img.put(REG["OPT_SCAL"], osc)
    # STEP_IN[slot]
    base = REG["STEP_IN"] if slot == 0 else REG["STEP_IN1"]
    kmw = np.zeros((c.B, c.kmw), U32)
    for b in range(c.B):
        for j in range(c.L):
            if am[b, j]:
                kmw[b, j >> 5] |= U32(1) << U32(j & 31)
    sl = np.zeros(SLOT_BYTES // 4, U32)
    sl[0:c.N] = ids.reshape(-1).astype(np.int32).view(U32)
    sl[c.si["km"] // 4:c.si["km"] // 4 + kmw.size] = kmw.reshape(-1)
    sl[c.si["lab"] // 4:c.si["lab"] // 4 + c.B] = np.asarray(labels, np.int32).view(U32)
    sl[c.si["w"] // 4:c.si["w"] // 4 + c.B] = np.ones(c.B, f32).view(U32)
    img.put(base, sl)
    return img, ttot


def csrs(c, inp, slot=0):
    """CSR values the host writes before the doorbell (docs/isa.md section 5)."""
    return dict(PROG_BASE=REG["PROG"], STEP_T=int(inp["step_s"]), SEED=int(inp["seed"]),
                SLOT_DELTA=REG["STEP_IN1"] - REG["STEP_IN"], DOORBELL_SLOT=slot)


def compile_all(c, slot=0):
    W, inp, ids, am, labels = load_inputs(c)
    img, ttot = build_image(c, W, inp, ids, am, labels, slot)
    items = program(c, ttot)
    flat = flatten(items, c, None, int(inp["step_s"]))
    need = place_fences(flat, c)
    prog, nops, ndesc = assemble(items, need, c)
    assert prog.size * 4 <= 262144
    img.put(REG["PROG"], prog)
    return dict(prog=prog, img=img, nops=nops, ndesc=ndesc, flat_ops=len(flat), csr=csrs(c, inp, slot), inp=inp,
                ids=ids, am=am, labels=labels, W=W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--task", default="ce")
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "compile"))
    a = ap.parse_args()
    c = Cfg(a.layers, a.batch, a.seq, a.task)
    r = compile_all(c)
    os.makedirs(a.out, exist_ok=True)
    tag = f"l{a.layers}_b{a.batch}_s{a.seq}"
    r["prog"].tofile(os.path.join(a.out, f"prog_{tag}.bin"))
    np.savez(os.path.join(a.out, f"image_{tag}.npz"), **{f"r{addr:08x}": arr for addr, arr in r["img"].reg})
    info = dict(cfg=vars(a), descriptors=r["ndesc"], unrolled_ops=r["flat_ops"], bytes=int(r["prog"].size * 4),
                ops=r["nops"], csr=r["csr"])
    with open(os.path.join(a.out, f"prog_{tag}.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"COMPILE {tag}: {r['ndesc']} descriptors ({r['flat_ops']} ops unrolled), {r['prog'].size * 4} bytes; "
          + " ".join(f"{k}={v}" for k, v in sorted(r["nops"].items())))


if __name__ == "__main__":
    main()
