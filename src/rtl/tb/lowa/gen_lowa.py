#!/usr/bin/env python3
"""Vector generator for tb_lw_top (P5).  Goldens only from model/hwmodel (H.lowa_merge, H.lowa_sg,
H.adamw, H.step_scalars, H.mul); tables from sw/lowa_tables.py.  Writes OUTDIR/<case>.{desc,mem.hex,exp.hex}.
  gen_lowa.py <case> <outdir>      cases: merge_unit merge_real_row merge_real_trn sg_unit sg_edge sg_real_l0q sg_real_l11q
                                          scal_s0 scal_peak adamw_s0 adamw_clip adamw_noclip adamw_real_coef adamw_real_head
                                          merge_gemm nomerge_gemm (tb_lw_gemm: gm_top + lw_top merge, mode 2)
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "sw"))
sys.path.insert(0, os.path.join(ROOT, "tb", "gemm"))
import hwmodel as H  # noqa: E402
import lowa_tables as LT  # noqa: E402
from gen_gemm import Mem, f2u, rand_mat  # noqa: E402

f32, U32 = np.float32, np.uint32
RUNS = os.path.join(ROOT, "model", "runs")
GAMMA, LR0, WD, B1, B2 = f32(0.552427), (f32(0.166667), f32(5e-3)), f32(0.01), f32(0.9), f32(0.999)
B1C, B2C, EPS, WARMUP, EPOCHS = f32(0.1), f32(0.001), f32(1e-8), 140, 30


class Case:
    def __init__(self, name, mode=0):
        self.name, self.mode, self.mem, self.d, self.exp = name, mode, Mem(), {}, []

    def place_u32(self, arr):
        a = self.mem.alloc(arr.size)
        self.mem.chunks.append((a, np.ascontiguousarray(arr, U32).reshape(-1)))
        return a

    def write(self, outdir):
        p = os.path.join(outdir, self.name)
        self.mem.write(p + ".mem.hex")
        n = 0
        with open(p + ".exp.hex", "w") as f:
            for a, arr in self.exp:
                for i, v in enumerate(f2u(arr).reshape(-1) if arr.dtype != U32 else arr.reshape(-1)):
                    f.write(f"{a + i:x} {int(v):08x}\n")
                    n += 1
        with open(p + ".desc", "w") as f:
            f.write(f"mode {self.mode:x}\nexp_n {n:x}\n")
            for k, v in self.d.items():
                f.write(f"{k} {int(v) & 0xFFFFFFFF:x}\n")


def synth_eflat(r, m, n, k, edge=True):
    """k centres (some on the edges / overlapping when edge), taps as SGWALinear: reflect(c + off)."""
    ci = r.integers(0, m, k)
    cj = r.integers(0, n, k)
    if edge:
        ci[0], cj[0] = 0, 0
        ci[1], cj[1] = m - 1, 2
        ci[2], cj[2] = 5, n - 1
        ci[3], cj[3] = ci[0] + 1, cj[0] + 1          # overlaps atom 0 (colliding cells)
        ci[4], cj[4] = ci[0], cj[0]                  # identical centre: every cell collides
    off = np.arange(-3, 4)
    rows = LT.reflect(ci[:, None] + off[None, :], m)
    cols = LT.reflect(cj[:, None] + off[None, :], n)
    di = np.array([t[0] + 3 for t in LT.TAPS])
    dj = np.array([t[1] + 3 for t in LT.TAPS])
    return (rows[:, di] * n + cols[:, dj]).reshape(-1)


def hval():
    return np.load(os.path.join(RUNS, "model_fp32.npz"))["lowa.0.hval"].astype(f32)


def place_tables(c, t, h, coef):
    return dict(hits=4 * c.place_u32(t["hits"]), dir=4 * c.place_u32(t["dir"]), ctr=4 * c.place_u32(t["centres"]),
                h=4 * c.mem.place(h), coef=4 * c.mem.place(coef))


def merge_case(c, W0, eflat, h, coef, gamma, trn, m, n):
    """One GEMM's W-tile sequence: nb over the B rows (ROW: W rows, TRN: W cols), kc over the other dim."""
    t = LT.build(eflat, m, n)
    W = H.lowa_merge(W0, eflat, coef, h, gamma)
    tb = place_tables(c, t, h, coef)
    c.d.update(m_trn=int(trn), m_nb=(n if trn else m) // 64, m_mb=1, m_kc0=(m if trn else n) // 64, m_kc1=0, m_kc2=0,
               m_ms0=1, m_ms1=0, m_ms2=0, m_hits_a=tb["hits"], m_dir_a=tb["dir"], m_coef_a=tb["coef"], m_hits_b=0, m_dir_b=0,
               m_coef_b=0, m_h_base=tb["h"], m_gamma=int(f2u(np.array([gamma], f32))[0]),
               w0_base=4 * c.mem.place(W0), w0_rs=4 * n)
    tiles = []
    for nb in range(c.d["m_nb"]):
        for kc in range(c.d["m_kc0"]):
            blk = W[64 * kc:64 * kc + 64, 64 * nb:64 * nb + 64].T if trn else W[64 * nb:64 * nb + 64, 64 * kc:64 * kc + 64]
            tiles.append(np.ascontiguousarray(blk, f32).reshape(-1))      # [n, k]
    c.exp.append((0, np.concatenate(tiles)))
    return c


def merge_unit(c):
    r = np.random.default_rng(5)
    m = n = 128
    eflat = synth_eflat(r, m, n, 8)
    coef = rand_mat(r, (8,), 120, 128)
    W0 = rand_mat(r, (m, n))
    t = LT.build(eflat, m, n)
    assert (t["cnt"] == 0).sum() > 0, "unit case should have an empty block"
    return merge_case(c, W0, eflat, hval(), coef, GAMMA, False, m, n)


def merge_real(c, trn):
    z = np.load(os.path.join(RUNS, "model_fp32.npz"))
    inp = np.load(os.path.join(RUNS, "stepB_l0f32.npz"))
    W0 = z["roberta.encoder.layer.0.attention.self.query.base_layer.weight"].astype(f32)
    coef = inp["pre_coef"][0].astype(f32)
    return merge_case(c, W0, z["lowa.0.eflat"], z["lowa.0.hval"].astype(f32), coef, GAMMA, trn, 768, 768)


def gemm_qproj(c, merge):
    """tb_lw_gemm: Q = X Wq^T + bq of layer 0 (4096 x 768 x 768, P3 real_qproj) with the GEMM engine streaming
    the frozen W0 and lw_merge adding the 256 atoms on load (merge = 1), or streaming L1's merged weight with the
    merge engine idle (merge = 0, the P3 case: the tile-rate reference).  Expected Q = L1 gemm on L1's merged W."""
    from gen_gemm import Real
    R = Real()
    z = np.load(os.path.join(RUNS, "model_fp32.npz"))
    W0 = z["roberta.encoder.layer.0.attention.self.query.base_layer.weight"].astype(f32)
    coef, eflat, h = R.inp["pre_coef"][0].astype(f32), z["lowa.0.eflat"], z["lowa.0.hval"].astype(f32)
    gamma = f32(z["lowa.0.gamma"])
    assert gamma == GAMMA
    Wm = H.lowa_merge(W0, eflat, coef, h, gamma)
    assert np.array_equal(f2u(Wm), f2u(R.merged("query")))
    bq = R.layer("attention.self.query.base_layer.bias")
    Q = H.gemm(R.x, Wm, bq)
    M, N, K = R.N, 768, 768
    # nb1 = nb2 = 1: one batch (gm_ctrl's batch counts; 0 would mean 65,536 batches, as gen_gemm.Case defaults)
    c.d.update(m=M, n=N, k0=K, nb1=1, nb2=1, a_base0=4 * c.mem.place(R.x), a_rs=4 * K,
               b_base0=4 * c.mem.place(W0 if merge else Wm), b_rs=4 * K, bias_base=4 * c.mem.place(bq), ep_bias=1)
    cb = c.mem.alloc(M * N)
    c.d.update(c_base=4 * cb, c_rs=4 * N, m_en=int(merge))
    if merge:
        t = LT.build(eflat, 768, 768)
        tb = place_tables(c, t, h, coef)
        c.d.update(m_trn=0, m_nb=N // 64, m_mb=M // 64, m_kc0=K // 64, m_kc1=0, m_kc2=0, m_ms0=1, m_ms1=0, m_ms2=0,
                   m_hits_a=tb["hits"], m_dir_a=tb["dir"], m_coef_a=tb["coef"], m_hits_b=0, m_dir_b=0, m_coef_b=0,
                   m_h_base=tb["h"], m_gamma=int(f2u(np.array([gamma], f32))[0]))
    c.exp.append((cb, Q))
    return c


def sg_case(c, dY, X, eflat, h, gamma, tok):
    t = LT.build(eflat, 768, 768)
    g = H.lowa_sg(dY, X, t["rows"], t["cols"], h, gamma)
    tb = place_tables(c, t, h, np.zeros(256, f32))
    gb = c.mem.alloc(256)
    c.d.update(op=0, a0=4 * c.mem.place(dY), a1=4 * 768, a2=4 * c.mem.place(X), a3=4 * 768,
               a4=tb["ctr"], a5=tb["h"], a6=4 * gb, a7=int(f2u(np.array([gamma], f32))[0]), len=tok)
    c.exp.append((gb, g))
    return c


def sg_unit(c):
    r = np.random.default_rng(7)
    tok = 5
    eflat = synth_eflat(r, 768, 768, 256, edge=False)
    return sg_case(c, rand_mat(r, (tok, 768)), rand_mat(r, (tok, 768)), eflat, hval(), GAMMA, tok)


def sg_edge(c):
    """edge-folded atoms, overlapping atoms, one token, row stride 2 x 768 (CLS-row style gather)."""
    r = np.random.default_rng(8)
    eflat = synth_eflat(r, 768, 768, 256, edge=True)
    dY, X = rand_mat(r, (6, 768)), rand_mat(r, (6, 768))
    t = LT.build(eflat, 768, 768)
    g = H.lowa_sg(dY[::2], X[::2], t["rows"], t["cols"], hval(), GAMMA)
    tb = place_tables(c, t, hval(), np.zeros(256, f32))
    gb = c.mem.alloc(256)
    c.d.update(op=0, a0=4 * c.mem.place(dY), a1=4 * 2 * 768, a2=4 * c.mem.place(X), a3=4 * 2 * 768,
               a4=tb["ctr"], a5=tb["h"], a6=4 * gb, a7=int(f2u(np.array([GAMMA], f32))[0]), len=3)
    c.exp.append((gb, g))
    return c


def sg_real_calls(out):
    """The L1 step on the step-B batch with H.lowa_sg hooked: call n -> (dY, X, rows, cols, h, gamma, grad).
    Call order = step.py's loop (l = 11 .. 0, query then value): n = 0 is module 22 (layer-11 query),
    n = 22 is module 0 (layer-0 query).  Cached in <out>/sg_real_cache.npz."""
    cache = os.path.join(out, "sg_real_cache.npz")
    keep = {0: 22, 22: 0}
    if not os.path.exists(cache):
        import hwmodel.step as S
        rec, cnt = {}, [0]
        orig = H.lowa_sg

        def w(dY, X, rows, cols, h, gamma):
            g = orig(dY, X, rows, cols, h, gamma)
            n = cnt[0]
            cnt[0] += 1
            if n in keep:
                rec[f"m{keep[n]}"] = (np.array(dY, f32), np.array(X, f32), np.array(rows), np.array(cols), np.array(h, f32), f32(gamma), g)
            return g
        H.lowa_sg = w
        W = S.Weights(os.path.join(RUNS, "model_fp32.npz"))
        S.step(W, dict(np.load(os.path.join(RUNS, "stepB_l0f32.npz"))), verbose=False)
        H.lowa_sg = orig
        assert cnt[0] == 24
        np.savez(cache, **{f"{k}.{i}": v for k, t in rec.items() for i, v in enumerate(t)})
    z = np.load(cache)
    return {m: tuple(z[f"m{m}.{i}"] for i in range(7)) for m in keep.values()}


def sg_real(c, module, cls_only, out):
    """All 256 coefficient gradients of a real module from the step-B backward (layer-0 query: all 4096
    tokens; layer-11 query: only the 32 CLS rows, every other dQ row is exactly zero, dataflow.md D6)."""
    dY, X, rows, cols, h, gamma, g_step = sg_real_calls(out)[module]
    t = LT.build(np.load(os.path.join(RUNS, "model_fp32.npz"))[f"lowa.{module}.eflat"], 768, 768)
    assert np.array_equal(t["rows"], rows) and np.array_equal(t["cols"], cols), "table rows/cols differ from the step's"
    l1 = np.load(os.path.join(RUNS, "stepB_l1.npz"))
    assert np.array_equal(f2u(g_step), f2u(l1["g_coef"][module])), "hooked call disagrees with stepB_l1"
    if cls_only:
        L = 128
        assert not np.any(dY.reshape(-1, L, 768)[:, 1:, :]), "non-CLS rows of dQ are not all zero"
        dY, X = np.ascontiguousarray(dY[::L]), np.ascontiguousarray(X[::L])
    g = H.lowa_sg(dY, X, rows, cols, h, gamma)
    assert np.array_equal(f2u(g), f2u(g_step)), "pruned / replayed call disagrees with the step"
    cb, hb, gb = c.place_u32(t["centres"]), c.mem.place(h), c.mem.alloc(256)
    c.d.update(op=0, a0=4 * c.mem.place(dY), a1=4 * 768, a2=4 * c.mem.place(X), a3=4 * 768,
               a4=4 * cb, a5=4 * hb, a6=4 * gb, a7=int(f2u(np.array([gamma], f32))[0]), len=dY.shape[0])
    c.exp.append((gb, g))
    return c


def scalars(s, ttot, b1t_prev, b2t_prev):
    inv_w, inv_tw = f32(1.0 / WARMUP), f32(1.0 / (ttot - WARMUP))
    return H.step_scalars(s, WARMUP, ttot, inv_w, inv_tw, LR0, WD, B1, B2, b1t_prev, b2t_prev), inv_w, inv_tw


def prev_products(s):
    b1t, b2t = f32(1.0), f32(1.0)
    for _ in range(s):
        b1t, b2t = H.mul(b1t, B1)[0], H.mul(b2t, B2)[0]
    return b1t, b2t


def scal_case(c, s, ttot=3000):
    b1p, b2p = prev_products(s)
    o, inv_w, inv_tw = scalars(s, ttot, b1p, b2p)
    const = np.array([inv_w, inv_tw, LR0[0], LR0[1], WD, B1, B2, B1C, B2C, EPS, 0, 0, 0, 0, 0, 0], f32)
    clip = f32(0.37)
    block = np.zeros(24, f32)
    block[0], block[1], block[15], block[23] = b1p, b2p, clip, clip
    cb, sb = 4 * c.mem.place(const), 4 * c.mem.place(block)
    c.d.update(op=1, a0=s, a1=WARMUP, a2=ttot, a3=cb, a4=sb)
    exp = np.array([o["b1t"], o["b2t"], o["bc1"], o["rbc2"], o["fr"], o["lr"][0], o["lr"][1], 0,
                    o["decay"][0], o["step"][0], o["rbc2"], B1C, B2, B2C, EPS, clip,
                    o["decay"][1], o["step"][1], o["rbc2"], B1C, B2, B2C, EPS, clip], f32)
    c.exp.append((sb // 4, exp))
    return c


def adamw_build(c, p0, g, m0, v0, s, grp, clip_en, clipc, ttot=3000):
    b1p, b2p = prev_products(s)
    o, _, _ = scalars(s, ttot, b1p, b2p)
    sc = np.array([o["decay"][grp], o["step"][grp], o["rbc2"], B1C, B2, B2C, EPS], f32)
    p, m, v = p0.copy(), m0.copy(), v0.copy()
    gg = H.mul(g, clipc) if clip_en else g.copy()
    H.adamw(p, np.ascontiguousarray(gg), m, v, sc)
    block = np.concatenate([sc, [clipc]]).astype(f32)
    pb, gb, mb, vb, sb = (c.mem.place(x) for x in (p0, g, m0, v0, block))
    c.d.update(op=2, a0=4 * pb, a1=4 * gb, a2=4 * mb, a3=4 * vb, a4=4 * sb, len=p0.size, flag=int(clip_en))
    c.exp += [(pb, p), (mb, m), (vb, v)]
    return c


def adamw_unit(c, s, clip_en, clipc):
    r = np.random.default_rng(11 + s)
    n = 100
    p, g = rand_mat(r, (n,), 118, 130), rand_mat(r, (n,), 110, 130)
    m, v = rand_mat(r, (n,), 110, 128), np.abs(rand_mat(r, (n,), 100, 120))
    if s == 0:
        m[:], v[:] = 0, 0
    return adamw_build(c, p, g, m, v, s, 0, clip_en, f32(clipc))


def adamw_real(c, head):
    inp = np.load(os.path.join(RUNS, "stepB_l0f32.npz"))
    l1 = np.load(os.path.join(RUNS, "stepB_l1.npz"))
    s, ttot = int(inp["step_s"]), EPOCHS * int(inp["n_batches"])
    clipc = f32(l1["clip"])
    keys = ("dw", "db", "ow", "ob") if head else ("coef",)
    cat = lambda tag: np.concatenate([np.asarray(inp[tag + k] if (tag + k) in inp.files else l1[tag + k], f32).reshape(-1) for k in keys])
    p0, m0, v0 = cat("pre_"), cat("pre_m_"), cat("pre_v_")
    g = np.concatenate([np.asarray(l1["g_" + k], f32).reshape(-1) for k in keys])
    adamw_build(c, p0, g, m0, v0, s, 1 if head else 0, True, clipc, ttot)
    # cross-check against the L1 step's own post values (the same entry points, so this must hold)
    l1cat = lambda tag: np.concatenate([np.asarray(l1[tag + k], f32).reshape(-1) for k in keys])   # L1's, not L0's post_*
    post = [l1cat("post_"), l1cat("post_m_"), l1cat("post_v_")]
    for (a, arr), want in zip(c.exp[-3:], post):
        assert np.array_equal(f2u(arr), f2u(want)), "generator disagrees with stepB_l1 post values"
    return c


def build(case, out):
    c = Case(case, 2 if case.endswith("_gemm") else 1 if case.startswith("merge") else 0)
    fn = {"merge_unit": merge_unit, "merge_real_row": lambda c: merge_real(c, False), "merge_real_trn": lambda c: merge_real(c, True),
          "sg_unit": sg_unit, "sg_edge": sg_edge, "merge_gemm": lambda c: gemm_qproj(c, True), "nomerge_gemm": lambda c: gemm_qproj(c, False),
          "sg_real_l0q": lambda c: sg_real(c, 0, False, out), "sg_real_l11q": lambda c: sg_real(c, 22, True, out),
          "scal_s0": lambda c: scal_case(c, 0), "scal_peak": lambda c: scal_case(c, 140),
          "adamw_s0": lambda c: adamw_unit(c, 0, True, 0.5), "adamw_clip": lambda c: adamw_unit(c, 140, True, 0.123),
          "adamw_noclip": lambda c: adamw_unit(c, 700, False, 1.0),
          "adamw_real_coef": lambda c: adamw_real(c, False), "adamw_real_head": lambda c: adamw_real(c, True)}[case]
    return fn(c)


def main():
    case, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    c = build(case, out)
    c.write(out)
    print(f"{case}: {sum(a.size for _, a in c.exp)} expected words")


if __name__ == "__main__":
    main()
