#!/usr/bin/env python3
"""Program interpreter (P6c): walks a compiled LoWA-TA program descriptor by descriptor, decodes it exactly as the
sequencer does (sw/isa.py: header, REL term adds, LOOP / SETCSR / FENCE / END) and executes every engine
descriptor on a flat DRAM image with the L1 per-op entry points (model/hwmodel), following each engine's
descriptor contract (the semantics the P3-P5 engine testbench generators encode).  Program order is the
execution order (the FENCEs make the concurrent schedule equivalent to it).

Used by `make compiler-check` (sw/check_compiler.py: interpreter vs the L1 step) and as the golden of the
reduced-program RTL run.
"""
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import as_strided

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model"))
import hwmodel as H  # noqa: E402
import isa  # noqa: E402
import lowa_tables as LT  # noqa: E402
import rng  # noqa: E402

f32, U32 = np.float32, np.uint32
D = 768
DS = rng.DROP_SCALE
SITES = {v: k for k, v in rng.SITE.items()}


def bits2f(u):
    return np.array([u], U32).view(f32)[0]


class Machine:
    def __init__(self, nbytes, seed, step_t, slot_delta=0, slot=0):
        self.mem = np.zeros(nbytes // 4, U32)
        self.mf = self.mem.view(f32)
        self.seed, self.step_t = seed, step_t
        self.term = [0] * 16
        self.term[isa.T_STEP] = step_t
        self.term[isa.T_SLOT] = slot_delta if slot else 0
        self.str = [0] * 16
        self.gn = []
        self.kcache = {}
        self.trace = []

    # ------------------------------------------------------------------ memory helpers
    def load(self, img):
        for a, arr in img.reg:
            self.mem[a // 4:a // 4 + arr.size] = arr

    def view(self, base, shape, strides, dtype=f32):
        assert base % 4 == 0 and all(s % 4 == 0 for s in strides)
        arr = self.mf if dtype == f32 else self.mem.view(dtype)
        hi = base + sum((n - 1) * s for n, s in zip(shape, strides)) + 4
        assert min(shape) > 0 and hi <= self.mem.size * 4, f"access {base:#x}..{hi:#x} outside the image"
        return as_strided(arr[base // 4:], shape=shape, strides=strides, writeable=True)

    def rd(self, base, shape, strides, dtype=f32):
        return np.ascontiguousarray(self.view(base, shape, strides, dtype))

    def mat(self, base, rows, rs, cols):
        return self.rd(base, (rows, cols), (rs, 4))

    def wmat(self, base, rs, arr):
        arr = np.asarray(arr, f32)
        self.view(base, arr.shape, (rs, 4))[...] = arr

    def vec(self, base, n, dtype=f32):
        return self.rd(base, (n,), (4,), dtype)

    def keep(self, site, layer, idx):
        """keep bits of mask (seed, STEP_T, site, layer) at element indices idx (any shape)."""
        n = int(idx.max()) + 1
        key = (site, layer)
        if key not in self.kcache or self.kcache[key].size < n:
            self.kcache[key] = rng.keep_mask((n,), self.seed, self.step_t, SITES[site], layer)
        return self.kcache[key][idx]

    # ------------------------------------------------------------------ engines
    def merged(self, f, s, slot, base, rows, cols, rs):
        """stored W segment [rows x cols] with the LoWA atoms of table set `slot` added (merge-on-load)."""
        W0 = self.mat(base, rows, rs, cols)
        hits = self.vec(f[f"m_hits_{slot}"], 6400, U32).astype(np.int64)
        dirw = self.vec(f[f"m_dir_{slot}"], 256, U32).astype(np.int64)
        eflat = np.full(6400, -1, np.int64)
        for blk in range(256):
            st, cnt = dirw[blk] & 0xFFFF, (dirw[blk] >> 16) & 0xFF
            a, b = blk // 16, blk % 16
            for hv in hits[st:st + cnt]:
                r, c_, tap, j = hv & 63, (hv >> 6) & 63, (hv >> 12) & 31, (hv >> 17) & 255
                eflat[25 * j + tap] = (64 * a + r) * cols + 64 * b + c_
        assert (eflat >= 0).all(), "hit tables do not cover every atom entry"
        coef = self.vec(f[f"m_coef_{slot}"], 256)
        h = self.vec(f["m_h_base"], 25)
        return H.lowa_merge(W0, eflat, coef, h, bits2f(f["m_gamma"]))

    def gemm(self, f):
        M, N, nb1, nb2 = f["m"], f["n"], f["nb1"], f["nb2"]
        A, Bm = [], []
        for i in range(3):
            k = f[f"k{i}"]
            if not k:
                continue
            ab, bb = f[f"a_base{i}"], f[f"b_base{i}"]
            if f["a_trn"]:
                a = self.rd(ab, (nb1, nb2, M, k), (f["a_bs1"], f["a_bs2"], 4, f["a_rs"]))
            else:
                a = self.rd(ab, (nb1, nb2, M, k), (f["a_bs1"], f["a_bs2"], f["a_rs"], 4))
            ms = f[f"m_ms{i}"] if f["m_en"] else 0
            if ms:
                assert nb1 == nb2 == 1
                rows, cols = (k, N) if f["b_trn"] else (N, k)
                Wm = self.merged(f, i, "ab"[ms - 1], bb, rows, cols, f["b_rs"])
                b = (Wm.T if f["b_trn"] else Wm)[None, None]
            elif f["b_trn"]:
                b = self.rd(bb, (nb1, nb2, N, k), (f["b_bs1"], f["b_bs2"], 4, f["b_rs"]))
            else:
                b = self.rd(bb, (nb1, nb2, N, k), (f["b_bs1"], f["b_bs2"], f["b_rs"], 4))
            A.append(a)
            Bm.append(b)
        for s in A[:-1]:
            assert s.shape[-1] % 64 == 0, "GEMM_ACC2 segment not a whole number of K chunks"
        A = np.ascontiguousarray(np.concatenate(A, -1), f32)
        Bm = np.ascontiguousarray(np.concatenate(Bm, -1), f32)
        K = A.shape[-1]
        C = np.empty((nb1, nb2, M, N), f32)
        H.gemm_raw(nb1, nb2, M, N, K, A, (nb2 * M * K, M * K, K, 1), Bm, (nb2 * N * K, N * K, K, 1), C,
                   (nb2 * M * N, M * N, N, 1))
        cst = (f["c_bs1"], f["c_bs2"], f["c_rs"], 4)
        Y = C
        if f["ep_bias"]:
            Y = H.add_bias(Y, self.vec(f["bias_base"], N))
        if f["ep_scale"]:
            Y = H.scale(Y, f32(2.0 ** f["ep_sexp"]))
        if f["ep_gelu"]:
            self.view(f["o2_base"], Y.shape, cst)[...] = Y
            Y = H.gelu(Y)
        if f["ep_gelu_bwd"]:
            Y = H.gelu_bwd(Y, self.rd(f["aux_base"], Y.shape, cst))
        if f["ep_drop"]:
            idx = (np.arange(nb1)[:, None, None, None] * f["c_bs1"] + np.arange(nb2)[None, :, None, None] * f["c_bs2"]
                   + np.arange(M)[None, None, :, None] * f["c_rs"]) // 4 + np.arange(N)[None, None, None, :]
            Y = H.drop(Y, self.keep(f["site"], f["layer"], idx), bits2f(f["ep_ds"]))
        if f["ep_resid"]:
            Y = H.add(Y, self.rd(f["aux_base"], Y.shape, cst))
        self.view(f["c_base"], Y.shape, cst)[...] = Y

    def vmask(self, f, rows, cols):
        idx = f["mk_base"] + np.arange(rows)[:, None] * f["mk_rs"] + np.arange(cols)[None, :]
        return self.keep(f["site"], f["layer"], idx)

    def kmask(self, f, nb, L):
        w = self.rd(f["cv0_base"], (nb, (L + 31) // 32), (f["km_rs"], 4), U32)
        j = np.arange(L)
        return ((w[:, j >> 5] >> (j & 31).astype(U32)) & 1).astype(np.uint8)

    def vecop(self, f):
        vop = isa.VOPS[f["vop"]]
        rows, cols = f["rows"], f["cols"]
        A = lambda: self.mat(f["a_base"], rows, f["a_rs"], cols)
        Bv = lambda: self.mat(f["b_base"], rows, f["b_rs"], cols)
        out = lambda y: self.wmat(f["c_base"], f["c_rs"], np.asarray(y, f32).reshape(rows, cols))
        st = lambda k: self.vec(f[k], rows)
        if vop in ("LN_FWD", "EMB"):
            assert f["c0"] == int(np.array([1.0 / cols], f32).view(U32)[0]) and f["c1"] == int(H.LN_EPS.view(U32))
        if vop == "LN_FWD":
            y, mu, rs = H.ln_fwd(A(), self.vec(f["cv1_base"], cols), self.vec(f["cv2_base"], cols))
            out(y)
            if f["st_en"]:
                self.mf[f["st0_base"] // 4:f["st0_base"] // 4 + rows] = mu
                self.mf[f["st1_base"] // 4:f["st1_base"] // 4 + rows] = rs
        elif vop == "LN_BWD":
            out(H.ln_bwd(A(), st("st0_base"), st("st1_base"), self.vec(f["cv0_base"], cols), Bv()))
        elif vop in ("SMAX_FWD", "SMAX_REC"):
            L = cols
            nb, nh = rows // f["rows_per_b"], f["rows_per_b"] // L
            S = A().reshape(nb, nh, L, L)
            km = self.kmask(f, nb, L)
            if vop == "SMAX_FWD":
                _, Pd, m, r = H.smax_fwd(S, km, self.vmask(f, rows, cols).reshape(S.shape), bits2f(f["ds"]))
                out(Pd)
                if f["st_en"]:
                    self.mf[f["st0_base"] // 4:f["st0_base"] // 4 + rows] = m.reshape(-1)
                    self.mf[f["st1_base"] // 4:f["st1_base"] // 4 + rows] = r.reshape(-1)
            else:   # P = mul(exp(sub(s, m)), r) with the stashed m, r; masked keys -> +0
                m, r = st("st0_base"), st("st1_base")
                Sx = S.reshape(rows, cols)
                e = H.exp(H.sub(Sx, np.repeat(m[:, None], cols, 1)))
                kmr = np.repeat(km, nh * L, 0)
                e = np.where(kmr.astype(bool), e, f32(0.0)).astype(f32)
                out(H.mul(e, np.repeat(r[:, None], cols, 1)))
        elif vop == "SMAX_BWD":
            out(H.smax_bwd(A(), Bv(), self.vmask(f, rows, cols), bits2f(f["ds"]), bits2f(f["c0"])))
        elif vop == "EMB":
            L = f["emb_l"]
            ids = self.vec(f["ids_base"], rows, np.int32).astype(np.int64).reshape(-1, L)
            nz = (ids != 1).astype(np.int64)
            pos = (np.cumsum(nz, 1) * nz + 1).reshape(-1)
            ids = ids.reshape(-1)
            word = np.stack([self.vec(f["a_base"] + int(i) * f["a_rs"], cols) for i in ids])
            ptab = np.stack([self.vec(f["b_base"] + int(p) * f["b_rs"], cols) for p in pos])
            e = H.add(H.add(word, self.vec(f["cv0_base"], cols)), ptab)
            y, _, _ = H.ln_fwd(e, self.vec(f["cv1_base"], cols), self.vec(f["cv2_base"], cols))
            out(H.drop(y, self.vmask(f, rows, cols), bits2f(f["ds"])))
        elif vop in ("CE", "MSE"):
            z = A()
            w = self.vec(f["st1_base"], rows)
            if vop == "CE":
                loss, dz = H.ce(z, self.vec(f["st0_base"], rows, np.int32).astype(np.int64), w, bits2f(f["c0"]))
            else:
                loss, dz = H.mse(z, self.vec(f["st0_base"], rows), w, bits2f(f["c0"]))
            out(dz)
            self.mf[f["o2_base"] // 4] = loss
        elif vop == "GN_ROWS":
            if f["g_reset"]:
                self.gn = []
            g = A()
            norms, _, _ = H.gnorm_clip(g.reshape(-1).copy(), [0, g.size], [cols])
            self.gn.append(g)
            self.mf[f["o2_base"] // 4] = norms[0]
        elif vop == "GN_FIN":
            off = np.cumsum([0] + [g.size for g in self.gn])
            flat = np.concatenate([g.reshape(-1) for g in self.gn]).astype(f32)
            _, G, c = H.gnorm_clip(flat, off, [g.shape[1] for g in self.gn], eps=bits2f(f["c0"]))
            self.mf[f["o2_base"] // 4:f["o2_base"] // 4 + 2] = [G, c]
        elif vop == "E_ADD":
            out(H.add(A(), Bv()))
        elif vop == "E_MUL":
            out(H.mul(A(), Bv()))
        elif vop == "E_SCALE":
            out(H.mul(A(), bits2f(f["c0"])))
        elif vop == "E_DROP":
            out(H.drop(A(), self.vmask(f, rows, cols), bits2f(f["ds"])))
        elif vop == "E_TANH":
            out(H.tanh(A()))
        elif vop == "E_TANHB":
            out(H.tanh_bwd(Bv(), A()))
        elif vop == "E_COPY":
            out(A())
        else:
            raise NotImplementedError(vop)

    def lowa(self, f):
        if f["op"] == 0:
            n = f["len"]
            dY = self.mat(f["a0"], n, f["a1"], D)
            X = self.mat(f["a2"], n, f["a3"], D)
            ctr = self.vec(f["a4"], 256, U32).astype(np.int64)
            ci, cj = ctr >> 16, ctr & 0xFFFF
            di = np.array([t[0] for t in LT.TAPS])
            dj = np.array([t[1] for t in LT.TAPS])
            rows, cols = LT.reflect(ci[:, None] + di[None], D), LT.reflect(cj[:, None] + dj[None], D)
            g = H.lowa_sg(dY, X, rows, cols, self.vec(f["a5"], 25), bits2f(f["a7"]))
            self.mf[f["a6"] // 4:f["a6"] // 4 + 256] = g
        elif f["op"] == 1:
            cst = self.vec(f["a3"], 10)
            blk = self.vec(f["a4"], 24).copy()
            o = H.step_scalars(int(f["a0"]), int(f["a1"]), int(f["a2"]), cst[0], cst[1], (cst[2], cst[3]), cst[4],
                               cst[5], cst[6], blk[0], blk[1])
            new = [o["b1t"], o["b2t"], o["bc1"], o["rbc2"], o["fr"], o["lr"][0], o["lr"][1], 0.0]
            for g in range(2):
                new += [o["decay"][g], o["step"][g], o["rbc2"], cst[7], cst[6], cst[8], cst[9], blk[15 + 8 * g]]
            self.mf[f["a4"] // 4:f["a4"] // 4 + 24] = np.array(new, f32)
        else:
            n = f["len"]
            sl = lambda k: slice(f[k] // 4, f[k] // 4 + n)
            sc = self.vec(f["a4"], 8)
            g = self.mf[sl("a1")].copy()
            if f["flag"]:
                g = H.mul(g, sc[7])
            p, m, v = self.mf[sl("a0")].copy(), self.mf[sl("a2")].copy(), self.mf[sl("a3")].copy()
            H.adamw(p, np.ascontiguousarray(g), m, v, np.ascontiguousarray(sc[:7]))
            self.mf[sl("a0")], self.mf[sl("a2")], self.mf[sl("a3")] = p, m, v

    def dma(self, f):
        for i2 in range(f["n2"]):
            for i1 in range(f["n1"]):
                s = (f["src"] + i2 * f["s_s2"] + i1 * f["s_s1"]) // 4
                d = (f["dst"] + i2 * f["d_s2"] + i1 * f["d_s1"]) // 4
                self.mem[d:d + f["n0"]] = self.mem[s:s + f["n0"]].copy()

    # ------------------------------------------------------------------ sequencer
    def run(self, prog_base, max_desc=1 << 20, verbose=False):
        pc = prog_base
        loop = None
        n = 0
        while True:
            w0 = int(self.mem[pc // 4])
            nsl = ((w0 >> 21) & 3) + 1
            words = self.mem[pc // 4:pc // 4 + 16 * nsl]
            h, name, f, _ = isa.decode_words(words, self.term)
            nxt = pc + 64 * nsl
            n += 1
            assert n < max_desc
            self.trace.append((pc, name))
            if name == "LOOP":
                loop = dict(start=nxt, end=nxt + f["body"], left=f["count"], dir=f["dir"])
                self.term[isa.T_LIDX] = f["start"]
                for k in range(1, 13):
                    self.term[k] = 0
            elif name == "SETCSR":
                if 1 <= f["csr"] <= 12:
                    self.str[f["csr"]] = f["value"]
            elif name == "END":
                self.step_t += 1
                return n
            elif name in ("FENCE", "NOP"):
                pass
            elif name in ("GEMM", "GEMM_ACC2"):
                self.gemm(f)
            elif name in ("VEC", "LOSS", "GNORM"):
                self.vecop(f)
            elif name in ("LOWA_SG", "LOWA_SCAL", "ADAMW"):
                self.lowa(f)
            elif name == "DMA_COPY":
                self.dma(f)
            if verbose:
                print(f"{pc:#x} {name} {isa.VOPS[f['vop']] if 'vop' in f else ''}", flush=True)
            pc = nxt
            if loop is not None and pc == loop["end"]:
                loop["left"] -= 1
                if loop["left"] > 0:
                    sgn = -1 if loop["dir"] else 1
                    self.term[isa.T_LIDX] += sgn
                    for k in range(1, 13):
                        self.term[k] = (self.term[k] + sgn * self.str[k]) & 0xFFFFFFFF
                    pc = loop["start"]
                else:
                    loop = None
