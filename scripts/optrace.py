#!/usr/bin/env python
"""Scalar-level operation traces of ONE training update step on ONE adapted
RoBERTa-base module (query, 768 x 768; b = 2 tokens), for the seven methods of the paper's
Table III / hwcost table.  Every row is one scalar MULT / ADD (or a rare
DIV / SQRT / MAX / MIN / ROUND).  Plus `lowa_sampled`: LoWA with the opt-in
sampled (SDDMM) coefficient backward, SGWALinear(sampled_grad=True).

  env/bin/python scripts/optrace.py --plan            # count pass + torch verification only
  env/bin/python scripts/optrace.py --write -j 40     # + write the gzip traces
  env/bin/python scripts/optrace.py --readback -j 40  # count rows FROM the files, replay-check
  # forward side only (1_build, 2_fold, 1_forward) at another token count, own folder:
  env/bin/python scripts/optrace.py --forward_only --b 4096 --out results/optrace_forward_b4096 --write --readback -j 40

CPU only (CUDA_VISIBLE_DEVICES is forced to "" here; no GPU is touched).
Outputs: results/optrace/  (<method>.csv.gz or <method>.partNNNN.csv.gz shards,
counts_trace.csv, counts_roberta_step.csv, counts_gemma_step.csv [formula only], plan.json).  See results/optrace/README.md.

Design.  Each method is a deterministic PROGRAM (prog_<method>) that computes
every scalar in numpy float32 in a fixed execution order and hands each
vectorised block of ops to Tracer.emit (one "step").  A count pass runs the
program once (counts, verification against the repo's torch modules); the
writer re-runs the same program in worker processes, each formatting only its
contiguous range of steps (one shard = contiguous steps, <= SHARD_ROWS rows),
so shard order == execution order and temporaries t<N> are globally unique
(SSA: every Op name is written exactly once).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse, ctypes, csv, glob, hashlib, json, math, subprocess, sys, tempfile, time, zlib
from collections import defaultdict
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import hw_cost as H                                              # noqa: E402
from haar_adapter import (haar_lengths, haar_analysis_unnorm,    # noqa: E402
                          haar_synthesis_unnorm)
from qwha_adapter import _wht_sync_free                          # noqa: E402
from qwha_hadamard import get_hadK                               # noqa: E402
from loca_dct_utils import DCT_CACHE                             # noqa: E402

OUT = os.path.join(ROOT, "results", "optrace")
D = 768                        # roberta-base hidden; query / value are 768 x 768
B_TOK = 2                      # reduced token count of the trace (--b overrides)
FWD_ONLY = False               # --forward_only: trace 1_build/2_fold/1_forward only, stop before backward
FWD_PHASES = ("1_build", "2_fold", "1_forward")
TOK_BLOCK = 16                 # LoCA / LYRA per-token forward is emitted in token blocks (bounded memory)
REAL_B, REAL_MODULES = 32 * 128, 24          # roberta-base step: batch 32 x max_length 128; q,v x 12
GEMMA = dict(D=2048, B=32 * 128, MODULES=36)  # formula-only second table
SHARD_ROWS = 50_000_000
CH = 1 << 20                   # rows-per-group formatted per C call
METHODS = ["lowa", "lowa_sampled", "fourierft", "wave1", "wave2", "loca", "qwha", "lyra"]
# AdamW (torch.optim.AdamW, single-tensor kernel); state is synthetic, step 10 -> 11
LR, BETA1, BETA2, EPS, WD, STEP = 5e-2, 0.9, 0.999, 1e-8, 0.01, 11
f32 = np.float32
PHASES = ["1_build", "1_forward", "2_fold", "3_backward", "4_adamw"]
OPS = ["MULT", "ADD", "DIV", "SQRT", "MAX", "MIN", "ROUND"]
OPC = {o: i for i, o in enumerate(OPS)}
PHC = {p: i for i, p in enumerate(PHASES)}

# --------------------------------------------------------------------------- #
#  C row formatter (compiled once per run)                                     #
# --------------------------------------------------------------------------- #
C_SRC = r"""
#include <stdint.h>
#include <string.h>
#include <stdio.h>
static inline char *put_i64(char *o, int64_t v) {
    char b[24]; int n = 0;
    if (v < 0) { *o++ = '-'; v = -v; }
    do { b[n++] = '0' + (char)(v % 10); v /= 10; } while (v);
    while (n) *o++ = b[--n];
    return o;
}
static inline char *put_name(char *o, int16_t pid, int64_t i, int64_t j, const char *pre,
                             const int32_t *poff, const int32_t *plen, const int8_t *pkind) {
    if (pid < 0) return o;
    int neg = pid & 0x4000; pid &= 0x3fff;
    int q = (pkind[pid] == 0 && j >= 0);          /* "P[i,j]" contains a comma: CSV-quote it */
    if (q) *o++ = '"';
    if (neg) *o++ = '-';
    memcpy(o, pre + poff[pid], plen[pid]); o += plen[pid];
    int8_t k = pkind[pid];
    if (k == 1) { o = put_i64(o, i); }
    else if (k == 0) {
        *o++ = '['; o = put_i64(o, i);
        if (j >= 0) { *o++ = ','; o = put_i64(o, j); }
        *o++ = ']';
        if (q) *o++ = '"';
    }
    return o;
}
static inline char *put_f(char *o, float v) { return o + snprintf(o, 32, "%.9g", (double)v); }
int64_t fmt_rows(int64_t n, const uint8_t *op, const uint8_t *ph,
                 const int16_t *pa, const int64_t *ia, const int64_t *ja,
                 const int16_t *pb, const int64_t *ib, const int64_t *jb,
                 const int16_t *po, const int64_t *io, const int64_t *jo,
                 const float *va, const float *vb, const float *vo,
                 const char *pre, const int32_t *poff, const int32_t *plen, const int8_t *pkind,
                 const char *ops, const int32_t *ooff, const int32_t *olen,
                 const char *phs, const int32_t *phoff, const int32_t *phlen, char *out) {
    char *o = out;
    for (int64_t r = 0; r < n; r++) {
        memcpy(o, ops + ooff[op[r]], olen[op[r]]); o += olen[op[r]]; *o++ = ',';
        o = put_name(o, pa[r], ia[r], ja[r], pre, poff, plen, pkind); *o++ = ',';
        o = put_name(o, pb[r], ib[r], jb[r], pre, poff, plen, pkind); *o++ = ',';
        o = put_name(o, po[r], io[r], jo[r], pre, poff, plen, pkind); *o++ = ',';
        o = put_f(o, va[r]); *o++ = ',';
        if (pb[r] >= 0) o = put_f(o, vb[r]);
        *o++ = ',';
        o = put_f(o, vo[r]); *o++ = ',';
        memcpy(o, phs + phoff[ph[r]], phlen[ph[r]]); o += phlen[ph[r]];
        *o++ = '\n';
    }
    return (int64_t)(o - out);
}
"""
_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        h = hashlib.sha1(C_SRC.encode()).hexdigest()[:12]
        d = os.path.join(tempfile.gettempdir(), f"optrace_{h}")
        so = os.path.join(d, "libfmt.so")
        if not os.path.exists(so):
            os.makedirs(d, exist_ok=True)
            src = os.path.join(d, "fmt.c")
            open(src, "w").write(C_SRC)
            tmp = so + f".{os.getpid()}"
            subprocess.check_call(["gcc", "-O3", "-shared", "-fPIC", "-o", tmp, src])
            os.replace(tmp, so)
        _LIB = ctypes.CDLL(so)
        _LIB.fmt_rows.restype = ctypes.c_int64
    return _LIB


def _tab(strs):
    b = "".join(strs).encode()
    off = np.cumsum([0] + [len(s.encode()) for s in strs[:-1]]).astype(np.int32)
    ln = np.array([len(s.encode()) for s in strs], np.int32)
    return b, off, ln


# --------------------------------------------------------------------------- #
#  Symbolic names: (prefix id, i, j); kind 0 = P[i] / P[i,j], 1 = temp t<i>,   #
#  2 = literal / named constant.  Bit 0x4000 of the prefix id = negated operand #
# --------------------------------------------------------------------------- #
PREFIXES, PID = [], {}
NEG = np.int16(0x4000)


def pid(name, kind=0):
    key = (name, kind)
    if key not in PID:
        PID[key] = len(PREFIXES)
        PREFIXES.append(key)
    return PID[key]


class N:
    __slots__ = ("p", "i", "j")

    def __init__(self, p, i=-1, j=-1, raw=False):
        if raw:
            self.p, self.i, self.j = p, i, j
            return
        self.p, self.i, self.j = np.broadcast_arrays(np.asarray(p, np.int16),
                                                     np.asarray(i, np.int64),
                                                     np.asarray(j, np.int64))

    @property
    def shape(self):
        return self.p.shape

    def __getitem__(self, k):
        return N(self.p[k], self.i[k], self.j[k], raw=True)

    def reshape(self, *s):
        return N(self.p.reshape(*s), self.i.reshape(*s), self.j.reshape(*s), raw=True)

    @property
    def T(self):
        return N(self.p.T, self.i.T, self.j.T, raw=True)

    def neg(self):
        return N(self.p ^ NEG, self.i, self.j, raw=True)

    def copy(self):
        return N(self.p.copy(), self.i.copy(), self.j.copy(), raw=True)

    def __setitem__(self, k, o):
        self.p[k], self.i[k], self.j[k] = o.p, o.i, o.j

    @staticmethod
    def cat(lst, axis=-1):
        return N(np.concatenate([x.p for x in lst], axis), np.concatenate([x.i for x in lst], axis),
                 np.concatenate([x.j for x in lst], axis), raw=True)

    def bc(self, shape):
        return N(np.broadcast_to(self.p, shape), np.broadcast_to(self.i, shape),
                 np.broadcast_to(self.j, shape), raw=True)


def lit(name):
    return N(pid(name, 2))


def nm(name, i=-1, j=-1):
    return N(pid(name), i, j)


def ar(n):
    return np.arange(n, dtype=np.int64)


def grid(name, R, C):
    return nm(name, ar(R)[:, None], ar(C)[None, :])


# --------------------------------------------------------------------------- #
#  Tracer                                                                       #
# --------------------------------------------------------------------------- #
class Done(Exception):
    pass


class Tracer:
    def __init__(self, method, mode="count", shard_of_step=None, shards=()):
        self.method, self.mode = method, mode
        self.next_tmp, self.step = 0, 0
        self.counts = defaultdict(int)
        self.step_rows = []
        self.sos = shard_of_step
        self.shards = set(shards)
        self.last_step = (max(i for i, s in enumerate(shard_of_step) if s in self.shards)
                          if mode == "write" else None)
        self.cur, self.fh, self.z = None, None, None
        self.written = defaultdict(int)
        self.bytes_raw = 0

    def tmp(self, shape):
        n = int(np.prod(shape))
        base = self.next_tmp
        self.next_tmp += n
        return N(pid("t", 1), (base + ar(n)).reshape(shape))

    # one step = one vectorised block; groups are interleaved element-major
    def emit(self, phase, groups):
        shape = np.shape(groups[0][6])
        L = int(np.prod(shape))
        sid = self.step
        self.step += 1
        for g in groups:
            self.counts[(phase, g[0])] += L
        if self.mode == "count":
            self.step_rows.append(L * len(groups))
            return
        if self.sos[sid] not in self.shards:
            if sid > self.last_step:
                raise Done
            return
        self._open(self.sos[sid])
        flat = []
        for (op, A, Bn, O, va, vb, vo) in groups:
            def fl(x, dt):
                return np.ascontiguousarray(np.broadcast_to(np.asarray(x, dt), shape)).reshape(-1)
            ent = [OPC[op]]
            for nme in (A, Bn, O):
                if nme is None:
                    ent += [np.full(1, -1, np.int16), np.zeros(1, np.int64), np.zeros(1, np.int64)]
                else:
                    ent += [fl(nme.p, np.int16), fl(nme.i, np.int64), fl(nme.j, np.int64)]
            ent += [fl(va, f32), fl(0 if vb is None else vb, f32) if vb is not None else np.zeros(1, f32),
                    fl(vo, f32)]
            flat.append((ent, Bn is None))
        G = len(groups)
        pre, poff, plen = _tab([p for p, _ in PREFIXES])
        pkind = np.array([k for _, k in PREFIXES], np.int8)
        ops, ooff, olen = _tab(OPS)
        phs, phoff, phlen = _tab(PHASES)
        for c0 in range(0, L, CH):
            c1 = min(L, c0 + CH)
            n = c1 - c0
            cols = []
            for f in range(10):
                if f == 0:
                    a = np.empty((n, G), np.uint8)
                    for g, (ent, _) in enumerate(flat):
                        a[:, g] = ent[0]
                else:
                    dt = ent_dtype(f)
                    a = np.empty((n, G), dt)
                    for g, (ent, un) in enumerate(flat):
                        x = ent[f]
                        a[:, g] = x[0] if x.size == 1 and L > 1 or (un and f in (4, 5, 6, 8)) else x[c0:c1]
                cols.append(a.reshape(-1))
            ph = np.full(n * G, PHC[phase], np.uint8)
            R = n * G
            buf = np.empty(R * 200 + 1024, np.uint8)
            P = lambda a: a.ctypes.data_as(ctypes.c_void_p)
            nb = lib().fmt_rows(ctypes.c_int64(R), P(cols[0]), P(ph), P(cols[1]), P(cols[2]), P(cols[3]),
                                P(cols[4]), P(cols[5]), P(cols[6]), P(cols[7]), P(cols[8]), P(cols[9]),
                                P(ent_vals(flat, 0, c0, c1, G, L)), P(ent_vals(flat, 1, c0, c1, G, L)),
                                P(ent_vals(flat, 2, c0, c1, G, L)),
                                ctypes.c_char_p(pre), P(poff), P(plen), P(pkind),
                                ctypes.c_char_p(ops), P(ooff), P(olen),
                                ctypes.c_char_p(phs), P(phoff), P(phlen), P(buf))
            mv = memoryview(buf)[:nb]
            self.bytes_raw += nb
            self.fh.write(self.z.compress(mv))
        for g in groups:
            self.written[(phase, g[0])] += L

    def _open(self, s):
        if self.cur == s:
            return
        self.close()
        self.cur = s
        path = shard_path(self.method, s, self.nshards)
        self.fh = open(path + ".tmp", "wb")
        self.z = zlib.compressobj(1, zlib.DEFLATED, 31)
        self.fh.write(self.z.compress(b"OPERATION,InA,InB,Op,InA_val,InB_val,Op_val,PHASE\n"))

    def close(self):
        if self.fh is not None:
            self.fh.write(self.z.flush())
            self.fh.close()
            path = shard_path(self.method, self.cur, self.nshards)
            os.replace(path + ".tmp", path)
            self.fh = None


def ent_dtype(f):
    return np.int16 if f in (1, 4, 7) else np.int64


def ent_vals(flat, which, c0, c1, G, L):
    n = c1 - c0
    a = np.empty((n, G), f32)
    for g, (ent, un) in enumerate(flat):
        x = ent[10 + which]
        a[:, g] = x[0] if (x.size == 1 and L > 1) or (un and which == 1) else x[c0:c1]
    return a.reshape(-1)


def shard_path(method, s, nshards):
    if nshards == 1:
        return os.path.join(OUT, f"{method}.csv.gz")
    return os.path.join(OUT, f"{method}.part{s:04d}.csv.gz")


# --------------------------------------------------------------------------- #
#  Op helpers (values exact fp32: one rounding per traced op)                  #
# --------------------------------------------------------------------------- #
def mul(tr, ph, A, va, B, vb, O=None):
    vo = (np.asarray(va, f32) * np.asarray(vb, f32)).astype(f32)
    O = tr.tmp(vo.shape) if O is None else O
    tr.emit(ph, [("MULT", A, B, O, va, vb, vo)])
    return O, vo


def add(tr, ph, A, va, B, vb, O=None):
    vo = (np.asarray(va, f32) + np.asarray(vb, f32)).astype(f32)
    O = tr.tmp(vo.shape) if O is None else O
    tr.emit(ph, [("ADD", A, B, O, va, vb, vo)])
    return O, vo


def div(tr, ph, A, va, B, vb, O=None):
    vo = (np.asarray(va, f32) / np.asarray(vb, f32)).astype(f32)
    O = tr.tmp(vo.shape) if O is None else O
    tr.emit(ph, [("DIV", A, B, O, va, vb, vo)])
    return O, vo


def tree_sum(tr, ph, A, va, O):
    """Sum over the last axis by pairwise (tree) reduction: L-1 ADDs per output."""
    va = np.asarray(va, f32)
    L = va.shape[-1]
    if L == 1:
        return A[..., 0], va[..., 0]
    while L > 1:
        h = L // 2
        a, b = A[..., 0:2 * h:2], A[..., 1:2 * h:2]
        x, y = va[..., 0:2 * h:2], va[..., 1:2 * h:2]
        s = (x + y).astype(f32)
        last = (L == 2)
        on = O.bc(s.shape[:-1])[..., None] if last else tr.tmp(s.shape)
        tr.emit(ph, [("ADD", a, b, on, x, y, s)])
        if L % 2:
            A, va = N.cat([on, A[..., -1:]]), np.concatenate([s, va[..., -1:]], -1)
        else:
            A, va = on, s
        L = h + L % 2
    return A[..., 0], va[..., 0]


def scatter_add(tr, ph, tgtN, tgtv, pos, srcN, srcv, finalN):
    """index_add: tgt[pos[r]] += src[r] in row order r (chained per target).
    tgtN/tgtv/finalN flat (T,).  Returns new (names, values) flat."""
    P = pos.size
    order = np.argsort(pos, kind="stable")
    sp = pos[order]
    first = np.r_[True, sp[1:] != sp[:-1]]
    grp_start = np.maximum.accumulate(np.where(first, np.arange(P), 0))
    rank = np.empty(P, np.int64)
    rank[order] = np.arange(P) - grp_start
    lastm = np.r_[sp[1:] != sp[:-1], True]
    is_last = np.empty(P, bool)
    is_last[order] = lastm
    curN, curv = tgtN.copy(), tgtv.astype(f32).copy()
    outN = tr.tmp((P,)).copy()
    outN[is_last] = finalN[pos[is_last]]
    inN = N(np.zeros(P, np.int16), np.zeros(P, np.int64), np.zeros(P, np.int64), raw=True)
    inv = np.zeros(P, f32)
    outv = np.zeros(P, f32)
    for r in range(int(rank.max()) + 1 if P else 0):
        sel = np.nonzero(rank == r)[0]
        p_ = pos[sel]
        inN[sel] = curN[p_]
        inv[sel] = curv[p_]
        outv[sel] = (curv[p_] + srcv[sel]).astype(f32)
        curv[p_] = outv[sel]
        curN[p_] = outN[sel]
    tr.emit(ph, [("ADD", inN, srcN, outN, inv, srcv, outv)])
    return curN, curv


def gemm_gtx(tr, ph, gN, gv, xN, xv, O):
    """O[i,j] = sum_t g[t,i] x[t,j]  (dense weight gradient G^T X)."""
    b, m = gv.shape
    n = xv.shape[1]
    A = gN.T[:, None, :].bc((m, n, b))
    Bn = xN.T[None, :, :].bc((m, n, b))
    va = np.broadcast_to(gv.T[:, None, :], (m, n, b))
    vb = np.broadcast_to(xv.T[None, :, :], (m, n, b))
    Pn, pv = mul(tr, ph, A, va, Bn, vb)
    return tree_sum(tr, ph, Pn, pv, O)


def adamw(tr, pN, p, gN, g, st):
    """torch.optim.AdamW single-tensor kernel, per element:
       p*=(1-lr wd); m.lerp_(g,1-b1); v=v*b2+(1-b2) g g; p += -ss * m/(sqrt(v)/bc2s+eps)"""
    ph = "4_adamw"
    k = p.size
    sh = p.shape
    bc1 = 1 - BETA1 ** STEP
    bc2 = 1 - BETA2 ** STEP
    c_wd, c_w1, c_b2, c_1b2 = f32(1 - LR * WD), f32(1 - BETA1), f32(BETA2), f32(1 - BETA2)
    c_bc2s, c_eps, c_nss = f32(bc2 ** 0.5), f32(EPS), f32(-(LR / bc1))
    m0, v0 = st["m"], st["v"]
    mN = N(pid("m"), pN.i, pN.j)
    vN = N(pid("v"), pN.i, pN.j)
    p1N, p1 = mul(tr, ph, pN, p, lit("one_minus_lr_wd"), c_wd)
    dN, d = add(tr, ph, gN, g, mN.neg(), -m0)
    eN, e = mul(tr, ph, lit("one_minus_beta1"), c_w1, dN, d)
    m1N, m1 = add(tr, ph, mN, m0, eN, e, N(pid("m_new"), pN.i, pN.j))
    aN, a = mul(tr, ph, vN, v0, lit("beta2"), c_b2)
    ggN, gg = mul(tr, ph, gN, g, gN, g)
    gg2N, gg2 = mul(tr, ph, ggN, gg, lit("one_minus_beta2"), c_1b2)
    v1N, v1 = add(tr, ph, aN, a, gg2N, gg2, N(pid("v_new"), pN.i, pN.j))
    sq = np.sqrt(v1).astype(f32)
    sqN = tr.tmp(sh)
    tr.emit(ph, [("SQRT", v1N, None, sqN, v1, None, sq)])
    d0N, d0 = div(tr, ph, sqN, sq, lit("bc2_sqrt"), c_bc2s)
    denN, den = add(tr, ph, d0N, d0, lit("eps"), c_eps)
    qN, q = div(tr, ph, m1N, m1, denN, den)
    sN, s = mul(tr, ph, qN, q, lit("neg_step_size"), c_nss)
    pnN, pn = add(tr, ph, p1N, p1, sN, s, N(pid(pname_new(pN)), pN.i, pN.j))
    return pn, m1, v1


def pname_new(pN):
    return PREFIXES[int(pN.p.flat[0]) & 0x3fff][0] + "_new"


# --------------------------------------------------------------------------- #
#  Transforms (textbook scalar algorithms)                                     #
# --------------------------------------------------------------------------- #
def fft_pass(tr, ph, reN, re, imN, im, inverse, outRe, outIm, rowblock=128):
    """Length-N complex FFT along axis 1.  N = 2^p: radix-2 (fft_pass_r2).
    N = 3 * 2^p (768): one mixed-radix Cooley-Tukey DIT step (Cooley & Tukey
    1965, general factor; e.g. Oppenheim & Schafer 3e, 9.5): three length-N/3
    radix-2 FFTs of x[3n+r], then for every k
        X[k] = F0[k'] + W^k F1[k'] + W^(2k) F2[k'],   k' = k mod N/3,
    i.e. 2 complex twiddle multiplies (4 MULT + 2 ADD each) + 2 complex adds
    (2 ADD each) = 8 MULT + 8 ADD per output."""
    R, Nn = re.shape
    if Nn & (Nn - 1) == 0:
        return fft_pass_r2(tr, ph, reN, re, imN, im, inverse, outRe, outIm)
    assert Nn % 3 == 0 and (Nn // 3) & (Nn // 3 - 1) == 0, Nn
    M = Nn // 3
    Fs = []
    for r in range(3):
        oR, oI = tr.tmp((R, M)), tr.tmp((R, M))
        Fs.append(fft_pass_r2(tr, ph, reN[:, r::3], np.ascontiguousarray(re[:, r::3]), imN[:, r::3],
                              np.ascontiguousarray(im[:, r::3]), inverse, oR, oI))
    ang = 2.0 * np.pi * np.arange(Nn) / Nn
    sgn = 1.0 if inverse else -1.0
    twr, twi = np.cos(ang).astype(f32), (sgn * np.sin(ang)).astype(f32)
    tagr, tagi = (f"wIr{Nn}_", f"wIi{Nn}_") if inverse else (f"wFr{Nn}_", f"wFi{Nn}_")
    kk = ar(Nn)
    kp = kk % M
    outRe_v, outIm_v = np.empty((R, Nn), f32), np.empty((R, Nn), f32)
    for r0 in range(0, R, rowblock):
        r1 = min(R, r0 + rowblock)
        sl = slice(r0, r1)
        F0r, F0i = Fs[0][1][sl][:, kp], Fs[0][3][sl][:, kp]
        F0rN, F0iN = Fs[0][0][sl][:, kp], Fs[0][2][sl][:, kp]
        groups, acc_r, acc_i, accN_r, accN_i = [], F0r, F0i, F0rN, F0iN
        for q in (1, 2):
            j = (q * kk) % Nn
            wr, wi = twr[j][None, :], twi[j][None, :]
            wrN, wiN = nm(tagr, j[None, :]), nm(tagi, j[None, :])
            br_, bi_ = Fs[q][1][sl][:, kp], Fs[q][3][sl][:, kp]
            brN, biN = Fs[q][0][sl][:, kp], Fs[q][2][sl][:, kp]
            p1 = (wr * br_).astype(f32); p2 = (wi * bi_).astype(f32); trr = (p1 - p2).astype(f32)
            p3 = (wr * bi_).astype(f32); p4 = (wi * br_).astype(f32); tii = (p3 + p4).astype(f32)
            nr = (acc_r + trr).astype(f32); ni = (acc_i + tii).astype(f32)
            t = tr.tmp(p1.shape + (6,))
            T = [t[..., z] for z in range(6)]
            if q == 2:
                OR, OI = outRe[sl], outIm[sl]
            else:
                o = tr.tmp(p1.shape + (2,))
                OR, OI = o[..., 0], o[..., 1]
            groups += [("MULT", wrN, brN, T[0], wr, br_, p1), ("MULT", wiN, biN, T[1], wi, bi_, p2),
                       ("ADD", T[0], T[1].neg(), T[2], p1, -p2, trr),
                       ("MULT", wrN, biN, T[3], wr, bi_, p3), ("MULT", wiN, brN, T[4], wi, br_, p4),
                       ("ADD", T[3], T[4], T[5], p3, p4, tii),
                       ("ADD", accN_r, T[2], OR, acc_r, trr, nr), ("ADD", accN_i, T[5], OI, acc_i, tii, ni)]
            acc_r, acc_i, accN_r, accN_i = nr, ni, OR, OI
        tr.emit(ph, [(g[0], g[1].bc(acc_r.shape), g[2].bc(acc_r.shape), g[3].bc(acc_r.shape),
                      np.broadcast_to(g[4], acc_r.shape), np.broadcast_to(g[5], acc_r.shape), g[6])
                     for g in groups])
        outRe_v[sl], outIm_v[sl] = acc_r, acc_i
    return outRe, outRe_v, outIm, outIm_v


def fft_pass_r2(tr, ph, reN, re, imN, im, inverse, outRe, outIm, rowblock=256):
    """Length-N complex FFT along axis 1 of (R, N) arrays: iterative radix-2
    decimation-in-time Cooley-Tukey (bit-reversed input order), every twiddle
    multiplied (incl. w=1), 4 MULT + 6 ADD per butterfly. Unnormalised."""
    R, Nn = re.shape
    lg = int(round(math.log2(Nn)))
    rev = np.array([int(format(x, f"0{lg}b")[::-1], 2) for x in range(Nn)])
    reN, re, imN, im = reN[:, rev], re[:, rev].copy(), imN[:, rev], im[:, rev].copy()
    ang = 2.0 * np.pi * np.arange(Nn // 2) / Nn
    sgn = 1.0 if inverse else -1.0
    twr, twi = np.cos(ang).astype(f32), (sgn * np.sin(ang)).astype(f32)
    tagr, tagi = (f"wIr{Nn}_", f"wIi{Nn}_") if inverse else (f"wFr{Nn}_", f"wFi{Nn}_")
    reN, imN = reN.copy(), imN.copy()
    h = 1
    while h < Nn:
        jj = ar(h) * (Nn // (2 * h))
        wr, wi = twr[jj], twi[jj]
        wrN, wiN = nm(tagr, jj), nm(tagi, jj)
        last = (2 * h == Nn)
        for r0 in range(0, R, rowblock):
            r1 = min(R, r0 + rowblock)
            sh = (r1 - r0, Nn // (2 * h), 2, h)
            v = lambda a: a[r0:r1].reshape(sh)
            vn = lambda a: a[r0:r1].reshape(sh)
            ar_, ai_, br_, bi_ = v(re)[:, :, 0], v(im)[:, :, 0], v(re)[:, :, 1], v(im)[:, :, 1]
            arN, aiN, brN, biN = vn(reN)[:, :, 0], vn(imN)[:, :, 0], vn(reN)[:, :, 1], vn(imN)[:, :, 1]
            bs = ar_.shape
            p1 = (wr * br_).astype(f32); p2 = (wi * bi_).astype(f32); trr = (p1 - p2).astype(f32)
            p3 = (wr * bi_).astype(f32); p4 = (wi * br_).astype(f32); tii = (p3 + p4).astype(f32)
            y0r = (ar_ + trr).astype(f32); y0i = (ai_ + tii).astype(f32)
            y1r = (ar_ - trr).astype(f32); y1i = (ai_ - tii).astype(f32)
            t = tr.tmp(bs + (6,))
            T = [t[..., q] for q in range(6)]
            if last:
                Y0r, Y0i = vn(outRe)[:, :, 0], vn(outIm)[:, :, 0]
                Y1r, Y1i = vn(outRe)[:, :, 1], vn(outIm)[:, :, 1]
            else:
                yt = tr.tmp(bs + (4,))
                Y0r, Y0i, Y1r, Y1i = (yt[..., q] for q in range(4))
            tr.emit(ph, [("MULT", wrN, brN, T[0], wr, br_, p1),
                         ("MULT", wiN, biN, T[1], wi, bi_, p2),
                         ("ADD", T[0], T[1].neg(), T[2], p1, -p2, trr),
                         ("MULT", wrN, biN, T[3], wr, bi_, p3),
                         ("MULT", wiN, brN, T[4], wi, br_, p4),
                         ("ADD", T[3], T[4], T[5], p3, p4, tii),
                         ("ADD", arN, T[2], Y0r, ar_, trr, y0r),
                         ("ADD", aiN, T[5], Y0i, ai_, tii, y0i),
                         ("ADD", arN, T[2].neg(), Y1r, ar_, -trr, y1r),
                         ("ADD", aiN, T[5].neg(), Y1i, ai_, -tii, y1i)])
            vre, vim = v(re), v(im)
            vre[:, :, 0], vim[:, :, 0], vre[:, :, 1], vim[:, :, 1] = y0r, y0i, y1r, y1i
            for arrN, a0, a1 in ((reN, Y0r, Y1r), (imN, Y0i, Y1i)):
                s_ = vn(arrN)
                s_[:, :, 0] = a0
                s_[:, :, 1] = a1
        h *= 2
    return reN, re, imN, im             # DIT on bit-reversed input -> natural order


def fft2(tr, ph, reN, re, imN, im, inverse, out1, out2):
    """2-D FFT by row-column: length-n FFT of every row, then of every column."""
    o1r, o1i = out1
    reN, re, imN, im = fft_pass(tr, ph, reN, re, imN, im, inverse, o1r, o1i)
    o2r, o2i = out2
    rN, r, iN, i = fft_pass(tr, ph, reN.T, np.ascontiguousarray(re.T), imN.T,
                            np.ascontiguousarray(im.T), inverse, o2r.T, o2i.T)
    return rN.T, np.ascontiguousarray(r.T), iN.T, np.ascontiguousarray(i.T)


def haar_analysis_tr(tr, ph, xN, x, lens, dn, zname):
    """Mallat/Haar analysis (haar_adapter.haar_analysis_unnorm) then *dn."""
    cur, curN, dets = x, xN, []
    for L in lens[1:]:
        v0, v1 = cur[:, 0::2], cur[:, 1::2]
        v0N, v1N = curN[:, 0::2], curN[:, 1::2]
        det = (v0 - v1).astype(f32)
        nxt = (v0 + v1).astype(f32)
        t = tr.tmp(det.shape + (2,))
        tr.emit(ph, [("ADD", v0N, v1N.neg(), t[..., 0], v0, -v1, det),
                     ("ADD", v0N, v1N, t[..., 1], v0, v1, nxt)])
        dets.append((t[..., 0], det))
        cur, curN = nxt, t[..., 1]
    zN = N.cat([curN] + [d[0] for d in dets[::-1]])
    zv = np.concatenate([cur] + [d[1] for d in dets[::-1]], -1)
    b, n = zv.shape
    return mul(tr, ph, zN, zv, nm("dn", ar(n)[None, :]), dn[None, :], grid(zname, b, n))


def haar_synthesis_tr(tr, ph, uN, u, lens, outname):
    b, m = u.shape
    r = lens[-1]
    t, tN = u[:, :r], uN[:, :r]
    off, offs = r, []
    for L in reversed(lens[1:]):
        e, eN = u[:, off:off + L], uN[:, off:off + L]
        offs.append((L, off))
        off += L
        s0 = (t + e).astype(f32)
        s1 = (t - e).astype(f32)
        if 2 * L == m:
            o = grid(outname, b, m).reshape(b, L, 2)
            O0, O1 = o[..., 0], o[..., 1]
        else:
            tt = tr.tmp((b, L, 2))
            O0, O1 = tt[..., 0], tt[..., 1]
        tr.emit(ph, [("ADD", tN, eN, O0, t, e, s0), ("ADD", tN, eN.neg(), O1, t, -e, s1)])
        t = np.stack([s0, s1], -1).reshape(b, 2 * L)
        tN = N.cat([O0[..., None], O1[..., None]]).reshape(b, 2 * L)
    return tN, t, offs


def haar_synthesis_adj_tr(tr, ph, gN, g, lens, offs, outname):
    """Adjoint of the synthesis cascade (grad wrt u)."""
    b, m = g.shape
    r = lens[-1]
    guN = grid(outname, b, m)
    gu = np.zeros((b, m), f32)
    cur, curN = g, gN
    for (L, off) in reversed(offs):
        ga, gb = cur[:, 0::2], cur[:, 1::2]
        gaN, gbN = curN[:, 0::2], curN[:, 1::2]
        gt = (ga + gb).astype(f32)
        ge = (ga - gb).astype(f32)
        geN = guN[:, off:off + L]
        gtN = guN[:, :r] if L == r else tr.tmp((b, L))
        tr.emit(ph, [("ADD", gaN, gbN, gtN, ga, gb, gt), ("ADD", gaN, gbN.neg(), geN, ga, -gb, ge)])
        gu[:, off:off + L] = ge
        cur, curN = gt, gtN
    gu[:, :r] = cur
    return guN, gu


def wht_tr(tr, ph, xN, x, outname, rowblock=512):
    """QWHA's _wht_sync_free for n = 2^p (K = 1): log2(n) stages of adjacent-pair
    butterflies on the (B, n/2, 2, w) view, then (/sqrt_n) * sqrt(n)."""
    R, n = x.shape
    hk, K = get_hadK(n)
    hadK = None if hk is None else hk.numpy().astype(f32)
    res_v = np.empty((R, n), f32)
    for r0 in range(0, R, rowblock):
        r1 = min(R, r0 + rowblock)
        Rb = r1 - r0
        inp, inpN = x[r0:r1].reshape(Rb, n, 1), xN[r0:r1].reshape(Rb, n, 1)
        while inp.shape[1] > K:
            s, w = inp.shape[1], inp.shape[2]
            i4, i4N = inp.reshape(Rb, s // 2, 2, w), inpN.reshape(Rb, s // 2, 2, w)
            a, bb = i4[:, :, 0, :], i4[:, :, 1, :]
            aN, bN = i4N[:, :, 0, :], i4N[:, :, 1, :]
            s0 = (a + bb).astype(f32)
            s1 = (a - bb).astype(f32)
            t = tr.tmp(s0.shape + (2,))
            tr.emit(ph, [("ADD", aN, bN, t[..., 0], a, bb, s0),
                         ("ADD", aN, bN.neg(), t[..., 1], a, -bb, s1)])
            inp = np.concatenate([s0, s1], -1)
            inpN = N.cat([t[..., 0], t[..., 1]], -1)
        if K > 1:                                       # hadK (K x K) @ inp, a plain matmul
            w = inp.shape[2]
            Pv = (hadK[None, :, None, :] * np.transpose(inp, (0, 2, 1))[:, None, :, :]).astype(f32)
            PN = tr.tmp(Pv.shape)
            hN = nm("hadK", ar(K)[None, :, None, None], ar(K)[None, None, None, :]).bc(Pv.shape)
            iN = N(*(np.transpose(c, (0, 2, 1))[:, None, :, :] for c in (inpN.p, inpN.i, inpN.j))).bc(Pv.shape)
            tr.emit(ph, [("MULT", hN, iN, PN, np.broadcast_to(hadK[None, :, None, :], Pv.shape),
                          np.broadcast_to(np.transpose(inp, (0, 2, 1))[:, None, :, :], Pv.shape), Pv)])
            oN = tr.tmp((Rb, K, w))
            inpN, inp = tree_sum(tr, ph, PN, Pv, oN)
            inpN = oN
        inp, inpN = inp.reshape(Rb, n), inpN.reshape(Rb, n)
        sq_t = f32(torch.tensor(n).sqrt().item())
        dN, dv = div(tr, ph, inpN, inp, lit("sqrt_n"), sq_t)
        oN = nm(outname, ar(r0, )[None] if False else (r0 + ar(Rb))[:, None], ar(n)[None, :])
        _, ov = mul(tr, ph, dN, dv, lit("sqrt_n"), f32(math.sqrt(n)), oN)
        res_v[r0:r1] = ov
    return grid(outname, R, n), res_v


# --------------------------------------------------------------------------- #
#  Shared context                                                               #
# --------------------------------------------------------------------------- #
def context(name):
    torch.manual_seed(0)
    mod, ntrain = H.build("lowa" if name == "lowa_sampled" else name, D)  # hw_cost config, f64 CPU
    mod = mod.float()
    if name == "lowa_sampled":
        mod.sampled_grad = True             # SGWALinear(sampled_grad=True): the SDDMM backward
    gx = torch.Generator().manual_seed(11)
    gg = torch.Generator().manual_seed(12)
    X = torch.randn(B_TOK, D, generator=gx)
    G = torch.randn(B_TOK, D, generator=gg)
    return mod, X, G


def opt_state(shape):
    g = torch.Generator().manual_seed(13)
    m = torch.randn(shape, generator=g) * 1e-2
    v = torch.rand(shape, generator=g) * 1e-4 + 1e-6
    return m, v


def torch_ref(name, mod, X, G, pname):
    """Real module forward, (y*G).sum() backward, torch AdamW step."""
    mod.train()
    if name == "loca":
        mod.dct_mode = "sparse"           # authors' sparse_idct2d: the Table III path
    p = dict(mod.named_parameters())[pname]
    for q in mod.parameters():
        q.requires_grad_(q is p)
    y = mod(X)
    (y * G).sum().backward()
    grad = p.grad.detach().clone()
    m, v = opt_state(p.shape)
    opt = torch.optim.AdamW([p], lr=LR, betas=(BETA1, BETA2), eps=EPS, weight_decay=WD, foreach=False)
    opt.state[p] = {"step": torch.tensor(float(STEP - 1)), "exp_avg": m.clone(), "exp_avg_sq": v.clone()}
    p0 = p.detach().clone()
    opt.step()
    return dict(y=y.detach().numpy(), grad=grad.numpy(), p_new=p.detach().numpy().copy(), p0=p0.numpy(),
                m=m.numpy(), v=v.numpy())


def X_names():
    return grid("x", B_TOK, D)


def G_names():
    return grid("g", B_TOK, D)


# --------------------------------------------------------------------------- #
#  Programs                                                                     #
# --------------------------------------------------------------------------- #
def prog_lowa(tr, mod, X, G, st, sampled=False):
    m, n, k = mod.m, mod.n, mod.k
    c = mod.coef.detach().numpy().astype(f32)
    gam = f32(mod.scaling)
    h = mod.hval.numpy().astype(f32)
    s = h.size
    ef = mod.eflat.numpy().astype(np.int64)
    W0 = mod.base_layer.weight.detach().numpy()
    ph = "1_build"
    gcN, gc = mul(tr, ph, nm("c", ar(k)), c, lit("gamma"), gam, nm("gc", ar(k)))
    vN, v = mul(tr, ph, gcN[:, None], gc[:, None], nm("h", ar(s)[None, :]), h[None, :],
                nm("v", ar(k)[:, None], ar(s)[None, :]))
    flatall = ar(m * n)
    W0N = nm("W0", flatall // n, flatall % n)
    WN = nm("W", flatall // n, flatall % n)
    _, Wv = scatter_add(tr, "2_fold", W0N, W0.reshape(-1), ef, vN.reshape(-1), v.reshape(-1), WN)
    Wv = Wv.reshape(m, n)
    if FWD_ONLY:
        return dict(W=Wv, y=X.numpy() @ Wv.T)
    ph = "3_backward"
    if sampled:
        # SGWALinear(sampled_grad=True): SDDMM over the UNION of atom supports only,
        # gW[i,l] = sum_t g[t,i] x[t,l] for (i,l) in U (b MULT + (b-1) ADD per cell),
        # then each atom entry reads its cell (gather, no arithmetic).
        uf, uinv = np.unique(ef, return_inverse=True)
        ui, uj = uf // n, uf % n
        A = G_names().T[ui]                                      # (|U|, b)
        Bn = X_names().T[uj]
        PN, pv = mul(tr, ph, A, G.numpy().T[ui], Bn, X.numpy().T[uj])
        SN, Sv = tree_sum(tr, ph, PN, pv, nm("gW", ui, uj))
        gv = Sv[uinv].reshape(k, s)
        gvN = SN[uinv].reshape(k, s)
    else:
        gWN = grid("gW", m, n)
        _, gW = gemm_gtx(tr, ph, G_names(), G.numpy(), X_names(), X.numpy(), gWN)
        gW = gW.reshape(m, n)
        gv = gW.reshape(-1)[ef].reshape(k, s)
        gvN = gWN.reshape(-1)[ef].reshape(k, s)
    pN, pv = mul(tr, ph, gvN, gv, nm("h", ar(s)[None, :]), h[None, :])
    ggcN, ggc = tree_sum(tr, ph, pN, pv, nm("ggc", ar(k)))
    gN, g = mul(tr, ph, ggcN, ggc, lit("gamma"), gam, nm("grad_c", ar(k)))
    pn, _, _ = adamw(tr, nm("c", ar(k)), c, gN, g, st)
    y = X.numpy() @ Wv.T                                 # excluded frozen-shape GEMM
    return dict(W=Wv, y=y, grad=g, p_new=pn)


def prog_fourierft(tr, mod, X, G, st):
    m, n, k = mod.m, mod.n, mod.n_frequency
    sp = mod.spectrum.detach().numpy().astype(f32)
    idx = mod.indices.numpy()
    sc = f32(mod.scaling)
    W0 = mod.base_layer.weight.detach().numpy()
    dense = np.zeros((m, n), f32)
    dense[idx[0], idx[1]] = sp
    ph = "1_build"
    zero = lit("0").bc((m, n))
    rN, r, iN, i = fft2(tr, ph, grid("S", m, n), dense, zero, np.zeros((m, n), f32), True,
                        (grid("R1r", m, n), grid("R1i", m, n)), (grid("Zu_r", m, n), grid("Zu_i", m, n)))
    inv = f32(1.0 / (m * n))
    Zr, Zi = (r * inv).astype(f32), (i * inv).astype(f32)
    tr.emit(ph, [("MULT", rN, lit("inv_mn"), grid("Zr", m, n), r, inv, Zr),
                 ("MULT", iN, lit("inv_mn"), grid("Zi", m, n), i, inv, Zi)])
    ref_Z = (r.copy(), i.copy(), Zr.copy(), Zi.copy())
    dWN, dW = mul(tr, ph, grid("Zr", m, n), Zr, lit("scaling"), sc, grid("dW", m, n))
    _, Wv = add(tr, "2_fold", grid("W0", m, n), W0, dWN, dW, grid("W", m, n))
    if FWD_ONLY:
        return dict(W=Wv, y=X.numpy() @ Wv.T, fftz=(ref_Z, dense))
    ph = "3_backward"
    gWN, gW = gemm_gtx(tr, ph, G_names(), G.numpy(), X_names(), X.numpy(), grid("gW", m, n))
    gRN, gR = mul(tr, ph, gWN, gW, lit("scaling"), sc, grid("gZr", m, n))
    rN, r, iN, i = fft2(tr, ph, gRN, gR, zero, np.zeros((m, n), f32), False,
                        (grid("B1r", m, n), grid("B1i", m, n)), (grid("gSu_r", m, n), grid("gSu_i", m, n)))
    gSr, gSi = (r * inv).astype(f32), (i * inv).astype(f32)
    tr.emit(ph, [("MULT", rN, lit("inv_mn"), grid("gSr", m, n), r, inv, gSr),
                 ("MULT", iN, lit("inv_mn"), grid("gSi", m, n), i, inv, gSi)])
    g = gSr[idx[0], idx[1]]
    gN = nm("gSr", idx[0], idx[1])
    pn, _, _ = adamw(tr, nm("c", ar(k)), sp, gN, g, st)
    y = X.numpy() @ Wv.T
    return dict(W=Wv, y=y, grad=g, p_new=pn, fft=(ref_Z, dense, gR, r, i))


def prog_wave(tr, mod, X, G, st):
    m, n, k, mu = mod.m, mod.n, mod.k, mod.mu
    sp = mod.spectrum.detach().numpy().astype(f32)
    sc = f32(mod.scaling)
    rows, cols, pidx = mod.rows.numpy(), mod.cols.numpy(), mod.pidx.numpy()
    dn, dm = mod.dn.numpy().astype(f32), mod.dm.numpy().astype(f32)
    ln, lm = mod.lens_n, mod.lens_m
    Xv, Gv = X.numpy(), G.numpy()
    b = Xv.shape[0]
    W0 = mod.base_layer.weight.detach().numpy()
    ph = "1_forward"
    valsN, vals = mul(tr, ph, nm("c", ar(k)), sp, lit("scaling"), sc, nm("vals", ar(k)))

    def fwd(ph, pre):
        zN, z = haar_analysis_tr(tr, ph, X_names(), Xv, ln, dn, pre + "z")
        cN, cv = mul(tr, ph, zN[:, cols], z[:, cols], valsN[pidx][None, :].bc((b, mu * k)),
                     np.broadcast_to(vals[pidx][None, :], (b, mu * k)))
        pos = (ar(b)[:, None] * m + rows[None, :]).reshape(-1)
        yN, yv = scatter_add(tr, ph, lit("0").bc((b * m,)), np.zeros(b * m, f32), pos,
                             cN.reshape(-1), cv.reshape(-1), grid(pre + "yc", b, m).reshape(-1))
        uN, u = mul(tr, ph, yN.reshape(b, m), yv.reshape(b, m), nm("dm", ar(m)[None, :]), dm[None, :],
                    grid(pre + "u", b, m))
        dyN, dy, offs = haar_synthesis_tr(tr, ph, uN, u, lm, pre + "dy")
        return z, dyN, dy, offs

    z, dyN, dy, offs = fwd(ph, "")
    yb = Xv @ W0.T                                       # excluded frozen GEMM output
    _, y = add(tr, ph, grid("yb", b, m), yb, dyN, dy, grid("y", b, m))
    if FWD_ONLY:
        return dict(y=y, dy=dy, z=z)
    ph = "3_backward"
    rz, _, _, offs = fwd(ph, "r_")                       # _HaarFn recomputes the chain
    guN, gu = haar_synthesis_adj_tr(tr, ph, G_names(), Gv, lm, offs, "gu")
    gycN, gyc = mul(tr, ph, guN, gu, nm("dm", ar(m)[None, :]), dm[None, :], grid("gyc", b, m))
    gcN, gc = gycN[:, rows], gyc[:, rows]
    pN, pv = mul(tr, ph, gcN, gc, grid("r_z", b, n)[:, cols], rz[:, cols])
    svN, sv = tree_sum(tr, ph, pN.T, pv.T, nm("gvp", ar(mu * k)))
    gvN, gv = scatter_add(tr, ph, lit("0").bc((k,)), np.zeros(k, f32), pidx, svN, sv, nm("gvals", ar(k)))
    gN, g = mul(tr, ph, gvN, gv, lit("scaling"), sc, nm("grad_c", ar(k)))
    pn, _, _ = adamw(tr, nm("c", ar(k)), sp, gN, g, st)
    return dict(y=y, dy=dy, grad=g, p_new=pn, z=z)


def prog_loca(tr, mod, X, G, st):
    m, n, k = mod.out_features, mod.in_features, mod.n_frequency
    a = mod.spectrum.detach().numpy().astype(f32)
    li = mod.spectrum_indices.detach().numpy().astype(f32)
    scale = f32(mod.scale)
    Cm = DCT_CACHE.get_matrix(m, torch.device("cpu"), torch.float32).numpy()
    Cn = DCT_CACHE.get_matrix(n, torch.device("cpu"), torch.float32).numpy()
    Xv, Gv = X.numpy(), G.numpy()
    b = Xv.shape[0]
    W0 = mod.base_layer.weight.detach().numpy()
    ph = "1_forward"
    liN = nm("loc", ar(2)[:, None], ar(k)[None, :])
    t1 = np.maximum(li, f32(0)).astype(f32)
    t1N = tr.tmp(li.shape)
    tr.emit(ph, [("MAX", liN, lit("0"), t1N, li, f32(0), t1)])
    t2 = np.minimum(t1, f32(0.999)).astype(f32)
    t2N = tr.tmp(li.shape)
    tr.emit(ph, [("MIN", t1N, lit("0.999"), t2N, t1, f32(0.999), t2)])
    mm1 = np.array([[m - 1], [n - 1]], f32)
    t3N, t3 = mul(tr, ph, t2N, t2, N(pid("dim_minus_1", 0), ar(2)[:, None]), mm1)
    Lv = np.round(t3).astype(f32)
    LN = nm("L", ar(2)[:, None], ar(k)[None, :])
    tr.emit(ph, [("ROUND", t3N, None, LN, t3, None, Lv)])
    L0, L1 = Lv[0].astype(np.int64), Lv[1].astype(np.int64)
    # A = C_M^T[:, L0] * a   (m, k)
    AN, A = mul(tr, ph, nm("Cm", L0[None, :], ar(m)[:, None]), Cm[L0].T, nm("c", ar(k)[None, :]), a[None, :],
                grid("A", m, k))
    # Y = A @ C_N[L1, :] as k rank-1 updates (outer-product accumulation)
    YN = grid("Y", m, n)
    acc, accN = None, None
    for i in range(k):
        P = (A[:, i:i + 1] * Cn[L1[i]][None, :]).astype(f32)
        PN = tr.tmp((m, n))
        mg = ("MULT", AN[:, i:i + 1], nm("Cn", L1[i], ar(n)[None, :]), PN, A[:, i:i + 1], Cn[L1[i]][None, :], P)
        if i == 0:
            tr.emit(ph, [mg])
            acc, accN = P, PN
        else:
            new = (acc + P).astype(f32)
            ON = YN if i == k - 1 else tr.tmp((m, n))
            tr.emit(ph, [mg, ("ADD", accN, PN, ON, acc, P, new)])
            acc, accN = new, ON
    dWN, dW = mul(tr, ph, YN, acc, lit("loca_scale"), scale, grid("dW", m, n))
    # unmerged apply: ya = x @ dW^T, emitted per block of TOK_BLOCK tokens (rows are
    # independent per token; for b <= TOK_BLOCK this is the single block of the b=2 trace)
    XN = X_names()
    ya = np.empty((b, m), f32)
    for t0 in range(0, b, TOK_BLOCK):
        t1 = min(b, t0 + TOK_BLOCK)
        bt = t1 - t0
        PN, P = mul(tr, ph, XN[t0:t1, None, :].bc((bt, m, n)), np.broadcast_to(Xv[t0:t1, None, :], (bt, m, n)),
                    dWN[None].bc((bt, m, n)), np.broadcast_to(dW[None], (bt, m, n)))
        _, ya[t0:t1] = tree_sum(tr, ph, PN, P, nm("ya", (t0 + ar(bt))[:, None], ar(m)[None, :]))
    yaN = grid("ya", b, m)
    yb = Xv @ W0.T
    _, y = add(tr, ph, grid("yb", b, m), yb, yaN, ya, grid("y", b, m))
    if FWD_ONLY:
        return dict(y=y, dW=dW)
    ph = "3_backward"
    gdWN, gdW = gemm_gtx(tr, ph, G_names(), Gv, X_names(), Xv, grid("gdW", m, n))
    gYN, gY = mul(tr, ph, gdWN, gdW, lit("loca_scale"), scale, grid("gY", m, n))
    gA = np.empty((m, k), f32)
    for i in range(k):
        PN, P = mul(tr, ph, gYN, gY, nm("Cn", L1[i], ar(n)[None, :]), Cn[L1[i]][None, :])
        _, gA[:, i] = tree_sum(tr, ph, PN, P, nm("gA", ar(m), i))
    gAN = grid("gA", m, k)
    PN, P = mul(tr, ph, gAN.T, gA.T, nm("Cm", L0[:, None], ar(m)[None, :]), Cm[L0])
    gN, g = tree_sum(tr, ph, PN, P, nm("grad_c", ar(k)))
    pn, _, _ = adamw(tr, nm("c", ar(k)), a, gN, g, st)
    return dict(y=y, dW=dW, grad=g, p_new=pn)


def prog_qwha(tr, mod, X, G, st):
    m, n, k = mod.out_features, mod.in_features, mod.n_frequency
    sp = mod.spectrum.detach().numpy().astype(f32)
    idx = mod.indices.numpy()
    sc = f32(mod.scaling)
    W0 = mod.base_layer.weight.detach().numpy()
    Xv, Gv = X.numpy(), G.numpy()
    b = Xv.shape[0]
    ph = "1_forward"
    WhN, Wh = wht_tr(tr, ph, grid("W0", m, n), W0, "Wh")
    v1N, v1 = mul(tr, ph, nm("c", ar(k)), sp, lit("scaling"), sc)
    v2N, v2 = mul(tr, ph, v1N, v1, lit("1"), f32(1))
    sqm = f32(math.sqrt(m))
    vN, v = div(tr, ph, v2N, v2, lit("sqrt_m"), sqm, nm("vals", ar(k)))
    flat = ar(m * n)
    pos = idx[0] * n + idx[1]
    _, Bw = scatter_add(tr, ph, WhN.reshape(-1), Wh.reshape(-1), pos, vN, v, nm("Bw", flat // n, flat % n))
    Bw = Bw.reshape(m, n)
    xhN, xh = wht_tr(tr, ph, X_names(), Xv, "xh")
    yh = (xh @ Bw.T).astype(f32)                         # excluded GEMM (replaces W0 x)
    _, y = div(tr, ph, grid("yh", b, m), yh, lit("n_in"), f32(n), grid("y", b, m))
    if FWD_ONLY:
        return dict(y=y, Wh=Wh, xh=xh)
    ph = "3_backward"
    gyN, gy = div(tr, ph, G_names(), Gv, lit("n_in"), f32(n), grid("gyh", b, m))
    gBN, gB = gemm_gtx(tr, ph, gyN, gy, xhN, xh, grid("gBw", m, n))
    gB = gB.reshape(m, n)
    g0, g0N = gB[idx[0], idx[1]], nm("gBw", idx[0], idx[1])
    g1N, g1 = div(tr, ph, g0N, g0, lit("sqrt_m"), sqm)
    g2N, g2 = mul(tr, ph, g1N, g1, lit("1"), f32(1))
    gN, g = mul(tr, ph, g2N, g2, lit("scaling"), sc, nm("grad_c", ar(k)))
    pn, _, _ = adamw(tr, nm("c", ar(k)), sp, gN, g, st)
    return dict(y=y, grad=g, p_new=pn, Wh=Wh, xh=xh)


def prog_lyra(tr, mod, X, G, st):
    m, n, p, q = mod.out_features, mod.in_features, mod.p, mod.q
    S = mod.coeffs.detach().numpy().astype(f32)
    Ci, Co = mod.dct_in.numpy().astype(f32), mod.dct_out.numpy().astype(f32)
    sc = f32(mod.scaling)
    W0 = mod.base_layer.weight.detach().numpy()
    Xv, Gv = X.numpy(), G.numpy()
    b = Xv.shape[0]
    ph = "1_forward"
    # the three per-token contractions, emitted per block of TOK_BLOCK tokens (for b <= TOK_BLOCK:
    # one block, identical to the b=2 trace)
    XN = X_names()
    xp, so, do = np.empty((b, q), f32), np.empty((b, p), f32), np.empty((b, m), f32)
    for t0 in range(0, b, TOK_BLOCK):
        t1 = min(b, t0 + TOK_BLOCK)
        bt, tt = t1 - t0, t0 + ar(t1 - t0)
        PN, P = mul(tr, ph, XN[t0:t1, None, :].bc((bt, q, n)), np.broadcast_to(Xv[t0:t1, None, :], (bt, q, n)),
                    grid("Cin", q, n)[None].bc((bt, q, n)), np.broadcast_to(Ci[None], (bt, q, n)))
        xpN, xp[t0:t1] = tree_sum(tr, ph, PN, P, nm("xp", tt[:, None], ar(q)[None, :]))
        xpN = nm("xp", tt[:, None], ar(q)[None, :])
        PN, P = mul(tr, ph, xpN[:, None, :].bc((bt, p, q)), np.broadcast_to(xp[t0:t1, None, :], (bt, p, q)),
                    grid("S", p, q)[None].bc((bt, p, q)), np.broadcast_to(S[None], (bt, p, q)))
        _, so[t0:t1] = tree_sum(tr, ph, PN, P, nm("so", tt[:, None], ar(p)[None, :]))
        soN = nm("so", tt[:, None], ar(p)[None, :])
        PN, P = mul(tr, ph, soN[:, None, :].bc((bt, m, p)), np.broadcast_to(so[t0:t1, None, :], (bt, m, p)),
                    grid("Cout", p, m).T[None].bc((bt, m, p)), np.broadcast_to(Co.T[None], (bt, m, p)))
        _, do[t0:t1] = tree_sum(tr, ph, PN, P, nm("do", tt[:, None], ar(m)[None, :]))
    xpN, doN = grid("xp", b, q), grid("do", b, m)
    sdN, sd = mul(tr, ph, lit("scaling"), sc, doN, do, grid("sd", b, m))
    yb = Xv @ W0.T
    _, y = add(tr, ph, grid("yb", b, m), yb, sdN, sd, grid("y", b, m))
    if FWD_ONLY:
        return dict(y=y)
    ph = "3_backward"
    gdN, gd = mul(tr, ph, G_names(), Gv, lit("scaling"), sc, grid("gdo", b, m))
    PN, P = mul(tr, ph, gdN[:, None, :].bc((b, p, m)), np.broadcast_to(gd[:, None, :], (b, p, m)),
                grid("Cout", p, m)[None].bc((b, p, m)), np.broadcast_to(Co[None], (b, p, m)))
    gsoN, gso = tree_sum(tr, ph, PN, P, grid("gso", b, p))
    PN, P = mul(tr, ph, gsoN.T[:, None, :].bc((p, q, b)), np.broadcast_to(gso.T[:, None, :], (p, q, b)),
                xpN.T[None].bc((p, q, b)), np.broadcast_to(xp.T[None], (p, q, b)))
    gN, g = tree_sum(tr, ph, PN, P, grid("grad_S", p, q))
    pn, _, _ = adamw(tr, grid("S", p, q), S, gN, g, st)
    return dict(y=y, grad=g, p_new=pn)


PROGS = {"lowa": (prog_lowa, "coef"),
         "lowa_sampled": (lambda *a: prog_lowa(*a, sampled=True), "coef"), "fourierft": (prog_fourierft, "spectrum"),
         "wave1": (prog_wave, "spectrum"), "wave2": (prog_wave, "spectrum"),
         "loca": (prog_loca, "spectrum"), "qwha": (prog_qwha, "spectrum"), "lyra": (prog_lyra, "coeffs")}


def run_prog(name, tr):
    PREFIXES.clear(); PID.clear()
    mod, X, G = context(name)
    prog, pname = PROGS[name]
    p = dict(mod.named_parameters())[pname]
    m0, v0 = opt_state(p.shape)
    st = dict(m=m0.numpy().astype(f32), v=v0.numpy().astype(f32))
    with torch.no_grad():
        out = prog(tr, mod, X, G, st)
    return out, mod, X, G, pname


# --------------------------------------------------------------------------- #
#  Exact op-count formulas (per module, b tokens)                              #
# --------------------------------------------------------------------------- #
def fft1d(N):
    """(MULT, ADD) of one length-N complex FFT as traced."""
    if N & (N - 1) == 0:
        bf = (N // 2) * int(round(math.log2(N)))
        return 4 * bf, 6 * bf
    a, b = fft1d(N // 3)
    return 3 * a + 8 * N, 3 * b + 8 * N


def wht1d(n):
    """(ADD, MULT) of one QWHA length-n WHT before the /sqrt_n * sqrt(n) pair."""
    K = get_hadK(n)[1]
    st = int(round(math.log2(n // K)))
    return n * st + (n * (K - 1) if K > 1 else 0), (n * K if K > 1 else 0)


_UNION = {}


def lowa_union(m, n, k=256, K=3, seed=H.SEED):
    """|U| = number of distinct cells in the union of LoWA's k atom supports
    (atoms may overlap; boundary folds may repeat a cell).  Read off the repo's
    own SGWALinear buffers (built on the meta device: no m x n allocation)."""
    key = (m, n, k, K, seed)
    if key not in _UNION:
        from sgwa_adapter import SGWALinear
        lay = SGWALinear(torch.nn.Linear(n, m, bias=False, device="meta"), k=k, K=K, seed=seed)
        _UNION[key] = int(lay.urow.numel())
    return _UNION[key]


def formulas(name, b, m=D, n=D, k=256, K=3, s=25, p=16, q=16):
    ad = dict(MULT=6 * k, ADD=5 * k, DIV=2 * k, SQRT=k)        # AdamW per trained scalar
    if name == "lowa_sampled":
        u = lowa_union(m, n, k, K)
        return {"1_build": dict(MULT=k + k * s), "2_fold": dict(ADD=k * s),
                "3_backward": dict(MULT=u * b + k * s + k, ADD=u * (b - 1) + k * (s - 1)),
                "4_adamw": ad}
    if name == "lowa":
        return {"1_build": dict(MULT=k + k * s), "2_fold": dict(ADD=k * s),
                "3_backward": dict(MULT=b * m * n + k * s + k, ADD=(b - 1) * m * n + k * (s - 1)),
                "4_adamw": ad}
    if name == "fourierft":
        fM = m * fft1d(n)[0] + n * fft1d(m)[0]
        fA = m * fft1d(n)[1] + n * fft1d(m)[1]
        return {"1_build": dict(MULT=fM + 2 * m * n + m * n, ADD=fA), "2_fold": dict(ADD=m * n),
                "3_backward": dict(MULT=b * m * n + m * n + fM + 2 * m * n, ADD=(b - 1) * m * n + fA),
                "4_adamw": ad}
    if name in ("wave1", "wave2"):
        mu = 1 if name == "wave1" else 2
        rn, rm = haar_lengths(n)[-1], haar_lengths(m)[-1]
        fm = b * (n + mu * k + m)
        fa = b * (2 * (n - rn) + mu * k + 2 * (m - rm))
        return {"1_forward": dict(MULT=k + fm, ADD=fa + b * m),
                "3_backward": dict(MULT=fm + b * m + b * mu * k + k,
                                   ADD=fa + 2 * b * (m - rm) + (b - 1) * mu * k + mu * k),
                "4_adamw": ad}
    if name == "loca":
        return {"1_forward": dict(MAX=2 * k, MIN=2 * k, ROUND=2 * k,
                                  MULT=2 * k + m * k + k * m * n + m * n + b * m * n,
                                  ADD=(k - 1) * m * n + b * m * (n - 1) + b * m),
                "3_backward": dict(MULT=b * m * n + m * n + k * m * n + k * m,
                                   ADD=(b - 1) * m * n + k * m * (n - 1) + k * (m - 1)),
                "4_adamw": ad}
    if name == "qwha":
        wa, wm = wht1d(n)
        return {"1_forward": dict(ADD=(m + b) * wa + k, DIV=(m + b) * n + k + b * m,
                                  MULT=(m + b) * (n + wm) + 2 * k),
                "3_backward": dict(DIV=b * m + k, MULT=b * m * n + 2 * k, ADD=(b - 1) * m * n),
                "4_adamw": ad}
    if name == "lyra":
        return {"1_forward": dict(MULT=b * q * n + b * p * q + b * m * p + b * m,
                                  ADD=b * q * (n - 1) + b * p * (q - 1) + b * m * (p - 1) + b * m),
                "3_backward": dict(MULT=b * m + b * p * m + p * q * b, ADD=b * p * (m - 1) + p * q * (b - 1)),
                "4_adamw": dict(MULT=6 * p * q, ADD=5 * p * q, DIV=2 * p * q, SQRT=p * q)}


FORMULA_TEXT = {
    "lowa": {"1_build": "MULT k + k s", "2_fold": "ADD k s",
             "3_backward": "MULT b m n + k s + k ; ADD (b-1) m n + k (s-1)"},
    "lowa_sampled": {"1_build": "MULT k + k s", "2_fold": "ADD k s",
                     "3_backward": "MULT u b + k s + k ; ADD u (b-1) + k (s-1)   "
                                   "[u = |union of the k atom supports| = {u} here, <= k s]"},
    "fourierft": {"1_build": "MULT FM + 3 m n ; ADD FA   [FM,FA = m*fft(n) + n*fft(m); fft(2^p): 4,6 x (N/2)p; "
                             "fft(3*2^p) = 3 fft(N/3) + (8N, 8N)]",
                  "2_fold": "ADD m n",
                  "3_backward": "MULT b m n + 3 m n + FM ; ADD (b-1) m n + FA"},
    "wave": {"1_forward": "MULT k + b (n + mu k + m) ; ADD b (2(n-r_n) + mu k + 2(m-r_m)) + b m",
             "3_backward": "MULT b (n + mu k + m) + b m + b mu k + k ; "
                           "ADD b (2(n-r_n) + mu k + 2(m-r_m)) + 2 b (m-r_m) + b mu k"},
    "loca": {"1_forward": "MAX 2k, MIN 2k, ROUND 2k ; MULT 2k + m k + k m n + m n + b m n ; "
                          "ADD (k-1) m n + b m (n-1) + b m",
             "3_backward": "MULT b m n + m n + k m n + k m ; ADD (b-1) m n + k m (n-1) + k (m-1)"},
    "qwha": {"1_forward": "ADD (m+b)(n log2(n/K) + n(K-1)[K>1]) + k ; DIV (m+b) n + k + b m ; "
                          "MULT (m+b)(n + nK[K>1]) + 2k   [K = QWHA hadK block: 12 at n=768, 1 at 2048]",
             "3_backward": "DIV b m + k ; MULT b m n + 2k ; ADD (b-1) m n"},
    "lyra": {"1_forward": "MULT b(qn + pq + mp + m) ; ADD b(q(n-1) + p(q-1) + m(p-1) + m)",
             "3_backward": "MULT b m + b p m + p q b ; ADD b p (m-1) + p q (b-1)"},
    "adamw": "per trained scalar: MULT 6, ADD 5, DIV 2, SQRT 1",
}


# --------------------------------------------------------------------------- #
#  Driver                                                                       #
# --------------------------------------------------------------------------- #
def mx(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max()), float(np.abs(b).max())


def verify(name, out, mod, X, G, pname):
    mod2, X2, G2 = context(name)
    ref = torch_ref(name, mod2, X2, G2, pname)
    v = {}
    v["forward_y"] = mx(out["y"], ref["y"])
    if not FWD_ONLY:
        v["grad"] = mx(out["grad"].reshape(-1), ref["grad"].reshape(-1))
        v["p_new"] = mx(out["p_new"].reshape(-1), ref["p_new"].reshape(-1))
    if name == "lowa_sampled" and not FWD_ONLY:                # torch sampled backward vs torch dense autograd
        md, Xd, Gd = context("lowa")
        v["grad_torch_sampled_vs_dense_autograd"] = mx(ref["grad"].reshape(-1),
                                                       torch_ref("lowa", md, Xd, Gd, pname)["grad"].reshape(-1))
    # AdamW alone, fed torch's own gradient (isolates the optimiser trace)
    mod3, _, _ = context(name)
    with torch.no_grad():
        if name in ("lowa", "lowa_sampled"):
            W = (mod3.base_layer.weight + mod3.get_delta_weight()).numpy()
            v["merged_W"] = mx(out["W"], W)
        if name == "fourierft":
            W = (mod3.base_layer.weight + mod3.get_delta_weight()).numpy()
            v["merged_W"] = mx(out["W"], W)
            if FWD_ONLY:
                (r, i, Zr, Zi), dense = out["fftz"]
            else:
                (r, i, Zr, Zi), dense, gR, br, bi = out["fft"]
            Zt = torch.fft.ifft2(torch.from_numpy(dense))
            v["fft_ifft2_vs_torch"] = mx(Zr + 1j * Zi, Zt.numpy())
            if not FWD_ONLY:
                Bt = torch.fft.fft2(torch.from_numpy(gR).to(torch.complex64))
                v["fft_fft2_unnorm_vs_torch"] = mx(br + 1j * bi, Bt.numpy())
        if name in ("wave1", "wave2"):
            zt = haar_analysis_unnorm(X, mod3.lens_n) * mod3.dn
            v["haar_analysis_vs_torch"] = mx(out["z"], zt.numpy())
            v["haar_delta_vs_torch"] = mx(out["dy"], mod3.delta_apply(X).numpy())
        if name == "loca":
            mod3.dct_mode = "sparse"
            v["dW_vs_torch_sparse"] = mx(out["dW"], mod3.get_delta_weight().numpy())
            mod3.dct_mode = "default"
            v["dW_vs_torch_default_dense_idct"] = mx(out["dW"], mod3.get_delta_weight().numpy())
        if name == "qwha":
            v["wht_W0_vs_torch"] = mx(out["Wh"], _wht_sync_free(mod3.base_layer.weight).numpy())
            v["wht_x_vs_torch"] = mx(out["xh"], _wht_sync_free(X).numpy())
    return v


def plan(name):
    tr = Tracer(name, "count")
    t0 = time.time()
    out, mod, X, G, pname = run_prog(name, tr)
    ver = verify(name, out, mod, X, G, pname)
    rows = np.array(tr.step_rows, np.int64)
    sos, s, acc = [], 0, 0
    for r in rows:
        if acc and acc + r > SHARD_ROWS:
            s += 1
            acc = 0
        sos.append(s)
        acc += int(r)
    counts = {f"{p}|{o}": c for (p, o), c in tr.counts.items()}
    fcounts = formulas(name, B_TOK)
    fl = {f"{p}|{o}": c for p, d in fcounts.items() for o, c in d.items()
          if c and (not FWD_ONLY or p in FWD_PHASES)}
    return dict(method=name, b=B_TOK, forward_only=FWD_ONLY, rows=int(rows.sum()), steps=len(rows), nshards=s + 1, shard_of_step=sos,
                counts=counts, formula_counts=fl, formula_match=(counts == fl), verify=ver,
                secs=time.time() - t0)


def write_task(args):
    name, sos, shards, nshards = args
    tr = Tracer(name, "write", sos, shards)
    tr.nshards = nshards
    try:
        run_prog(name, tr)
    except Done:
        pass
    tr.close()
    return name, shards, {f"{p}|{o}": c for (p, o), c in tr.written.items()}, tr.bytes_raw


def readback_one(path):
    """Count rows per (PHASE, OPERATION) from the file; replay every 1009th row."""
    cnt = defaultdict(int)
    bad, chk, worst = 0, 0, 0.0
    fn = {"MULT": lambda a, b: f32(a) * f32(b), "ADD": lambda a, b: f32(a) + f32(b),
          "DIV": lambda a, b: f32(a) / f32(b), "SQRT": lambda a, b: np.sqrt(f32(a)),
          "MAX": lambda a, b: max(f32(a), f32(b)), "MIN": lambda a, b: min(f32(a), f32(b)),
          "ROUND": lambda a, b: f32(np.round(f32(a)))}
    p = subprocess.Popen(["bash", "-c", f"zcat '{path}'"], stdout=subprocess.PIPE, bufsize=1 << 24)
    it = iter(p.stdout)
    hdr = next(it)
    assert hdr == b"OPERATION,InA,InB,Op,InA_val,InB_val,Op_val,PHASE\n", hdr
    for ln, line in enumerate(it):
        op = line[:line.index(b",")]
        ph = line[line.rfind(b",") + 1:-1]
        cnt[(ph, op)] += 1
        if ln % 1009 == 0:
            f = line[:-1].rsplit(b",", 4)          # names may be quoted "P[i,j]"; values never are
            o = op.decode()
            a = float(f[1]); bb = float(f[2]) if f[2] else 0.0
            want = fn[o](a, bb)
            got = f32(float(f[3]))
            chk += 1
            if not (want == got):
                bad += 1
                worst = max(worst, abs(float(want) - float(got)))
    p.wait()
    return path, {f"{a.decode()}|{b.decode()}": c for (a, b), c in cnt.items()}, chk, bad, worst


def main():
    global B_TOK, FWD_ONLY, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--readback", action="store_true")
    ap.add_argument("-j", type=int, default=32)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--b", type=int, default=B_TOK, help="token count b of the traced step (default 2)")
    ap.add_argument("--forward_only", action="store_true",
                    help="trace only the forward-side phases (1_build, 2_fold, 1_forward); "
                         "programs stop before 3_backward / 4_adamw")
    ap.add_argument("--out", default=None, help="output folder (default results/optrace)")
    a = ap.parse_args()
    B_TOK, FWD_ONLY = a.b, a.forward_only
    if a.out:
        OUT = os.path.abspath(a.out)
    os.makedirs(OUT, exist_ok=True)
    meths = a.methods.split(",")
    pj = os.path.join(OUT, "plan.json")
    plans = json.load(open(pj)) if os.path.exists(pj) else {}
    for mm in meths:                       # a stored plan made at another b / scope is stale
        if mm in plans and (plans[mm].get("b", 2) != B_TOK or plans[mm].get("forward_only", False) != FWD_ONLY):
            if not a.plan and not a.write:
                sys.exit(f"{mm}: plan.json has b={plans[mm].get('b', 2)}, forward_only="
                         f"{plans[mm].get('forward_only', False)}; rerun with the same --b/--forward_only")
            del plans[mm]
    if a.plan or a.write and any(mm not in plans for mm in meths):
        for mm in meths:
            if a.plan or mm not in plans:
                pl = plan(mm)
                plans[mm] = pl
                print(mm, pl["rows"], "rows", pl["nshards"], "shards", "formula_match", pl["formula_match"],
                      f"{pl['secs']:.0f}s", json.dumps(pl["verify"]), flush=True)
                json.dump(plans, open(pj, "w"))
    if a.write:
        lib()
        tasks = []
        for mm in meths:
            pl = plans[mm]
            for f in glob.glob(os.path.join(OUT, f"{mm}.part*.csv.gz*")) + \
                    glob.glob(os.path.join(OUT, f"{mm}.csv.gz*")):
                os.remove(f)
            ns = pl["nshards"]
            per = 1
            for s0 in range(0, ns, per):
                tasks.append((mm, pl["shard_of_step"], list(range(s0, min(ns, s0 + per))), ns))
        tasks.sort(key=lambda t: -plans[t[0]]["rows"] / plans[t[0]]["nshards"] * len(t[2]))
        t0 = time.time()
        written = defaultdict(lambda: defaultdict(int))
        raw = defaultdict(int)
        with get_context("fork").Pool(a.j, maxtasksperchild=1) as pool:
            for mm, sh, w, nb in pool.imap_unordered(write_task, tasks):
                for kk, c in w.items():
                    written[mm][kk] += c
                raw[mm] += nb
                print(f"[{time.time()-t0:.0f}s] {mm} shards {sh} done", flush=True)
        for mm in meths:
            plans[mm]["written_counts"] = dict(written[mm])
            plans[mm]["bytes_raw"] = raw[mm]
            files = sorted(glob.glob(os.path.join(OUT, f"{mm}.part*.csv.gz")) +
                           glob.glob(os.path.join(OUT, f"{mm}.csv.gz")))
            plans[mm]["files"] = [os.path.basename(f) for f in files]
            plans[mm]["bytes_gz"] = sum(os.path.getsize(f) for f in files)
            assert plans[mm]["written_counts"] == plans[mm]["counts"], mm
        json.dump(plans, open(pj, "w"))
    if a.readback:
        paths = []
        for mm in meths:
            paths += [os.path.join(OUT, f) for f in plans[mm]["files"]]
        res = {}
        with get_context("fork").Pool(a.j) as pool:
            for path, cnt, chk, bad, worst in pool.imap_unordered(readback_one, paths):
                res[path] = (cnt, chk, bad, worst)
        for mm in meths:
            tot = defaultdict(int)
            chk = bad = 0
            worst = 0.0
            for f in plans[mm]["files"]:
                cnt, c, bd, w = res[os.path.join(OUT, f)]
                for kk, v in cnt.items():
                    tot[kk] += v
                chk += c; bad += bd; worst = max(worst, w)
            plans[mm]["readback_counts"] = dict(tot)
            plans[mm]["readback_match"] = dict(tot) == plans[mm]["counts"]
            plans[mm]["replay"] = dict(checked=chk, mismatched=bad, worst=worst)
            print(mm, "readback_match", plans[mm]["readback_match"], plans[mm]["replay"], flush=True)
        json.dump(plans, open(pj, "w"))
    if FWD_ONLY:
        write_tables_forward(plans)
    else:
        write_tables(plans)


def write_tables_forward(plans):
    """counts_trace_forward.csv: per method x forward phase, counted from the files (readback),
    with the closed-form formula evaluated at that method's traced b and an exact-equality check."""
    rows = []
    for mm in METHODS:
        if mm not in plans:
            continue
        pl = plans[mm]
        b = pl["b"]
        src = pl.get("readback_counts") or pl["counts"]
        fo = formulas(mm, b)
        ft = FORMULA_TEXT["wave" if mm.startswith("wave") else mm]
        if mm == "lowa_sampled":
            ft = {kk: vv.replace("{u}", str(lowa_union(D, D))) for kk, vv in ft.items()}
        T, TF = defaultdict(int), defaultdict(int)
        for ph in FWD_PHASES:
            if ph not in fo:
                continue
            c = {o: src.get(f"{ph}|{o}", 0) for o in OPS}
            f_ = {o: fo[ph].get(o, 0) for o in OPS}
            for o in OPS:
                T[o] += c[o]; TF[o] += f_[o]
            rows.append([mm, b, ph, c["MULT"], c["ADD"], sum(c[o] for o in OPS[2:]),
                         ";".join(f"{o}={c[o]}" for o in OPS[2:] if c[o]), sum(c.values()),
                         ft.get(ph, ""), f_["MULT"], f_["ADD"], sum(f_[o] for o in OPS[2:]), c == f_,
                         "readback" if pl.get("readback_counts") else "generator"])
        rows.append([mm, b, "FORWARD_TOTAL", T["MULT"], T["ADD"], sum(T[o] for o in OPS[2:]),
                     ";".join(f"{o}={T[o]}" for o in OPS[2:] if T[o]), sum(T.values()), "",
                     TF["MULT"], TF["ADD"], sum(TF[o] for o in OPS[2:]), dict(T) == dict(TF), ""])
    with open(os.path.join(OUT, "counts_trace_forward.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "b", "phase", "MULT", "ADD", "other", "other_breakdown", "rows",
                    "formula (m=n=768, k=256)", "formula_MULT", "formula_ADD", "formula_other",
                    "formula_at_b_equals_readback", "source"])
        w.writerows(rows)


def write_tables(plans):
    rows = []
    for mm in METHODS:
        if mm not in plans:
            continue
        pl = plans[mm]
        src = pl.get("readback_counts") or pl["counts"]
        for ph in PHASES:
            c = {o: src.get(f"{ph}|{o}", 0) for o in OPS}
            if not any(c.values()):
                continue
            rows.append([mm, ph, c["MULT"], c["ADD"], sum(c[o] for o in OPS[2:]),
                         ";".join(f"{o}={c[o]}" for o in OPS[2:] if c[o]),
                         "readback" if pl.get("readback_counts") else "generator"])
        tot = defaultdict(int)
        for kk, v in src.items():
            tot[kk.split("|")[1]] += v
        rows.append([mm, "TOTAL", tot["MULT"], tot["ADD"], sum(tot[o] for o in OPS[2:]),
                     ";".join(f"{o}={tot[o]}" for o in OPS[2:] if tot[o]),
                     "readback" if pl.get("readback_counts") else "generator"])
    with open(os.path.join(OUT, "counts_trace.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "phase", "MULT", "ADD", "other", "other_breakdown", "source"])
        w.writerows(rows)
    step_table(plans, "counts_roberta_step.csv", D, REAL_B, REAL_MODULES, True)
    step_table(plans, "counts_gemma_step.csv", GEMMA["D"], GEMMA["B"], GEMMA["MODULES"], False)


def step_table(plans, fname, d, bb, nmod, check):
    rows = []
    for mm in METHODS:
        ft = FORMULA_TEXT["wave" if mm.startswith("wave") else mm]
        if mm == "lowa_sampled":
            ft = {kk: vv.replace("{u}", f"{lowa_union(d, d)} at d={d}") for kk, vv in ft.items()}
        small = formulas(mm, B_TOK)
        big = formulas(mm, bb, m=d, n=d)
        src = None
        if mm in plans:
            src = plans[mm].get("readback_counts") or plans[mm]["counts"]
        T = defaultdict(int)
        for ph in PHASES:
            if ph not in big:
                continue
            match = (all(src.get(f"{ph}|{o}", 0) == small[ph].get(o, 0) for o in OPS)
                     if (check and src is not None) else "n/a (no trace at this width)")
            o = {x: big[ph].get(x, 0) * nmod for x in OPS}
            for x in OPS:
                T[x] += o[x]
            rows.append([mm, ph, o["MULT"], o["ADD"], sum(o[x] for x in OPS[2:]),
                         ";".join(f"{x}={o[x]}" for x in OPS[2:] if o[x]),
                         ft.get(ph, FORMULA_TEXT["adamw"]), match])
        rows.append([mm, "TOTAL", T["MULT"], T["ADD"], sum(T[x] for x in OPS[2:]),
                     ";".join(f"{x}={T[x]}" for x in OPS[2:] if T[x]), "", ""])
    with open(os.path.join(OUT, fname), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "phase", "MULT", "ADD", "other", "other_breakdown",
                    f"formula_per_module (x{nmod} modules, b={bb}, m=n={d}, k=256)",
                    "formula_at_b2_equals_trace"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
