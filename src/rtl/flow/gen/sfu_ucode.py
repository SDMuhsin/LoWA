#!/usr/bin/env python3
"""Microcode of hw/fp/fp_sfu.v: the docs/arith.md section 3 algorithms as sequences of fma ops.

    sfu_ucode.py emit  <out.v>     write hw/fp/fp_sfu_ucode.v
    sfu_ucode.py check [n]         execute every micro-program op by op with the L1 `fma`
                                    (plus the same integer pre/post steps as the RTL) and compare
                                    with the L1 `exp recip rsqrt sqrt tanh erf log` entry points
The check validates the PROGRAMS (it is not the RTL test; tb/fp/tb_fp_sfu.sv is).

Every rounded op is one fma:  mul(x, y) = fma(x, y, -0),  add(x, y) = fma(x, 1, y),
sub(x, y) = fma(x, 1, -y); these are bit-identical to L1's add/mul/sub (arith.md section 1).
Operand sources: A in {R0..R3, KAB} (optionally negated), B in {R0..R3, KAB, P1, P2},
C in {R0..R3, KC} (optionally negated).  KAB / KC are per-step ROM constants; P1 = 2^(e1-127),
P2 = 2^(e2-127) are built from the per-thread exponent fields e1, e2.
Write-back side effects: EXPN (exp: n = bits(t) - bits(1.5*2^23), e1 = (n>>1)+127,
e2 = (n - (n>>1)) + 127) and RSPLIT (recip of a computed value: R[dst] = significand with
exponent field 127, k = -E, e1 = (k>>1)+127, e2 = (k-(k>>1))+127).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model"))

FNS = ["exp", "recip", "rsqrt", "sqrt", "tanh", "erf", "log"]
# ------------------------------------------------------------------------------- constants
ONE, NONE_, TWO, HALF, NZ = 0x3F800000, 0xBF800000, 0x40000000, 0x3F000000, 0x80000000
L2E, MAGIC, NMAGIC = 0x3FB8AA3B, 0x4B400000, 0xCB400000
LN2HI, LN2LO = 0x3F317218, 0xB102E308
NLN2HI, NLN2LO = LN2HI ^ 0x80000000, LN2LO ^ 0x80000000
RC0, RC1 = 0x3FB4B4B5, 0x3EF0F0F1
EXP_C = [0x3F800000, 0x3F000000, 0x3E2AAAAA, 0x3D2AAA41, 0x3C088986, 0x3AB6CAB2, 0x394F4769]
TANH_C = [0xBEAAAAAB, 0x3E08887D, 0xBD5D0B49, 0x3CB30437, 0xBC1019E8, 0x3B604FFA, 0xBA9B451C,
          0x39A05C7B, 0xB82CB570]
ERFS_C = [0x3E0375D4, 0xBEC093A2, 0x3DE71710, 0xBCDC083E, 0x3BAA6D6D, 0xBA5574D0, 0x38AF88C7]
ERFL_C = [0x3E037DA2, 0x3F22ECFB, 0x3DD393B0, 0xBCA3B3D8, 0x3A233D73, 0x3AA0702B, 0xBA021F57,
          0x38D82E17, 0xB746876D, 0x351F8958]
LOG_C = [0xBEFFFFFC, 0x3EAAAAA8, 0xBE800213, 0x3E4CD1FA, 0xBE2A13B6, 0x3E1138C6, 0xBE079BB2,
         0x3E04245D, 0xBD9C187A]

ASEL = {"R0": 0, "R1": 1, "R2": 2, "R3": 3, "K": 4}
BSEL = {"R0": 0, "R1": 1, "R2": 2, "R3": 3, "K": 4, "P1": 5, "P2": 6}
CSEL = {"R0": 0, "R1": 1, "R2": 2, "R3": 3, "K": 4}
WB = {None: 0, "EXPN": 1, "RSPLIT": 2}


class Op:
    """dst = fma(+-A, B, +-C).  A/B/C: register name, 'P1'/'P2', or an int (ROM constant)."""

    def __init__(self, dst, a, b, c, nega=False, negc=False, wb=None):
        self.dst, self.a, self.b, self.c, self.nega, self.negc, self.wb = dst, a, b, c, nega, negc, wb
        ka = a if isinstance(a, int) else None
        kb = b if isinstance(b, int) else None
        assert ka is None or kb is None, "only one of A, B may be a ROM constant"
        self.kab = ka if ka is not None else kb
        self.kc = c if isinstance(c, int) else None


def mul(d, a, b, **kw):
    return Op(d, a, b, NZ, **kw)


def horner(dst, coefs, x):
    """p = c_d; p = fma(p, x, c_k), k = d-1..0  (first op uses c_d directly as A)."""
    d = len(coefs) - 1
    ops = [Op(dst, coefs[d], x, coefs[d - 1])]
    for k in range(d - 2, -1, -1):
        ops.append(Op(dst, dst, x, coefs[k]))
    return ops


def exp_full(x, negx, t, r, p, out):
    # G has degree 6 over EXP_C = c0..c6, Horner down to c0, then fma(G, r, 1)
    ops = [Op(t, x, L2E, MAGIC, nega=negx, wb="EXPN"),
           Op(t, t, ONE, NMAGIC),
           Op(r, t, NLN2HI, x, negc=negx),
           Op(r, t, NLN2LO, r)]
    ops += horner(p, EXP_C, r)
    ops += [Op(p, p, r, ONE), mul(p, p, "P1"), mul(out, p, "P2")]
    return ops


def recip_core(m, y, e, out):
    ops = [Op(y, m, RC1, RC0, nega=True)]                        # y = fma(-m, 8/17, 24/17)
    for _ in range(3):
        ops += [Op(e, m, y, ONE, nega=True), Op(y, y, e, y)]     # e = fma(-m, y, 1); y = fma(y, e, y)
    ops += [mul(y, y, "P1"), mul(out, y, "P2")]
    return ops


def rsqrt_iter():
    ops = []
    for _ in range(3):
        ops += [mul("R2", "R1", "R1"),                           # t = y*y
                Op("R2", "R0", "R2", ONE, nega=True),            # e = fma(-m, t, 1)
                mul("R3", "R1", HALF),                           # h = y*0.5
                Op("R1", "R3", "R2", "R1")]                      # y = fma(h, e, y)
    return ops


OUT = "OUT"
PROGS = {
    # fn: {br: [ops]}  (br only for tanh / erf; the short branch is front-padded with NOPs)
    "exp": {0: exp_full("R0", False, "R1", "R2", "R3", OUT)},
    "recip": {0: recip_core("R0", "R1", "R2", OUT)},
    "rsqrt": {0: rsqrt_iter() + [mul(OUT, "R1", "P1")]},
    "sqrt": {0: rsqrt_iter() + [mul("R2", "R0", "R1"),             # s = m*y
                                Op("R3", "R2", "R2", "R0", nega=True),  # r = fma(-s, s, m)
                                mul("R1", "R1", HALF),              # h = y*0.5
                                Op("R2", "R3", "R1", "R2"),         # s = fma(r, h, s)
                                mul(OUT, "R2", "P1")]},             # s * 2^k
    "tanh": {0: [mul("R2", "R0", "R0")]                               # z = x*x
             + horner("R3", TANH_C, "R2")                             # P(z)
             + [mul("R1", "R0", "R2"),                                # x*z
                Op(OUT, "R1", "R3", "R0")],                           # fma(x*z, P, x)
             1: [mul("R0", "R1", TWO)]                                # 2a
             + exp_full("R0", False, "R1", "R2", "R3", "R3")          # e = exp(2a)
             + [Op("R0", "R3", ONE, ONE, wb="RSPLIT")]                # d = add(e, 1) -> m(d), k
             + recip_core("R0", "R1", "R2", "R1")                     # recip(d)
             + [Op(OUT, TWO, "R1", ONE, nega=True)]},                 # fma(-2, recip, 1)
    "erf": {0: [mul("R2", "R0", "R0")]                                # z = x*x
            + horner("R3", ERFS_C, "R2")                              # Q(z)
            + [Op(OUT, "R0", "R3", "R0")],                            # fma(x, Q, x)
            1: horner("R2", ERFL_C, "R1")                             # R(a)
            + [Op("R2", "R2", "R1", "R1")]                            # t = fma(R, a, a)
            + exp_full("R2", True, "R3", "R0", "R1", "R1")            # exp(-t)
            + [Op(OUT, "R1", ONE, ONE, nega=True)]},                  # sub(1, e) = fma(-e, 1, 1)
    "log": {0: [Op("R2", "R0", ONE, NONE_)]                           # f = sub(mb, 1)
            + horner("R3", LOG_C, "R2")                               # Q(f)
            + [mul("R0", "R2", "R2"),                                 # f*f
               Op("R0", "R0", "R3", "R2"),                            # fma(f*f, Q, f)
               Op("R0", "R1", LN2LO, "R0"),                           # fma(E, ln2lo, r)
               Op(OUT, "R1", LN2HI, "R0")]},                          # fma(E, ln2hi, r)
}
# output sign OR: recip (sign of x), tanh (both branches), erf long branch
ORS = {"exp": {0: 0}, "recip": {0: 1}, "rsqrt": {0: 0}, "sqrt": {0: 0}, "tanh": {0: 1, 1: 1},
       "erf": {0: 0, 1: 1}, "log": {0: 0}}


def length(fn):
    return max(len(p) for p in PROGS[fn].values())


def schedule(fn, br):
    """list of N entries: None (NOP) or Op; short branches are front-padded."""
    ops = PROGS[fn].get(br, PROGS[fn][0])
    return [None] * (length(fn) - len(ops)) + ops


def ctl_word(fn, br, step):
    """-> dict of control fields for (fn, br, step)."""
    sch = schedule(fn, br)
    op = sch[step]
    last = int(step == len(sch) - 1)
    d = dict(iss=0, asel=0, nega=0, bsel=0, csel=0, negc=0, wen=0, dst=0, wb=0, last=last,
             ors=ORS[fn].get(br, 0) if last else 0, kab=0, kc=0)
    if op is None:
        return d

    def src(v, table):
        return table["K"] if isinstance(v, int) else table[v]
    d.update(iss=1, asel=src(op.a, ASEL), nega=int(op.nega), bsel=src(op.b, BSEL),
             csel=src(op.c, CSEL), negc=int(op.negc), kab=op.kab or 0, kc=op.kc or 0,
             wb=WB[op.wb])
    if op.dst != OUT:
        d.update(wen=1, dst=int(op.dst[1]))
    else:
        assert last, f"{fn}/{br}: OUT before the last step"
    return d


# ------------------------------------------------------------------------------- emit
def emit(path):
    lines = []
    for fi, fn in enumerate(FNS):
        brs = [0, 1] if fn in ("tanh", "erf") else [0]
        for br in brs:
            for st in range(length(fn)):
                d = ctl_word(fn, br, st)
                sel = f"{{3'd{fi}, 1'b{br}, 5'd{st}}}" if len(brs) == 2 else f"{{3'd{fi}, 1'b?, 5'd{st}}}"
                lines.append(
                    f"            {sel:<22}: begin c = {{1'b{d['iss']}, 3'd{d['asel']}, 1'b{d['nega']}, "
                    f"3'd{d['bsel']}, 3'd{d['csel']}, 1'b{d['negc']}, 1'b{d['wen']}, 2'd{d['dst']}, "
                    f"2'd{d['wb']}, 1'b{d['last']}, 1'b{d['ors']}}}; "
                    f"kab = 32'h{d['kab']:08X}; kc = 32'h{d['kc']:08X}; end"
                    + ("  // NOP (branch padding)" if not d["iss"] else ""))
    lens = ", ".join(f"{fn} {length(fn)}" for fn in FNS)
    src = f"""`default_nettype none
// fp_sfu_ucode: micro-program ROM of fp_sfu (combinational).
// GENERATED by flow/gen/sfu_ucode.py -- do not edit; `make gen-check` fails if stale.
// Steps per element (incl. branch padding): {lens}; fn 7 = 1 step (reserved, NaN).
// c = {{iss, asel[2:0], nega, bsel[2:0], csel[2:0], negc, wen, dst[1:0], wb[1:0], last, ors}}
//   asel: 0-3 R0-R3, 4 KAB   bsel: 0-3 R0-R3, 4 KAB, 5 P1, 6 P2   csel: 0-3 R0-R3, 4 KC
//   wb: 0 none, 1 EXPN, 2 RSPLIT
module fp_sfu_ucode (
    input  wire [2:0]  fn,
    input  wire        br,
    input  wire [4:0]  step,
    output reg  [18:0] c,
    output reg  [31:0] kab,
    output reg  [31:0] kc
);
    always @* begin
        casez ({{fn, br, step}})
{chr(10).join(lines)}
            default: begin c = {{1'b0, 3'd0, 1'b0, 3'd0, 3'd0, 1'b0, 1'b0, 2'd0, 2'd0, 1'b1, 1'b0}}; kab = 32'h0; kc = 32'h0; end
        endcase
    end
endmodule
`default_nettype wire
"""
    with open(path, "w") as f:
        f.write(src)


# ------------------------------------------------------------------------------- check
def check(n):
    import hwmodel as H
    sys.path.insert(0, os.path.join(HERE, "..", "..", "tb", "fp"))
    import gen_vectors as G
    U32 = np.uint32
    f2u, u2f = G.f2u, G.u2f
    r = np.random.default_rng(1)
    bad_total = 0
    for fn in FNS:
        x = np.concatenate([G.sfu_random(fn, n, r), np.asarray(G.SPECIAL, U32)])
        st = pre(fn, x)
        R = [st["R0"], st["R1"], np.zeros_like(x), np.zeros_like(x)]
        e1, e2, s = st["e1"].copy(), st["e2"].copy(), st["s"]
        y = np.zeros_like(x)
        for br in sorted(set(int(b) for b in np.unique(st["br"]))):
            m = st["br"] == br
            Rm = [v[m].copy() for v in R]
            e1m, e2m = e1[m].copy(), e2[m].copy()
            for step in range(length(fn)):
                d = ctl_word(fn, br, step)
                if not d["iss"]:
                    continue

                def pick(sel, k, p_ok):
                    if sel < 4:
                        return Rm[sel]
                    if sel == 4:
                        return np.full(int(m.sum()), k, U32)
                    assert p_ok
                    return ((e1m if sel == 5 else e2m).astype(U32) & 0xFF) << 23
                A = pick(d["asel"], d["kab"], False) ^ U32(d["nega"] << 31)
                B = pick(d["bsel"], d["kab"], True)
                C = pick(d["csel"], d["kc"], False) ^ U32(d["negc"] << 31)
                out = f2u(H.fma(u2f(A), u2f(B), u2f(C)))
                if d["wb"] == 1:
                    nn = out.astype(np.int64) - MAGIC
                    n1 = nn >> 1
                    e1m, e2m = (n1 + 127) & 0xFF, (nn - n1 + 127) & 0xFF
                if d["wb"] == 2:
                    k = 127 - ((out >> 23) & 0xFF).astype(np.int64)
                    k1 = k >> 1
                    e1m, e2m = (k1 + 127) & 0xFF, (k - k1 + 127) & 0xFF
                    out = (out & U32(0x007FFFFF)) | U32(0x3F800000)
                if d["wen"]:
                    Rm[d["dst"]] = out
                if d["last"]:
                    if d["ors"]:
                        out = out | (s[m] << 31)
                    y[m] = out
        sp = st["spec"]
        y = np.where(sp != 0, st["specv"], y).astype(U32)
        want = f2u(getattr(H, fn)(u2f(x)))
        bad = int((y != want).sum())
        bad_total += bad
        print(f"  {fn:6s} {len(x):9d} inputs  {length(fn):2d} steps  mismatches {bad}")
        if bad:
            i = np.nonzero(y != want)[0][:5]
            print("    e.g.", [(hex(int(x[j])), hex(int(y[j])), hex(int(want[j]))) for j in i])
    return bad_total


def pre(fn, x):
    """Accept-time integer pre-processing, exactly as fp_sfu.v does it (mirrored here)."""
    U32 = np.uint32
    x = x.astype(U32)
    s = (x >> 31).astype(U32)
    ex = ((x >> 23) & 0xFF).astype(np.int64)
    fr = x & U32(0x7FFFFF)
    isnan = (ex == 255) & (fr != 0)
    isinf = (ex == 255) & (fr == 0)
    isz = ex == 0
    xd = np.where(isz, s << 31, x).astype(U32)
    a = xd & U32(0x7FFFFFFF)
    z = np.zeros_like(x)
    spec = np.zeros(len(x), np.int64)
    specv = np.zeros_like(x)

    def setsp(mask, val):
        newm = mask & (spec == 0)
        specv[newm] = np.asarray(val, U32)[newm] if np.ndim(val) else val
        spec[newm] = 1
    QN = U32(0x7FC00000)
    st = dict(R0=z.copy(), R1=z.copy(), e1=np.zeros(len(x), np.int64), e2=np.zeros(len(x), np.int64),
              s=s, br=np.zeros(len(x), np.int64))
    E = ex - 127
    mant1 = (fr | U32(0x3F800000)).astype(U32)
    if fn == "exp":
        setsp(isnan, QN)
        setsp((s == 0) & (a > 0x42B20000), U32(0x7F800000))
        setsp((s == 1) & (a > 0x42D00000), U32(0))
        st["R0"] = xd
    elif fn == "recip":
        setsp(isnan, QN)
        setsp(isinf, s << 31)
        setsp(isz, (s << 31) | U32(0x7F800000))
        k = -E
        k1 = k >> 1
        st["e1"], st["e2"] = (k1 + 127) & 0xFF, (k - k1 + 127) & 0xFF
        st["R0"] = mant1
    elif fn in ("rsqrt", "sqrt"):
        setsp(isnan, QN)
        if fn == "rsqrt":
            setsp(isz, (s << 31) | U32(0x7F800000))
            setsp(s == 1, QN)
            setsp(isinf, U32(0))
        else:
            setsp(isz, s << 31)
            setsp(s == 1, QN)
            setsp(isinf, U32(0x7F800000))
        k = E >> 1
        m = (fr | ((127 + E - 2 * k).astype(U32) << 23)).astype(U32)
        st["R0"] = m
        st["R1"] = (U32(0x5F3759DF) - (m >> 1)).astype(U32)
        st["e1"] = ((-k if fn == "rsqrt" else k) + 127) & 0xFF
    elif fn in ("tanh", "erf"):
        setsp(isnan, QN)
        lim = 0x41100000 if fn == "tanh" else 0x407AE148
        setsp(a >= lim, (s << 31) | U32(0x3F800000))
        st["br"] = (a >= (0x3F800000 if fn == "tanh" else 0x3F6C0000)).astype(np.int64)
        st["R0"], st["R1"] = xd, a
    elif fn == "log":
        setsp(isnan, QN)
        setsp(isz, U32(0xFF800000))
        setsp(s == 1, QN)
        setsp(isinf, U32(0x7F800000))
        big = mant1 > 0x3FB504F3
        mb = np.where(big, mant1 - U32(0x00800000), mant1).astype(U32)
        Ei = E + big
        st["R0"] = mb
        st["R1"] = Ei.astype(np.float32).view(U32)        # |E| <= 128: exact
    st["spec"], st["specv"] = spec, specv
    return st


if __name__ == "__main__":
    if sys.argv[1] == "emit":
        emit(sys.argv[2])
    elif sys.argv[1] == "check":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
        b = check(n)
        print("UCODE CHECK", "PASS" if b == 0 else f"FAIL {b}")
        sys.exit(1 if b else 0)
