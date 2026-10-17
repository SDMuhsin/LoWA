#!/usr/bin/env python3
"""Vector-unit micro-programs (P4): element / scalar kernels and the per-op phase tables.
    vu_ucode.py emit  <out.v>     write hw/vec/vu_ucode.v (kernel ROM + phase ROM)
    vu_ucode.py check [seed]      run every vector op through the kernel emulator with the L1
                                  primitives (fma / cmp / sfu) and compare with the L1 per-op
                                  entry points (ln_fwd, smax_fwd, ce, gnorm_clip, ...), bit for bit
The check validates the PROGRAMS (it is not the RTL test; tb/vec/ is).

Lane model (hw/vec/vu_lane.v): per element a slot holds r0 = A[j], r1 = B[j], r2..r4 = cvec0..2[j],
r5 = temp, flags keep / kmask / is_y.  Ops run on one fp_fma (mul = fma(x, y, -0), add = fma(x, 1, y),
sub = fma(x, 1, -y): bit-identical to L1, docs/arith.md), the SFU (push / pop in order), the MAX
unit (fp_cmp feedback), the ACC unit (fp_fma feedback, chunked sums of docs/arith.md section 2),
the OUT path and STORE (in-place write of A[j]).  Lane scalars s0..s3, TOT0/1 (acc results), MAXR,
XTOT (cross-lane result) are per lane; C0..C3, DS come from the descriptor.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# ----------------------------------------------------------------------------- encodings
OPT = dict(END=0, FMA=1, SFU=2, POP=3, MAX=4, ACC=5, OUT=6, STORE=7, MIN=8, MOV=9)
SRC = dict(r0=0, r1=1, r2=2, r3=3, r4=4, r5=5, s0=8, s1=9, s2=10, s3=11, TOT0=12, TOT1=13, MAXR=14,
           XTOT=15, C0=16, C1=17, C2=18, C3=19, ZERO=20, ONE=21, NZERO=22, DS=23, OHN=24, NINF=25)
DST = dict(r0=0, r1=1, r2=2, r3=3, r4=4, r5=5, s0=8, s1=9, s2=10, s3=11, none=15)
FN = dict(exp=0, recip=1, rsqrt=2, sqrt=3, tanh=4, erf=5, ln=6)
UOPW = 36      # opt 4 | srca 5 | srcb 5 | srcc 5 | dst 4 | nega negc zk zm capy 5 | fn 3 | accw 1 | last 1 | pad 3


def op(opt, dst="none", a="ZERO", b="ZERO", c="ZERO", nega=0, negc=0, zk=0, zm=0, capy=0, fn="exp", accw=0):
    return dict(opt=opt, dst=dst, a=a, b=b, c=c, nega=nega, negc=negc, zk=zk, zm=zm, capy=capy, fn=fn, accw=accw)


def mul(d, x, y, **k):   return op("FMA", d, x, y, "NZERO", **k)
def add(d, x, y, **k):   return op("FMA", d, x, "ONE", y, **k)
def sub(d, x, y, **k):   return op("FMA", d, x, "ONE", y, negc=1, **k)
def fma(d, x, y, z, **k): return op("FMA", d, x, y, z, **k)
def sfu(fn, x):          return op("SFU", a=x, fn=fn)
def pop(d):              return op("POP", d)
def mx(x, **k):          return op("MAX", a=x, **k)
def acc(w, x, y, **k):   return op("ACC", a=x, b=y, accw=w, **k)
def out(x, **k):         return op("OUT", a=x, **k)
def store(x, **k):       return op("STORE", a=x, **k)
def mn(d, x, y):         return op("MIN", d, x, y)
def mov(d, x):           return op("MOV", d, x)


EMB_PRE = [add("r0", "r0", "r2"), add("r0", "r0", "r1")]                  # e = add(add(w, tt), pos)
KERNELS = {
    # LayerNorm forward (dataflow.md 5)
    "K_SUM":      [acc(0, "r0", "ONE")],
    "K_SQDEV":    [sub("r5", "r0", "s0"), acc(0, "r5", "r5")],
    "K_LNOUT":    [sub("r5", "r0", "s0"), mul("r5", "r5", "s1"), fma("r5", "r5", "r3", "r4"), out("r5")],
    # softmax forward / recompute / backward
    "K_MAX":      [mx("r0", zm=1)],
    "K_EXPSUM":   [sub("r5", "r0", "s0"), sfu("exp", "r5"), pop("r5"), store("r5", zm=1), acc(0, "r5", "ONE", zm=1)],
    "K_SCALEOUT": [mul("r5", "r0", "s1"), mul("r5", "r5", "DS", zk=1), out("r5")],
    "K_SCALE":    [mul("r5", "r0", "s1"), out("r5")],
    "K_DROPFMA":  [mul("r5", "r1", "DS", zk=1), acc(0, "r0", "r5")],
    "K_DSOUT":    [mul("r5", "r1", "DS", zk=1), sub("r5", "r5", "s0"), mul("r5", "r0", "r5"), mul("r5", "r5", "C0"), out("r5")],
    # LayerNorm backward
    "K_LNBSUMS":  [mul("r1", "r1", "r2"), acc(0, "r1", "ONE"), sub("r5", "r0", "s0"), mul("r5", "r5", "s1"), acc(1, "r1", "r5")],
    "K_LNBOUT":   [mul("r1", "r1", "r2"), sub("r5", "r0", "s0"), mul("r5", "r5", "s1"), sub("r1", "r1", "s2"),
                   mul("r5", "r5", "s3"), sub("r1", "r1", "r5"), mul("r5", "s1", "r1"), out("r5")],
    # embeddings: e = add(add(word, tt0), pos), then LN, then dropout
    "K_EMB_SUM":  EMB_PRE + [acc(0, "r0", "ONE")],
    "K_EMB_SQDEV": EMB_PRE + [sub("r5", "r0", "s0"), acc(0, "r5", "r5")],
    "K_EMB_OUT":  EMB_PRE + [sub("r5", "r0", "s0"), mul("r5", "r5", "s1"), fma("r5", "r5", "r3", "r4"),
                             mul("r5", "r5", "DS", zk=1), out("r5")],
    # grad norm rows, CE, MSE
    "K_GN":       [acc(0, "r0", "r0")],
    "K_CE_MAX":   [mx("r0", capy=1)],
    "K_CE_EXP":   [sub("r5", "r0", "s2"), sfu("exp", "r5"), pop("r5"), store("r5"), acc(0, "r5", "ONE")],
    "K_CE_OUT":   [mul("r5", "r0", "s3"), add("r5", "r5", "OHN"), mul("r5", "r5", "s1"), mul("r5", "r5", "C0"), out("r5")],
    "K_MSE":      [sub("r5", "r0", "s0"), mul("r1", "r5", "r5"), mul("s2", "r1", "s1"), mul("r5", "r5", "C1"),
                   mul("r5", "r5", "s1"), out("r5")],
    # element-wise
    "K_E_ADD":    [add("r5", "r0", "r1"), out("r5")],
    "K_E_MUL":    [mul("r5", "r0", "r1"), out("r5")],
    "K_E_SCALE":  [mul("r5", "r0", "C0"), out("r5")],
    "K_E_SCALER": [mul("r5", "r0", "s3"), out("r5")],
    "K_E_DROP":   [mul("r5", "r0", "DS", zk=1), out("r5")],
    "K_E_TANH":   [sfu("tanh", "r0"), pop("r5"), out("r5")],
    "K_E_TANHB":  [fma("r5", "r0", "r0", "ONE", nega=1), mul("r5", "r1", "r5"), out("r5")],
    "K_E_COPY":   [out("r0")],
    # scalar kernels (one element, j = 0)
    "S_LN_MU":    [mul("s0", "TOT0", "C0")],
    "S_LN_RSTD":  [mul("r5", "TOT0", "C0"), add("r5", "r5", "C1"), sfu("rsqrt", "r5"), pop("s1")],
    "S_MAXR":     [mov("s0", "MAXR")],
    "S_RECIP1":   [sfu("recip", "TOT0"), pop("s1")],
    "S_LNB":      [mul("s2", "TOT0", "C0"), mul("s3", "TOT1", "C0")],
    "S_TOT0":     [mov("s0", "TOT0")],
    "S_CE_MAXR":  [mov("s2", "MAXR")],
    "S_CE_FIN":   [sfu("ln", "TOT0"), pop("r5"), add("r5", "s2", "r5"), sub("r5", "r5", "s3"), mul("s2", "r5", "s1"),
                   sfu("recip", "TOT0"), pop("s3")],
    "S_LOSS":     [mul("s0", "XTOT", "C0")],
    "S_GN_ROW":   [sfu("sqrt", "XTOT"), pop("r5"), fma("s2", "r5", "r5", "s2"), mov("s0", "r5")],
    "S_GN_FIN":   [sfu("sqrt", "s2"), pop("r5"), mov("s0", "r5"), add("r5", "r5", "C0"), sfu("recip", "r5"), pop("r5"),
                   mn("s3", "r5", "ONE")],
}
KID = {k: i for i, k in enumerate(KERNELS)}
PCW = 4
assert max(len(v) for v in KERNELS.values()) <= (1 << PCW)

# ----------------------------------------------------------------------------- phases
# kind: ELEM (kernel over the row segments), SCAL (scalar kernel, one element), XL (cross-lane
# sequential add of a lane scalar over the tile's rows), LDS (load per-row scalars st0 -> s0,
# st1 -> s1), WST (write s0 -> st0, s1 -> st1 per row), W0 (write lane-0 s0 to o2), W03 (lane-0
# s0 -> o2, s3 -> o2 + 4).  final = runs once after the last tile (else per tile).
KIND = dict(END=0, ELEM=1, SCAL=2, XL=3, LDS=4, WST=5, W0=6, W03=7)
PHW = 20   # kind 3 | kid 6 | use_b use_cv0 use_cv1 use_cv2 use_keep use_km out chunk xl_s2 xl_chunk final 11


def ph(kind, kid=None, final=0, **f):
    d = dict(kind=kind, kid=KID[kid] if kid else 0, use_b=0, use_cv0=0, use_cv1=0, use_cv2=0, use_keep=0, use_km=0,
             out=0, chunk=0, xl_s2=0, xl_chunk=0, final=final)
    d.update(f)
    return d


VOPS = {
    "LN_FWD":   [ph("ELEM", "K_SUM", chunk=1), ph("SCAL", "S_LN_MU"), ph("ELEM", "K_SQDEV", chunk=1),
                 ph("SCAL", "S_LN_RSTD"), ph("ELEM", "K_LNOUT", use_cv1=1, use_cv2=1, out=1), ph("WST")],
    "LN_BWD":   [ph("LDS"), ph("ELEM", "K_LNBSUMS", use_b=1, use_cv0=1, chunk=1), ph("SCAL", "S_LNB"),
                 ph("ELEM", "K_LNBOUT", use_b=1, use_cv0=1, out=1)],
    "SMAX_FWD": [ph("ELEM", "K_MAX", use_km=1), ph("SCAL", "S_MAXR"), ph("ELEM", "K_EXPSUM", use_km=1),
                 ph("SCAL", "S_RECIP1"), ph("ELEM", "K_SCALEOUT", use_keep=1, out=1), ph("WST")],
    "SMAX_REC": [ph("LDS"), ph("ELEM", "K_EXPSUM", use_km=1), ph("ELEM", "K_SCALE", out=1)],
    "SMAX_BWD": [ph("ELEM", "K_DROPFMA", use_b=1, use_keep=1), ph("SCAL", "S_TOT0"),
                 ph("ELEM", "K_DSOUT", use_b=1, use_keep=1, out=1)],
    "EMB":      [ph("ELEM", "K_EMB_SUM", use_b=1, use_cv0=1, chunk=1), ph("SCAL", "S_LN_MU"),
                 ph("ELEM", "K_EMB_SQDEV", use_b=1, use_cv0=1, chunk=1), ph("SCAL", "S_LN_RSTD"),
                 ph("ELEM", "K_EMB_OUT", use_b=1, use_cv0=1, use_cv1=1, use_cv2=1, use_keep=1, out=1)],
    "CE":       [ph("LDS"), ph("ELEM", "K_CE_MAX"), ph("SCAL", "S_CE_MAXR"), ph("ELEM", "K_CE_EXP"),
                 ph("SCAL", "S_CE_FIN"), ph("XL", xl_s2=1), ph("ELEM", "K_CE_OUT", out=1),
                 ph("SCAL", "S_LOSS", final=1), ph("W0", final=1)],
    "MSE":      [ph("LDS"), ph("ELEM", "K_MSE", out=1), ph("XL", xl_s2=1),
                 ph("SCAL", "S_LOSS", final=1), ph("W0", final=1)],
    "GN_ROWS":  [ph("ELEM", "K_GN", chunk=1), ph("XL", xl_chunk=1),
                 ph("SCAL", "S_GN_ROW", final=1), ph("W0", final=1)],
    "GN_FIN":   [ph("SCAL", "S_GN_FIN", final=1), ph("W03", final=1)],
    "E_ADD":    [ph("ELEM", "K_E_ADD", use_b=1, out=1)],
    "E_MUL":    [ph("ELEM", "K_E_MUL", use_b=1, out=1)],
    "E_SCALE":  [ph("ELEM", "K_E_SCALE", out=1)],
    "E_SCALER": [ph("ELEM", "K_E_SCALER", out=1)],
    "E_DROP":   [ph("ELEM", "K_E_DROP", use_keep=1, out=1)],
    "E_TANH":   [ph("ELEM", "K_E_TANH", out=1)],
    "E_TANHB":  [ph("ELEM", "K_E_TANHB", use_b=1, out=1)],
    "E_COPY":   [ph("ELEM", "K_E_COPY", out=1)],
}
VID = {k: i for i, k in enumerate(VOPS)}
NPH = 10
assert max(len(v) for v in VOPS.values()) < NPH


# ----------------------------------------------------------------------------- emit
def enc_uop(o, last):
    v = OPT[o["opt"]] | (SRC[o["a"]] << 4) | (SRC[o["b"]] << 9) | (SRC[o["c"]] << 14) | (DST[o["dst"]] << 19)
    v |= (o["nega"] << 23) | (o["negc"] << 24) | (o["zk"] << 25) | (o["zm"] << 26) | (o["capy"] << 27)
    v |= (FN[o["fn"]] << 28) | (o["accw"] << 31) | (last << 32)
    return v


def enc_ph(p):
    v = KIND[p["kind"]] | (p["kid"] << 3)
    for i, f in enumerate(("use_b", "use_cv0", "use_cv1", "use_cv2", "use_keep", "use_km", "out", "chunk", "xl_s2",
                           "xl_chunk", "final")):
        v |= p[f] << (9 + i)
    return v


def emit(path):
    L = ["`default_nettype none",
         "// GENERATED by flow/gen/vu_ucode.py -- do not edit (make gen-check).",
         "// vu_ucode: the vector-unit kernel ROM (micro-op per kernel id and pc) and the phase ROM (phase",
         "// per vector op id and index).  Combinational; the sequencer registers the outputs.",
         "// uop: [3:0] opt (0 END 1 FMA 2 SFU 3 POP 4 MAX 5 ACC 6 OUT 7 STORE 8 MIN 9 MOV), [8:4] srca, [13:9] srcb,",
         "//      [18:14] srcc, [22:19] dst, [23] nega, [24] negc, [25] zk, [26] zm, [27] capy, [30:28] fn, [31] accw, [32] last",
         "// phase: [2:0] kind (0 END 1 ELEM 2 SCAL 3 XL 4 LDS 5 WST 6 W0 7 W03), [8:3] kid, [9] use_b, [10] use_cv0,",
         "//      [11] use_cv1, [12] use_cv2, [13] use_keep, [14] use_km, [15] out, [16] chunk, [17] xl_s2, [18] xl_chunk, [19] final",
         "module vu_ucode (",
         "    input  wire [5:0]  kid,",
         "    input  wire [3:0]  pc,",
         f"    output reg  [{UOPW - 1}:0] uop,",
         "    input  wire [4:0]  vop,",
         "    input  wire [3:0]  ph,",
         f"    output reg  [{PHW - 1}:0] phase",
         ");",
         "    always @* begin",
         f"        uop = {UOPW}'d0;",
         "        case ({kid, pc})"]
    for k, ops in KERNELS.items():
        for i, o in enumerate(ops):
            L.append(f"            {{6'd{KID[k]}, 4'd{i}}}: uop = {UOPW}'h{enc_uop(o, i == len(ops) - 1):09x};  // {k}")
    L += ["            default: ;", "        endcase", f"        phase = {PHW}'d0;", "        case ({vop, ph})"]
    for v, phs in VOPS.items():
        for i, p in enumerate(phs):
            L.append(f"            {{5'd{VID[v]}, 4'd{i}}}: phase = {PHW}'h{enc_ph(p):05x};  // {v} {p['kind']}")
    L += ["            default: ;", "        endcase", "    end", "endmodule", "`default_nettype wire", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))


# ----------------------------------------------------------------------------- emulator (check)
def check(seed):
    sys.path.insert(0, os.path.join(ROOT, "model"))
    import hwmodel as H  # noqa
    f32 = np.float32
    KC = 64
    NZ = np.uint32(0x80000000).view(f32)
    bad = 0

    def F(x):
        return f32(x)

    def hfma(a, b, c):
        return H.fma(a, b, c)[0]

    def hadd(a, b):
        return H.add(a, b)[0]

    class Acc:
        def __init__(self, chunk):
            self.p, self.tot, self.c, self.chunk = F(0), F(0), 0, chunk

        def push(self, x, y):
            self.p = hfma(x, y, self.p)
            if self.chunk:
                self.c += 1
                if self.c == KC:
                    self.tot, self.p, self.c = hadd(self.p, self.tot), F(0), 0

        def result(self):
            if not self.chunk:
                return self.p
            return hadd(self.p, self.tot) if self.c else self.tot

    class Lane:
        def __init__(self):
            self.s = [F(0)] * 4
            self.tot = [F(0), F(0)]
            self.maxr = F(0)
            self.xtot = F(0)

    def run_kernel(lane, kname, consts, ds, elems, chunk, A_store=None):
        """elems: list of dict(r0..r4, keep, km, is_y, j).  Returns outputs list."""
        accs = [Acc(chunk), Acc(chunk)]
        outs = []
        maxr = np.uint32(0xff800000).view(f32)
        for e in elems:
            r = [e.get("r0", F(0)), e.get("r1", F(0)), e.get("r2", F(0)), e.get("r3", F(0)), e.get("r4", F(0)), F(0)]
            sfuq = []

            def src(n):
                if n in SRC and SRC[n] < 6:
                    return r[SRC[n]]
                if n[0] == "s":
                    return lane.s[int(n[1])]
                return dict(TOT0=lane.tot[0], TOT1=lane.tot[1], MAXR=lane.maxr, XTOT=lane.xtot, C0=consts[0], C1=consts[1],
                            C2=consts[2], C3=consts[3], ZERO=F(0), ONE=F(1), NZERO=NZ, DS=ds,
                            OHN=(F(-1) if e.get("is_y") else NZ), NINF=np.uint32(0xff800000).view(f32))[n]

            def wr(n, v):
                if DST[n] < 6:
                    r[DST[n]] = v
                else:
                    lane.s[DST[n] - 8] = v

            def zero(o, v):
                if o["zk"] and not e.get("keep", 1):
                    return F(0)
                if o["zm"] and not e.get("km", 1):
                    return F(0)
                return v
            for o in KERNELS[kname]:
                t = o["opt"]
                if t == "FMA":
                    a = src(o["a"]); c = src(o["c"])
                    if o["nega"]:
                        a = (np.array(a, f32).view(np.uint32) ^ np.uint32(0x80000000)).view(f32)[()]
                    if o["negc"]:
                        c = (np.array(c, f32).view(np.uint32) ^ np.uint32(0x80000000)).view(f32)[()]
                    v = hfma(a, src(o["b"]), c)
                    wr(o["dst"], zero(o, v))
                elif t == "SFU":
                    sfuq.append(getattr(H, "log" if o["fn"] == "ln" else o["fn"])(np.array([src(o["a"])], f32))[0])
                elif t == "POP":
                    wr(o["dst"], sfuq.pop(0))
                elif t == "MAX":
                    if not (o["zm"] and not e.get("km", 1)):
                        maxr = H.fmax(maxr, src(o["a"]))[0]
                    if o["capy"] and e.get("is_y"):
                        lane.s[3] = src(o["a"])
                elif t == "ACC":
                    accs[o["accw"]].push(zero(o, src(o["a"])), src(o["b"]))
                elif t == "OUT":
                    outs.append(zero(o, src(o["a"])))
                elif t == "STORE":
                    A_store[e["j"]] = zero(o, src(o["a"]))
                elif t == "MIN":
                    wr(o["dst"], H.fmin(src(o["a"]), src(o["b"]))[0])
                elif t == "MOV":
                    wr(o["dst"], src(o["a"]))
        lane.tot = [accs[0].result(), accs[1].result()]
        lane.maxr = maxr
        return outs

    def run_vop(vop, rows, cols, A, B=None, cv=(None, None, None), keep=None, km=None, s_in=(None, None),
                consts=(0, 0, 0, 0), ds=F(0), labels=None, xl_state=None, lanes=None):
        """rows processed in tiles of 16 (lane = row); returns dict(out, st0, st1, w0, w03, lanes, xl)."""
        consts = [F(c) for c in consts]
        lanes = lanes or [Lane() for _ in range(16)]
        xl = xl_state if xl_state is not None else dict(p=F(0), tot=F(0), c=0)
        OUT = np.zeros((rows, cols), f32)
        st0, st1 = np.zeros(rows, f32), np.zeros(rows, f32)
        res = {}
        A = A.copy()
        for t0 in range(0, rows, 16):
            nr = min(16, rows - t0)
            for p in VOPS[vop]:
                if p["final"]:
                    continue
                for r in range(nr):
                    ln, row = lanes[r], t0 + r
                    if p["kind"] == "LDS":
                        ln.s[0] = s_in[0][row]; ln.s[1] = s_in[1][row]
                    elif p["kind"] == "ELEM":
                        elems = []
                        for j in range(cols):
                            e = dict(j=j, r0=A[row, j])
                            if p["use_b"]: e["r1"] = B[row, j]
                            if p["use_cv0"]: e["r2"] = cv[0][j]
                            if p["use_cv1"]: e["r3"] = cv[1][j]
                            if p["use_cv2"]: e["r4"] = cv[2][j]
                            if p["use_keep"]: e["keep"] = int(keep[row, j])
                            if p["use_km"]: e["km"] = int(km[row, j])
                            if labels is not None: e["is_y"] = int(labels[row] == j)
                            elems.append(e)
                        o = run_kernel(ln, list(KERNELS)[p["kid"]], consts, ds, elems, p["chunk"], A[row])
                        if p["out"]:
                            OUT[row, :] = o
                    elif p["kind"] == "SCAL":
                        run_kernel(ln, list(KERNELS)[p["kid"]], consts, ds, [dict(j=0)], 0)
                    elif p["kind"] == "WST":
                        st0[row], st1[row] = ln.s[0], ln.s[1]
                if p["kind"] == "XL":
                    for r in range(nr):
                        v = lanes[r].s[2] if p["xl_s2"] else lanes[r].tot[0]
                        xl["p"] = hfma(v, F(1), xl["p"])
                        if p["xl_chunk"]:
                            xl["c"] += 1
                            if xl["c"] == KC:
                                xl["tot"], xl["p"], xl["c"] = hadd(xl["p"], xl["tot"]), F(0), 0
        for p in VOPS[vop]:
            if not p["final"]:
                continue
            if p["kind"] == "SCAL":
                if p["kid"] in (KID["S_LOSS"], KID["S_GN_ROW"]):      # XL flush first
                    xt = (hadd(xl["p"], xl["tot"]) if xl["c"] else xl["tot"]) if p["kid"] == KID["S_GN_ROW"] else xl["p"]
                    for ln in lanes:
                        ln.xtot = xt
                for ln in lanes:
                    run_kernel(ln, list(KERNELS)[p["kid"]], consts, ds, [dict(j=0)], 0)
            elif p["kind"] == "W0":
                res["w0"] = lanes[0].s[0]
            elif p["kind"] == "W03":
                res["w0"], res["w3"] = lanes[0].s[0], lanes[0].s[3]
        res.update(out=OUT, st0=st0, st1=st1, A=A, lanes=lanes, xl=xl)
        return res

    rng = np.random.default_rng(seed)
    DS = np.uint32(0x3F8E38E4).view(f32)

    def rnd(*shape, lo=-2.0, hi=2.0):
        return rng.uniform(lo, hi, shape).astype(f32)

    def cmpbits(name, a, b):
        nonlocal bad
        a, b = np.ascontiguousarray(a, f32).view(np.uint32).reshape(-1), np.ascontiguousarray(b, f32).view(np.uint32).reshape(-1)
        m = int((a != b).sum())
        bad += m
        print(f"  {name}: {a.size} values, {m} mismatches")

    # LN fwd / bwd (d = 200: chunk tails) and emb
    rows, d = 37, 200
    x, g, b = rnd(rows, d), rnd(d), rnd(d)
    y, mu, rs = H.ln_fwd(x, g, b)
    r = run_vop("LN_FWD", rows, d, x, cv=(None, g, b), consts=(1.0 / d, 1e-5, 0, 0))
    cmpbits("ln_fwd y", r["out"], y); cmpbits("ln_fwd mu", r["st0"], mu); cmpbits("ln_fwd rstd", r["st1"], rs)
    dy = rnd(rows, d)
    dx = H.ln_bwd(x, mu, rs, g, dy)
    r = run_vop("LN_BWD", rows, d, x, B=dy, cv=(g, None, None), s_in=(mu, rs), consts=(1.0 / d, 0, 0, 0))
    cmpbits("ln_bwd dx", r["out"], dx)
    w, tt, pe = rnd(rows, d), rnd(d), rnd(rows, d)
    e = H.add(H.add(w, tt), pe)
    y, _, _ = H.ln_fwd(e, g, b)
    keep = (rng.uniform(size=(rows, d)) > 0.1).astype(np.uint8)
    y = H.drop(y, keep, DS)
    r = run_vop("EMB", rows, d, w, B=pe, cv=(tt, g, b), keep=keep, consts=(1.0 / d, 1e-5, 0, 0), ds=DS)
    cmpbits("emb", r["out"], y)
    # softmax fwd / rec / bwd: B=2, H=3, L=48 (masked keys, one fully masked row)
    Bn, Hn, L = 2, 3, 48
    S = rnd(Bn, Hn, L, L, lo=-30, hi=30)
    kmask = (rng.uniform(size=(Bn, L)) > 0.2).astype(np.uint8)
    kmask[1, :] = 0
    ka = (rng.uniform(size=(Bn, Hn, L, L)) > 0.1).astype(np.uint8)
    P, Pd, m, rr = H.smax_fwd(S, kmask, ka, DS)
    km_rows = np.repeat(kmask[:, None, None, :], Hn, 1).repeat(L, 2).reshape(-1, L)
    r = run_vop("SMAX_FWD", Bn * Hn * L, L, S.reshape(-1, L), keep=ka.reshape(-1, L), km=km_rows, ds=DS)
    cmpbits("smax_fwd P'", r["out"], Pd); cmpbits("smax m", r["st0"], m); cmpbits("smax r", r["st1"], rr)
    r = run_vop("SMAX_REC", Bn * Hn * L, L, S.reshape(-1, L), km=km_rows, s_in=(m.reshape(-1), rr.reshape(-1)))
    cmpbits("smax_rec P", r["out"], P)
    if os.environ.get("VU_DBG"):
        o = r["out"].reshape(Bn, Hn, L, L)
        bad = np.argwhere(o.view(np.uint32) != P.view(np.uint32))[:3]
        for b, h, i, j in bad:
            print("DBG", b, h, i, j, "km", kmask[b, j], "got", hex(o[b, h, i, j].view(np.uint32)), "want", hex(P[b, h, i, j].view(np.uint32)), "m", m[b, h, i], "r", rr[b, h, i],
                  "A", r["A"].reshape(Bn, Hn, L, L)[b, h, i, j], "S", S[b, h, i, j], "e", H.exp(H.sub(S[b, h, i, j], m[b, h, i]))[0])
        sys.exit(1)
    dPd = rnd(Bn, Hn, L, L)
    dS = H.smax_bwd(P, dPd, ka, DS, 0.125)
    r = run_vop("SMAX_BWD", Bn * Hn * L, L, P.reshape(-1, L), B=dPd.reshape(-1, L), keep=ka.reshape(-1, L), ds=DS,
                consts=(0.125, 0, 0, 0))
    cmpbits("smax_bwd dS", r["out"], dS)
    # CE / MSE
    Bn, C = 32, 3
    z = rnd(Bn, C, lo=-5, hi=5)
    yl = rng.integers(0, C, Bn)
    wv = np.ones(Bn, f32); wv[-3:] = 0
    inv_b = H.recip(np.array([29], f32))[0]
    loss, dz = H.ce(z, yl, wv, inv_b)
    r = run_vop("CE", Bn, C, z, s_in=(yl, wv), labels=yl, consts=(inv_b, 0, 0, 0))
    cmpbits("ce dz", r["out"], dz); cmpbits("ce loss", np.array([r["w0"]]), np.array([loss]))
    yt = rnd(Bn, 1)
    loss_m, dz_m = H.mse(z[:, :1], yt, wv, inv_b)
    r = run_vop("MSE", Bn, 1, z[:, :1], s_in=(yt.reshape(-1), wv), consts=(inv_b, H.mul(F(2), inv_b)[0], 0, 0))
    cmpbits("mse dz", r["out"], dz_m); cmpbits("mse loss", np.array([r["w0"]]), np.array([loss_m]))
    # gnorm: 3 tensors (rows x rowlen), chunk tails on rows and cols
    shapes = [(1, 256), (70, 200), (3, 77)]
    gs = [rnd(*s, lo=-0.1, hi=0.1) for s in shapes]
    off = np.cumsum([0] + [g.size for g in gs])
    flat = np.concatenate([g.reshape(-1) for g in gs])
    norms, G, c = H.gnorm_clip(flat.copy(), off, [s[1] for s in shapes])
    lanes, xl = None, None
    nn = []
    for i, g in enumerate(gs):
        r = run_vop("GN_ROWS", g.shape[0], g.shape[1], g, lanes=lanes, xl_state=None if i == 0 else None)
        lanes = r["lanes"]
        nn.append(r["w0"])
        if i == 0:
            pass
    cmpbits("gnorm norms", np.array(nn), norms)
    r = run_vop("GN_FIN", 0, 0, np.zeros((0, 0), f32), lanes=lanes, consts=(1e-6, 0, 0, 0))
    cmpbits("gnorm G", np.array([r["w0"]]), np.array([G])); cmpbits("gnorm c", np.array([r["w3"]]), np.array([c]))
    lanes[0].s[3] = r["w3"]
    r = run_vop("E_SCALER", 70, 200, gs[1], lanes=lanes)
    cmpbits("gnorm scaled", r["out"], H.mul(gs[1], c))
    # element-wise
    a, bb = rnd(20, 50), rnd(20, 50)
    keep = (rng.uniform(size=(20, 50)) > 0.1).astype(np.uint8)
    cmpbits("e_add", run_vop("E_ADD", 20, 50, a, B=bb)["out"], H.add(a, bb))
    cmpbits("e_mul", run_vop("E_MUL", 20, 50, a, B=bb)["out"], H.mul(a, bb))
    cmpbits("e_scale", run_vop("E_SCALE", 20, 50, a, consts=(0.3, 0, 0, 0))["out"], H.mul(a, F(0.3)))
    cmpbits("e_drop", run_vop("E_DROP", 20, 50, a, keep=keep, ds=DS)["out"], H.drop(a, keep, DS))
    th = H.tanh(a)
    cmpbits("e_tanh", run_vop("E_TANH", 20, 50, a)["out"], th)
    cmpbits("e_tanhb", run_vop("E_TANHB", 20, 50, th, B=bb)["out"], H.tanh_bwd(bb, th))
    cmpbits("e_copy", run_vop("E_COPY", 20, 50, a)["out"], a)
    print("VU-UCODE CHECK " + ("PASS" if bad == 0 else f"FAIL ({bad} mismatches)"))
    return bad


if __name__ == "__main__":
    if sys.argv[1] == "emit":
        emit(sys.argv[2])
    elif sys.argv[1] == "check":
        sys.exit(1 if check(int(sys.argv[2]) if len(sys.argv) > 2 else 1) else 0)
