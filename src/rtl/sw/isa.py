#!/usr/bin/env python3
"""LoWA-TA descriptor encoding (P6c, docs/isa.md section 1-3 as built).  One module shared by the compiler
(sw/compile.py), the program interpreter (sw/interp.py) and the sequencer testbench generators.

Descriptor = 16 * NSL words (NSL = 1..3 slots of 64 B), little-endian 32-bit words:
  w0          header  [7:0] OP  [11:8] UNIT  [15:12] WAIT  [19:16] SIG  [20] CLR  [22:21] NSL-1  [31:23] TAG
  w1 .. wR    REL     R = 2 * NSL words of 4-bit term selectors, nibble i (word 1 + i // 8, bits 4 (i % 8))
                      belongs to descriptor word i; header / REL nibbles are 0
  wR+1 ..     payload p0, p1, ... (13 / 27 / 41 words); the sequencer adds TERM[sel] to every word
TERM[0] = 0, TERM[1..12] = layer terms (LOOP: start 0, +/- STR[k] per iteration), TERM[13] = loop index,
TERM[14] = STEP_T (CSR), TERM[15] = slot delta (DOORBELL slot ? SLOT_DELTA : 0).
"""
import numpy as np

U32 = np.uint32
UNIT_SEQ, UNIT_DMA, UNIT_GEMM, UNIT_VEC, UNIT_LOWA = 0, 1, 2, 3, 4
OPS = {  # name: (opcode, unit, slots)
    "NOP": (0x00, UNIT_SEQ, 1), "GEMM": (0x01, UNIT_GEMM, 3), "GEMM_ACC2": (0x02, UNIT_GEMM, 3),
    "VEC": (0x10, UNIT_VEC, 2), "LOSS": (0x16, UNIT_VEC, 2), "GNORM": (0x22, UNIT_VEC, 2),
    "LOWA_SG": (0x20, UNIT_LOWA, 1), "LOWA_SCAL": (0x21, UNIT_LOWA, 1), "ADAMW": (0x23, UNIT_LOWA, 1),
    "DMA_COPY": (0x30, UNIT_DMA, 1), "FENCE": (0x31, UNIT_SEQ, 1), "LOOP": (0x32, UNIT_SEQ, 1),
    "SETCSR": (0x33, UNIT_SEQ, 1), "END": (0x3F, UNIT_SEQ, 1),
}
OPNAME = {v[0]: k for k, v in OPS.items()}
NONE = 0xF
T_LIDX, T_STEP, T_SLOT = 13, 14, 15
# FENCE unit mask bits
F_DMA, F_GEMM, F_VEC, F_LOWA = 1, 2, 4, 8
UNIT_FBIT = {UNIT_DMA: F_DMA, UNIT_GEMM: F_GEMM, UNIT_VEC: F_VEC, UNIT_LOWA: F_LOWA}

# payload word layouts (field name per word; packed words list (name, lsb, width))
GEMM_FIELDS = [
    [("m", 0, 16), ("n", 16, 16)], [("k0", 0, 16), ("k1", 16, 16)], [("k2", 0, 16), ("nb1", 16, 16)],
    [("nb2", 0, 16), ("a_trn", 16, 1), ("b_trn", 17, 1), ("ep_bias", 18, 1), ("ep_scale", 19, 1), ("ep_gelu", 20, 1),
     ("ep_gelu_bwd", 21, 1), ("ep_drop", 22, 1), ("ep_resid", 23, 1), ("m_en", 24, 1), ("m_trn", 25, 1)],
    [("ep_sexp", 0, 8), ("site", 8, 4)], [("layer", 0, 4)],
    "a_base0", "a_base1", "a_base2", "a_rs", "a_bs1", "a_bs2",
    "b_base0", "b_base1", "b_base2", "b_rs", "b_bs1", "b_bs2",
    "c_base", "c_rs", "c_bs1", "c_bs2", "bias_base", "aux_base", "o2_base", "ep_ds",
    [("m_nb", 0, 8), ("m_mb", 8, 8), ("m_kc0", 16, 8), ("m_kc1", 24, 8)],
    [("m_kc2", 0, 8), ("m_ms0", 8, 2), ("m_ms1", 10, 2), ("m_ms2", 12, 2)],
    "m_hits_a", "m_dir_a", "m_coef_a", "m_hits_b", "m_dir_b", "m_coef_b", "m_h_base", "m_gamma",
]
VEC_FIELDS = [
    [("vop", 0, 5), ("st_en", 8, 1), ("x_reset", 9, 1), ("g_reset", 10, 1), ("site", 12, 4)], [("layer", 0, 4)],
    [("rows", 0, 16), ("cols", 16, 16)], [("rows_per_b", 0, 16), ("emb_l", 16, 16)],
    "a_base", "a_rs", "b_base", "b_rs", "c_base", "c_rs", "cv0_base", "cv1_base", "cv2_base", "km_rs",
    "st0_base", "st1_base", "o2_base", "ids_base", "mk_base", "mk_rs", "c0", "c1", "c2", "c3", "ds",
]
LOWA_FIELDS = [[("op", 0, 2), ("flag", 8, 1)], "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "len"]
DMA_FIELDS = ["src", "dst", "n0", "n1", "s_s1", "d_s1", "n2", "s_s2", "d_s2"]
LOOP_FIELDS = ["body", "count", "start", "dir"]        # body = descriptors (bytes) of the body; dir 0 up, 1 down
SETCSR_FIELDS = ["csr", "value"]                        # csr 1..12: STR[k]; 16: perf marker
FENCE_FIELDS = ["mask"]
UNIT_FIELDS = {UNIT_GEMM: GEMM_FIELDS, UNIT_VEC: VEC_FIELDS, UNIT_LOWA: LOWA_FIELDS, UNIT_DMA: DMA_FIELDS}
SEQ_FIELDS = {"LOOP": LOOP_FIELDS, "SETCSR": SETCSR_FIELDS, "FENCE": FENCE_FIELDS, "END": [], "NOP": []}
LOWA_OP = {"LOWA_SG": 0, "LOWA_SCAL": 1, "ADAMW": 2}
VOPS = ["LN_FWD", "LN_BWD", "SMAX_FWD", "SMAX_REC", "SMAX_BWD", "EMB", "CE", "MSE", "GN_ROWS", "GN_FIN", "E_ADD",
        "E_MUL", "E_SCALE", "E_SCALER", "E_DROP", "E_TANH", "E_TANHB", "E_COPY"]
VID = {k: i for i, k in enumerate(VOPS)}


def fields_of(name):
    u = OPS[name][1]
    return SEQ_FIELDS[name] if u == UNIT_SEQ else UNIT_FIELDS[u]


def encode(name, f, rel=None, wait=NONE, sig=NONE, clr=0, tag=0):
    """f: field values; rel: {field name of an address word: term selector}.  Returns uint32[16 * NSL]."""
    op, unit, nsl = OPS[name]
    nw = 16 * nsl
    nrel = 2 * nsl
    w = np.zeros(nw, np.uint64)
    w[0] = op | unit << 8 | (wait & 0xF) << 12 | (sig & 0xF) << 16 | (clr & 1) << 20 | (nsl - 1) << 21 | (tag & 0x1FF) << 23
    lay = fields_of(name)
    assert len(lay) <= nw - 1 - nrel, name
    known = set()
    rel = rel or {}
    for i, spec in enumerate(lay):
        wi = 1 + nrel + i
        if isinstance(spec, str):
            w[wi] = int(f.get(spec, 0)) & 0xFFFFFFFF
            known.add(spec)
            names = [spec]
        else:
            v = 0
            for nm, lsb, wd in spec:
                x = int(f.get(nm, 0))
                assert 0 <= x < (1 << wd) or wd == 8 and -128 <= x < 128, (name, nm, x)
                v |= (x & ((1 << wd) - 1)) << lsb
                known.add(nm)
            w[wi] = v
            names = [nm for nm, _, _ in spec]
        sel = 0
        for nm in names:
            if nm in rel:
                assert sel == 0
                sel = rel[nm]
        w[1 + wi // 8] |= sel << (4 * (wi % 8))
    bad = (set(f) | set(rel)) - known
    assert not bad, (name, bad)
    return w.astype(U32)


def header(w0):
    w0 = int(w0)
    return dict(op=w0 & 0xFF, unit=(w0 >> 8) & 0xF, wait=(w0 >> 12) & 0xF, sig=(w0 >> 16) & 0xF, clr=(w0 >> 20) & 1,
                nsl=((w0 >> 21) & 3) + 1, tag=(w0 >> 23) & 0x1FF)


def decode_words(words, term):
    """words: uint32[16 * NSL] of one descriptor; term: 16 TERM values.  Returns (header, fields, payload words
    after the term adds) exactly as the sequencer's decoder forms them."""
    h = header(words[0])
    nsl = h["nsl"]
    nrel = 2 * nsl
    out = []
    for wi in range(1 + nrel, 16 * nsl):
        sel = (int(words[1 + wi // 8]) >> (4 * (wi % 8))) & 0xF
        out.append((int(words[wi]) + int(term[sel])) & 0xFFFFFFFF)
    name = OPNAME.get(h["op"])
    if name is None or OPS[name][1] != h["unit"] or OPS[name][2] != nsl:
        raise ValueError(f"descriptor decode error: {h}")
    f = {}
    for i, spec in enumerate(fields_of(name)):
        if isinstance(spec, str):
            f[spec] = out[i]
        else:
            for nm, lsb, wd in spec:
                f[nm] = (out[i] >> lsb) & ((1 << wd) - 1)
    if "ep_sexp" in f and f["ep_sexp"] >= 128:
        f["ep_sexp"] -= 256
    return h, name, f, out
