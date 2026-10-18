#!/usr/bin/env python3
"""LoWA-TA P0 budget calculator: bytes, MACs, cycles, DRAM traffic, SRAM macros, area.

Every number in docs/{memory_map,dataflow,budget}.md that is not a measured anchor is
printed by this script.  Usage:
    python3 src/rtl/docs/calc/budget.py                 # B=32, L=128, all configs
    python3 src/rtl/docs/calc/budget.py --L 41          # another padded length
No torch, no numpy: plain python3.
"""
import argparse
import math

# ---------------------------------------------------------------- workload (PLAN fixed decision 1)
D, H, DH, F, NL = 768, 12, 64, 3072, 12
VOCAB, NPOS = 50265, 514
K_ATOM, S_TAP, WIN = 256, 25, 7           # k centres, diamond taps, 2K+1 window
LOWA_TOK_OVH = 218                         # P5 measured per-token overhead of lw_sg (see cyc_lowa)
OPT_CYC_EL = 9.0                           # P5 measured AdamW cycles per element (lw_adamw)
N_MOD = 2 * NL                             # query, value
FP = 4                                     # bytes per fp32

# ---------------------------------------------------------------- physical anchors
SRAM_W, SRAM_H = 301.300, 224.930          # um, gf180mcu_ocd_ip_sram__sram256x8m8wm1 LEF SIZE
SRAM_B = 256                               # bytes per macro (256 x 8)
MACRO_MM2 = SRAM_W * SRAM_H / 1e6          # 0.06777 mm2
HALO = 1.25                                # placement halo/channel factor on macro area (assumption)
UTIL = 0.60                                # std-cell placement utilisation (assumption, prior chip ~?)
FLOP_UM2 = 8420.0 / 142                    # dfxtp_2, from GanAccelAsic mb_lane.stat (tt)
# Measured Yosys/abc tt areas from /workspace/GanAccelAsic/src/syn/build/*.stat (weak anchors):
A_FP32_ADD = 24142.8                       # mb_fp32_add_rne, combinational
A_FP16_MUL = 14793.5                       # mb_fp16_mul (11x11 significand)
# ASSUMPTIONS (P2 must measure): fp32 FMA PE = 24x24 array scaled from the fp16 mul by
# (24/11)^2 on 80 % of its area, + 2x an fp32 adder for the 3-wide align/add, + 15k
# normalise/round, + LAMBDA*150 pipeline flops + LAMBDA*32 accumulator flops + 100 operand flops.
LAMBDA = 4                                 # FMA latency = accumulators per PE
# MEASURED in P2 (docs/p2_report.md, `make synth-fp`, Yosys tt area incl. its pipeline flops):
# hw/fp/fp_fma at STAGES = LAMBDA = 4, and the SFU (fp_sfu LF=2, 4 threads, its own FMA inside).
FMA_UM2 = 119410.0                         # --fma-um2 overrides
SFU_UM2 = 222855.0                         # --sfu-um2 overrides
P0_MODEL = False                           # --p0: the P0 assumption instead of the measurement


def pe_area_p0(lam=LAMBDA):
    mul = 0.8 * A_FP16_MUL * (24 / 11) ** 2 + 0.2 * A_FP16_MUL
    add = 2 * A_FP32_ADD
    norm = 15000.0
    flops = (lam * 150 + lam * 32 + 100) * FLOP_UM2
    return mul + add + norm + flops


def pe_area_um2(lam=LAMBDA):
    """PE = measured pipelined FMA + lam accumulators + 100 operand flops."""
    if P0_MODEL:
        return pe_area_p0(lam)
    return FMA_UM2 + (lam * 32 + 100) * FLOP_UM2


def cfg_list():
    # R x C array, V vector lanes, dbuf = operand double-buffering, lane_macros = macros per 32-bit W lane
    # (one macro = 256 x 8 bit, so 4 macros = 256 fp32 words), xdeep = macro rows stacked in XBUF
    return [
        dict(name="full16", R=16, C=16, V=16, dbuf=2, lane_macros=4, xdeep=4),
        dict(name="full16-lean", R=16, C=16, V=16, dbuf=2, lane_macros=4, xdeep=1),
        dict(name="mid8", R=8, C=8, V=8, dbuf=2, lane_macros=4, xdeep=4),
        dict(name="shuttle2x1", R=2, C=1, V=1, dbuf=1, lane_macros=4, xdeep=1),
        dict(name="shuttle1x1", R=1, C=1, V=1, dbuf=1, lane_macros=4, xdeep=1),
    ]


def gemm_geom(c):
    R, C, lam = c["R"], c["C"], LAMBDA
    wbuf = C * c["lane_macros"]                       # macros per W buffer: C lanes x 32 bit
    xbuf = max(1, math.ceil(R / lam)) * 4 * c["xdeep"]  # X lanes need 1 word per lam cycles; xdeep macro rows
    words_w = wbuf * SRAM_B // FP
    words_x = xbuf * SRAM_B // FP
    bc = C * lam                                      # output columns per pass
    kc = words_w // bc                                # K chunk held per W block
    br = words_x // kc                                # token rows per X block
    acc_words = br * bc
    acc = max(4, math.ceil(acc_words * FP / SRAM_B))
    return dict(wbuf=wbuf, xbuf=xbuf, bc=bc, kc=kc, br=br, acc=acc,
                bpm=FP * (1.0 / br + 1.0 / bc))       # DRAM bytes per MAC (operand reload)


OVH_BLOCK = 100                            # P3 measured: cycles per output block (load/epilogue hand-off)
OVH_CHUNK = 24                             # P3 measured: cycles per (block, K chunk) (burst gaps, pass gating)


# P6a measured (docs/p6a_report.md section 4): the memory system (hw/mem/mm_top) on the 256-bit AXI4 port,
# DRAM model at 800 MB/s (32 B / cycle at 25 MHz), 24-cycle read latency, R and W sharing the bandwidth
# (+wfair=1, nominal).  EFF = useful beats / cycles of the run (keep-bit stand-in traffic excluded, ruling R7):
#   GEMM  real_ffn1_256 (16x16): 2,557,440 beats in 2,576,250 cycles = 0.993 (2,434,146 with an unlimited bus)
#   vector real_ln_fwd: 1,623,040 beats in 5,255,246 cycles = 0.309 (P6a, 1-beat bursts: outstanding-limited);
#   P6b (ruling R8, vu_loader per-row 4-beat bursts): 2,963,526 cycles = 0.548 (compute bound 2,656,302)
# Sensitivity (--dram-rprio): a DRAM that serves writes only on spare bandwidth while reads stream starves the
# GEMM's output writes and stalls its drain: real_ffn1_256 3,841,682 cycles = 0.666.
# An op costs max(compute cycles, bytes / (DRAM_BPC x EFF)), per GEMM row and per vector op.
DRAM_BPC = 32.0                            # bytes / cycle, 256-bit port
EFF_GEMM = 2557440 / 2576250               # 0.993
EFF_GEMM_RPRIO = 2557440 / 3841682         # 0.666
EFF_VEC = 1623040 / 2963526                # 0.548 (P6b; P6a 0.309)
VEC_B_PASS = 1623040 * 32 / 9437184        # 5.50 bytes per vector element-pass (real_ln_fwd: 3 reads + 1 write / 3)
DRAM_CHARGE = True                         # --no-dram: the pre-P6a compute-only step time

EP_RATE_GELU = 0.43                        # P3 measured: GELU epilogue elements / cycle (16 SFUs)
EP_RATE_GELU_BWD = 0.32                    # P3 measured: GELU' epilogue elements / cycle


def gemm_cycles(c, M, N, K, ep_rate=None, nb=1):
    """Output-stationary engine (P3, docs/p3_report.md section 5).  Per output block (br x bc)
    and K chunk the array needs micro-tiles x lam x kc cycles (a short last chunk costs a full
    drain slot) and the loader one cycle per 8-word beat of the X and W tiles (the bound when
    M < br); a block also cannot finish faster than its epilogue (ep_rate elements / cycle,
    default the drain width DW words / cycle; GELU rates measured above); plus the measured
    per-block / per-chunk overheads, the first tile's load and the last block's epilogue
    (exposed).  Full chunks stream with no skew bubble (the P0 model charged R + C per pass)."""
    R, C, lam = c["R"], c["C"], LAMBDA
    g = gemm_geom(c)
    bc, kc, br = g["bc"], g["kc"], g["br"]
    if ep_rate is None:
        ep_rate = max(1, R * C // kc)
    chunks = [kc] * (K // kc) + ([K % kc] if K % kc else [])
    shapes = [(br, M // br), (M % br, 1)] if M % br else [(br, M // br)]          # (rows, count)
    cshapes = [(bc, N // bc), (N % bc, 1)] if N % bc else [(bc, N // bc)]        # (cols, count)

    def block(rows, cols):
        mts = math.ceil(rows / R)
        blk = sum(max(mts * lam * kc, math.ceil((rows + cols) * kt / 8)) + OVH_CHUNK for kt in chunks)
        return max(blk, rows * cols / ep_rate) + OVH_BLOCK

    cyc = sum(block(r, cl) * nr * nc for r, nr in shapes for cl, nc in cshapes)
    first = math.ceil((shapes[0][0] + cshapes[0][0]) * chunks[0] / 8)              # first tile load
    last_ep = shapes[-1][0] * cshapes[-1][0] / ep_rate                             # last block's epilogue
    return int(cyc * nb + first + last_ep)          # nb batches stream in one descriptor


# P4 vector-unit rates (docs/p4_report.md section 4), measured in tb/vec on the real roberta-base
# tensors (V = 16 lanes, lane = row): cycles per element per lane = cycles / (tiles x row length).
# A row op costs, per 16-row tile and 256-word row segment, the larger of the segment load
# (one 8-word beat per row per operand on one 256-bit channel: 2 cycles per column per operand),
# the kernel (one fp_fma per lane under micro-ops, N ops per element; SFU passes 13 (exp) / 25
# (tanh) cycles per element) and the output beats (2 cycles per column), plus the per-tile
# scalar phases.  LN over d = 768 (3 segments) re-loads every segment for every pass.
VEC_RATE = {                                  # cycles per element per lane (V = 16), measured P4
    "LN_FWD": 13.51, "LN_BWD": 23.75, "SMAX_FWD": 23.44, "SMAX_REC": 22.4, "SMAX_BWD": 14.69, "EMB": 30.70,
    "GN": 9.74, "ELEM": 7.13, "TANH": 25.5, "CE": 60.0,     # ELEM: dual-operand add on 1024 x 768
}


def vec_cycles(op, rows, cols, V=16):
    """cycles of one vector descriptor (op over rows x cols, lane = row)"""
    return -(-rows // V) * cols * VEC_RATE[op]


# test-case model (tb/vec/run_tests.py): elements as counted by gen_vec.py (passes x rows x cols)
VEC_CASE = {"ln_fwd": ("LN_FWD", 3), "ln_bwd": ("LN_BWD", 2), "emb": ("EMB", 3), "gnorm": ("GN", 2), "elem": ("ELEM", 1),
            "smax": ("SMAX_FWD", 2.4), "loss": ("TANH", 1.1), "real_smax": ("SMAX_FWD", 3), "real_smax_bwd": ("SMAX_BWD", 2),
            "real_ln_fwd": ("LN_FWD", 3), "real_ln_bwd": ("LN_BWD", 2), "real_emb": ("EMB", 3), "real_loss": ("TANH", 2),
            "real_gnorm": ("GN", 2)}


def vec_cycles_case(case, elements, V=16):
    op, passes = VEC_CASE.get(case, ("ELEM", 1))
    return elements / passes * VEC_RATE[op] / V


def step(c, B, L, cls_prune=True, C_LAB=2):
    N = B * L
    E = B * H * L * L                                 # attention score elements
    g = gemm_geom(c)
    rows = []                                         # (name, M, N, K, count)

    def G(name, M, Nn, K, cnt=1):
        rows.append((name, M, Nn, K, cnt))

    for layer in range(NL):
        last = cls_prune and layer == NL - 1
        Nq = B if last else N                         # rows whose output is ever used
        # forward
        G("fwd_q", Nq, D, D); G("fwd_k", N, D, D); G("fwd_v", N, D, D)
        G("fwd_qk", (1 if last else L), L, DH, B * H)
        G("fwd_pv", (1 if last else L), DH, L, B * H)
        G("fwd_o", Nq, D, D); G("fwd_ffn1", Nq, F, D); G("fwd_ffn2", Nq, D, F)
        # backward (dX only; frozen weights get no gradient)
        G("bwd_ffn2_dx", Nq, F, D); G("bwd_ffn1_dx", Nq, D, F); G("bwd_o_dx", Nq, D, D)
        G("bwd_qk_recompute", (1 if last else L), L, DH, B * H)
        G("bwd_dP", (1 if last else L), L, DH, B * H)               # dO V^T
        G("bwd_dV", L, DH, (1 if last else L), B * H)               # P'^T dO
        G("bwd_dQ", (1 if last else L), DH, L, B * H)               # dS K
        if layer > 0:
            G("bwd_dK", L, DH, (1 if last else L), B * H)           # dS^T Q
            G("bwd_qkv_dx", N, D, 3 * D)                            # [dQ dK dV][Wq;Wk;Wv]
    # head
    G("head_dense", B, D, D); G("head_out", B, C_LAB, D)
    G("head_dW_dense", D, D, B); G("head_dW_out", C_LAB, D, B)
    G("head_dx_out", B, D, C_LAB); G("head_dx_dense", B, D, D)

    macs = sum(M * Nn * K * n for _, M, Nn, K, n in rows)
    ep_rates = {"fwd_ffn1": EP_RATE_GELU, "bwd_ffn2_dx": EP_RATE_GELU_BWD}   # GELU / GELU' epilogues
    cyc_gemm_c = sum(gemm_cycles(c, M, Nn, K, ep_rates.get(name), n) for name, M, Nn, K, n in rows)
    dram_gemm = sum((g["bpm"] * M * Nn * K + FP * M * Nn) * n for _, M, Nn, K, n in rows)
    # P6a: each GEMM costs max(array/loader cycles, its DRAM bytes at the measured bus efficiency)
    cyc_gemm_m = sum(max(gemm_cycles(c, M, Nn, K, ep_rates.get(name), n),
                         (g["bpm"] * M * Nn * K + FP * M * Nn) * n / (DRAM_BPC * EFF_GEMM)) for name, M, Nn, K, n in rows)
    cyc_gemm = cyc_gemm_m if DRAM_CHARGE else cyc_gemm_c

    # vector unit element-passes (epilogues fused into GEMM drain are not counted: hidden)
    Lq = 1 if cls_prune else L
    vec = (3 * D * N)                                   # embedding LN (stats x2 + normalise)
    vec += NL * (3 * E + 6 * N * D)                     # fwd: softmax 3 passes, 2 LN x 3 passes
    vec += NL * (2 * E + 6 * N * D)                     # bwd: softmax row-dot + dS, 2 LN-bwd x 3
    vec += NL * N * D                                   # recompute next-layer input X = LN2(z2)
    # P4: measured per-op rates (VEC_RATE, V = 16 lanes) instead of one element-pass per cycle per lane
    cyc_vec = (VEC_RATE["EMB"] * N * D
               + NL * (VEC_RATE["SMAX_FWD"] * E + 2 * VEC_RATE["LN_FWD"] * N * D)
               + NL * (VEC_RATE["SMAX_REC"] * E + VEC_RATE["SMAX_BWD"] * E + 2 * VEC_RATE["LN_BWD"] * N * D
                       + VEC_RATE["LN_FWD"] * N * D)) / c["V"]       # per-lane rates, V lanes
    vec_bytes = vec * VEC_B_PASS                        # P6a measured bytes per element-pass (P0: 4.0)
    cyc_vec_c = cyc_vec
    # P6a: per op, max(kernel cycles, its element-passes x VEC_B_PASS at the measured efficiency)
    vops = [(VEC_RATE["EMB"] * N * D, 3 * D * N)]
    vops += [(VEC_RATE["SMAX_FWD"] * E, 3 * E), (2 * VEC_RATE["LN_FWD"] * N * D, 6 * N * D),
             ((VEC_RATE["SMAX_REC"] + VEC_RATE["SMAX_BWD"]) * E, 2 * E), (2 * VEC_RATE["LN_BWD"] * N * D, 6 * N * D),
             (VEC_RATE["LN_FWD"] * N * D, N * D)] * NL
    assert sum(p for _, p in vops) == vec
    cyc_vec_m = sum(max(k / c["V"], p * VEC_B_PASS / (DRAM_BPC * EFF_VEC)) for k, p in vops)
    cyc_vec = cyc_vec_m if DRAM_CHARGE else cyc_vec_c

    # LoWA sampled gradient: per token per module, k atoms, WIN cycles each (window read)
    lowa_tokens = N * (N_MOD - 1) + B                   # last-layer query: only CLS rows nonzero
    # P5 measured (lw_sg, NL = 4): 2,778 cycles / token on the real layer-0 query (model 2,560): the row
    # load (D words, single-buffered) + the window-bound atom dispatch + LOWA_TOK_OVH (chain-slot wait of
    # ~6 cycles per 8 atoms, row-request latency, chain drain)
    cyc_lowa = lowa_tokens * (K_ATOM * WIN + D + LOWA_TOK_OVH)
    lowa_bytes = (lowa_tokens * 2 * D) * FP             # dY row + X row per token per module
    lowa_macs = lowa_tokens * K_ATOM * S_TAP * 2        # mul + fma

    # merge-on-load: hits re-applied on every W-block reload of q,v (fwd + bwd)
    reloads = math.ceil(N / g["br"])
    merge_hits = N_MOD * 2 * reloads * K_ATOM * S_TAP
    merge_bytes = merge_hits * 4                        # 32-bit hit-list entry

    # embeddings gather + optimizer
    n_head = D * D + D + D * C_LAB + C_LAB
    n_train = N_MOD * K_ATOM + n_head
    emb_bytes = N * D * FP + L * D * FP
    opt_bytes = n_train * FP * 7 + n_train * FP * 2      # p,g,m,v read; p,m,v write; + norm pass
    # P5 measured (lw_adamw, NSQ 2 / NRC 1, one element stream): 9.0 cycles / element on the real step-B
    # coefficients (55,463 cycles / 6,144) and head (5,329,353 / 592,130); the AdamW update runs in the
    # LoWA unit, not across the V vector lanes
    cyc_opt = n_train * OPT_CYC_EL
    cyc_misc = (N * D * 3) / c["V"]                     # embedding add + dropout

    # stash (written fwd, read bwd): per layer z1, z2, Q, K, V (Nd) + u (N F) + stats
    st_layer = (5 * N * D + N * F) * FP + (4 * N + 2 * B * H * L) * FP
    stash = NL * st_layer + N * D * FP                  # + X0 (embedding-dropout output)
    if cls_prune:
        st_last = (2 * N * D + 3 * B * D + B * F) * FP + (4 * N + 2 * B * H * L) * FP
        stash = (NL - 1) * st_layer + st_last + N * D * FP
    stash_bytes = 2 * stash

    total_dram = dram_gemm + vec_bytes + lowa_bytes + merge_bytes + emb_bytes + opt_bytes + stash_bytes
    cyc_serial = cyc_gemm + cyc_vec + cyc_lowa + cyc_opt + cyc_misc
    cyc_overlap = cyc_gemm + cyc_vec + cyc_opt + cyc_misc    # LoWA unit runs beside the array
    return dict(macs=macs, cyc_gemm=cyc_gemm, cyc_gemm_c=cyc_gemm_c, cyc_gemm_m=cyc_gemm_m, cyc_vec_c=cyc_vec_c,
                cyc_vec_m=cyc_vec_m, cyc_vec=cyc_vec, cyc_lowa=cyc_lowa,
                cyc_opt=cyc_opt + cyc_misc, cyc_serial=cyc_serial, cyc_overlap=cyc_overlap,
                dram_gemm=dram_gemm, dram_total=total_dram, stash=stash, lowa_macs=lowa_macs,
                merge_hits=merge_hits, n_train=n_train, rows=rows, util=macs / (cyc_gemm * c["R"] * c["C"]),
                vec_bytes=vec_bytes, lowa_bytes=lowa_bytes, stash_bytes=stash_bytes, opt_bytes=opt_bytes,
                emb_bytes=emb_bytes, merge_bytes=merge_bytes)


VU_RB = 256                                 # vector lane tile-buffer words (vu_lane RB)


def onchip_words(c):
    """On-chip RAM words (32 bit) as the RTL instantiates mm_sram_bank (P8c reconciliation, docs/p8c_report.md):
    gemm: X and W tile buffers (double-buffered) + the ACC (gm_drain keeps two halves: block n's epilogue reads
    one while block n+1 accumulates in the other); vec: two tile buffers per lane + 3 column vectors x 2 parities;
    lowa: dY and X row buffers (D words), atom accumulator and atom centres (256 each), merge block directory and
    coefficient buffers (512 each, two module slots).  The DMA return / write FIFOs of hw/mem are flops, not RAM."""
    g = gemm_geom(c)
    words_w = g["wbuf"] * SRAM_B // FP
    words_x = g["xbuf"] * SRAM_B // FP
    w = dict(gemm=c["dbuf"] * (words_w + words_x) + 2 * g["br"] * g["bc"],
             vec=(2 * c["V"] + 6) * VU_RB if c["V"] > 1 else 0,
             lowa=(2 * D + 256 + 256 + 2 * 512) if c["V"] > 1 else 0,
             dma=0)
    w["total"] = sum(w.values())
    return w


def area(c):
    g = gemm_geom(c)
    P = c["R"] * c["C"]
    w = onchip_words(c)
    # GF180 macro-equivalents (256 B each) of the RTL's on-chip RAM (P0 area model; the TSMC-65 port uses latch
    # register files, docs/p8b_report.md)
    m_gemm = math.ceil(w["gemm"] * FP / SRAM_B)
    m_vec = math.ceil(w["vec"] * FP / SRAM_B)          # lane tile buffers + column-vector banks
    m_lowa = math.ceil(w["lowa"] * FP / SRAM_B)
    m_dma = 0                                           # DMA FIFOs are flops
    macros = m_gemm + m_vec + m_lowa + m_dma
    pe = P * pe_area_um2() / 1e6
    sfu = 60000.0 if P0_MODEL else SFU_UM2              # P0: assumption; P2: measured fp_sfu
    vec = c["V"] * (pe_area_um2() + sfu) / 1e6          # lane FMA + SFU
    vec += 4 * c["V"] * c["V"] * 32 * FLOP_UM2 / 1e6     # in/out V x V transpose buffers, double-buffered
    lowa = 2 * pe_area_um2() / 1e6 if c["V"] > 1 else 0.0  # 4 mul + 4 FMA lanes ~ 2 PE-equiv.; shuttle shares the vector FMA
    n_rng = max(1, c["V"] // 2)                         # Threefry-2x32-13: 2 mask decisions / cycle each
    rng = n_rng * (32 * 1500 + 13 * 64 * FLOP_UM2) / 1e6  # 32 x 32-bit adders + 13 stage regs (assumption)
    ctrl = 0.8 if c["V"] > 1 else 0.15                  # sequencer, CSRs, DMA/AXI (assumption)
    logic = pe + vec + lowa + rng + ctrl
    sram = macros * MACRO_MM2
    die_core = logic / UTIL + sram * HALO
    return dict(P=P, macros=macros, m_gemm=m_gemm, sram_kb=macros * SRAM_B / 1024, pe=pe, words=w,
                logic=logic, sram=sram, core=die_core, geom=g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--fmhz", type=float, default=25.0)
    ap.add_argument("--fma-um2", type=float, default=None, help="pipelined fp32 FMA area (um2)")
    ap.add_argument("--sfu-um2", type=float, default=None, help="SFU area per vector lane (um2)")
    ap.add_argument("--p0", action="store_true", help="P0's FMA/SFU area assumptions (comparison)")
    ap.add_argument("--no-dram", action="store_true", help="compute-only GEMM / vector time (pre-P6a)")
    ap.add_argument("--dram-rprio", action="store_true", help="GEMM efficiency of a read-priority DRAM (0.666)")
    a = ap.parse_args()
    global FMA_UM2, SFU_UM2, P0_MODEL, DRAM_CHARGE, EFF_GEMM
    DRAM_CHARGE = not a.no_dram
    EFF_GEMM = EFF_GEMM_RPRIO if a.dram_rprio else EFF_GEMM
    FMA_UM2 = a.fma_um2 if a.fma_um2 is not None else FMA_UM2
    SFU_UM2 = a.sfu_um2 if a.sfu_um2 is not None else SFU_UM2
    P0_MODEL = a.p0
    B, L = a.B, (a.L + 15) // 16 * 16
    N = B * L
    f = a.fmhz * 1e6
    print(f"== workload B={B} L_pad={L} N={N}")
    n_backbone = VOCAB * D + NPOS * D + D + 2 * D + NL * (4 * D * D + 4 * D + 2 * D * F + F + D + 4 * D)
    print(f"backbone params {n_backbone:,}  = {n_backbone*FP/1e6:.1f} MB fp32")
    src = "P0 assumption" if P0_MODEL else f"measured FMA {FMA_UM2:,.0f} um2 + acc/operand flops"
    print(f"PE area ({src}, lam={LAMBDA}) = {pe_area_um2():,.0f} um2 (P0 model {pe_area_p0():,.0f});"
          f"  macro = {MACRO_MM2:.5f} mm2")
    for c in cfg_list():
        s = step(c, B, L)
        s0 = step(c, B, L, cls_prune=False)
        ar = area(c)
        g = ar["geom"]
        print(f"\n== {c['name']}: R x C = {c['R']}x{c['C']} (P={ar['P']}), V={c['V']}, lam={LAMBDA}")
        print(f"  GEMM geometry: Bc={g['bc']} Kc={g['kc']} Br={g['br']} | macros W={g['wbuf']} X={g['xbuf']} ACC={g['acc']}"
              f" x dbuf={c['dbuf']} | DRAM bytes/MAC={g['bpm']:.4f}")
        print(f"  MACs/step (CLS-pruned) {s['macs']/1e9:.1f} G  (unpruned {s0['macs']/1e9:.1f} G); array util {s['util']:.3f}")
        print(f"  cycles: gemm {s['cyc_gemm']/1e6:.1f} M, vector {s['cyc_vec']/1e6:.1f} M, lowa {s['cyc_lowa']/1e6:.1f} M, opt+misc {s['cyc_opt']/1e6:.2f} M")
        print(f"  DRAM charge (P6a, {DRAM_BPC:.0f} B/cycle = {DRAM_BPC*f/1e6:,.0f} MB/s at {a.fmhz} MHz, eff GEMM {EFF_GEMM:.3f} / vector {EFF_VEC:.3f}): gemm {s['cyc_gemm_c']/1e6:.1f} -> "
              f"{s['cyc_gemm_m']/1e6:.1f} M, vector {s['cyc_vec_c']/1e6:.1f} -> {s['cyc_vec_m']/1e6:.1f} M"
              + ("" if DRAM_CHARGE else "  (not charged: --no-dram)"))
        print(f"  step cycles: overlapped {s['cyc_overlap']/1e6:.1f} M  serial {s['cyc_serial']/1e6:.1f} M"
              f"  -> {s['cyc_overlap']/f:.1f} s / {s['cyc_serial']/f:.1f} s at {a.fmhz} MHz")
        print(f"  DRAM/step: total {s['dram_total']/1e9:.2f} GB (gemm operands {s['dram_gemm']/1e9:.2f}, stash w+r {s['stash_bytes']/1e9:.2f},"
              f" vector {s['vec_bytes']/1e9:.2f}, lowa {s['lowa_bytes']/1e9:.3f}, merge {s['merge_bytes']/1e6:.1f} MB,"
              f" emb {s['emb_bytes']/1e6:.1f} MB, opt {s['opt_bytes']/1e6:.1f} MB)")
        print(f"  avg DRAM BW {s['dram_total']/(s['cyc_overlap']/f)/1e6:.0f} MB/s; GEMM-phase peak {g['bpm']*ar['P']*f/1e6:.0f} MB/s")
        print(f"  stash {s['stash']/1e9:.3f} GB; LoWA MACs {s['lowa_macs']/1e6:.0f} M; merge hits {s['merge_hits']/1e6:.1f} M; trainable {s['n_train']:,}")
        wd = ar["words"]
        print(f"  on-chip RAM (RTL): {wd['total']:,} words = {wd['total']*FP/1024:.0f} KiB (gemm {wd['gemm']:,}, vec {wd['vec']:,},"
              f" lowa {wd['lowa']:,}; DMA FIFOs are flops)")
        print(f"  area: PE {ar['pe']:.1f} mm2, logic {ar['logic']:.1f} mm2, macros {ar['macros']} ({ar['sram_kb']:.1f} KiB) = {ar['sram']:.1f} mm2,"
              f" core est {ar['core']:.1f} mm2 (util {UTIL}, halo {HALO})")
    # host bytes / step
    host_in = N * 2 + N // 8 + B * 4 + 16
    print(f"\nhost->device per step: ids {N*2} + mask {N//8} + labels {B*4} + descriptor 16 = {host_in} B")


if __name__ == "__main__":
    main()
