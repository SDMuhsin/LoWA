`default_nettype none
// gm_ctrl: executes one GEMM descriptor over its tile sequence (docs/dataflow.md section 2):
//   for b1, b2 (batches): for nb (W block of BC columns): for mb (X block of BR rows):
//     for seg (K segments, GEMM_ACC2 = 3): for kc (K chunks of KC, the last may be short):
//       load X(mb, kc), W(nb, kc) into the next buffer   (gm_loader, one block ahead)
//       for mt (micro-tiles of ROWS rows with rows < M): one pass of KT*LAMBDA cycles
//       schedule the drain of every pass (gm_drain)    ; after the last kc: epilogue (gm_epilogue)
// Three threads: the loop generator (block descriptors into a FIFO), the loader feed (tile
// requests one block ahead, buffers alternate per block; a buffer is held against the loader
// (x_hold / w_hold) from the moment its tile is loaded until the cycle after the block's last
// pass started, the feeders' busy flags cover the rest of that pass) and the pass sequencer,
// which starts a
// pass when (docs/p3_report.md section 3): the block's tiles are loaded (and the merge engine is
// not busy on them), the W feeder is ready, the X feeder has no lane conflict (previous pass on
// the same buffer with KT mod XL != 0 must have finished), the drain engine will be free when
// this pass's drain starts (A' >= A_prev + TOT, and A' - A_prev - TOT even when < 5: the RMW
// write tail of the previous drain then meets this drain's reads on the other parity lanes), the
// previously scheduled drain's last injection precedes this pass's first capture, and, for the
// first chunk of an output block, the epilogue has released that ACC half.  Full chunks then
// stream back to back with no bubble.
// Drain start A = pass start + DD + 1, DD = LAMBDA*KT + DOFF, DOFF = ROWS + DW + L + 1 - LAMBDA
// (section 3 derives DOFF >= ROWS + DW + L - LAMBDA; one cycle of margin is kept).
module gm_ctrl #(
    parameter ROWS   = 16,
    parameter COLS   = 16,
    parameter LAMBDA = 4,
    parameter QW     = 2,                   // log2 LAMBDA
    parameter DW     = 4,
    parameter FMA_STAGES = 2,
    parameter BR     = 64,
    parameter BRW    = 6,                   // log2 BR
    parameter BC     = 64,
    parameter BCW    = 6,                   // log2 BC
    parameter KC     = 64,
    parameter KW     = 6,
    parameter XL     = 8,
    parameter RW     = 4,
    parameter MTW    = 2,
    parameter NW     = 7
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          start,
    output reg           busy,
    output reg           done,
    // descriptor
    input  wire [15:0]   d_m, d_n, d_k0, d_k1, d_k2, d_nb1, d_nb2,
    input  wire [31:0]   d_a_base0, d_a_base1, d_a_base2, d_a_rs, d_a_bs1, d_a_bs2,
    input  wire          d_a_trn,
    input  wire [31:0]   d_b_base0, d_b_base1, d_b_base2, d_b_rs, d_b_bs1, d_b_bs2,
    input  wire          d_b_trn,
    input  wire [31:0]   d_c_base, d_c_rs, d_c_bs1, d_c_bs2,
    input  wire [31:0]   d_bias_base, d_aux_base, d_o2_base,
    // loader
    output reg           tq_valid,
    input  wire          tq_ready,
    output reg           tq_kind,
    output reg           tq_buf,
    output reg           tq_trn,
    output reg  [31:0]   tq_addr,
    output reg  [31:0]   tq_rs,
    output reg  [NW-1:0] tq_n,
    output reg  [KW:0]   tq_kt,
    input  wire          tile_done,
    input  wire          tile_done_kind,
    input  wire          tile_done_buf,
    input  wire [1:0]    merge_busy,
    output wire [1:0]    x_hold,        // X / W buffer holds a loaded or running block: the loader
    output wire [1:0]    w_hold,        // must not write it (released after the block's last pass start)
    // feeders
    output reg           ps_start,
    output reg           ps_buf,
    output reg  [MTW-1:0] ps_mt,
    output reg  [KW:0]   ps_kt,
    input  wire          x_idle,
    input  wire          w_ready,
    // drain
    output reg           dr_start,
    output reg           dr_half,
    output reg  [MTW-1:0] dr_mt,
    output reg           dr_first,
    output reg           dr_last,
    input  wire          dr_free,
    // epilogue
    output reg           ob_valid,
    input  wire          ob_ready,
    output reg           ob_half,
    output reg  [NW-1:0] ob_rows, ob_cols,
    output reg  [31:0]   ob_c, ob_aux, ob_o2, ob_bias, ob_cidx,
    input  wire          ep_done,
    input  wire          ep_done_half,
    input  wire          ep_idle
);
    localparam DOFF = ROWS + DW + FMA_STAGES + 1 - LAMBDA;    // + LAMBDA*KT = DD   (margin 1)
    localparam NG   = COLS / DW;
    localparam RL   = ROWS * LAMBDA;
    localparam TOT  = NG * RL;                                // drain engine busy cycles
    localparam LI   = (NG - 1) * (RL - DW) + (LAMBDA - 1) * (ROWS - 1);   // last injection - A - 2
    localparam DDW  = 10;                                     // LAMBDA*KC + DOFF < 1024
    /* verilator lint_off WIDTHTRUNC */   // sized copies of integer parameters (16-bit wrap arithmetic)
    localparam [15:0]    BC_16 = BC, BR_16 = BR, KC_16 = KC;
    localparam [DDW-1:0] DOFF_E = DOFF;
    localparam [15:0]    TOT_16 = TOT;
    localparam [15:0]    LI2_16 = LI + 2;
    localparam [15:0]    CB_16  = FMA_STAGES + 2 - LAMBDA;    // capture offset (margin 1), may wrap
    /* verilator lint_on WIDTHTRUNC */

    // ================================================================== loop generator
    reg          g_act;                     // generating
    reg  [15:0]  b1, b2, nb, mb;
    reg  [1:0]   seg;
    reg  [15:0]  kc_rem;                    // K left in the segment
    reg  [15:0]  m_rem, n_rem;              // rows / cols left
    reg  [31:0]  oa_b1, oa_b2, oa_mb, oa_kc;
    reg  [31:0]  ob_b1, ob_b2, ob_nb, ob_kc;
    reg  [31:0]  oc_b1, oc_b2, oc_mb, oc_nb;
    reg          g_half;
    wire [31:0]  a_seg = (seg == 2'd0) ? d_a_base0 : (seg == 2'd1) ? d_a_base1 : d_a_base2;
    wire [31:0]  b_seg = (seg == 2'd0) ? d_b_base0 : (seg == 2'd1) ? d_b_base1 : d_b_base2;
    wire [1:0]   last_seg = (d_k2 != 16'd0) ? 2'd2 : (d_k1 != 16'd0) ? 2'd1 : 2'd0;
    /* verilator lint_off WIDTHEXPAND */
    /* verilator lint_off WIDTHTRUNC */   // comparisons of 16-bit counts with the block constants
    wire [KW:0]  g_kt = (kc_rem > KC) ? KC : kc_rem[KW:0];
    wire [NW-1:0] g_rows = (m_rem > BR) ? BR : m_rem[NW-1:0];
    wire [NW-1:0] g_cols = (n_rem > BC) ? BC : n_rem[NW-1:0];
    wire         g_last_kc = (kc_rem <= KC);
    wire         g_last_mb = (m_rem <= BR);
    wire         g_last_nb = (n_rem <= BC);
    /* verilator lint_on WIDTHTRUNC */
    /* verilator lint_on WIDTHEXPAND */
    wire         g_last_seg = (seg == last_seg);
    wire         g_first = (seg == 2'd0) & (oa_kc == 32'd0);
    wire         g_last  = g_last_kc & g_last_seg;
    wire         g_last_b2 = (b2 == d_nb2 - 1'b1);
    wire         g_last_b1 = (b1 == d_nb1 - 1'b1);
    // steps
    wire [31:0]  a_mb_step = d_a_trn ? (BR * 4) : (d_a_rs << BRW);
    wire [31:0]  a_kc_step = d_a_trn ? (d_a_rs << KW) : (KC * 4);
    wire [31:0]  b_nb_step = d_b_trn ? (BC * 4) : (d_b_rs << BCW);
    wire [31:0]  b_kc_step = d_b_trn ? (d_b_rs << KW) : (KC * 4);
    wire [31:0]  c_mb_step = d_c_rs << BRW;
    wire [31:0]  c_nb_step = BC * 4;
    wire [31:0]  c_off = oc_b1 + oc_b2 + oc_mb + oc_nb;
    // block FIFO (depth 4, two readers)
    localparam BW = 32 + 32 + 1 + NW + (KW+1)          // x: addr rs trn n kt
                  + 32 + 32 + 1 + NW                  // w: addr rs trn n
                  + 1 + 1 + 1 + (MTW+1)               // first last half mt_cnt
                  + 32 + 32;                          // c_off, c_nb offset (bias)
    reg  [BW-1:0] bf [0:3];
    reg  [2:0]    bf_wp, bf_lp, bf_cp;      // write, loader-read, compute-read pointers
    wire          bf_full = (bf_wp - bf_cp) == 3'd4;
    wire          g_emit = g_act & ~bf_full;
    /* verilator lint_off WIDTHTRUNC */
    /* verilator lint_off WIDTHEXPAND */
    wire [MTW:0]  g_mtc = (g_rows + ROWS - 1) >> RW;
    /* verilator lint_on WIDTHEXPAND */
    /* verilator lint_on WIDTHTRUNC */
    wire [31:0]   x_addr = a_seg + oa_b1 + oa_b2 + oa_mb + oa_kc;
    wire [31:0]   w_addr = b_seg + ob_b1 + ob_b2 + ob_nb + ob_kc;
    always @(posedge clk) begin
        if (!rst_n) begin
            g_act <= 1'b0; bf_wp <= 3'd0;
        end else begin
            if (start & ~busy) begin
                g_act <= 1'b1; b1 <= 16'd0; b2 <= 16'd0; nb <= 16'd0; mb <= 16'd0; seg <= 2'd0;
                kc_rem <= d_k0; m_rem <= d_m; n_rem <= d_n; g_half <= 1'b0;
                oa_b1 <= 32'd0; oa_b2 <= 32'd0; oa_mb <= 32'd0; oa_kc <= 32'd0;
                ob_b1 <= 32'd0; ob_b2 <= 32'd0; ob_nb <= 32'd0; ob_kc <= 32'd0;
                oc_b1 <= 32'd0; oc_b2 <= 32'd0; oc_mb <= 32'd0; oc_nb <= 32'd0;
            end else if (g_emit) begin
                bf[bf_wp[1:0]] <= {x_addr, d_a_rs, d_a_trn, g_rows, g_kt,
                                   w_addr, d_b_rs, d_b_trn, g_cols,
                                   g_first, g_last, g_half, g_mtc, c_off, oc_nb};
                bf_wp <= bf_wp + 1'b1;
                // advance: kc -> seg -> mb -> nb -> b2 -> b1
                if (!g_last_kc) begin
                    kc_rem <= kc_rem - KC_16; oa_kc <= oa_kc + a_kc_step; ob_kc <= ob_kc + b_kc_step;
                end else begin
                    oa_kc <= 32'd0; ob_kc <= 32'd0;
                    if (!g_last_seg) begin
                        seg <= seg + 1'b1;
                        kc_rem <= (seg == 2'd0) ? d_k1 : d_k2;
                    end else begin
                        seg <= 2'd0; kc_rem <= d_k0; g_half <= ~g_half;
                        if (!g_last_mb) begin
                            mb <= mb + 1'b1; m_rem <= m_rem - BR_16;
                            oa_mb <= oa_mb + a_mb_step; oc_mb <= oc_mb + c_mb_step;
                        end else begin
                            mb <= 16'd0; m_rem <= d_m; oa_mb <= 32'd0; oc_mb <= 32'd0;
                            if (!g_last_nb) begin
                                nb <= nb + 1'b1; n_rem <= n_rem - BC_16;
                                ob_nb <= ob_nb + b_nb_step; oc_nb <= oc_nb + c_nb_step;
                            end else begin
                                nb <= 16'd0; n_rem <= d_n; ob_nb <= 32'd0; oc_nb <= 32'd0;
                                if (!g_last_b2) begin
                                    b2 <= b2 + 1'b1;
                                    oa_b2 <= oa_b2 + d_a_bs2; ob_b2 <= ob_b2 + d_b_bs2; oc_b2 <= oc_b2 + d_c_bs2;
                                end else begin
                                    b2 <= 16'd0; oa_b2 <= 32'd0; ob_b2 <= 32'd0; oc_b2 <= 32'd0;
                                    if (!g_last_b1) begin
                                        b1 <= b1 + 1'b1;
                                        oa_b1 <= oa_b1 + d_a_bs1; ob_b1 <= ob_b1 + d_b_bs1; oc_b1 <= oc_b1 + d_c_bs1;
                                    end else begin
                                        g_act <= 1'b0;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_g = (|b1) | (|b2) | (|nb) | (|mb);
    /* verilator lint_on UNUSEDSIGNAL */

    // ================================================================== loader feed
    // block i uses buffer i & 1; requests W then X
    wire          lf_avail = (bf_lp != bf_wp);
    wire [BW-1:0] lf_b = bf[bf_lp[1:0]];
    reg           lf_phase;                 // 0: W request, 1: X request
    reg           lf_buf;
    wire [31:0]   lf_xa  = lf_b[BW-1 -: 32];
    wire [31:0]   lf_xrs = lf_b[BW-33 -: 32];
    wire          lf_xtrn = lf_b[BW-65];
    wire [NW-1:0] lf_xn  = lf_b[BW-66 -: NW];
    wire [KW:0]   lf_kt  = lf_b[BW-66-NW -: KW+1];
    wire [31:0]   lf_wa  = lf_b[BW-67-NW-KW -: 32];
    wire [31:0]   lf_wrs = lf_b[BW-99-NW-KW -: 32];
    wire          lf_wtrn = lf_b[BW-131-NW-KW];
    wire [NW-1:0] lf_wn  = lf_b[BW-132-NW-KW -: NW];
    always @(posedge clk) begin
        if (!rst_n) begin
            tq_valid <= 1'b0; bf_lp <= 3'd0; lf_phase <= 1'b0; lf_buf <= 1'b0;
        end else begin
            if (tq_valid & tq_ready) begin
                tq_valid <= 1'b0;
                if (lf_phase) begin bf_lp <= bf_lp + 1'b1; lf_buf <= ~lf_buf; end
                lf_phase <= ~lf_phase;
            end else if (~tq_valid & lf_avail) begin
                tq_valid <= 1'b1; tq_buf <= lf_buf; tq_kt <= lf_kt;
                tq_kind <= ~lf_phase;
                tq_trn  <= lf_phase ? lf_xtrn : lf_wtrn;
                tq_addr <= lf_phase ? lf_xa : lf_wa;
                tq_rs   <= lf_phase ? lf_xrs : lf_wrs;
                tq_n    <= lf_phase ? lf_xn : lf_wn;
            end
            if (start & ~busy) begin lf_buf <= 1'b0; lf_phase <= 1'b0; end
        end
    end

    // ================================================================== pass sequencer
    reg  [1:0]    x_ld, w_ld;               // tiles loaded per buffer (pending computation)
    reg  [1:0]    buf_inuse;               // block being computed on the buffer
    reg           rel_pend, rel_buf;        // release the buffer the cycle after the last pass start
    assign x_hold = x_ld | buf_inuse;
    assign w_hold = w_ld | buf_inuse;
    reg  [1:0]    half_busy;
    wire          ct_avail = (bf_cp != bf_wp);
    wire [BW-1:0] ct_b = bf[bf_cp[1:0]];
    wire [KW:0]   ct_kt   = ct_b[BW-66-NW -: KW+1];
    wire [NW-1:0] ct_rows = ct_b[BW-66 -: NW];
    wire [NW-1:0] ct_cols = ct_b[BW-132-NW-KW -: NW];
    wire          ct_first = ct_b[MTW+1+64+2];
    wire          ct_last  = ct_b[MTW+1+64+1];
    wire          ct_half  = ct_b[MTW+1+64];
    wire [MTW:0]  ct_mtc   = ct_b[64 +: MTW+1];
    wire [31:0]   ct_coff  = ct_b[63:32];
    wire [31:0]   ct_cnb   = ct_b[31:0];
    reg           ct_buf;
    reg  [MTW:0]  ct_mt;
    reg           ct_inblk;                 // passes of the block in progress (mt > 0)
    wire          tiles_ok = x_ld[ct_buf] & w_ld[ct_buf] & ~merge_busy[ct_buf];
    /* verilator lint_off WIDTHEXPAND */
    /* verilator lint_off WIDTHTRUNC */
    wire          x_ok = x_idle | (ct_inblk & ((ct_kt & (XL - 1)) == 0));   // KT mod XL == 0
    wire [DDW-1:0] ktl  = ct_kt << QW;                                // LAMBDA * KT
    /* verilator lint_on WIDTHTRUNC */
    wire [DDW-1:0] dd   = ktl + DOFF_E;
    // drain scheduling against the previously scheduled drain (start time last_a, 16-bit times):
    //   (a) this drain A' = now + dd + 2 >= last_a + TOT            (engine free)
    //   (b) first capture of this pass (now + 4 + LAMBDA*KT - LAMBDA + 1 + L, margin 1)
    //       >= last injection of the previous drain (last_a + 2 + LI)
    reg  [15:0]   last_a;
    reg           have_a;
    reg  [15:0]   now;                      // free-running cycle counter (16-bit wrap arithmetic)
    wire [15:0]   target = now + {{(16-DDW){1'b0}}, dd} + 16'd1;   // drain fires at target (+1: ps_start is registered)
    wire [15:0]   lhs_a = now + {{(16-DDW){1'b0}}, dd} + 16'd2;
    wire [15:0]   rhs_a = last_a + TOT_16;
    wire [15:0]   lhs_b = now + {{(16-DDW){1'b0}}, ktl} + 16'd4 + CB_16;
    wire [15:0]   rhs_b = last_a + LI2_16;
    wire [15:0]   dif_a = lhs_a - rhs_a;
    /* verilator lint_off UNUSEDSIGNAL */   // only the sign bit of dif_b is used
    wire [15:0]   dif_b = lhs_b - rhs_b;
    /* verilator lint_on UNUSEDSIGNAL */
    // (c) the previous drain's RMW write tail (5 cycles) may overlap this drain's first reads only
    //     with an even offset, so that reads and writes stay on different parity lanes
    wire          tail_ok = (dif_a[15:3] != 13'd0) | (dif_a[2:0] > 3'd4) | ~dif_a[0];
    wire          dr_ok = ~have_a | (~dif_a[15] & ~dif_b[15] & tail_ok);
    // (d) pass starts are LAMBDA-aligned to the previous pass start: the X feeder's lane sets of
    //     two overlapping passes are disjoint only for that alignment (gm_xfeed), and the drain
    //     gap of equal-KT passes is then even by construction
    reg  [15:0]   last_t0;
    /* verilator lint_off UNUSEDSIGNAL */   // only the low QW bits of the spacing are used
    wire [15:0]   spacing = now + 16'd1 - last_t0;
    /* verilator lint_on UNUSEDSIGNAL */
    wire          align_ok = ~have_a | (spacing[QW-1:0] == {QW{1'b0}});
    wire          last_pass = (ct_mt == ct_mtc - 1'b1);
    /* verilator lint_on WIDTHEXPAND */
    wire          ob_push = ct_last & last_pass;
    wire          half_ok = ~(ct_first & ~ct_inblk) | ~half_busy[ct_half];
    wire          ob_ok = ~ob_push | ~ob_valid | ob_ready;
    wire          sched_ok;                 // drain schedule FIFO has room
    wire          can_start = ct_avail & (ct_inblk | tiles_ok) & w_ready & x_ok & dr_ok & align_ok & half_ok & ob_ok & sched_ok;
    always @(posedge clk) begin
        if (!rst_n) begin
            ps_start <= 1'b0; ct_mt <= {(MTW+1){1'b0}}; ct_inblk <= 1'b0; ct_buf <= 1'b0; bf_cp <= 3'd0;
            x_ld <= 2'b00; w_ld <= 2'b00; half_busy <= 2'b00; ob_valid <= 1'b0; now <= 16'd0;
            have_a <= 1'b0; buf_inuse <= 2'b00; rel_pend <= 1'b0;
        end else begin
            now <= now + 1'b1;
            ps_start <= 1'b0;
            rel_pend <= 1'b0;
            if (rel_pend) buf_inuse[rel_buf] <= 1'b0;
            if (tile_done) begin
                if (tile_done_kind) w_ld[tile_done_buf] <= 1'b1;
                else                x_ld[tile_done_buf] <= 1'b1;
            end
            if (ep_done) half_busy[ep_done_half] <= 1'b0;
            if (ob_valid & ob_ready) ob_valid <= 1'b0;
            if (can_start) begin
                ps_start <= 1'b1; ps_buf <= ct_buf; ps_mt <= ct_mt[MTW-1:0]; ps_kt <= ct_kt;
                last_a <= target + 16'd1; have_a <= 1'b1; last_t0 <= now + 16'd1;
                if (ct_first & ~ct_inblk) half_busy[ct_half] <= 1'b1;
                if (~ct_inblk) buf_inuse[ct_buf] <= 1'b1;
                if (last_pass) begin rel_pend <= 1'b1; rel_buf <= ct_buf; end
                if (ob_push) begin
                    ob_valid <= 1'b1; ob_half <= ct_half; ob_rows <= ct_rows; ob_cols <= ct_cols;
                    ob_c    <= d_c_base + ct_coff;
                    ob_aux  <= d_aux_base + ct_coff;
                    ob_o2   <= d_o2_base + ct_coff;
                    ob_bias <= d_bias_base + ct_cnb;
                    ob_cidx <= ct_coff >> 2;
                end
                if (last_pass) begin
                    ct_mt <= {(MTW+1){1'b0}}; ct_inblk <= 1'b0; bf_cp <= bf_cp + 1'b1;
                    ct_buf <= ~ct_buf;
                    x_ld[ct_buf] <= 1'b0; w_ld[ct_buf] <= 1'b0;
                end else begin
                    ct_mt <= ct_mt + 1'b1; ct_inblk <= 1'b1;
                end
            end
            if (start & ~busy) begin ct_buf <= 1'b0; x_ld <= 2'b00; w_ld <= 2'b00; have_a <= 1'b0; end
        end
    end

    // ================================================================== drain schedule (depth 4)
    // entries {target time, half, mt, first, last}; the head fires when now == target
    localparam SW = 16 + 1 + MTW + 2;
    reg  [SW-1:0] sf [0:3];
    reg  [2:0]    sf_wp, sf_rp;
    wire          sf_empty = (sf_wp == sf_rp);
    assign sched_ok = (sf_wp - sf_rp) != 3'd4;
    wire [SW-1:0] sh = sf[sf_rp[1:0]];
    wire          fire = ~sf_empty & (sh[SW-1 -: 16] == now);
    always @(posedge clk) begin
        if (!rst_n) begin
            sf_wp <= 3'd0; sf_rp <= 3'd0; dr_start <= 1'b0;
        end else begin
            if (can_start) begin
                sf[sf_wp[1:0]] <= {target, ct_half, ct_mt[MTW-1:0], ct_first, ct_last};
                sf_wp <= sf_wp + 1'b1;
            end
            dr_start <= fire;
            if (fire) begin
                sf_rp <= sf_rp + 1'b1;
                dr_half <= sh[MTW+2]; dr_mt <= sh[2 +: MTW]; dr_first <= sh[1]; dr_last <= sh[0];
            end
        end
    end

    // ================================================================== busy / done
    wire all_idle = ~g_act & (bf_cp == bf_wp) & ~ct_inblk & ~tq_valid & (bf_lp == bf_wp)
                    & sf_empty & ~dr_start & dr_free & ~ob_valid & ep_idle;
    reg  [3:0] settle;
    always @(posedge clk) begin
        if (!rst_n) begin
            busy <= 1'b0; done <= 1'b0; settle <= 4'd0;
        end else begin
            done <= 1'b0;
            if (start & ~busy) begin
                busy <= 1'b1; settle <= 4'd8;
            end else if (busy) begin
                if (settle != 4'd0) settle <= settle - 1'b1;
                else if (all_idle) begin busy <= 1'b0; done <= 1'b1; end
            end
        end
    end
    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_c = (|ct_b) | (|lf_b) | (|sh);
    /* verilator lint_on UNUSEDSIGNAL */
endmodule
`default_nettype wire
