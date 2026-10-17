`default_nettype none
// gm_epilogue: reads the finished output block from the ACC half the array is not using, applies
// the epilogue chain of docs/dataflow.md section 2 in the isa.md order and writes C:
//   y  = add(acc, bias[col])                      (EP_BIAS)
//   y  = mul(y, 2^sexp)                           (EP_SCALE, gm_pow2: exact)
//   y  = gelu(y), u = y stored to out2            (EP_GELU, gm_gelu)       \ exclusive
//   y  = mul(y, gelu_grad(u)), u from aux         (EP_GELU_BWD, gm_gelu)   /
//   y  = keep ? mul(y, DS) : +0                   (EP_DROP, keep bits from the mask channel)
//   y  = add(y, aux)                              (EP_RESID, aux = residual from rd1)
// DW elements per cycle (one ACC word: columns LAMBDA*(DW*g + j) + q, j = 0..DW-1) through
// fixed-latency lanes; the GELU chains go through gm_gelu one element per cycle (NSFU SFUs).
// Groups of GWD = DW*LAMBDA consecutive columns of a row are assembled and written as
// GWD/8 beats (word mask for columns >= n_cols); rows >= n_rows are never read.
// One block descriptor (valid/ready) per output block; a micro-tile is read only after the
// drain engine reported its final chunk sum (fin_*); ep_done(half) when the block is written.
module gm_epilogue #(
    parameter ROWS   = 16,
    parameter LAMBDA = 4,
    parameter QW     = 2,
    parameter DW     = 4,
    parameter GW     = 2,                   // log2 (COLS / DW) column groups
    parameter RW     = 4,                   // log2 ROWS
    parameter MTW    = 2,                   // log2 (BR / ROWS)
    parameter AW     = 9,                   // ACC address width
    parameter BC     = 64,                  // columns per block
    parameter NW     = 7,                   // width of row / column counts
    parameter NSFU   = 16,
    parameter NROB   = 64,
    parameter ROBW   = 6,
    parameter DWW    = 2,                   // log2 DW (1 when DW = 1)
    parameter GWW    = 4,                   // log2 (DW*LAMBDA)
    parameter NASM   = 4                    // assembler group entries
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              rst_n_g,       // reset copy for the GELU pool (gm_top's reset tree)
    // static epilogue configuration
    input  wire              ep_bias,
    input  wire              ep_scale,
    input  wire [7:0]        ep_sexp,
    input  wire              ep_gelu,
    input  wire              ep_gelu_bwd,
    input  wire              ep_drop,
    input  wire              ep_resid,
    input  wire [31:0]       ep_ds,
    // block descriptors
    input  wire              ob_valid,
    output wire              ob_ready,
    input  wire              ob_half,
    input  wire [NW-1:0]     ob_rows,       // 1 .. BR
    input  wire [NW-1:0]     ob_cols,       // 1 .. BC
    input  wire [31:0]       ob_c,          // C tile byte address
    input  wire [31:0]       ob_aux,        // aux tile byte address (residual / u)
    input  wire [31:0]       ob_o2,         // out2 tile byte address (u store)
    input  wire [31:0]       ob_bias,       // bias byte address of this column block
    input  wire [31:0]       ob_cidx,       // element index of C tile (0, 0) (mask index base)
    input  wire [31:0]       ob_rs,         // C row stride, bytes (also aux / out2)
    // drain -> tile final
    input  wire              fin_valid,
    input  wire              fin_half,
    output reg               ep_done,
    output reg               ep_done_half,
    output wire              idle,
    // ACC read port
    input  wire [1:0]        acc_block,     // a drain is active on that half: no reads
    output reg               acc_en,
    output reg               acc_half,
    output reg               acc_par,
    output reg  [AW-1:0]     acc_addr,
    input  wire [DW*32-1:0]  acc_q,
    // memory read channel 1 (bias, aux)
    output reg               rd_req_valid,
    input  wire              rd_req_ready,
    output reg  [31:0]       rd_req_addr,
    output reg  [6:0]        rd_req_len,
    input  wire              rd_dat_valid,
    output wire              rd_dat_ready,
    input  wire [255:0]      rd_dat,
    // mask channel (keep bits for GWD consecutive elements)
    output reg               mk_req_valid,
    input  wire              mk_req_ready,
    output reg  [31:0]       mk_req_idx,
    input  wire              mk_rsp_valid,
    input  wire [DW*LAMBDA-1:0] mk_rsp_keep,
    // write channel
    output reg               wr_valid,
    input  wire              wr_ready,
    output reg  [31:0]       wr_addr,
    output reg  [255:0]      wr_data,
    output reg  [7:0]        wr_mask,
    output wire              err
);
    localparam GWD  = DW * LAMBDA;          // words per group
    localparam NBG  = (GWD + 7) / 8;        // beats per group per plane
    localparam NBB  = (BC + 7) / 8;         // bias beats
    /* verilator lint_off WIDTHTRUNC */   // sized copies of integer parameters (<= 64)
    localparam [6:0] BC_LEN = BC;           // bias burst length (words)
    localparam [6:0] GWD_LEN = GWD;         // aux burst length (words)
    localparam [3:0] NBG_4 = NBG;
    /* verilator lint_on WIDTHTRUNC */
    localparam use_aux = 1'b0;              // (placeholder for readability)
    wire need_aux  = ep_resid | ep_gelu_bwd;
    wire gelu_mode = ep_gelu | ep_gelu_bwd;

    // ------------------------------------------------------------------ block state
    reg            run;                     // a block is in progress
    reg            half;
    reg  [NW-1:0]  n_rows, n_cols;
    reg  [31:0]    c_tile, aux_tile, o2_tile, bias_tile, cidx, c_rs;
    reg  [MTW:0]   fin_cnt [0:1];
    assign ob_ready = ~run;
    wire ob_fire = ob_valid & ob_ready;
    /* verilator lint_off WIDTHTRUNC */
    /* verilator lint_off WIDTHEXPAND */
    wire [NW-1:0] g_cnt_w = (n_cols + GWD - 1) >> GWW;  // groups per row (ceil)
    wire [MTW:0]  mt_cnt = (n_rows + ROWS - 1) >> RW;   // micro-tiles with rows
    wire [NW-1:0] g_last_w = g_cnt_w - 1'b1;
    wire [MTW:0]  mt_last_w = mt_cnt - 1'b1;
    /* verilator lint_on WIDTHEXPAND */
    /* verilator lint_on WIDTHTRUNC */

    // ------------------------------------------------------------------ bias registers
    reg  [BC*32-1:0] bias_r;                // word c at bias_r[c*32 +: 32]
    reg            bias_pend;               // bias beats still expected
    reg  [3:0]     bias_b;

    // ------------------------------------------------------------------ group sequencer (prefetch)
    // walks (mt, r, g): issues the aux read and the mask request of every group ahead of the
    // reader, bounded by the FIFO credits
    localparam AFD = 4, MFD = 8;
    reg  [MTW-1:0] s_mt;  reg [RW-1:0] s_r;  reg [GW-1:0] s_g;
    reg            s_act;
    reg  [31:0]    s_aux_row, s_cidx_row;   // row bases
    reg  [2:0]     a_cred;                  // aux groups issued - consumed (<= AFD)
    reg  [3:0]     m_cred;                  // mask groups issued - consumed (<= MFD)
    wire [RW-1:0]  s_rows_in_mt_w = rows_in_tile(s_mt);
    function [RW-1:0] rows_in_tile;         // last row index of micro-tile mt (rows - 1 in tile)
        input [MTW-1:0] m;
        reg [NW-1:0] rem;
        begin
            rem = n_rows - {m, {RW{1'b0}}};
            /* verilator lint_off WIDTHTRUNC */
            /* verilator lint_off WIDTHEXPAND */
            rows_in_tile = (rem > ROWS) ? ROWS - 1 : rem - 1;
            /* verilator lint_on WIDTHEXPAND */
            /* verilator lint_on WIDTHTRUNC */
        end
    endfunction
    wire s_last_g = ({{(NW-GW){1'b0}}, s_g} == g_last_w);
    wire s_last_r = (s_r == s_rows_in_mt_w);
    wire s_last_mt = ({1'b0, s_mt} == mt_last_w);
    wire s_need_a = need_aux;
    wire s_need_m = ep_drop;
    wire s_a_ok = ~s_need_a | ((a_cred != AFD) & ~rd_req_valid) ;
    wire s_m_ok = ~s_need_m | ((m_cred != MFD) & ~mk_req_valid);
    wire s_go = s_act & ~bias_pend & s_a_ok & s_m_ok;
    wire s_last_grp = s_last_g & s_last_r & s_last_mt;
    wire [31:0] s_goff = {{(30-GW-GWW){1'b0}}, s_g, {(GWW+2){1'b0}}};   // GWD*g*4 bytes
    wire [31:0] s_gidx = {{(32-GW-GWW){1'b0}}, s_g, {GWW{1'b0}}};       // GWD*g elements
    always @(posedge clk) begin
        if (!rst_n) begin
            s_act <= 1'b0; rd_req_valid <= 1'b0; mk_req_valid <= 1'b0;
        end else begin
            if (rd_req_valid & rd_req_ready) rd_req_valid <= 1'b0;
            if (mk_req_valid & mk_req_ready) mk_req_valid <= 1'b0;
            if (ob_fire) begin
                s_act <= 1'b1; s_mt <= {MTW{1'b0}}; s_r <= {RW{1'b0}}; s_g <= {GW{1'b0}};
                s_aux_row <= ob_aux; s_cidx_row <= ob_cidx;
                if (ep_bias) begin
                    rd_req_valid <= 1'b1; rd_req_addr <= ob_bias; rd_req_len <= BC_LEN;
                end
            end else if (s_go) begin
                if (s_need_a) begin
                    rd_req_valid <= 1'b1; rd_req_addr <= s_aux_row + s_goff; rd_req_len <= GWD_LEN;
                end
                if (s_need_m) begin
                    mk_req_valid <= 1'b1; mk_req_idx <= s_cidx_row + s_gidx;
                end
                if (s_last_g) begin
                    s_g <= {GW{1'b0}};
                    if (s_last_r) begin
                        s_r <= {RW{1'b0}};
                        if (s_last_mt) s_act <= 1'b0;
                        else s_mt <= s_mt + 1'b1;
                    end else begin
                        s_r <= s_r + 1'b1;
                    end
                    s_aux_row <= s_aux_row + c_rs;
                    s_cidx_row <= s_cidx_row + (c_rs >> 2);
                end else begin
                    s_g <= s_g + 1'b1;
                end
            end
        end
    end
    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_lg = s_last_grp | use_aux;
    /* verilator lint_on UNUSEDSIGNAL */

    // ------------------------------------------------------------------ rd1 response consumer
    // in order: bias beats (if ep_bias), then aux groups of NBG beats each -> aux FIFO
    // FIFOs are flat vectors read through one-hot pointers (cm_ohmux, docs/p3_report.md section 6)
    reg  [AFD*GWD*32-1:0] af;               // entry e at af[e*GWD*32 +: GWD*32]
    reg  [2:0]        af_wp, af_rp;
    localparam AREP = 8;
    (* keep *) reg [AREP*AFD-1:0] af_ohr;   // one-hot copies of af_rp[1:0]
    wire              af_empty = (af_wp == af_rp);
    reg  [3:0]        a_b;                  // beat within the group
    assign rd_dat_ready = 1'b1;             // bias regs / aux FIFO always have room (credits)
    always @(posedge clk) begin
        if (!rst_n) begin
            bias_pend <= 1'b0; bias_b <= 4'd0; af_wp <= 3'd0; a_b <= 4'd0;
        end else begin
            if (ob_fire & ep_bias) begin bias_pend <= 1'b1; bias_b <= 4'd0; end
            if (rd_dat_valid) begin
                if (bias_pend) begin
                    bias_b <= bias_b + 1'b1;
                    /* verilator lint_off WIDTHEXPAND */
                    if (bias_b == NBB - 1) bias_pend <= 1'b0;
                    /* verilator lint_on WIDTHEXPAND */
                end else begin
                    /* verilator lint_off WIDTHEXPAND */
                    if (a_b == NBG - 1) begin
                    /* verilator lint_on WIDTHEXPAND */
                        a_b <= 4'd0; af_wp <= af_wp + 1'b1;
                    end else begin
                        a_b <= a_b + 1'b1;
                    end
                end
            end
        end
    end
    // data writes of the bias registers / aux FIFO: one enable per word (no shared decode net)
    genvar bw, ae_, ab_, aw_;
    generate
        for (bw = 0; bw < BC; bw = bw + 1) begin : g_bias_wr
            /* verilator lint_off WIDTHEXPAND */
            wire en = rd_dat_valid & bias_pend & (bias_b == bw / 8);
            /* verilator lint_on WIDTHEXPAND */
            always @(posedge clk) if (en) bias_r[bw*32 +: 32] <= rd_dat[(bw % 8)*32 +: 32];
        end
        for (ae_ = 0; ae_ < AFD; ae_ = ae_ + 1) begin : g_af_e
            for (ab_ = 0; ab_ < NBG; ab_ = ab_ + 1) begin : g_af_b
                /* verilator lint_off WIDTHEXPAND */
                wire en = rd_dat_valid & ~bias_pend & (af_wp[1:0] == ae_) & (a_b == ab_);
                /* verilator lint_on WIDTHEXPAND */
                for (aw_ = 0; aw_ < 8; aw_ = aw_ + 1) begin : g_af_w
                    if (ab_*8 + aw_ < GWD) begin : g_in
                        always @(posedge clk) if (en) af[(ae_*GWD + ab_*8 + aw_)*32 +: 32] <= rd_dat[aw_*32 +: 32];
                    end
                end
            end
        end
    endgenerate
    // mask FIFO
    reg  [MFD*GWD-1:0] mf;
    reg  [3:0]     mf_wp, mf_rp;
    reg  [MFD-1:0] mf_oh;                   // one-hot mf_rp[2:0]
    wire           mf_empty = (mf_wp == mf_rp);
    always @(posedge clk) begin
        if (!rst_n) mf_wp <= 4'd0;
        else if (mk_rsp_valid) begin mf[mf_wp[2:0]*GWD +: GWD] <= mk_rsp_keep; mf_wp <= mf_wp + 1'b1; end
    end

    // ------------------------------------------------------------------ main reader
    reg  [MTW-1:0] m_mt;  reg [RW-1:0] m_r;  reg [GW-1:0] m_g;  reg [QW-1:0] m_q;
    reg            m_act, m_ingrp;
    reg  [31:0]    m_c_row, m_o2_row;
    wire [RW-1:0]  m_rows_in_mt_w = rows_in_tile(m_mt);
    wire m_last_g = ({{(NW-GW){1'b0}}, m_g} == g_last_w);
    wire m_last_r = (m_r == m_rows_in_mt_w);
    wire m_last_mt = ({1'b0, m_mt} == mt_last_w);
    wire [31:0] m_goff = {{(30-GW-GWW){1'b0}}, m_g, {(GWW+2){1'b0}}};
    /* verilator lint_off WIDTHEXPAND */
    wire m_last_q = (m_q == LAMBDA - 1);
    wire m_tile_ok = (fin_cnt[half] > m_mt);
    /* verilator lint_on WIDTHEXPAND */
    reg  [2:0]     asm_cred;                // free assembler entries not yet claimed
    wire gelu_room;
    wire m_grp_ok = m_act & ~m_ingrp & m_tile_ok & (~need_aux | ~af_empty) & (~ep_drop | ~mf_empty)
                    & (asm_cred != 3'd0) & (~gelu_mode | gelu_room) & ~bias_pend & ~acc_block[half];
    wire m_issue = (m_grp_ok | m_ingrp) & ~acc_block[half];
    // one-hot copies of m_q and m_g (bias / aux lane selects), kept in step with the counters
    localparam NGO  = 1 << GW;              // one-hot width of m_g
    localparam MREP = 2;
    (* keep *) reg [MREP*LAMBDA-1:0] mq_ohr;
    (* keep *) reg [MREP*NGO-1:0]    mg_ohr;
    wire [GWD*32-1:0] aux_head;
    wire [GWD-1:0]    mk_head;
    cm_ohmux #(.W(GWD*32), .N(AFD), .REP(AREP)) u_afh (.sel(af_ohr), .d(af), .y(aux_head));
    cm_ohmux #(.W(GWD), .N(MFD), .REP(1)) u_mfh (.sel(mf_oh), .d(mf), .y(mk_head));
    wire asm_free;                          // an assembler entry was released this cycle
    integer oi;
    always @(posedge clk) begin
        if (!rst_n) begin
            m_act <= 1'b0; m_ingrp <= 1'b0; af_rp <= 3'd0; mf_rp <= 4'd0; asm_cred <= NASM;
            a_cred <= 3'd0; m_cred <= 4'd0; m_q <= {QW{1'b0}};
            for (oi = 0; oi < AREP; oi = oi + 1) af_ohr[oi*AFD +: AFD] <= {{(AFD-1){1'b0}}, 1'b1};
            mf_oh <= {{(MFD-1){1'b0}}, 1'b1};
            for (oi = 0; oi < MREP; oi = oi + 1) begin
                mq_ohr[oi*LAMBDA +: LAMBDA] <= {{(LAMBDA-1){1'b0}}, 1'b1};
                mg_ohr[oi*NGO +: NGO] <= {{(NGO-1){1'b0}}, 1'b1};
            end
        end else begin
            // credits
            a_cred <= a_cred + {2'd0, (s_go & s_need_a)} - {2'd0, (m_issue & m_last_q & need_aux)};
            m_cred <= m_cred + {3'd0, (s_go & s_need_m)} - {3'd0, (m_issue & m_last_q & ep_drop)};
            asm_cred <= asm_cred - {2'd0, m_grp_ok} + {2'd0, asm_free};
            if (ob_fire) begin
                m_act <= 1'b1; m_mt <= {MTW{1'b0}}; m_r <= {RW{1'b0}}; m_g <= {GW{1'b0}}; m_q <= {QW{1'b0}};
                m_c_row <= ob_c; m_o2_row <= ob_o2;
                for (oi = 0; oi < MREP; oi = oi + 1) begin
                    mq_ohr[oi*LAMBDA +: LAMBDA] <= {{(LAMBDA-1){1'b0}}, 1'b1};
                    mg_ohr[oi*NGO +: NGO] <= {{(NGO-1){1'b0}}, 1'b1};
                end
            end else if (m_issue) begin
                if (m_last_q) begin
                    m_ingrp <= 1'b0;
                    m_q <= {QW{1'b0}};
                    for (oi = 0; oi < MREP; oi = oi + 1) mq_ohr[oi*LAMBDA +: LAMBDA] <= {{(LAMBDA-1){1'b0}}, 1'b1};
                    if (need_aux) begin
                        af_rp <= af_rp + 1'b1;
                        for (oi = 0; oi < AREP; oi = oi + 1)
                            af_ohr[oi*AFD +: AFD] <= {af_ohr[oi*AFD +: AFD-1], af_ohr[oi*AFD + AFD-1]};
                    end
                    if (ep_drop) begin
                        mf_rp <= mf_rp + 1'b1;
                        mf_oh <= {mf_oh[MFD-2:0], mf_oh[MFD-1]};
                    end
                    if (m_last_g) begin
                        m_g <= {GW{1'b0}};
                        for (oi = 0; oi < MREP; oi = oi + 1) mg_ohr[oi*NGO +: NGO] <= {{(NGO-1){1'b0}}, 1'b1};
                        m_c_row <= m_c_row + c_rs; m_o2_row <= m_o2_row + c_rs;
                        if (m_last_r) begin
                            m_r <= {RW{1'b0}};
                            if (m_last_mt) m_act <= 1'b0;
                            else m_mt <= m_mt + 1'b1;
                        end else begin
                            m_r <= m_r + 1'b1;
                        end
                    end else begin
                        m_g <= m_g + 1'b1;
                        for (oi = 0; oi < MREP; oi = oi + 1)
                            mg_ohr[oi*NGO +: NGO] <= {mg_ohr[oi*NGO +: NGO-1], mg_ohr[oi*NGO + NGO-1]};
                    end
                end else begin
                    m_ingrp <= 1'b1;
                    m_q <= m_q + 1'b1;
                    for (oi = 0; oi < MREP; oi = oi + 1)
                        mq_ohr[oi*LAMBDA +: LAMBDA] <= {mq_ohr[oi*LAMBDA +: LAMBDA-1], mq_ohr[oi*LAMBDA + LAMBDA-1]};
                end
            end
        end
    end
    // ACC read issue (stage 0) and the per-read metadata
    localparam OW = GWW;                    // word offset within the group
    // meta: {mt, r, g, q, last_q, last_ob, we[DW], keep[DW]} + aux[DW] + bias[DW] + addresses
    localparam MW0 = MTW + RW + GW + QW + 2 + 2 * DW;
    reg  [MW0-1:0]    meta0;
    reg  [DW*32-1:0]  aux0, bias0;
    reg  [31:0]       addr0, addr2_0;
    reg               v0;
    // lane selects through the one-hot copies (lane li uses copy li mod MREP): aux word
    // LAMBDA*li + q of the head group, bias word GWD*g + LAMBDA*li + q of the block
    wire [DW*32-1:0]  aux_sel, bias_sel;
    genvar sli, sg, sq;
    generate
        for (sli = 0; sli < DW; sli = sli + 1) begin : g_sel0
            localparam RC = sli % MREP;
            cm_ohmux #(.W(32), .N(LAMBDA), .REP(1)) u_aux (.sel(mq_ohr[RC*LAMBDA +: LAMBDA]),
                .d(aux_head[sli*LAMBDA*32 +: LAMBDA*32]), .y(aux_sel[sli*32 +: 32]));
            wire [NGO*LAMBDA*32-1:0] bcand;
            wire [NGO*LAMBDA-1:0]    bsel;
            for (sg = 0; sg < NGO; sg = sg + 1) begin : g_g
                for (sq = 0; sq < LAMBDA; sq = sq + 1) begin : g_q
                    if (sg*GWD + sli*LAMBDA + sq < BC) begin : g_in
                        assign bcand[(sg*LAMBDA + sq)*32 +: 32] = bias_r[(sg*GWD + sli*LAMBDA + sq)*32 +: 32];
                    end else begin : g_out
                        assign bcand[(sg*LAMBDA + sq)*32 +: 32] = 32'd0;
                    end
                    assign bsel[sg*LAMBDA + sq] = mg_ohr[RC*NGO + sg] & mq_ohr[RC*LAMBDA + sq];
                end
            end
            cm_ohmux #(.W(32), .N(NGO*LAMBDA), .REP(1)) u_bias (.sel(bsel), .d(bcand), .y(bias_sel[sli*32 +: 32]));
        end
    endgenerate
    integer li;
    always @(posedge clk) begin
        if (!rst_n) begin acc_en <= 1'b0; v0 <= 1'b0; end
        else begin
            acc_en <= m_issue; v0 <= m_issue;
        end
        acc_half <= half; acc_par <= m_q[0];
        acc_addr <= {m_mt, m_r, m_g, m_q[QW-1:1]};
        for (li = 0; li < DW; li = li + 1) begin
            // column within the block: LAMBDA*(DW*g + j) + q ; within the group: LAMBDA*j + q
            /* verilator lint_off WIDTHEXPAND */
            meta0[li]      <= mk_head[li * LAMBDA + m_q];
            meta0[DW + li] <= ({m_g, {GWW{1'b0}}} + li * LAMBDA + m_q) < n_cols;
            aux0[li*32 +: 32]  <= aux_sel[li*32 +: 32];
            bias0[li*32 +: 32] <= bias_sel[li*32 +: 32];
            /* verilator lint_on WIDTHEXPAND */
        end
        meta0[MW0-1 -: MTW+RW+GW+QW] <= {m_mt, m_r, m_g, m_q};
        meta0[2*DW+1] <= m_last_q;
        meta0[2*DW]   <= m_last_q & m_last_g & m_last_r & m_last_mt;
        addr0   <= m_c_row  + m_goff;
        addr2_0 <= m_o2_row + m_goff;
    end

    // ------------------------------------------------------------------ lanes
    // s1: ACC Q valid (macro) ; s2: registered
    wire              v2;
    wire [MW0-1:0]    meta2;
    wire [DW*32-1:0]  aux2, bias2;
    wire [63:0]       ad2;
    cm_delay #(.W(1), .N(2), .RST(1)) u_v2 (.clk(clk), .rst_n(rst_n), .d(v0), .q(v2));
    cm_delay #(.W(MW0+2*DW*32+64), .N(2), .RST(0)) u_m2 (.clk(clk), .rst_n(rst_n),
        .d({meta0, aux0, bias0, addr0, addr2_0}), .q({meta2, aux2, bias2, ad2}));
    reg  [DW*32-1:0]  acc2;
    always @(posedge clk) acc2 <= acc_q;
    // s2..s4: bias add ; s4: select, scale ; s5: registered v3
    wire [DW*32-1:0]  yb4, v2d4;
    wire              v4;
    wire [MW0-1:0]    meta4;
    wire [DW*32-1:0]  aux4;
    wire [63:0]       ad4;
    cm_delay #(.W(1), .N(2), .RST(1)) u_v4 (.clk(clk), .rst_n(rst_n), .d(v2), .q(v4));
    cm_delay #(.W(MW0+DW*32+64+DW*32), .N(2), .RST(0)) u_m4 (.clk(clk), .rst_n(rst_n),
        .d({meta2, aux2, ad2, acc2}), .q({meta4, aux4, ad4, v2d4}));
    reg  [DW*32-1:0]  v5;
    reg               vv5;
    reg  [MW0-1:0]    meta5;
    reg  [DW*32-1:0]  aux5;
    reg  [63:0]       ad5;
    genvar j;
    generate
        for (j = 0; j < DW; j = j + 1) begin : g_lane
            wire        bv;
            fp_add #(.STAGES(2)) u_badd (.clk(clk), .rst_n(rst_n), .in_valid(v2),
                .a(acc2[j*32 +: 32]), .b(bias2[j*32 +: 32]), .sub(1'b0), .out_valid(bv), .y(yb4[j*32 +: 32]));
            wire [31:0] vb = ep_bias ? yb4[j*32 +: 32] : v2d4[j*32 +: 32];
            wire [31:0] vs;
            gm_pow2 u_sc (.x(vb), .e(ep_sexp), .y(vs));
            always @(posedge clk) v5[j*32 +: 32] <= ep_scale ? vs : vb;
            /* verilator lint_off UNUSEDSIGNAL */
            wire unused_bv = bv;
            /* verilator lint_on UNUSEDSIGNAL */
        end
    endgenerate
    always @(posedge clk) begin
        if (!rst_n) vv5 <= 1'b0; else vv5 <= v4;
        meta5 <= meta4; aux5 <= aux4; ad5 <= ad4;
    end

    // ---- direct path: s5: drop mul ; s6: select ; s6..s8: resid add ; s8: select -> assembler
    wire [DW*32-1:0]  ym6, v5d6, yr8, v6d8;
    wire              v6, v8;
    wire [MW0-1:0]    meta6, meta8;
    wire [DW*32-1:0]  aux6;
    wire [63:0]       ad6, ad8;
    cm_delay #(.W(1), .N(1), .RST(1)) u_v6 (.clk(clk), .rst_n(rst_n), .d(vv5 & ~gelu_mode), .q(v6));
    cm_delay #(.W(MW0+DW*32+64+DW*32), .N(1), .RST(0)) u_m6 (.clk(clk), .rst_n(rst_n),
        .d({meta5, aux5, ad5, v5}), .q({meta6, aux6, ad6, v5d6}));
    reg  [DW*32-1:0]  v6s;
    cm_delay #(.W(1), .N(2), .RST(1)) u_v8 (.clk(clk), .rst_n(rst_n), .d(v6), .q(v8));
    cm_delay #(.W(MW0+64+DW*32), .N(2), .RST(0)) u_m8 (.clk(clk), .rst_n(rst_n),
        .d({meta6, ad6, v6s}), .q({meta8, ad8, v6d8}));
    generate
        for (j = 0; j < DW; j = j + 1) begin : g_lane2
            wire mv, rv;
            fp_mul #(.STAGES(1)) u_dmul (.clk(clk), .rst_n(rst_n), .in_valid(vv5),
                .a(v5[j*32 +: 32]), .b(ep_ds), .out_valid(mv), .y(ym6[j*32 +: 32]));
            wire keep6 = meta6[j];
            always @* v6s[j*32 +: 32] = ep_drop ? (keep6 ? ym6[j*32 +: 32] : 32'd0) : v5d6[j*32 +: 32];
            fp_add #(.STAGES(2)) u_radd (.clk(clk), .rst_n(rst_n), .in_valid(v6),
                .a(v6s[j*32 +: 32]), .b(aux6[j*32 +: 32]), .sub(1'b0), .out_valid(rv), .y(yr8[j*32 +: 32]));
            /* verilator lint_off UNUSEDSIGNAL */
            wire unused_v = mv | rv;
            /* verilator lint_on UNUSEDSIGNAL */
        end
    endgenerate
    wire [DW*32-1:0] out8 = ep_resid ? yr8 : v6d8;

    // ---- GELU path: s5 -> FIFO (8 words of DW elements) -> serializer -> gm_gelu
    // The reader claims LAMBDA FIFO entries per group before it issues the group's ACC reads
    // (gf_cred); the ROB of gm_gelu throttles the serializer.
    localparam GFW = DW*32 + DW*32 + MW0 + 64;
    localparam GREP = 4;
    reg  [8*GFW-1:0] gf;
    reg  [3:0]     gf_wp, gf_rp;
    (* keep *) reg [GREP*8-1:0] gf_ohr;     // one-hot copies of gf_rp[2:0]
    reg  [3:0]     gf_cred;
    wire           gf_empty = (gf_wp == gf_rp);
    wire           g_pop;
    /* verilator lint_off WIDTHEXPAND */
    assign gelu_room = (gf_cred >= LAMBDA);
    /* verilator lint_on WIDTHEXPAND */
    reg  [7:0]     gfw_oh;                  // one-hot gf_wp[2:0]: per-entry write enables
    genvar gw;
    generate
        for (gw = 0; gw < 8; gw = gw + 1) begin : g_gf_wr
            wire en = vv5 & gelu_mode & gfw_oh[gw];
            always @(posedge clk) if (en) gf[gw*GFW +: GFW] <= {v5, aux5, meta5, ad5};
        end
    endgenerate
    always @(posedge clk) begin
        if (!rst_n) begin gf_wp <= 4'd0; gf_cred <= 4'd8; gfw_oh <= 8'd1; end
        else begin
            if (vv5 & gelu_mode) begin
                gf_wp <= gf_wp + 1'b1;
                gfw_oh <= {gfw_oh[6:0], gfw_oh[7]};
            end
            /* verilator lint_off WIDTHTRUNC */
            gf_cred <= gf_cred - ((m_grp_ok & gelu_mode) ? LAMBDA : 0) + {3'd0, g_pop};
            /* verilator lint_on WIDTHTRUNC */
        end
    end
    wire [GFW-1:0]   gh;
    cm_ohmux #(.W(GFW), .N(8), .REP(GREP)) u_gfh (.sel(gf_ohr), .d(gf), .y(gh));
    wire [DW*32-1:0] gh_v   = gh[GFW-1 -: DW*32];
    wire [DW*32-1:0] gh_aux = gh[GFW-1-DW*32 -: DW*32];
    wire [MW0-1:0]   gh_m   = gh[64 +: MW0];
    wire [63:0]      gh_ad  = gh[63:0];
    reg  [DWW-1:0]   gs_j;                  // element index j within the FIFO word
    wire             g_in_ready;
    wire             g_issue = ~gf_empty & g_in_ready;
    /* verilator lint_off WIDTHEXPAND */
    wire             g_last_j = (gs_j == DW - 1);
    /* verilator lint_on WIDTHEXPAND */
    assign g_pop = g_issue & g_last_j;
    integer gi;
    always @(posedge clk) begin
        if (!rst_n) begin
            gf_rp <= 4'd0; gs_j <= {DWW{1'b0}};
            for (gi = 0; gi < GREP; gi = gi + 1) gf_ohr[gi*8 +: 8] <= 8'd1;
        end else if (g_issue) begin
            if (g_last_j) begin
                gs_j <= {DWW{1'b0}}; gf_rp <= gf_rp + 1'b1;
                for (gi = 0; gi < GREP; gi = gi + 1) gf_ohr[gi*8 +: 8] <= {gf_ohr[gi*8 +: 7], gf_ohr[gi*8 + 7]};
            end else gs_j <= gs_j + 1'b1;
        end
    end
    // gelu meta: {off[OW], we (bit 66), last_in_group (65), last_ob (64), addr, addr2}
    localparam GMW = OW + 3 + 64;
    wire [QW-1:0] gq = gh_m[2*DW+2 +: QW];
    /* verilator lint_off UNUSEDSIGNAL */   // LAMBDA*j + q < GWD: the high bits are zero
    wire [OW+DWW-1:0] g_off_w = ({{OW{1'b0}}, gs_j} << QW) | {{DWW{1'b0}}, {{(OW-QW){1'b0}}, gq}};
    /* verilator lint_on UNUSEDSIGNAL */
    wire [OW-1:0] g_off = g_off_w[OW-1:0];
    /* verilator lint_off WIDTHEXPAND */
    wire          g_we  = gh_m[DW + gs_j];
    /* verilator lint_on WIDTHEXPAND */
    wire          g_lastq = gh_m[2*DW+1];
    wire          g_lastob = gh_m[2*DW];
    wire [GMW-1:0] g_meta = {g_off, g_we, g_lastq & g_last_j, g_lastob & g_last_j, gh_ad};
    wire          go_valid;
    wire [31:0]   go_y, go_u;
    /* verilator lint_off UNUSEDSIGNAL */   // go_meta[64] (last of block) is implied by the counters
    wire [GMW-1:0] go_meta;
    /* verilator lint_on UNUSEDSIGNAL */
    wire          g_err;
    gm_gelu #(.NSFU(NSFU), .NROB(NROB), .ROBW(ROBW), .MW(GMW), .LF(2)) u_gelu (
        .clk(clk), .rst_n(rst_n_g), .in_valid(g_issue), .in_ready(g_in_ready),
        .in_bwd(ep_gelu_bwd), .in_bypass(~g_we),
        .in_u(ep_gelu_bwd ? gh_aux[gs_j*32 +: 32] : gh_v[gs_j*32 +: 32]),
        .in_dg(gh_v[gs_j*32 +: 32]), .in_meta(g_meta),
        .out_valid(go_valid), .out_y(go_y), .out_u(go_u), .out_meta(go_meta), .err(g_err));

    // ------------------------------------------------------------------ assembler
    // NASM entries of a group: out / out2 words, we mask, addresses; words arrive tagged with
    // their offset; the group completes on last_in_group.  Entries are allocated in order.
    // Storage is flat; every word has its own write enable (entry one-hot x offset one-hot) and
    // the writer reads its beat through a one-hot (entry, beat) select, so no pointer flop fans
    // out to the whole NASM x GWD x 32 bit file (docs/p3_report.md section 6).
    reg  [NASM*GWD*32-1:0] asm_o, asm_o2;
    reg  [NASM*GWD-1:0]    asm_we;
    reg  [NASM*64-1:0]     asm_ad;
    reg  [NASM-1:0]   asm_done;
    reg  [2:0]        as_wp, as_rp;         // fill pointer, write pointer (NASM <= 4)
    reg  [NASM-1:0]   as_wp_oh;             // one-hot as_wp[1:0]
    wire              wr_pop;               // the writer released an entry (declared early for Icarus)
    wire [QW-1:0]     q8 = meta8[2*DW+2 +: QW];
    wire [LAMBDA-1:0] q8_oh;
    // gm_gelu results re-registered with the word offset decoded (one more cycle of latency on
    // the GELU return path; the done condition counts gr_v)
    reg               gr_v, gr_we, gr_lastg;
    reg  [31:0]       gr_y, gr_u;
    reg  [GWD-1:0]    gr_off;
    reg  [63:0]       gr_ad;
    integer ai;
    always @(posedge clk) begin
        if (!rst_n) gr_v <= 1'b0; else gr_v <= go_valid;
        gr_y <= go_y; gr_u <= go_u; gr_we <= go_meta[66]; gr_lastg <= go_meta[65]; gr_ad <= go_meta[63:0];
        /* verilator lint_off WIDTHEXPAND */
        for (ai = 0; ai < GWD; ai = ai + 1) gr_off[ai] <= (go_meta[GMW-1 -: OW] == ai);
        /* verilator lint_on WIDTHEXPAND */
    end
    wire as_adv = (v8 & meta8[2*DW+1]) | (gr_v & gr_lastg);
    always @(posedge clk) begin
        if (!rst_n) begin as_wp <= 3'd0; as_wp_oh <= {{(NASM-1){1'b0}}, 1'b1}; asm_done <= {NASM{1'b0}}; end
        else begin
            if (v8 & meta8[2*DW+1]) asm_done[as_wp[1:0]] <= 1'b1;
            if (gr_v & gr_lastg)    asm_done[as_wp[1:0]] <= 1'b1;
            if (as_adv) begin as_wp <= as_wp + 1'b1; as_wp_oh <= {as_wp_oh[NASM-2:0], as_wp_oh[NASM-1]}; end
            if (wr_pop) asm_done[as_rp[1:0]] <= 1'b0;
        end
    end
    genvar ae, ao, aq;
    generate
        for (aq = 0; aq < LAMBDA; aq = aq + 1) begin : g_q8
            /* verilator lint_off WIDTHEXPAND */
            assign q8_oh[aq] = (q8 == aq);
            /* verilator lint_on WIDTHEXPAND */
        end
        for (ae = 0; ae < NASM; ae = ae + 1) begin : g_asm
            wire sel_w = as_wp_oh[ae];
            for (ao = 0; ao < GWD; ao = ao + 1) begin : g_word
                // direct path: lane ao / LAMBDA at sub-row q8 = ao mod LAMBDA ; GELU path: offset ao
                wire en8 = v8 & sel_w & q8_oh[ao % LAMBDA];
                wire eng = gr_v & sel_w & gr_off[ao];
                always @(posedge clk) begin
                    if (en8) begin
                        asm_o[(ae*GWD + ao)*32 +: 32] <= out8[(ao / LAMBDA)*32 +: 32];
                        asm_we[ae*GWD + ao] <= meta8[DW + ao / LAMBDA];
                    end
                    if (eng) begin
                        asm_o[(ae*GWD + ao)*32 +: 32]  <= gr_y;
                        asm_o2[(ae*GWD + ao)*32 +: 32] <= gr_u;
                        asm_we[ae*GWD + ao] <= gr_we;
                    end
                end
            end
            always @(posedge clk) begin
                if (v8 & sel_w)  asm_ad[ae*64 +: 64] <= ad8;
                if (gr_v & sel_w) asm_ad[ae*64 +: 64] <= gr_ad;
            end
        end
    endgenerate

    // ------------------------------------------------------------------ writer
    localparam NB2  = 2 * NBG;              // beats per entry: plane 0 beats then plane 1 beats
    localparam BW   = 256 + 8;              // beat candidate {data, word mask}
    localparam WREP = 4;
    reg  [3:0]  w_b;                        // beat index
    (* keep *) reg [WREP*NASM-1:0] as_rp_ohr;   // one-hot copies of as_rp[1:0]
    (* keep *) reg [WREP*NB2-1:0]  wb_ohr;      // one-hot copies of w_b
    wire        w_avail = asm_done[as_rp[1:0]];
    wire [3:0]  w_nb = ep_gelu ? {NBG_4[2:0], 1'b0} : NBG_4;
    wire        w_plane = (w_b >= NBG_4);
    wire [3:0]  w_bb = w_plane ? w_b - NBG_4 : w_b;
    wire        w_fire = wr_valid & wr_ready;
    wire        w_last = (w_b == w_nb - 1'b1);
    assign wr_pop = w_fire & w_last;
    assign asm_free = wr_pop;
    integer wi;
    always @(posedge clk) begin
        if (!rst_n) begin
            wr_valid <= 1'b0; w_b <= 4'd0; as_rp <= 3'd0;
            for (wi = 0; wi < WREP; wi = wi + 1) begin
                as_rp_ohr[wi*NASM +: NASM] <= {{(NASM-1){1'b0}}, 1'b1};
                wb_ohr[wi*NB2 +: NB2] <= {{(NB2-1){1'b0}}, 1'b1};
            end
        end else begin
            if (w_fire) begin
                if (w_last) begin
                    w_b <= 4'd0; as_rp <= as_rp + 1'b1;
                    for (wi = 0; wi < WREP; wi = wi + 1) begin
                        as_rp_ohr[wi*NASM +: NASM] <= {as_rp_ohr[wi*NASM +: NASM-1], as_rp_ohr[wi*NASM + NASM-1]};
                        wb_ohr[wi*NB2 +: NB2] <= {{(NB2-1){1'b0}}, 1'b1};
                    end
                end else begin
                    w_b <= w_b + 1'b1;
                    for (wi = 0; wi < WREP; wi = wi + 1)
                        wb_ohr[wi*NB2 +: NB2] <= {wb_ohr[wi*NB2 +: NB2-1], wb_ohr[wi*NB2 + NB2-1]};
                end
            end
            if (~wr_valid | w_fire) begin
                wr_valid <= w_avail & ~(w_fire & w_last);
            end
        end
    end
    // beat candidates (entry e, beat b): plane b / NBG, words (b mod NBG)*8 .. +7 (0 beyond GWD)
    wire [NASM*NB2*BW-1:0]   bcand;
    wire [WREP*NASM*NB2-1:0] bsel;
    genvar ce, cb, cw, cr;
    generate
        for (ce = 0; ce < NASM; ce = ce + 1) begin : g_ce
            for (cb = 0; cb < NB2; cb = cb + 1) begin : g_cb
                localparam CP = cb / NBG;
                localparam CB = cb % NBG;
                for (cw = 0; cw < 8; cw = cw + 1) begin : g_cw
                    if (CB*8 + cw < GWD) begin : g_in
                        if (CP == 0) begin : g_p0
                            assign bcand[(ce*NB2 + cb)*BW + cw*32 +: 32] = asm_o[(ce*GWD + CB*8 + cw)*32 +: 32];
                        end else begin : g_p1
                            assign bcand[(ce*NB2 + cb)*BW + cw*32 +: 32] = asm_o2[(ce*GWD + CB*8 + cw)*32 +: 32];
                        end
                        assign bcand[(ce*NB2 + cb)*BW + 256 + cw] = asm_we[ce*GWD + CB*8 + cw];
                    end else begin : g_out
                        assign bcand[(ce*NB2 + cb)*BW + cw*32 +: 32] = 32'd0;
                        assign bcand[(ce*NB2 + cb)*BW + 256 + cw] = 1'b0;
                    end
                end
                for (cr = 0; cr < WREP; cr = cr + 1) begin : g_cr
                    assign bsel[cr*NASM*NB2 + ce*NB2 + cb] = as_rp_ohr[cr*NASM + ce] & wb_ohr[cr*NB2 + cb];
                end
            end
        end
    endgenerate
    wire [BW-1:0] beat;
    wire [63:0]   w_ad;
    cm_ohmux #(.W(BW), .N(NASM*NB2), .REP(WREP)) u_beat (.sel(bsel), .d(bcand), .y(beat));
    cm_ohmux #(.W(64), .N(NASM), .REP(1)) u_wad (.sel(as_rp_ohr[NASM-1:0]), .d(asm_ad), .y(w_ad));
    always @* begin
        wr_data = beat[255:0];
        wr_mask = beat[263:256];
        wr_addr = (w_plane ? w_ad[31:0] : w_ad[63:32]) + {23'd0, w_bb, 5'd0};
    end

    // ------------------------------------------------------------------ block completion
    reg  m_done_seen;
    wire all_written = ~(|asm_done) & ~wr_valid & ~v0 & ~v2 & ~v4 & ~vv5 & ~v6 & ~v8 & gf_empty & ~go_valid & ~gr_v;
    reg  [ROBW+1:0] g_inflight;             // elements issued to gm_gelu and not yet in the assembler
    always @(posedge clk) begin
        if (!rst_n) g_inflight <= {(ROBW+2){1'b0}};
        else g_inflight <= g_inflight + {{(ROBW+1){1'b0}}, g_issue} - {{(ROBW+1){1'b0}}, gr_v};
    end
    always @(posedge clk) begin
        if (!rst_n) begin
            run <= 1'b0; ep_done <= 1'b0; fin_cnt[0] <= {(MTW+1){1'b0}}; fin_cnt[1] <= {(MTW+1){1'b0}};
            m_done_seen <= 1'b0;
        end else begin
            ep_done <= 1'b0;
            if (fin_valid) fin_cnt[fin_half] <= fin_cnt[fin_half] + 1'b1;
            if (ob_fire) begin
                run <= 1'b1; half <= ob_half; n_rows <= ob_rows; n_cols <= ob_cols;
                c_tile <= ob_c; aux_tile <= ob_aux; o2_tile <= ob_o2; bias_tile <= ob_bias;
                cidx <= ob_cidx; c_rs <= ob_rs; m_done_seen <= 1'b0;
            end else if (run) begin
                if (~m_act & ~s_act) m_done_seen <= 1'b1;
                if (m_done_seen & all_written & (g_inflight == {(ROBW+2){1'b0}})) begin
                    run <= 1'b0; ep_done <= 1'b1; ep_done_half <= half;
                    fin_cnt[half] <= {(MTW+1){1'b0}};
                end
            end
        end
    end
    assign idle = ~run & ~ob_valid;
    assign err = g_err;
    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_x = (|c_tile) | (|aux_tile) | (|o2_tile) | (|bias_tile) | (|cidx) | (|w_ad) | (|gh_m);
    /* verilator lint_on UNUSEDSIGNAL */
endmodule
`default_nettype wire
