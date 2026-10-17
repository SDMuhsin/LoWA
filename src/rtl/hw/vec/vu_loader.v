`default_nettype none
// vu_loader: executes one step's loads for the vector unit (docs/p4_report.md section 2.1).
//   rd0: the row-segment tiles of operand A and B: for each column block of 8 * 2^HBL words, one
//        request per row (len <= 8 * 2^HBL words, a contiguous multi-beat burst of that row; P6b,
//        ruling R8), beats land in the row's lane holder FIFO (2^HBL beats) and drain one word per
//        cycle into the lane's tile buffer (the DMA contract: a beat carries 8 consecutive words of one
//        DRAM row; the beats of one request go to the same lane).
//        Row addresses: base + idx * rs, idx = row0 + r, or the gathered id (EMB word table) /
//        position id computed from the ids (pos = 1 + cumsum(id != 1) * (id != 1), per sequence
//        of d_emb_l tokens) for operand B.
//   rd1: per-row scalars (st0 / st1 -> lane s0 / s1, one word per row), the ids of a gathered
//        tile, and the column vectors cvec0..2 (segment of the row length, or the key-mask bit row
//        b_idx when ld_km) into the cvec banks of parity ld_cvpar.
// Both channels run concurrently; ld_done when every write has landed.
module vu_loader #(
    parameter V    = 16,
    parameter VW   = 4,
    parameter RBW  = 8,
    parameter HBL  = 2                   // log2 beats per rd0 row request = lane holder FIFO depth (0..3)
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         unit_start,          // descriptor start: reset the position-id state
    input  wire         ld_start,
    output reg          ld_done,
    input  wire         ld_a, ld_b, ld_abuf, ld_gather, ld_km, ld_cvpar,
    input  wire [2:0]   ld_cv,
    input  wire [1:0]   ld_sreg,
    input  wire [15:0]  ld_row0, ld_seg_off, ld_b_idx,
    input  wire [VW:0]  ld_nrows,
    input  wire [RBW:0] ld_seg_len,
    // descriptor
    input  wire [31:0]  d_a_base, d_a_rs, d_b_base, d_b_rs,
    input  wire [31:0]  d_cv0_base, d_cv1_base, d_cv2_base, d_km_rs,
    input  wire [31:0]  d_st0_base, d_st1_base, d_ids_base,
    input  wire [15:0]  d_cols, d_emb_l,
    // memory read channel 0 (tiles)
    output reg          rd0_req_valid,
    input  wire         rd0_req_ready,
    output reg  [31:0]  rd0_req_addr,
    output reg  [6:0]   rd0_req_len,
    input  wire         rd0_dat_valid,
    output wire         rd0_dat_ready,
    input  wire [255:0] rd0_dat,
    // memory read channel 1 (scalars, ids, column vectors)
    output reg          rd1_req_valid,
    input  wire         rd1_req_ready,
    output reg  [31:0]  rd1_req_addr,
    output reg  [6:0]   rd1_req_len,
    input  wire         rd1_dat_valid,
    output wire         rd1_dat_ready,
    input  wire [255:0] rd1_dat,
    // lane tile writes
    output wire [V-1:0] lt_we,
    output wire [V-1:0] lt_buf,            // per lane: the buffer its holder drains into
    output wire [V*RBW-1:0] lt_addr,
    output wire [V*32-1:0] lt_d,
    // lane scalar writes
    output wire [V-1:0] sld_v,
    output reg  [1:0]   sld_sel,
    output wire [V*32-1:0] sld_d,
    // cvec bank writes
    output reg  [2:0]   cv_we,
    output reg          cv_wpar,
    output reg  [RBW-1:0] cv_waddr,
    output reg  [31:0]  cv_wd
);
    /* verilator lint_off WIDTHTRUNC */   // V < 2**(VW+1) by construction
    localparam [VW:0] VV = V;
    /* verilator lint_on WIDTHTRUNC */
    // ------------------------------------------------------------------ row address table
    reg  [31:0] rowa [0:V-1];                 // operand A row addresses
    reg  [31:0] rowb [0:V-1];
    wire [V*32-1:0] ids_v;                    // gathered ids of the tile (per-lane registers below)
    reg  [15:0] pos_cnt, pos_cnt_tile, seqpos, seqpos_tile, saved_row0;
    reg         ids_ok;                       // ids of this tile are in the table
    reg  [2:0]  ast;                          // address generation state
    reg  [VW:0] ar;
    reg         a_which;                      // 0 = A table, 1 = B table
    reg  [31:0] m_a, m_b;
    reg  [31:0] m_p1;
    /* verilator lint_off UNUSEDSIGNAL */
    reg  [VW:0] m_r1, m_r2;
    /* verilator lint_on UNUSEDSIGNAL */
    reg         m_v1, m_v2, m_w1, m_w2;
    reg  [31:0] m_base1, m_base2;
    reg  [15:0] cnt_run;
    reg         addr_ok;
    wire [31:0] id_cur = ids_v[ar[VW-1:0]*32 +: 32];
    wire        id_nz = (id_cur != 32'd1);
    wire [15:0] pos_nz = id_nz ? 16'd1 : 16'd0;
    wire [15:0] pos_val = id_nz ? cnt_run + 16'd2 : 16'd1;
    localparam [2:0] A_IDLE = 3'd0, A_WAITIDS = 3'd1, A_GEN = 3'd2, A_FLUSH = 3'd3;
    always @(posedge clk) begin
        m_v1 <= 1'b0; m_v2 <= m_v1; m_w2 <= m_w1; m_r2 <= m_r1; m_base2 <= m_base1;
        m_p1 <= m_a * m_b;                                    // product of the operands registered one cycle earlier
        if (m_v2) begin
            if (m_w2) rowb[m_r2[VW-1:0]] <= m_base2 + m_p1; else rowa[m_r2[VW-1:0]] <= m_base2 + m_p1;
        end
        if (!rst_n) begin
            ast <= A_IDLE; addr_ok <= 1'b0; ar <= {(VW+1){1'b0}}; a_which <= 1'b0; saved_row0 <= 16'hFFFF;
            pos_cnt <= 16'd0; seqpos <= 16'd0; m_w1 <= 1'b0; m_r1 <= {(VW+1){1'b0}}; m_a <= 32'd0; m_b <= 32'd0;
            m_base1 <= 32'd0; cnt_run <= 16'd0; pos_cnt_tile <= 16'd0; seqpos_tile <= 16'd0;
        end else begin
            if (unit_start) begin saved_row0 <= 16'hFFFF; pos_cnt <= 16'd0; seqpos <= 16'd0; end
            case (ast)
                A_IDLE: if (ld_start & (ld_a | ld_b)) begin
                    addr_ok <= 1'b0; ar <= {(VW+1){1'b0}}; a_which <= ~ld_a;
                    if (ld_gather) begin
                        ast <= A_WAITIDS;
                    end else ast <= A_GEN;
                end
                A_WAITIDS: if (ids_ok) begin
                    // position ids: continue the running count of the sequence, or restart it
                    if (ld_row0 != saved_row0) begin
                        saved_row0 <= ld_row0;
                        if (seqpos >= d_emb_l) begin cnt_run <= 16'd0; seqpos_tile <= 16'd0; pos_cnt_tile <= 16'd0; end
                        else begin cnt_run <= pos_cnt; seqpos_tile <= seqpos; pos_cnt_tile <= pos_cnt; end
                    end else cnt_run <= pos_cnt_tile;
                    ast <= A_GEN;
                end
                A_GEN: begin
                    m_v1 <= 1'b1; m_w1 <= a_which; m_r1 <= ar;
                    if (a_which) begin
                        m_base1 <= d_b_base; m_b <= d_b_rs;
                        m_a <= ld_gather ? {16'd0, pos_val} : {16'd0, ld_row0 + {{(16-VW-1){1'b0}}, ar}};
                        if (ld_gather) cnt_run <= cnt_run + pos_nz;
                    end else begin
                        m_base1 <= d_a_base; m_b <= d_a_rs;
                        m_a <= ld_gather ? id_cur : {16'd0, ld_row0 + {{(16-VW-1){1'b0}}, ar}};
                    end
                    if (ar == VV - 1) begin
                        if (a_which | ~ld_b) ast <= A_FLUSH;
                        else begin a_which <= 1'b1; ar <= {(VW+1){1'b0}}; end
                    end else ar <= ar + 1'b1;
                end
                A_FLUSH: if (~m_v1 & ~m_v2) begin
                    ast <= A_IDLE; addr_ok <= 1'b1;
                    if (ld_gather) begin pos_cnt <= cnt_run; seqpos <= seqpos_tile + {{(16-VW-1){1'b0}}, VV}; end
                end
                default: ast <= A_IDLE;
            endcase
        end
    end

    // ------------------------------------------------------------------ rd0: tile requests
    // P6b (ruling R8): one request per row per block of HB beats (BWD = 8 HB words, <= 64), rows inner:
    // each request is a contiguous HB-beat burst of one row (a segment <= BWD words is one request per row);
    // the lane holder is an HB-beat FIFO so a whole burst lands without stalling the channel.
    localparam HB  = 1 << HBL;
    localparam BWL = HBL + 3;                 // log2 words per block
    localparam CBW = RBW + 1 - BWL;           // block counter width
    localparam [6:0] BWD7 = 7'd8 << HBL;
    localparam integer HBI = HB;
    localparam integer HBMI = HB - 1;
    localparam [2:0] HBM  = HBMI[2:0];
    localparam [3:0] HB4  = HBI[3:0];
    localparam PI = (HBL == 0) ? 1 : HBL;     // holder FIFO index width
    reg  [2:0]  r0st;
    reg         r0_op;                        // 0 = A, 1 = B
    reg  [CBW-1:0] r0_cb;                     // column block of BWD words
    reg  [VW:0] r0_r;
    /* verilator lint_off UNUSEDSIGNAL */   // segpb[BWL-1:0], m_r2[VW], lenp15[3:0]: shifted out
    wire [RBW:0] segpb = ld_seg_len + {{(RBW+1-BWL){1'b0}}, {BWL{1'b1}}};
    /* verilator lint_on UNUSEDSIGNAL */
    wire [CBW-1:0] nblk = segpb[RBW:BWL];
    wire [RBW:0] blk_rem = ld_seg_len - {r0_cb, {BWL{1'b0}}};
    wire [6:0]  blk_len = (blk_rem > {{(RBW-6){1'b0}}, BWD7}) ? BWD7 : blk_rem[6:0];
    localparam [2:0] R_IDLE = 3'd0, R_WAIT = 3'd1, R_REQ = 3'd2, R_DONE = 3'd3;
    reg         r0_active;
    always @(posedge clk) begin
        if (!rst_n) begin
            r0st <= R_IDLE; rd0_req_valid <= 1'b0; r0_op <= 1'b0; r0_cb <= {CBW{1'b0}}; r0_r <= {(VW+1){1'b0}}; r0_active <= 1'b0;
        end else begin
            case (r0st)
                R_IDLE: if (ld_start & (ld_a | ld_b)) begin
                    r0st <= R_WAIT; r0_op <= ~ld_a; r0_cb <= {CBW{1'b0}}; r0_r <= {(VW+1){1'b0}}; r0_active <= 1'b1;
                end else r0_active <= 1'b0;
                R_WAIT: if (addr_ok) r0st <= R_REQ;
                R_REQ: begin
                    if (~rd0_req_valid | rd0_req_ready) begin
                        rd0_req_valid <= 1'b1;
                        rd0_req_addr <= (r0_op ? rowb[r0_r[VW-1:0]] : rowa[r0_r[VW-1:0]]) +
                                        {14'd0, ld_seg_off + {{(15-RBW){1'b0}}, r0_cb, {BWL{1'b0}}}, 2'b00};
                        rd0_req_len <= blk_len;
                        if (r0_r == ld_nrows - 1) begin
                            r0_r <= {(VW+1){1'b0}};
                            if (r0_cb == nblk - 1'b1) begin
                                r0_cb <= {CBW{1'b0}};
                                if (r0_op | ~ld_b) r0st <= R_DONE; else r0_op <= 1'b1;
                            end else r0_cb <= r0_cb + 1'b1;
                        end else r0_r <= r0_r + 1'b1;
                    end
                end
                R_DONE: if (rd0_req_ready) begin rd0_req_valid <= 1'b0; r0st <= R_IDLE; end
                default: r0st <= R_IDLE;
            endcase
            if ((r0st == R_IDLE) & rd0_req_ready) rd0_req_valid <= 1'b0;
        end
    end
    // rd0 responses: beat k of (row, block) -> lane holder FIFO -> tile buffer (one word per cycle per lane)
    reg          b0_op;
    reg  [CBW-1:0] b0_cb;
    reg  [VW:0]  b0_r;
    reg  [2:0]   b0_k;                        // beat within the request
    reg          b0_active;
    wire [RBW:0] b_w0 = {b0_cb, {BWL{1'b0}}} + {{(RBW-5){1'b0}}, b0_k, 3'b000};   // word offset of the beat
    wire [RBW:0] b_rem = ld_seg_len - b_w0;
    wire [3:0]   b_len = (b_rem > 8) ? 4'd8 : b_rem[3:0];
    wire         b_lastk = (b_rem <= 8) | (b0_k == HBM);
    wire [V-1:0] hn_nz;                       // lane holder non-empty
    wire [V-1:0] hn_full;                     // lane holder full
    wire         tgt_free = ~hn_full[b0_r[VW-1:0]];
    assign rd0_dat_ready = b0_active & tgt_free;
    wire         beat0 = rd0_dat_valid & rd0_dat_ready;
    always @(posedge clk) begin
        if (!rst_n) begin
            b0_active <= 1'b0; b0_op <= 1'b0; b0_cb <= {CBW{1'b0}}; b0_r <= {(VW+1){1'b0}}; b0_k <= 3'd0;
        end else begin
            if (ld_start & (ld_a | ld_b)) begin
                b0_active <= 1'b1; b0_op <= ~ld_a; b0_cb <= {CBW{1'b0}}; b0_r <= {(VW+1){1'b0}}; b0_k <= 3'd0;
            end
            if (beat0) begin
                if (!b_lastk) b0_k <= b0_k + 3'd1;
                else begin
                    b0_k <= 3'd0;
                    if (b0_r == ld_nrows - 1) begin
                        b0_r <= {(VW+1){1'b0}};
                        if (b0_cb == nblk - 1'b1) begin
                            b0_cb <= {CBW{1'b0}};
                            if (b0_op | ~ld_b) b0_active <= 1'b0; else b0_op <= 1'b1;
                        end else b0_cb <= b0_cb + 1'b1;
                    end else b0_r <= b0_r + 1'b1;
                end
            end
        end
    end
    genvar gl;
    generate
        for (gl = 0; gl < V; gl = gl + 1) begin : g_hold
            reg  [255:0] f_d [0:HB-1];
            reg  [3:0]   f_n [0:HB-1];        // valid words of the beat (1..8)
            reg  [RBW-1:0] f_a [0:HB-1];      // tile word address of word 0
            reg          f_op [0:HB-1];
            reg  [3:0]   f_cnt;
            reg  [2:0]   wp, rp;
            reg  [2:0]   h_i;                 // next word index of the head beat
            reg          we_r, buf_r;
            reg  [RBW-1:0] a_r;
            reg  [31:0]  d_r;
            wire         mine = beat0 & (b0_r[VW-1:0] == gl);
            wire [PI-1:0] rpi = rp[PI-1:0] & HBM[PI-1:0];
            wire [PI-1:0] wpi = wp[PI-1:0] & HBM[PI-1:0];
            wire         ne = (f_cnt != 4'd0);
            wire         pop = ne & ({1'b0, h_i} == f_n[rpi] - 4'd1);
            always @(posedge clk) begin
                if (mine) begin
                    f_d[wpi] <= rd0_dat; f_n[wpi] <= b_len; f_a[wpi] <= b_w0[RBW-1:0]; f_op[wpi] <= b0_op;
                end
                if (!rst_n) begin f_cnt <= 4'd0; wp <= 3'd0; rp <= 3'd0; h_i <= 3'd0; we_r <= 1'b0; end
                else begin
                    if (mine) wp <= wp + 3'd1;
                    if (pop) begin rp <= rp + 3'd1; h_i <= 3'd0; end
                    else if (ne) h_i <= h_i + 3'd1;
                    f_cnt <= f_cnt + {3'd0, mine} - {3'd0, pop};
                    we_r <= ne;
                end
                buf_r <= f_op[rpi] ? ~ld_abuf : ld_abuf;
                a_r <= f_a[rpi] + {{(RBW-3){1'b0}}, h_i};
                d_r <= f_d[rpi][h_i * 32 +: 32];
            end
            assign hn_nz[gl] = ne;
            assign hn_full[gl] = (f_cnt == HB4);
            assign lt_we[gl] = we_r;
            assign lt_buf[gl] = buf_r;
            assign lt_addr[gl*RBW +: RBW] = a_r;
            assign lt_d[gl*32 +: 32] = d_r;
        end
    endgenerate
    wire        hold_any = |hn_nz;
    wire        rd0_busy = r0_active | b0_active | (r0st != R_IDLE);

    // ------------------------------------------------------------------ rd1: scalars, ids, cvecs
    // job order: sreg0, sreg1, ids, cv0, cv1, cv2; each job = ceil(len / 64) requests
    reg  [2:0]  j1;                           // request-side job index (0..5), 6 = done
    reg  [2:0]  jb;                           // response-side job index
    reg  [1:0]  r1st;
    reg  [15:0] j_off;                        // word offset within the job
    reg         r1_active, b1_active;
    wire [15:0] cv_words = ld_km ? ((d_cols + 16'd31) >> 5) : {{(15-RBW){1'b0}}, ld_seg_len};
    // job enables / lengths as wires (Icarus does not add signals read inside functions to @* / assign
    // sensitivity, P3 section 7.9): bit j of jen = job j enabled and non-empty
    wire [15:0] len_rows = {{(15-VW){1'b0}}, ld_nrows};
    wire [15:0] len_seg = {{(15-RBW){1'b0}}, ld_seg_len};
    wire [5:0]  jen = {ld_cv[2] & (ld_seg_len != 0), ld_cv[1] & (ld_seg_len != 0), ld_cv[0] & (cv_words != 16'd0),
                       ld_gather & (ld_a | ld_b) & (ld_nrows != 0), ld_sreg[1] & (ld_nrows != 0), ld_sreg[0] & (ld_nrows != 0)};
    function [15:0] jlen;
        input [2:0] j;
        input [15:0] lr, lc, ls;
        begin
            case (j)
                3'd0, 3'd1, 3'd2: jlen = lr;
                3'd3: jlen = lc;
                3'd4, 3'd5: jlen = ls;
                default: jlen = 16'd0;
            endcase
        end
    endfunction
    function [31:0] job_addr;
        input [2:0] j;
        begin
            case (j)
                3'd0: job_addr = d_st0_base + {14'd0, ld_row0, 2'b00};
                3'd1: job_addr = d_st1_base + {14'd0, ld_row0, 2'b00};
                3'd2: job_addr = d_ids_base + {14'd0, ld_row0, 2'b00};
                3'd3: job_addr = ld_km ? (d_cv0_base + ld_b_idx * d_km_rs) : (d_cv0_base + {14'd0, ld_seg_off, 2'b00});
                3'd4: job_addr = d_cv1_base + {14'd0, ld_seg_off, 2'b00};
                default: job_addr = d_cv2_base + {14'd0, ld_seg_off, 2'b00};
            endcase
        end
    endfunction
    wire [15:0] j_rem = jlen(j1, len_rows, cv_words, len_seg) - j_off;
    wire [6:0]  j_len = (j_rem > 16'd64) ? 7'd64 : j_rem[6:0];
    wire        j1_en = (j1 < 3'd6) & jen[j1];
    always @(posedge clk) begin
        if (!rst_n) begin
            j1 <= 3'd6; r1st <= 2'd0; rd1_req_valid <= 1'b0; j_off <= 16'd0; r1_active <= 1'b0;
        end else begin
            case (r1st)
                2'd0: begin
                    if (ld_start) begin j1 <= 3'd0; j_off <= 16'd0; r1st <= 2'd1; r1_active <= 1'b1; end
                    else r1_active <= 1'b0;
                    if (rd1_req_ready) rd1_req_valid <= 1'b0;
                end
                2'd1: begin                                   // find the next enabled job / issue
                    if (~rd1_req_valid | rd1_req_ready) begin
                        if (j1 == 3'd6) begin rd1_req_valid <= 1'b0; r1st <= 2'd0; end
                        else if (~j1_en) begin j1 <= j1 + 3'd1; j_off <= 16'd0; rd1_req_valid <= 1'b0; end
                        else begin
                            rd1_req_valid <= 1'b1; rd1_req_addr <= job_addr(j1) + {14'd0, j_off, 2'b00}; rd1_req_len <= j_len;
                            if (j_rem <= 16'd64) begin j1 <= j1 + 3'd1; j_off <= 16'd0; end
                            else j_off <= j_off + 16'd64;
                        end
                    end
                end
                default: r1st <= 2'd0;
            endcase
        end
    end
    // rd1 responses: one beat = 8 words; sreg / ids words go to 8 lanes at once, cvec words one per cycle
    reg  [255:0] c_d;
    reg  [3:0]   c_n;
    reg  [2:0]   c_i;
    reg  [RBW-1:0] c_a;
    reg  [2:0]   c_job;
    reg  [15:0]  jb_wofs;                     // word offset of the current beat within the job
    wire [15:0]  jb_rem = jlen(jb, len_rows, cv_words, len_seg) - jb_wofs;
    wire [3:0]   jb_blen = (jb_rem > 16'd8) ? 4'd8 : jb_rem[3:0];
    wire         jb_is_cv = (jb >= 3'd3);
    assign rd1_dat_ready = b1_active & (jb != 3'd6) & (~jb_is_cv | (c_n == 4'd0));
    wire         beat1 = rd1_dat_valid & rd1_dat_ready;
    // next enabled job after jb, and the first enabled job (priority over the jen wire)
    reg  [2:0]   jb_next, jb_first;
    integer      jj;
    always @* begin
        jb_next = 3'd6;
        for (jj = 5; jj >= 0; jj = jj - 1) if (jen[jj] && (jj > {29'd0, jb})) jb_next = jj[2:0];
        jb_first = 3'd6;
        for (jj = 5; jj >= 0; jj = jj - 1) if (jen[jj]) jb_first = jj[2:0];
    end
    always @(posedge clk) begin
        cv_we <= 3'b000;
        if (!rst_n) begin
            cv_we <= 3'b000;
            jb <= 3'd6; b1_active <= 1'b0; c_n <= 4'd0; jb_wofs <= 16'd0; ids_ok <= 1'b0; c_i <= 3'd0; c_job <= 3'd0;
        end else begin
            if (ld_start) begin
                jb <= jb_first; jb_wofs <= 16'd0; b1_active <= (jb_first != 3'd6); ids_ok <= ~(ld_gather & (ld_a | ld_b));
            end
            if (beat1) begin
                if (jb_is_cv) begin
                    c_d <= rd1_dat; c_n <= jb_blen; c_i <= 3'd0; c_a <= jb_wofs[RBW-1:0]; c_job <= jb;
                end else sld_sel <= {1'b0, jb[0]};
                if (jb_rem <= 16'd8) begin
                    jb_wofs <= 16'd0;
                    if (jb == 3'd2) ids_ok <= 1'b1;
                    jb <= jb_next;
                    if (jb_next == 3'd6) b1_active <= 1'b0;
                end else jb_wofs <= jb_wofs + 16'd8;
            end
            if (c_n != 4'd0) begin
                c_n <= c_n - 4'd1; c_i <= c_i + 3'd1;
                cv_we <= (c_job == 3'd3) ? 3'b001 : (c_job == 3'd4) ? 3'b010 : 3'b100;
                cv_waddr <= c_a + {{(RBW-3){1'b0}}, c_i};
                cv_wd <= c_d[c_i * 32 +: 32];
                cv_wpar <= ld_cvpar;
            end
        end
    end
    // per-lane capture of scalar / id beats (beat words 0..7 = lanes jb_wofs .. jb_wofs + 7)
    wire        sc_beat = beat1 & ~jb_is_cv;
    generate
        for (gl = 0; gl < V; gl = gl + 1) begin : g_sc
            reg  [31:0] id_r, sd_r;
            reg         sv_r;
            /* verilator lint_off WIDTHTRUNC */   // genvar / 8 fits VW+1 bits
            wire [VW:0] gl_grp = gl / 8;
            /* verilator lint_on WIDTHTRUNC */
            wire        hit = sc_beat & ((jb_wofs[VW:0] >> 3) == gl_grp) & (gl < ld_nrows);
            always @(posedge clk) begin
                if (!rst_n) sv_r <= 1'b0;
                else sv_r <= hit & (jb != 3'd2);
                if (hit & (jb == 3'd2)) id_r <= rd1_dat[(gl % 8) * 32 +: 32];
                if (hit) sd_r <= rd1_dat[(gl % 8) * 32 +: 32];
            end
            assign ids_v[gl*32 +: 32] = id_r;
            assign sld_v[gl] = sv_r;
            assign sld_d[gl*32 +: 32] = sd_r;
        end
    endgenerate
    wire rd1_busy = r1_active | b1_active | (r1st != 2'd0) | (c_n != 4'd0);

    // ------------------------------------------------------------------ done
    wire lt_we_any = |lt_we;
    reg  ld_run;
    always @(posedge clk) begin
        ld_done <= 1'b0;
        if (!rst_n) ld_run <= 1'b0;
        else begin
            if (ld_start) ld_run <= 1'b1;
            else if (ld_run & ~rd0_busy & ~rd1_busy & ~hold_any & ~lt_we_any & (ast == A_IDLE)) begin
                ld_run <= 1'b0; ld_done <= 1'b1;
            end
        end
    end
endmodule
`default_nettype wire
