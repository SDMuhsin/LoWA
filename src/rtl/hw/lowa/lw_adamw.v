`default_nettype none
// lw_adamw: streaming AdamW (docs/dataflow.md section 6.3, torch _multi_tensor_adamw order, L1
// lw_adamw) over a contiguous range of n elements: p, g, m, v are streamed in through two read
// channels (rd0: p and m, rd1: g and v, 16-word requests alternating between the two streams of a
// channel), p', m', v' are streamed out through the write channel (8-word beats, word mask on the
// tail).  Per element, exactly:
//   g'  = clip_en ? mul(g, c) : g                    (the clip factor c = word 7 of the scalar block)
//   pp  = mul(p, decay);  mm = fma(b1c, sub(g', m), m);  vv = fma(b2c, mul(g', g'), mul(v, b2))
//   den = add(mul(sqrt(vv), rbc2), eps);  p' = sub(pp, mul(step, mul(mm, recip(den))))
// Three fixed-latency segments joined by FIFOs: A (muls / fmas, 1 element per cycle), B (sqrt on a
// pool of NSQ fp_sfu, round-robin, outputs in order because the latency is fixed), C (recip on NRC
// fp_sfu, then the final muls / sub).  Throughput ~ min(NSQ / 17, NRC / 9) elements per cycle.
// Scalar block (8 words at d_sc_base, written by lw_scal / GN_FIN): decay, step, rbc2, b1c, b2, b2c, eps, c.
module lw_adamw #(
    parameter NSQ = 2,
    parameter NRC = 1
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    output wire         busy,
    output reg          done,
    input  wire [31:0]  d_p_base, d_g_base, d_m_base, d_v_base, d_n, d_sc_base,
    input  wire         d_clip_en,
    output wire         rd0_req_valid,
    input  wire         rd0_req_ready,
    output wire [31:0]  rd0_req_addr,
    output wire [6:0]   rd0_req_len,
    input  wire         rd0_dat_valid,
    output wire         rd0_dat_ready,
    input  wire [255:0] rd0_dat,
    output reg          rd1_req_valid,
    input  wire         rd1_req_ready,
    output reg  [31:0]  rd1_req_addr,
    output reg  [6:0]   rd1_req_len,
    input  wire         rd1_dat_valid,
    output wire         rd1_dat_ready,
    input  wire [255:0] rd1_dat,
    output reg          wr_valid,
    input  wire         wr_ready,
    output reg  [31:0]  wr_addr,
    output reg  [255:0] wr_data,
    output reg  [7:0]   wr_mask
);
    localparam [2:0] S_IDLE = 3'd0, S_SC = 3'd1, S_RUN = 3'd2;
    reg  [2:0]  st;
    assign busy = (st != S_IDLE);
    // ------------------------------------------------------------------ scalars (rd0 while S_SC)
    reg         sc_go;
    wire        sc_busy, sc_done, sc_wv, sc_rqv, sc_dr;
    wire [31:0] sc_wd, sc_rqa;
    wire [15:0] sc_idx;
    wire [6:0]  sc_rql;
    lw_rdseq u_sc (.clk(clk), .rst_n(rst_n), .go(sc_go), .base(d_sc_base), .nwords(16'd8), .busy(sc_busy), .done(sc_done),
                   .req_valid(sc_rqv), .req_ready(rd0_req_ready & (st == S_SC)), .req_addr(sc_rqa), .req_len(sc_rql),
                   .dat_valid(rd0_dat_valid & (st == S_SC)), .dat_ready(sc_dr), .dat(rd0_dat),
                   .w_valid(sc_wv), .w_ready(1'b1), .w_d(sc_wd), .w_idx(sc_idx));
    reg  [31:0] k_decay, k_step, k_rbc2, k_b1c, k_b2, k_b2c, k_eps, k_clip;
    always @(posedge clk) if ((st == S_SC) & sc_wv) case (sc_idx[2:0])
        3'd0: k_decay <= sc_wd;  3'd1: k_step <= sc_wd;  3'd2: k_rbc2 <= sc_wd;  3'd3: k_b1c <= sc_wd;
        3'd4: k_b2 <= sc_wd;     3'd5: k_b2c <= sc_wd;   3'd6: k_eps <= sc_wd;   default: k_clip <= sc_wd;
    endcase
    // ------------------------------------------------------------------ input streams: 0 p, 1 m (rd0); 2 g, 3 v (rd1)
    reg  [31:0] s_addr [0:3];
    reg  [31:0] s_rem  [0:3];                 // words still to request
    reg  [2:0]  s_infl [0:3];                 // beats requested, not yet delivered
    wire [3:0]  bf_in_valid, bf_in_ready, bf_out_valid, bf_out_ready;
    wire [255:0] bf_out [0:3];
    wire [1:0]  bf_count [0:3];
    genvar gs;
    generate
        for (gs = 0; gs < 4; gs = gs + 1) begin : g_bf
            lw_fifo #(.W(256), .LOG2(1)) u_bf (.clk(clk), .rst_n(rst_n), .in_valid(bf_in_valid[gs]), .in_ready(bf_in_ready[gs]),
                                                .in_d((gs < 2) ? rd0_dat : rd1_dat), .out_valid(bf_out_valid[gs]), .out_ready(bf_out_ready[gs]),
                                                .out_d(bf_out[gs]), .count(bf_count[gs]));
        end
    endgenerate
    // per channel: turn (which of its two streams requests next) and a route queue {stream, two_beats}
    reg         turn0, turn1;
    reg         q0_valid;                     // streamer's rd0 request (rd0 carries the scalar reader during S_SC)
    reg  [31:0] q0_addr;
    reg  [6:0]  q0_len;
    assign rd0_req_valid = (st == S_SC) ? sc_rqv : q0_valid;
    assign rd0_req_addr  = (st == S_SC) ? sc_rqa : q0_addr;
    assign rd0_req_len   = (st == S_SC) ? sc_rql : q0_len;
    wire        rq0_in_ready, rq0_out_valid, rq1_in_ready, rq1_out_valid;
    wire [1:0]  rq0_d, rq1_d;
    wire [2:0]  rq0_cnt, rq1_cnt;
    reg         rq0_pop, rq1_pop;
    reg         b0_second, b1_second;         // second beat of a two-beat request in progress
    wire [1:0]  cs0 = {1'b0, turn0}, cs1 = {1'b1, turn1};   // stream index requesting next on each channel
    wire [31:0] rem0 = s_rem[cs0], rem1 = s_rem[cs1];
    wire [6:0]  len0 = (rem0 > 32'd16) ? 7'd16 : rem0[6:0];
    wire [6:0]  len1 = (rem1 > 32'd16) ? 7'd16 : rem1[6:0];
    wire        two0 = (len0 > 7'd8), two1 = (len1 > 7'd8);
    wire        cred0 = ({3'b000, bf_count[cs0]} + {2'b00, s_infl[cs0]} + 5'd2 <= 5'd2);   // one request in flight per stream (2 beats)
    wire        cred1 = ({3'b000, bf_count[cs1]} + {2'b00, s_infl[cs1]} + 5'd2 <= 5'd2);
    wire        acc0 = q0_valid & rd0_req_ready & (st == S_RUN);
    wire        acc1 = rd1_req_valid & rd1_req_ready;
    lw_fifo #(.W(2), .LOG2(2)) u_rq0 (.clk(clk), .rst_n(rst_n), .in_valid(acc0), .in_ready(rq0_in_ready),
                                       .in_d({two0, turn0}), .out_valid(rq0_out_valid), .out_ready(rq0_pop), .out_d(rq0_d), .count(rq0_cnt));
    lw_fifo #(.W(2), .LOG2(2)) u_rq1 (.clk(clk), .rst_n(rst_n), .in_valid(acc1), .in_ready(rq1_in_ready),
                                       .in_d({two1, turn1}), .out_valid(rq1_out_valid), .out_ready(rq1_pop), .out_d(rq1_d), .count(rq1_cnt));
    wire [1:0]  dst0 = {1'b0, rq0_d[0]}, dst1 = {1'b1, rq1_d[0]};
    wire        run_dat0 = rd0_dat_valid & (st == S_RUN) & rq0_out_valid;
    wire        run_dat1 = rd1_dat_valid & rq1_out_valid;
    assign bf_in_valid = {run_dat1 & (dst1 == 2'd3), run_dat1 & (dst1 == 2'd2), run_dat0 & (dst0 == 2'd1), run_dat0 & (dst0 == 2'd0)};
    assign rd0_dat_ready = (st == S_SC) ? sc_dr : (rq0_out_valid & bf_in_ready[dst0]);
    assign rd1_dat_ready = rq1_out_valid & bf_in_ready[dst1];
    wire        beat0 = run_dat0 & bf_in_ready[dst0];
    wire        beat1 = run_dat1 & bf_in_ready[dst1];
    always @* begin
        rq0_pop = beat0 & (~rq0_d[1] | b0_second);
        rq1_pop = beat1 & (~rq1_d[1] | b1_second);
    end
    integer i;
    always @(posedge clk)
        if (!rst_n) begin
            q0_valid <= 1'b0; rd1_req_valid <= 1'b0; turn0 <= 1'b0; turn1 <= 1'b0; b0_second <= 1'b0; b1_second <= 1'b0;
            for (i = 0; i < 4; i = i + 1) begin s_infl[i] <= 3'd0; s_rem[i] <= 32'd0; end
        end else begin
            if (st == S_IDLE & start) begin
                s_addr[0] <= d_p_base; s_addr[1] <= d_m_base; s_addr[2] <= d_g_base; s_addr[3] <= d_v_base;
                for (i = 0; i < 4; i = i + 1) begin s_rem[i] <= d_n; s_infl[i] <= 3'd0; end
                turn0 <= 1'b0; turn1 <= 1'b0; b0_second <= 1'b0; b1_second <= 1'b0;
            end
            if (st == S_RUN) begin                    // S_SC: rd0 carries the scalar reader
                for (i = 0; i < 4; i = i + 1)
                    s_infl[i] <= s_infl[i] + ((acc0 & (cs0 == i[1:0])) ? (two0 ? 3'd2 : 3'd1) : 3'd0)
                                           + ((acc1 & (cs1 == i[1:0])) ? (two1 ? 3'd2 : 3'd1) : 3'd0)
                                           - {2'b00, beat0 & (dst0 == i[1:0])} - {2'b00, beat1 & (dst1 == i[1:0])};
                if (acc0) begin
                    q0_valid <= 1'b0; turn0 <= ~turn0;
                    s_addr[cs0] <= s_addr[cs0] + {23'd0, q0_len, 2'b00};
                    s_rem[cs0] <= s_rem[cs0] - {25'd0, q0_len};
                end else if (~q0_valid & (rem0 != 32'd0) & cred0 & rq0_in_ready) begin
                    q0_valid <= 1'b1; q0_addr <= s_addr[cs0]; q0_len <= len0;
                end
                if (acc1) begin
                    rd1_req_valid <= 1'b0; turn1 <= ~turn1;
                    s_addr[cs1] <= s_addr[cs1] + {23'd0, rd1_req_len, 2'b00};
                    s_rem[cs1] <= s_rem[cs1] - {25'd0, rd1_req_len};
                end else if (~rd1_req_valid & (rem1 != 32'd0) & cred1 & rq1_in_ready) begin
                    rd1_req_valid <= 1'b1; rd1_req_addr <= s_addr[cs1]; rd1_req_len <= len1;
                end
                if (beat0) b0_second <= rq0_d[1] & ~b0_second;
                if (beat1) b1_second <= rq1_d[1] & ~b1_second;
            end
        end
    // ------------------------------------------------------------------ element issue (segment A)
    reg  [31:0] e_in;                         // elements issued
    reg  [2:0]  hi;                           // word index in the head beats
    wire        f1_in_ready;
    wire [4:0]  f1_count;
    wire        heads = &bf_out_valid;
    wire        a_issue = (st == S_RUN) & heads & (e_in != d_n) & (f1_count <= 5'd8);                  // + 6 pipeline stages <= 16
    wire        a_last_word = (hi == 3'd7) | (e_in + 32'd1 == d_n);
    assign bf_out_ready = {4{a_issue & a_last_word}};
    wire [7:0]  hi_oh = 8'd1 << hi;
    wire [31:0] in_p, in_m, in_g, in_v;
    cm_ohmux #(.W(32), .N(8), .REP(1)) u_hp (.sel(hi_oh), .d(bf_out[0]), .y(in_p));
    cm_ohmux #(.W(32), .N(8), .REP(1)) u_hm (.sel(hi_oh), .d(bf_out[1]), .y(in_m));
    cm_ohmux #(.W(32), .N(8), .REP(1)) u_hg (.sel(hi_oh), .d(bf_out[2]), .y(in_g));
    cm_ohmux #(.W(32), .N(8), .REP(1)) u_hv (.sel(hi_oh), .d(bf_out[3]), .y(in_v));
    always @(posedge clk)
        if (!rst_n) begin e_in <= 32'd0; hi <= 3'd0; end
        else begin
            if (st == S_IDLE & start) begin e_in <= 32'd0; hi <= 3'd0; end
            if (a_issue) begin e_in <= e_in + 32'd1; hi <= a_last_word ? 3'd0 : hi + 3'd1; end
        end
    // A0: g1 = mul(g, c); operands delayed 2
    reg  [5:0]  av;
    always @(posedge clk) if (!rst_n) av <= 6'd0; else av <= {av[4:0], a_issue};
    wire        gc_ov;
    wire [31:0] gc_y, p_d2, g_d2, m_d2, v_d2, m_d4, pp_y, d_y, gg_y, vb_y, mm_y, vv_y, pp_d6;
    fp_mul #(.STAGES(2)) u_gc (.clk(clk), .rst_n(rst_n), .in_valid(a_issue), .a(in_g), .b(k_clip), .out_valid(gc_ov), .y(gc_y));
    cm_delay #(.W(32), .N(2)) u_dp (.clk(clk), .rst_n(rst_n), .d(in_p), .q(p_d2));
    cm_delay #(.W(32), .N(2)) u_dg (.clk(clk), .rst_n(rst_n), .d(in_g), .q(g_d2));
    cm_delay #(.W(32), .N(2)) u_dm (.clk(clk), .rst_n(rst_n), .d(in_m), .q(m_d2));
    cm_delay #(.W(32), .N(2)) u_dv (.clk(clk), .rst_n(rst_n), .d(in_v), .q(v_d2));
    wire [31:0] gs2 = d_clip_en ? gc_y : g_d2;
    wire        pp_ov, d_ov, gg_ov, vb_ov, mm_ov, vv_ov;
    fp_mul #(.STAGES(2)) u_pp (.clk(clk), .rst_n(rst_n), .in_valid(av[1]), .a(p_d2), .b(k_decay), .out_valid(pp_ov), .y(pp_y));
    fp_add #(.STAGES(2)) u_d  (.clk(clk), .rst_n(rst_n), .in_valid(av[1]), .a(gs2), .b(m_d2), .sub(1'b1), .out_valid(d_ov), .y(d_y));
    fp_mul #(.STAGES(2)) u_gg (.clk(clk), .rst_n(rst_n), .in_valid(av[1]), .a(gs2), .b(gs2), .out_valid(gg_ov), .y(gg_y));
    fp_mul #(.STAGES(2)) u_vb (.clk(clk), .rst_n(rst_n), .in_valid(av[1]), .a(v_d2), .b(k_b2), .out_valid(vb_ov), .y(vb_y));
    cm_delay #(.W(32), .N(2)) u_dm4 (.clk(clk), .rst_n(rst_n), .d(m_d2), .q(m_d4));
    fp_fma #(.STAGES(2)) u_mm (.clk(clk), .rst_n(rst_n), .in_valid(av[3]), .a(k_b1c), .b(d_y), .c(m_d4), .out_valid(mm_ov), .y(mm_y));
    fp_fma #(.STAGES(2)) u_vv (.clk(clk), .rst_n(rst_n), .in_valid(av[3]), .a(k_b2c), .b(gg_y), .c(vb_y), .out_valid(vv_ov), .y(vv_y));
    cm_delay #(.W(32), .N(2)) u_dpp (.clk(clk), .rst_n(rst_n), .d(pp_y), .q(pp_d6));
    wire unused_a = gc_ov | pp_ov | d_ov | gg_ov | vb_ov | mm_ov | vv_ov | sc_busy | sc_done | (|sc_idx[15:3]) | (|rq0_cnt) | (|rq1_cnt);
    // FIFO1 {pp, mm, vv}
    wire        f1_out_valid, f1_pop;
    wire [95:0] f1_d;
    lw_fifo #(.W(96), .LOG2(4)) u_f1 (.clk(clk), .rst_n(rst_n), .in_valid(av[5]), .in_ready(f1_in_ready), .in_d({pp_d6, mm_y, vv_y}),
                                       .out_valid(f1_out_valid), .out_ready(f1_pop), .out_d(f1_d), .count(f1_count));
    // ------------------------------------------------------------------ segment B: sqrt pool
    wire        f2_in_ready, f2_out_valid, f2_pop, sb_in_ready, sb_out_valid;
    wire [5:0]  f2_count;
    wire [127:0] f2_d;
    wire [95:0] sb_d;
    wire [4:0]  sb_count;
    reg  [NSQ-1:0] sq_oh;                    // round-robin unit (one-hot)
    wire [NSQ-1:0] sq_rdy, sq_ov;
    wire [32*NSQ-1:0] sq_yv;
    wire        sq_issue = f1_out_valid & (|(sq_rdy & sq_oh)) & sb_in_ready & (f2_count <= 6'd16);   // + NSQ x 4 threads + 4 stages <= 32
    assign f1_pop = sq_issue;
    genvar gq;
    generate
        for (gq = 0; gq < NSQ; gq = gq + 1) begin : g_sq
            wire [3:0] tg;
            fp_sfu #(.LF(2), .NTHR(4), .TAGW(4)) u_s (.clk(clk), .rst_n(rst_n), .in_valid(sq_issue & sq_oh[gq]), .in_ready(sq_rdy[gq]),
                                                       .in_fn(3'd3), .in_x(f1_d[31:0]), .in_tag(4'd0), .out_valid(sq_ov[gq]), .out_y(sq_yv[gq*32 +: 32]), .out_tag(tg));
            wire unused_t = |tg;
        end
    endgenerate
    generate
        if (NSQ > 1) begin : g_sqrot
            always @(posedge clk) if (!rst_n) sq_oh <= {{(NSQ - 1){1'b0}}, 1'b1}; else if (sq_issue) sq_oh <= {sq_oh[NSQ-2:0], sq_oh[NSQ-1]};
        end else begin : g_sqone
            always @(posedge clk) sq_oh <= 1'b1;
        end
    endgenerate
    wire        sq_any = |sq_ov;
    wire [31:0] sq_y;
    cm_ohmux #(.W(32), .N(NSQ), .REP(1)) u_sqy (.sel(sq_ov), .d(sq_yv), .y(sq_y));
    lw_fifo #(.W(96), .LOG2(4)) u_sb (.clk(clk), .rst_n(rst_n), .in_valid(sq_issue), .in_ready(sb_in_ready), .in_d(f1_d),
                                       .out_valid(sb_out_valid), .out_ready(sq_any), .out_d(sb_d), .count(sb_count));
    reg  [3:0]  bv;
    always @(posedge clk) if (!rst_n) bv <= 4'd0; else bv <= {bv[2:0], sq_any};
    wire        sr_ov, den_ov;
    wire [31:0] sr_y, den_y;
    wire [95:0] sb_d4;
    fp_mul #(.STAGES(2)) u_sr (.clk(clk), .rst_n(rst_n), .in_valid(sq_any), .a(sq_y), .b(k_rbc2), .out_valid(sr_ov), .y(sr_y));
    fp_add #(.STAGES(2)) u_den (.clk(clk), .rst_n(rst_n), .in_valid(bv[1]), .a(sr_y), .b(k_eps), .sub(1'b0), .out_valid(den_ov), .y(den_y));
    cm_delay #(.W(96), .N(4)) u_sbd (.clk(clk), .rst_n(rst_n), .d(sb_d), .q(sb_d4));
    lw_fifo #(.W(128), .LOG2(5)) u_f2 (.clk(clk), .rst_n(rst_n), .in_valid(bv[3]), .in_ready(f2_in_ready), .in_d({sb_d4, den_y}),
                                        .out_valid(f2_out_valid), .out_ready(f2_pop), .out_d(f2_d), .count(f2_count));
    wire unused_b = sr_ov | den_ov | sb_out_valid | (|sb_count) | f2_in_ready | f1_in_ready;
    // ------------------------------------------------------------------ segment C: recip pool, final ops
    wire        sc_in_ready, sc_out_valid, of_in_ready, of_out_valid, of_pop;
    wire [95:0] sc_d, of_d;
    wire [63:0] sc_d6;
    wire [4:0]  sc_count;
    wire [5:0]  of_count;
    reg  [NRC-1:0] rc_oh;
    wire [NRC-1:0] rc_rdy, rc_ov;
    wire [32*NRC-1:0] rc_yv;
    wire        rc_issue = f2_out_valid & (|(rc_rdy & rc_oh)) & sc_in_ready & (of_count <= 6'd16);   // + NRC x 4 threads + 6 stages <= 32
    assign f2_pop = rc_issue;
    genvar gr;
    generate
        for (gr = 0; gr < NRC; gr = gr + 1) begin : g_rc
            wire [3:0] tg;
            fp_sfu #(.LF(2), .NTHR(4), .TAGW(4)) u_s (.clk(clk), .rst_n(rst_n), .in_valid(rc_issue & rc_oh[gr]), .in_ready(rc_rdy[gr]),
                                                       .in_fn(3'd1), .in_x(f2_d[31:0]), .in_tag(4'd0), .out_valid(rc_ov[gr]), .out_y(rc_yv[gr*32 +: 32]), .out_tag(tg));
            wire unused_t = |tg;
        end
    endgenerate
    generate
        if (NRC > 1) begin : g_rcrot
            always @(posedge clk) if (!rst_n) rc_oh <= {{(NRC - 1){1'b0}}, 1'b1}; else if (rc_issue) rc_oh <= {rc_oh[NRC-2:0], rc_oh[NRC-1]};
        end else begin : g_rcone
            always @(posedge clk) rc_oh <= 1'b1;
        end
    endgenerate
    wire        rc_any = |rc_ov;
    wire [31:0] rc_y;
    cm_ohmux #(.W(32), .N(NRC), .REP(1)) u_rcy (.sel(rc_ov), .d(rc_yv), .y(rc_y));
    lw_fifo #(.W(96), .LOG2(4)) u_scf (.clk(clk), .rst_n(rst_n), .in_valid(rc_issue), .in_ready(sc_in_ready), .in_d(f2_d[127:32]),
                                        .out_valid(sc_out_valid), .out_ready(rc_any), .out_d(sc_d), .count(sc_count));
    reg  [5:0]  cv;
    always @(posedge clk) if (!rst_n) cv <= 6'd0; else cv <= {cv[4:0], rc_any};
    wire        t_ov, u_ov, pn_ov;
    wire [31:0] t_y, u_y, pn_y;
    fp_mul #(.STAGES(2)) u_t (.clk(clk), .rst_n(rst_n), .in_valid(rc_any), .a(sc_d[63:32]), .b(rc_y), .out_valid(t_ov), .y(t_y));
    fp_mul #(.STAGES(2)) u_u (.clk(clk), .rst_n(rst_n), .in_valid(cv[1]), .a(k_step), .b(t_y), .out_valid(u_ov), .y(u_y));
    cm_delay #(.W(64), .N(6)) u_scd (.clk(clk), .rst_n(rst_n), .d(sc_d[63:0]), .q(sc_d6));
    wire [31:0] pp_d4;
    cm_delay #(.W(32), .N(4)) u_ppd (.clk(clk), .rst_n(rst_n), .d(sc_d[95:64]), .q(pp_d4));
    fp_add #(.STAGES(2)) u_pn (.clk(clk), .rst_n(rst_n), .in_valid(cv[3]), .a(pp_d4), .b(u_y), .sub(1'b1), .out_valid(pn_ov), .y(pn_y));
    lw_fifo #(.W(96), .LOG2(5)) u_of (.clk(clk), .rst_n(rst_n), .in_valid(cv[5]), .in_ready(of_in_ready), .in_d({pn_y, sc_d6}),
                                       .out_valid(of_out_valid), .out_ready(of_pop), .out_d(of_d), .count(of_count));
    wire unused_c = t_ov | u_ov | pn_ov | sc_out_valid | (|sc_count) | of_in_ready;
    // ------------------------------------------------------------------ output: 8-word groups -> 3 beats
    reg  [31:0] e_out;                        // elements collected
    reg  [2:0]  oi;                           // word index in the group
    reg  [1:0]  ob;                           // beat being written: 0 p, 1 m, 2 v
    reg         o_wr;                         // group complete, writing
    reg  [255:0] ob_p, ob_m, ob_v;
    reg  [7:0]  ob_mask;
    reg  [31:0] wa_p, wa_m, wa_v;
    assign of_pop = of_out_valid & ~o_wr;
    wire        o_last = (e_out + 32'd1 == d_n);
    always @(posedge clk)
        if (!rst_n) begin
            e_out <= 32'd0; oi <= 3'd0; o_wr <= 1'b0; wr_valid <= 1'b0; ob <= 2'd0; ob_mask <= 8'd0;
        end else begin
            if (st == S_IDLE & start) begin
                e_out <= 32'd0; oi <= 3'd0; ob_mask <= 8'd0; wa_p <= d_p_base; wa_m <= d_m_base; wa_v <= d_v_base;
            end
            if (of_pop) begin
                ob_p[oi*32 +: 32] <= of_d[95:64]; ob_m[oi*32 +: 32] <= of_d[63:32]; ob_v[oi*32 +: 32] <= of_d[31:0];
                ob_mask[oi] <= 1'b1;
                e_out <= e_out + 32'd1; oi <= oi + 3'd1;
                if ((oi == 3'd7) | o_last) begin
                    o_wr <= 1'b1; ob <= 2'd0; wr_valid <= 1'b1; wr_addr <= wa_p; wr_data <= ob_p; wr_mask <= ob_mask;
                    wr_data[oi*32 +: 32] <= of_d[95:64]; wr_mask[oi] <= 1'b1;
                end
            end
            if (o_wr & wr_ready) begin
                case (ob)
                    2'd0: begin ob <= 2'd1; wr_addr <= wa_m; wr_data <= ob_m; end
                    2'd1: begin ob <= 2'd2; wr_addr <= wa_v; wr_data <= ob_v; end
                    default: begin
                        o_wr <= 1'b0; wr_valid <= 1'b0; oi <= 3'd0; ob_mask <= 8'd0;
                        wa_p <= wa_p + 32'd32; wa_m <= wa_m + 32'd32; wa_v <= wa_v + 32'd32;
                    end
                endcase
            end
        end
    // ------------------------------------------------------------------ FSM
    wire fin = (st == S_RUN) & (e_out == d_n) & ~o_wr;
    always @(posedge clk)
        if (!rst_n) begin st <= S_IDLE; sc_go <= 1'b0; done <= 1'b0; end
        else begin
            sc_go <= 1'b0; done <= 1'b0;
            case (st)
                S_IDLE: if (start) begin st <= S_SC; sc_go <= 1'b1; end
                S_SC: if (sc_done) st <= S_RUN;
                S_RUN: if (fin) begin st <= S_IDLE; done <= 1'b1; end
                default: st <= S_IDLE;
            endcase
        end
endmodule
`default_nettype wire
