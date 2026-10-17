`default_nettype none
// lw_top: the LoWA unit (P5): descriptor-level controller over lw_sg (LOWA_SG), lw_scal (LOWA_SCAL),
// lw_adamw (ADAMW) on the main queue (one op at a time, memory channels rd0 / rd1 / wr as the GEMM and
// vector units) and lw_merge on its own queue (m_*), which runs beside a GEMM descriptor with
// MERGE != none through rd2 and the GEMM engine's mg_* / w_tile_done / merge_busy ports.
// Main descriptor (d_op): 0 LOWA_SG  a0 dY base, a1 dY row stride, a2 X base, a3 X row stride, a4 centres,
//                           a5 h taps, a6 grad out, a7 gamma, len[15:0] tokens
//                         1 LOWA_SCAL a0 s, a1 warmup, a2 T, a3 CONST block, a4 OPT_SCAL block
//                         2 ADAMW    a0 p, a1 g, a2 m, a3 v, a4 scalar block, len n, flag clip_en
// Merge descriptor: see lw_merge.  Resets: one cm_rstbuf per engine.
module lw_top #(
    parameter NL  = 4,                   // sampled-gradient lanes
    parameter NSQ = 2,                   // AdamW sqrt SFUs
    parameter NRC = 1                    // AdamW recip SFUs
) (
    input  wire         clk,
    input  wire         rst_n,
    // main queue
    input  wire         start,
    output wire         busy,
    output wire         done,
    input  wire [1:0]   d_op,
    input  wire [31:0]  d_a0, d_a1, d_a2, d_a3, d_a4, d_a5, d_a6, d_a7, d_len,
    input  wire         d_flag,
    // merge queue
    input  wire         m_start,
    output wire         m_busy,
    output wire         m_done,
    input  wire         m_trn,
    input  wire [7:0]   m_nb, m_mb, m_kc0, m_kc1, m_kc2,
    input  wire [1:0]   m_ms0, m_ms1, m_ms2,
    input  wire [31:0]  m_hits_a, m_dir_a, m_coef_a, m_hits_b, m_dir_b, m_coef_b, m_h_base, m_gamma,
    output wire [15:0]  m_stat_tiles,
    // memory channels
    output wire         rd0_req_valid,
    input  wire         rd0_req_ready,
    output wire [31:0]  rd0_req_addr,
    output wire [6:0]   rd0_req_len,
    input  wire         rd0_dat_valid,
    output wire         rd0_dat_ready,
    input  wire [255:0] rd0_dat,
    output wire         rd1_req_valid,
    input  wire         rd1_req_ready,
    output wire [31:0]  rd1_req_addr,
    output wire [6:0]   rd1_req_len,
    input  wire         rd1_dat_valid,
    output wire         rd1_dat_ready,
    input  wire [255:0] rd1_dat,
    output wire         rd2_req_valid,
    input  wire         rd2_req_ready,
    output wire [31:0]  rd2_req_addr,
    output wire [6:0]   rd2_req_len,
    input  wire         rd2_dat_valid,
    output wire         rd2_dat_ready,
    input  wire [255:0] rd2_dat,
    output wire         wr_valid,
    input  wire         wr_ready,
    output wire [31:0]  wr_addr,
    output wire [255:0] wr_data,
    output wire [7:0]   wr_mask,
    // GEMM engine merge port
    output wire         mg_en,
    output wire         mg_buf,
    output wire         mg_we,
    output wire [3:0]   mg_lane,
    output wire [7:0]   mg_addr,
    output wire [31:0]  mg_wdata,
    input  wire [31:0]  mg_rdata,
    input  wire         w_tile_done,
    input  wire         w_tile_buf,
    output wire [1:0]   merge_busy
);
    wire rst_sg, rst_sc, rst_ad, rst_mg;
    cm_rstbuf u_r0 (.clk(clk), .rst_n(rst_n), .rst_n_q(rst_sg));
    cm_rstbuf u_r1 (.clk(clk), .rst_n(rst_n), .rst_n_q(rst_sc));
    cm_rstbuf u_r2 (.clk(clk), .rst_n(rst_n), .rst_n_q(rst_ad));
    cm_rstbuf u_r3 (.clk(clk), .rst_n(rst_n), .rst_n_q(rst_mg));
    // ------------------------------------------------------------------ main queue
    reg  [1:0]  cur;
    reg         act;
    wire        sg_busy, sg_done, sc_busy, sc_done, ad_busy, ad_done;
    wire        go_sg = start & ~act & (d_op == 2'd0);
    wire        go_sc = start & ~act & (d_op == 2'd1);
    wire        go_ad = start & ~act & (d_op == 2'd2);
    assign busy = act;
    assign done = (cur == 2'd0) ? sg_done : (cur == 2'd1) ? sc_done : ad_done;
    always @(posedge clk)
        if (!rst_n) begin act <= 1'b0; cur <= 2'd0; end
        else begin
            if (start & ~act) begin act <= 1'b1; cur <= d_op; end
            else if (act & done) act <= 1'b0;
        end
    wire        is_sg = (cur == 2'd0), is_sc = (cur == 2'd1), is_ad = (cur == 2'd2);
    // channel muxes
    wire        sg_r0v, sg_r0dr, sg_r1v, sg_r1dr, sg_wv, sc_r0v, sc_r0dr, sc_wv, ad_r0v, ad_r0dr, ad_r1v, ad_r1dr, ad_wv;
    wire [31:0] sg_r0a, sg_r1a, sg_wa, sc_r0a, sc_wa, ad_r0a, ad_r1a, ad_wa;
    wire [6:0]  sg_r0l, sg_r1l, sc_r0l, ad_r0l, ad_r1l;
    wire [255:0] sg_wd, sc_wd, ad_wd;
    wire [7:0]  sg_wm, sc_wm, ad_wm;
    assign rd0_req_valid = (is_sg & sg_r0v) | (is_sc & sc_r0v) | (is_ad & ad_r0v);
    assign rd0_req_addr  = is_sg ? sg_r0a : is_sc ? sc_r0a : ad_r0a;
    assign rd0_req_len   = is_sg ? sg_r0l : is_sc ? sc_r0l : ad_r0l;
    assign rd0_dat_ready = is_sg ? sg_r0dr : is_sc ? sc_r0dr : ad_r0dr;
    assign rd1_req_valid = (is_sg & sg_r1v) | (is_ad & ad_r1v);
    assign rd1_req_addr  = is_sg ? sg_r1a : ad_r1a;
    assign rd1_req_len   = is_sg ? sg_r1l : ad_r1l;
    assign rd1_dat_ready = is_sg ? sg_r1dr : ad_r1dr;
    assign wr_valid = (is_sg & sg_wv) | (is_sc & sc_wv) | (is_ad & ad_wv);
    assign wr_addr  = is_sg ? sg_wa : is_sc ? sc_wa : ad_wa;
    assign wr_data  = is_sg ? sg_wd : is_sc ? sc_wd : ad_wd;
    assign wr_mask  = is_sg ? sg_wm : is_sc ? sc_wm : ad_wm;
    lw_sg #(.D(768), .NL(NL)) u_sg (
        .clk(clk), .rst_n(rst_sg), .start(go_sg), .busy(sg_busy), .done(sg_done),
        .d_dy_base(d_a0), .d_dy_rs(d_a1), .d_x_base(d_a2), .d_x_rs(d_a3), .d_ctr_base(d_a4), .d_h_base(d_a5),
        .d_gamma(d_a7), .d_grad_base(d_a6), .d_tok(d_len[15:0]),
        .rd0_req_valid(sg_r0v), .rd0_req_ready(rd0_req_ready & is_sg), .rd0_req_addr(sg_r0a), .rd0_req_len(sg_r0l),
        .rd0_dat_valid(rd0_dat_valid & is_sg), .rd0_dat_ready(sg_r0dr), .rd0_dat(rd0_dat),
        .rd1_req_valid(sg_r1v), .rd1_req_ready(rd1_req_ready & is_sg), .rd1_req_addr(sg_r1a), .rd1_req_len(sg_r1l),
        .rd1_dat_valid(rd1_dat_valid & is_sg), .rd1_dat_ready(sg_r1dr), .rd1_dat(rd1_dat),
        .wr_valid(sg_wv), .wr_ready(wr_ready & is_sg), .wr_addr(sg_wa), .wr_data(sg_wd), .wr_mask(sg_wm));
    lw_scal u_sc (
        .clk(clk), .rst_n(rst_sc), .start(go_sc), .busy(sc_busy), .done(sc_done),
        .d_s(d_a0), .d_w(d_a1), .d_t(d_a2), .d_const_base(d_a3), .d_scal_base(d_a4),
        .rd_req_valid(sc_r0v), .rd_req_ready(rd0_req_ready & is_sc), .rd_req_addr(sc_r0a), .rd_req_len(sc_r0l),
        .rd_dat_valid(rd0_dat_valid & is_sc), .rd_dat_ready(sc_r0dr), .rd_dat(rd0_dat),
        .wr_valid(sc_wv), .wr_ready(wr_ready & is_sc), .wr_addr(sc_wa), .wr_data(sc_wd), .wr_mask(sc_wm));
    lw_adamw #(.NSQ(NSQ), .NRC(NRC)) u_ad (
        .clk(clk), .rst_n(rst_ad), .start(go_ad), .busy(ad_busy), .done(ad_done),
        .d_p_base(d_a0), .d_g_base(d_a1), .d_m_base(d_a2), .d_v_base(d_a3), .d_n(d_len), .d_sc_base(d_a4), .d_clip_en(d_flag),
        .rd0_req_valid(ad_r0v), .rd0_req_ready(rd0_req_ready & is_ad), .rd0_req_addr(ad_r0a), .rd0_req_len(ad_r0l),
        .rd0_dat_valid(rd0_dat_valid & is_ad), .rd0_dat_ready(ad_r0dr), .rd0_dat(rd0_dat),
        .rd1_req_valid(ad_r1v), .rd1_req_ready(rd1_req_ready & is_ad), .rd1_req_addr(ad_r1a), .rd1_req_len(ad_r1l),
        .rd1_dat_valid(rd1_dat_valid & is_ad), .rd1_dat_ready(ad_r1dr), .rd1_dat(rd1_dat),
        .wr_valid(ad_wv), .wr_ready(wr_ready & is_ad), .wr_addr(ad_wa), .wr_data(ad_wd), .wr_mask(ad_wm));
    wire unused_b = sg_busy | sc_busy | ad_busy;
    // ------------------------------------------------------------------ merge queue
    lw_merge u_mg (
        .clk(clk), .rst_n(rst_mg), .start(m_start), .busy(m_busy), .done(m_done),
        .d_trn(m_trn), .d_nb(m_nb), .d_mb(m_mb), .d_kc0(m_kc0), .d_kc1(m_kc1), .d_kc2(m_kc2),
        .d_ms0(m_ms0), .d_ms1(m_ms1), .d_ms2(m_ms2),
        .d_hits_a(m_hits_a), .d_dir_a(m_dir_a), .d_coef_a(m_coef_a), .d_hits_b(m_hits_b), .d_dir_b(m_dir_b), .d_coef_b(m_coef_b),
        .d_h_base(m_h_base), .d_gamma(m_gamma),
        .rd_req_valid(rd2_req_valid), .rd_req_ready(rd2_req_ready), .rd_req_addr(rd2_req_addr), .rd_req_len(rd2_req_len),
        .rd_dat_valid(rd2_dat_valid), .rd_dat_ready(rd2_dat_ready), .rd_dat(rd2_dat),
        .mg_en(mg_en), .mg_buf(mg_buf), .mg_we(mg_we), .mg_lane(mg_lane), .mg_addr(mg_addr), .mg_wdata(mg_wdata), .mg_rdata(mg_rdata),
        .w_tile_done(w_tile_done), .w_tile_buf(w_tile_buf), .merge_busy(merge_busy), .stat_tiles(m_stat_tiles));
endmodule
`default_nettype wire
