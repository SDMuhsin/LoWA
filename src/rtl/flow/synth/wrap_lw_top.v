`default_nettype none
// Synthesis/STA wrapper (not part of the design): registers every input of lw_top so that all timing
// paths are register to register (the memory-side and descriptor inputs come from flops in P6).
module wrap_lw_top #(parameter NL = 4, parameter NSQ = 2, parameter NRC = 1) (
    input  wire          clk, rst_n, start, d_flag, m_start, m_trn,
    input  wire [1:0]    d_op, m_ms0, m_ms1, m_ms2,
    input  wire [31:0]   d_a0, d_a1, d_a2, d_a3, d_a4, d_a5, d_a6, d_a7, d_len,
    input  wire [7:0]    m_nb, m_mb, m_kc0, m_kc1, m_kc2,
    input  wire [31:0]   m_hits_a, m_dir_a, m_coef_a, m_hits_b, m_dir_b, m_coef_b, m_h_base, m_gamma,
    output wire          busy, done, m_busy, m_done,
    output wire [15:0]   m_stat_tiles,
    output wire          rd0_req_valid, rd1_req_valid, rd2_req_valid, rd0_dat_ready, rd1_dat_ready, rd2_dat_ready, wr_valid,
    output wire [31:0]   rd0_req_addr, rd1_req_addr, rd2_req_addr, wr_addr,
    output wire [6:0]    rd0_req_len, rd1_req_len, rd2_req_len,
    output wire [255:0]  wr_data,
    output wire [7:0]    wr_mask,
    input  wire          rd0_req_ready, rd0_dat_valid, rd1_req_ready, rd1_dat_valid, rd2_req_ready, rd2_dat_valid, wr_ready,
    input  wire [255:0]  rd0_dat, rd1_dat, rd2_dat,
    output wire          mg_en, mg_buf, mg_we,
    output wire [3:0]    mg_lane,
    output wire [7:0]    mg_addr,
    output wire [31:0]   mg_wdata,
    input  wire [31:0]   mg_rdata,
    input  wire          w_tile_done, w_tile_buf,
    output wire [1:0]    merge_busy
);
    reg          start_q, d_flag_q, m_start_q, m_trn_q, r0rr, r0dv, r1rr, r1dv, r2rr, r2dv, wrr, wtd, wtb;
    reg  [1:0]   d_op_q, ms0, ms1, ms2;
    reg  [31:0]  a0, a1, a2, a3, a4, a5, a6, a7, ln, ha, da, ca, hb, db, cb, hh, gm, mgr;
    reg  [7:0]   nb, mb, k0, k1, k2;
    reg  [255:0] r0d, r1d, r2d;
    always @(posedge clk) begin
        start_q <= start; d_flag_q <= d_flag; m_start_q <= m_start; m_trn_q <= m_trn; d_op_q <= d_op; ms0 <= m_ms0; ms1 <= m_ms1; ms2 <= m_ms2;
        a0 <= d_a0; a1 <= d_a1; a2 <= d_a2; a3 <= d_a3; a4 <= d_a4; a5 <= d_a5; a6 <= d_a6; a7 <= d_a7; ln <= d_len;
        nb <= m_nb; mb <= m_mb; k0 <= m_kc0; k1 <= m_kc1; k2 <= m_kc2;
        ha <= m_hits_a; da <= m_dir_a; ca <= m_coef_a; hb <= m_hits_b; db <= m_dir_b; cb <= m_coef_b; hh <= m_h_base; gm <= m_gamma;
        r0rr <= rd0_req_ready; r0dv <= rd0_dat_valid; r1rr <= rd1_req_ready; r1dv <= rd1_dat_valid; r2rr <= rd2_req_ready; r2dv <= rd2_dat_valid;
        wrr <= wr_ready; r0d <= rd0_dat; r1d <= rd1_dat; r2d <= rd2_dat; mgr <= mg_rdata; wtd <= w_tile_done; wtb <= w_tile_buf;
    end
    lw_top #(.NL(NL), .NSQ(NSQ), .NRC(NRC)) u (
        .clk(clk), .rst_n(rst_n), .start(start_q), .busy(busy), .done(done), .d_op(d_op_q),
        .d_a0(a0), .d_a1(a1), .d_a2(a2), .d_a3(a3), .d_a4(a4), .d_a5(a5), .d_a6(a6), .d_a7(a7), .d_len(ln), .d_flag(d_flag_q),
        .m_start(m_start_q), .m_busy(m_busy), .m_done(m_done), .m_trn(m_trn_q), .m_nb(nb), .m_mb(mb), .m_kc0(k0), .m_kc1(k1), .m_kc2(k2),
        .m_ms0(ms0), .m_ms1(ms1), .m_ms2(ms2), .m_hits_a(ha), .m_dir_a(da), .m_coef_a(ca), .m_hits_b(hb), .m_dir_b(db), .m_coef_b(cb),
        .m_h_base(hh), .m_gamma(gm), .m_stat_tiles(m_stat_tiles),
        .rd0_req_valid(rd0_req_valid), .rd0_req_ready(r0rr), .rd0_req_addr(rd0_req_addr), .rd0_req_len(rd0_req_len),
        .rd0_dat_valid(r0dv), .rd0_dat_ready(rd0_dat_ready), .rd0_dat(r0d),
        .rd1_req_valid(rd1_req_valid), .rd1_req_ready(r1rr), .rd1_req_addr(rd1_req_addr), .rd1_req_len(rd1_req_len),
        .rd1_dat_valid(r1dv), .rd1_dat_ready(rd1_dat_ready), .rd1_dat(r1d),
        .rd2_req_valid(rd2_req_valid), .rd2_req_ready(r2rr), .rd2_req_addr(rd2_req_addr), .rd2_req_len(rd2_req_len),
        .rd2_dat_valid(r2dv), .rd2_dat_ready(rd2_dat_ready), .rd2_dat(r2d),
        .wr_valid(wr_valid), .wr_ready(wrr), .wr_addr(wr_addr), .wr_data(wr_data), .wr_mask(wr_mask),
        .mg_en(mg_en), .mg_buf(mg_buf), .mg_we(mg_we), .mg_lane(mg_lane), .mg_addr(mg_addr), .mg_wdata(mg_wdata), .mg_rdata(mgr),
        .w_tile_done(wtd), .w_tile_buf(wtb), .merge_busy(merge_busy));
endmodule
`default_nettype wire
