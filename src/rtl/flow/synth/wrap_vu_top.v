`default_nettype none
// Synthesis/STA wrapper (not part of the design): registers every input of vu_top (the memory-side
// and descriptor inputs come from flops in P6) so all timing paths are register to register.
module wrap_vu_top #(parameter V = 16, parameter VW = 4, parameter NS = 8, parameter SW = 3) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          start,
    output wire          busy, done,
    input  wire [4:0]    d_vop,
    input  wire [15:0]   d_rows, d_cols, d_rows_per_b, d_emb_l,
    input  wire          d_st_en, d_x_reset, d_g_reset,
    input  wire [31:0]   d_a_base, d_a_rs, d_b_base, d_b_rs, d_c_base, d_c_rs, d_cv0_base, d_cv1_base, d_cv2_base, d_km_rs,
    input  wire [31:0]   d_st0_base, d_st1_base, d_o2_base, d_ids_base, d_mk_base, d_mk_rs, d_c0, d_c1, d_c2, d_c3, d_ds,
    output wire          rd0_req_valid, rd1_req_valid, mk_req_valid, wr_valid, rd0_dat_ready, rd1_dat_ready,
    output wire [31:0]   rd0_req_addr, rd1_req_addr, mk_req_idx, wr_addr,
    output wire [6:0]    rd0_req_len, rd1_req_len,
    output wire [255:0]  wr_data,
    output wire [7:0]    wr_mask,
    input  wire          rd0_req_ready, rd0_dat_valid, rd1_req_ready, rd1_dat_valid, mk_req_ready, mk_rsp_valid, wr_ready,
    input  wire [255:0]  rd0_dat, rd1_dat,
    input  wire [15:0]   mk_rsp_keep
);
    reg          start_q, d_st_en_q, d_x_reset_q, d_g_reset_q;
    reg  [4:0]   d_vop_q;
    reg  [15:0]  d_rows_q, d_cols_q, d_rows_per_b_q, d_emb_l_q;
    reg  [31:0]  a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20;
    reg          r0rr, r0dv, r1rr, r1dv, mkrr, mkrv, wrr;
    reg  [255:0] r0d, r1d;
    reg  [15:0]  mkk;
    always @(posedge clk) begin
        start_q <= start; d_st_en_q <= d_st_en; d_x_reset_q <= d_x_reset; d_g_reset_q <= d_g_reset; d_vop_q <= d_vop;
        d_rows_q <= d_rows; d_cols_q <= d_cols; d_rows_per_b_q <= d_rows_per_b; d_emb_l_q <= d_emb_l;
        a0 <= d_a_base; a1 <= d_a_rs; a2 <= d_b_base; a3 <= d_b_rs; a4 <= d_c_base; a5 <= d_c_rs; a6 <= d_cv0_base;
        a7 <= d_cv1_base; a8 <= d_cv2_base; a9 <= d_km_rs; a10 <= d_st0_base; a11 <= d_st1_base; a12 <= d_o2_base;
        a13 <= d_ids_base; a14 <= d_mk_base; a15 <= d_mk_rs; a16 <= d_c0; a17 <= d_c1; a18 <= d_c2; a19 <= d_c3; a20 <= d_ds;
        r0rr <= rd0_req_ready; r0dv <= rd0_dat_valid; r1rr <= rd1_req_ready; r1dv <= rd1_dat_valid; mkrr <= mk_req_ready;
        mkrv <= mk_rsp_valid; wrr <= wr_ready; r0d <= rd0_dat; r1d <= rd1_dat; mkk <= mk_rsp_keep;
    end
    vu_top #(.V(V), .VW(VW), .NS(NS), .SW(SW)) u (
        .clk(clk), .rst_n(rst_n), .start(start_q), .busy(busy), .done(done),
        .d_vop(d_vop_q), .d_rows(d_rows_q), .d_cols(d_cols_q), .d_rows_per_b(d_rows_per_b_q), .d_emb_l(d_emb_l_q),
        .d_st_en(d_st_en_q), .d_x_reset(d_x_reset_q), .d_g_reset(d_g_reset_q),
        .d_a_base(a0), .d_a_rs(a1), .d_b_base(a2), .d_b_rs(a3), .d_c_base(a4), .d_c_rs(a5),
        .d_cv0_base(a6), .d_cv1_base(a7), .d_cv2_base(a8), .d_km_rs(a9), .d_st0_base(a10), .d_st1_base(a11), .d_o2_base(a12),
        .d_ids_base(a13), .d_mk_base(a14), .d_mk_rs(a15), .d_c0(a16), .d_c1(a17), .d_c2(a18), .d_c3(a19), .d_ds(a20),
        .rd0_req_valid(rd0_req_valid), .rd0_req_ready(r0rr), .rd0_req_addr(rd0_req_addr), .rd0_req_len(rd0_req_len),
        .rd0_dat_valid(r0dv), .rd0_dat_ready(rd0_dat_ready), .rd0_dat(r0d),
        .rd1_req_valid(rd1_req_valid), .rd1_req_ready(r1rr), .rd1_req_addr(rd1_req_addr), .rd1_req_len(rd1_req_len),
        .rd1_dat_valid(r1dv), .rd1_dat_ready(rd1_dat_ready), .rd1_dat(r1d),
        .mk_req_valid(mk_req_valid), .mk_req_ready(mkrr), .mk_req_idx(mk_req_idx), .mk_rsp_valid(mkrv), .mk_rsp_keep(mkk),
        .wr_valid(wr_valid), .wr_ready(wrr), .wr_addr(wr_addr), .wr_data(wr_data), .wr_mask(wr_mask));
endmodule
`default_nettype wire
