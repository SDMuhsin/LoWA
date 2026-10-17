`default_nettype none
// Synthesis/STA wrapper (not part of the design): registers every input of gm_top (the memory-side
// and descriptor inputs come from flops in P6) so all timing paths are register to register.
module wrap_gm_top #(parameter ROWS = 16, parameter COLS = 16, parameter NSFU = 16) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          start,
    output wire          busy, done, arr_active,
    output wire [7:0]    err,
    input  wire [15:0]   d_m, d_n, d_k0, d_k1, d_k2, d_nb1, d_nb2,
    input  wire [31:0]   d_a_base0, d_a_base1, d_a_base2, d_a_rs, d_a_bs1, d_a_bs2,
    input  wire [31:0]   d_b_base0, d_b_base1, d_b_base2, d_b_rs, d_b_bs1, d_b_bs2,
    input  wire [31:0]   d_c_base, d_c_rs, d_c_bs1, d_c_bs2, d_bias_base, d_aux_base, d_o2_base, d_ep_ds,
    input  wire          d_a_trn, d_b_trn, d_ep_bias, d_ep_scale, d_ep_gelu, d_ep_gelu_bwd, d_ep_drop, d_ep_resid,
    input  wire [7:0]    d_ep_sexp,
    output wire          rd0_req_valid, rd1_req_valid, mk_req_valid, wr_valid, rd0_dat_ready, rd1_dat_ready,
    output wire [31:0]   rd0_req_addr, rd1_req_addr, mk_req_idx, wr_addr, mg_rdata,
    output wire [6:0]    rd0_req_len, rd1_req_len,
    output wire [255:0]  wr_data,
    output wire [7:0]    wr_mask,
    output wire          w_tile_done, w_tile_buf,
    input  wire          rd0_req_ready, rd0_dat_valid, rd1_req_ready, rd1_dat_valid, mk_req_ready, mk_rsp_valid, wr_ready,
    input  wire [255:0]  rd0_dat, rd1_dat,
    input  wire [4*((ROWS*COLS >= 64) ? ROWS*COLS/64 : 1)-1:0] mk_rsp_keep,
    input  wire          mg_en, mg_buf, mg_we,
    input  wire [((COLS > 8) ? (COLS > 16 ? 5 : 4) : 3)-1:0] mg_lane,
    input  wire [7:0]    mg_addr,
    input  wire [31:0]   mg_wdata,
    input  wire [1:0]    merge_busy
);
    localparam MKW = 4 * ((ROWS*COLS >= 64) ? ROWS*COLS/64 : 1);
    localparam MGW = (COLS > 8) ? (COLS > 16 ? 5 : 4) : 3;
    reg          start_r, rst_r;
    reg [15:0]   m_r, n_r, k0_r, k1_r, k2_r, nb1_r, nb2_r;
    reg [31:0]   ab0_r, ab1_r, ab2_r, ars_r, abs1_r, abs2_r, bb0_r, bb1_r, bb2_r, brs_r, bbs1_r, bbs2_r;
    reg [31:0]   cb_r, crs_r, cbs1_r, cbs2_r, bias_r, aux_r, o2_r, ds_r;
    reg          atrn_r, btrn_r, eb_r, es_r, eg_r, egb_r, ed_r, er_r;
    reg [7:0]    sexp_r;
    reg          r0rr_r, r0dv_r, r1rr_r, r1dv_r, mkrr_r, mkrv_r, wrr_r, mge_r, mgb_r, mgw_r;
    reg [255:0]  r0d_r, r1d_r;
    reg [MKW-1:0] mkk_r;
    reg [MGW-1:0] mgl_r;
    reg [7:0]    mga_r;
    reg [31:0]   mgd_r;
    reg [1:0]    mb_r;
    always @(posedge clk) begin
        start_r <= start; rst_r <= rst_n; m_r <= d_m; n_r <= d_n; k0_r <= d_k0; k1_r <= d_k1; k2_r <= d_k2; nb1_r <= d_nb1; nb2_r <= d_nb2;
        ab0_r <= d_a_base0; ab1_r <= d_a_base1; ab2_r <= d_a_base2; ars_r <= d_a_rs; abs1_r <= d_a_bs1; abs2_r <= d_a_bs2;
        bb0_r <= d_b_base0; bb1_r <= d_b_base1; bb2_r <= d_b_base2; brs_r <= d_b_rs; bbs1_r <= d_b_bs1; bbs2_r <= d_b_bs2;
        cb_r <= d_c_base; crs_r <= d_c_rs; cbs1_r <= d_c_bs1; cbs2_r <= d_c_bs2; bias_r <= d_bias_base; aux_r <= d_aux_base;
        o2_r <= d_o2_base; ds_r <= d_ep_ds; atrn_r <= d_a_trn; btrn_r <= d_b_trn; eb_r <= d_ep_bias; es_r <= d_ep_scale;
        eg_r <= d_ep_gelu; egb_r <= d_ep_gelu_bwd; ed_r <= d_ep_drop; er_r <= d_ep_resid; sexp_r <= d_ep_sexp;
        r0rr_r <= rd0_req_ready; r0dv_r <= rd0_dat_valid; r1rr_r <= rd1_req_ready; r1dv_r <= rd1_dat_valid;
        mkrr_r <= mk_req_ready; mkrv_r <= mk_rsp_valid; wrr_r <= wr_ready; r0d_r <= rd0_dat; r1d_r <= rd1_dat;
        mkk_r <= mk_rsp_keep; mge_r <= mg_en; mgb_r <= mg_buf; mgw_r <= mg_we; mgl_r <= mg_lane; mga_r <= mg_addr;
        mgd_r <= mg_wdata; mb_r <= merge_busy;
    end
    (* keep_hierarchy *) gm_top #(.ROWS(ROWS), .COLS(COLS), .NSFU(NSFU)) u_dut (
        .clk(clk), .rst_n(rst_r), .start(start_r), .busy(busy), .done(done), .err(err), .arr_active(arr_active),
        .d_m(m_r), .d_n(n_r), .d_k0(k0_r), .d_k1(k1_r), .d_k2(k2_r), .d_nb1(nb1_r), .d_nb2(nb2_r),
        .d_a_base0(ab0_r), .d_a_base1(ab1_r), .d_a_base2(ab2_r), .d_a_rs(ars_r), .d_a_bs1(abs1_r), .d_a_bs2(abs2_r), .d_a_trn(atrn_r),
        .d_b_base0(bb0_r), .d_b_base1(bb1_r), .d_b_base2(bb2_r), .d_b_rs(brs_r), .d_b_bs1(bbs1_r), .d_b_bs2(bbs2_r), .d_b_trn(btrn_r),
        .d_c_base(cb_r), .d_c_rs(crs_r), .d_c_bs1(cbs1_r), .d_c_bs2(cbs2_r), .d_bias_base(bias_r), .d_aux_base(aux_r), .d_o2_base(o2_r),
        .d_ep_bias(eb_r), .d_ep_scale(es_r), .d_ep_gelu(eg_r), .d_ep_gelu_bwd(egb_r), .d_ep_drop(ed_r), .d_ep_resid(er_r),
        .d_ep_sexp(sexp_r), .d_ep_ds(ds_r),
        .rd0_req_valid(rd0_req_valid), .rd0_req_ready(r0rr_r), .rd0_req_addr(rd0_req_addr), .rd0_req_len(rd0_req_len),
        .rd0_dat_valid(r0dv_r), .rd0_dat_ready(rd0_dat_ready), .rd0_dat(r0d_r),
        .rd1_req_valid(rd1_req_valid), .rd1_req_ready(r1rr_r), .rd1_req_addr(rd1_req_addr), .rd1_req_len(rd1_req_len),
        .rd1_dat_valid(r1dv_r), .rd1_dat_ready(rd1_dat_ready), .rd1_dat(r1d_r),
        .mk_req_valid(mk_req_valid), .mk_req_ready(mkrr_r), .mk_req_idx(mk_req_idx), .mk_rsp_valid(mkrv_r), .mk_rsp_keep(mkk_r),
        .wr_valid(wr_valid), .wr_ready(wrr_r), .wr_addr(wr_addr), .wr_data(wr_data), .wr_mask(wr_mask),
        .mg_en(mge_r), .mg_buf(mgb_r), .mg_we(mgw_r), .mg_lane(mgl_r), .mg_addr(mga_r), .mg_wdata(mgd_r), .mg_rdata(mg_rdata),
        .w_tile_done(w_tile_done), .w_tile_buf(w_tile_buf), .merge_busy(mb_r));
endmodule
`default_nettype wire
