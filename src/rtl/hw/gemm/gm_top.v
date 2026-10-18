`default_nettype none
// gm_top: the LoWA-TA GEMM engine (P3): a ROWS x COLS output-stationary array (gm_array of
// gm_pe), the X / W double-buffered tile feeders (gm_xfeed / gm_wfeed on mm_sram_bank), the
// tile loader (gm_loader), the chunk-sum drain with the ACC banks (gm_drain), the epilogue
// (gm_epilogue with gm_gelu's fp_sfu pool) and the tile-level controller (gm_ctrl).
// It executes one GEMM descriptor (d_*; held stable from `start` until `done`):
//   C[M x N] = A[M x K] * B^T (+ epilogue) over nb1 x nb2 batches, K = k0 + k1 + k2 segments
//   (GEMM_ACC2), A / B in ROW (lane = matrix row) or TRN (lane = matrix column) mode.
// Memory side (P6 attaches the DMA / AXI; the P3 testbench a memory model):
//   rd0: tile read requests (addr, len words) + 256-bit data beats in order
//   rd1: epilogue reads (bias, residual / u)         wr: 256-bit beats + word mask
//   mk : dropout keep bits (DW*LAMBDA per request, element index = C word offset)
// P5 attaches the merge engine to mg_* (one-word RMW on the next W buffer) + merge_busy.
// Every width is a plain constant expression of the parameters (docs/rtl_conventions.md).
module gm_top #(
    parameter ROWS       = 16,              // 4, 8 or 16 (multiple of LAMBDA)
    parameter COLS       = 16,              // 4, 8 or 16
    parameter LAMBDA     = 4,               // accumulators per PE = W rows per column (fixed: 4)
    parameter FMA_STAGES = 2,
    parameter NSFU       = 16,
    parameter NROB       = 64,
    parameter ROBW       = 6
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          start,
    output wire          busy,
    output wire          done,
    output wire [7:0]    err,               // sticky error flags (see below)
    output wire          arr_active,        // a pass is streaming (for utilisation counters)
    // descriptor
    input  wire [15:0]   d_m, d_n, d_k0, d_k1, d_k2, d_nb1, d_nb2,
    input  wire [31:0]   d_a_base0, d_a_base1, d_a_base2, d_a_rs, d_a_bs1, d_a_bs2,
    input  wire          d_a_trn,
    input  wire [31:0]   d_b_base0, d_b_base1, d_b_base2, d_b_rs, d_b_bs1, d_b_bs2,
    input  wire          d_b_trn,
    input  wire [31:0]   d_c_base, d_c_rs, d_c_bs1, d_c_bs2,
    input  wire [31:0]   d_bias_base, d_aux_base, d_o2_base,
    input  wire          d_ep_bias, d_ep_scale, d_ep_gelu, d_ep_gelu_bwd, d_ep_drop, d_ep_resid,
    input  wire [7:0]    d_ep_sexp,
    input  wire [31:0]   d_ep_ds,
    // memory read channel 0 (tiles)
    output wire          rd0_req_valid,
    input  wire          rd0_req_ready,
    output wire [31:0]   rd0_req_addr,
    output wire [6:0]    rd0_req_len,
    input  wire          rd0_dat_valid,
    output wire          rd0_dat_ready,
    input  wire [255:0]  rd0_dat,
    // memory read channel 1 (epilogue)
    output wire          rd1_req_valid,
    input  wire          rd1_req_ready,
    output wire [31:0]   rd1_req_addr,
    output wire [6:0]    rd1_req_len,
    input  wire          rd1_dat_valid,
    output wire          rd1_dat_ready,
    input  wire [255:0]  rd1_dat,
    // mask channel
    output wire          mk_req_valid,
    input  wire          mk_req_ready,
    output wire [31:0]   mk_req_idx,
    input  wire          mk_rsp_valid,
    input  wire [LAMBDA*((ROWS*COLS >= 64) ? ROWS*COLS/64 : 1)-1:0] mk_rsp_keep,
    // write channel
    output wire          wr_valid,
    input  wire          wr_ready,
    output wire [31:0]   wr_addr,
    output wire [255:0]  wr_data,
    output wire [7:0]    wr_mask,
    // merge engine hook (P5)
    input  wire          mg_en,
    input  wire          mg_buf,
    input  wire          mg_we,
    input  wire [((COLS > 8) ? (COLS > 16 ? 5 : 4) : 3)-1:0] mg_lane,
    input  wire [7:0]    mg_addr,
    input  wire [31:0]   mg_wdata,
    output wire [31:0]   mg_rdata,
    output wire          w_tile_done,
    output wire          w_tile_buf,
    input  wire [1:0]    merge_busy
);
    // ------------------------------------------------------------------ derived widths
    localparam QW   = 2;                                    // log2 LAMBDA (LAMBDA = 4)
    localparam KC   = 64, KW = 6;
    localparam BR   = 64, BRW = 6;
    localparam BC   = COLS * LAMBDA;
    localparam BCW  = (COLS == 4) ? 4 : (COLS == 8) ? 5 : (COLS == 16) ? 6 : 7;
    localparam RW   = (ROWS == 4) ? 2 : (ROWS == 8) ? 3 : (ROWS == 16) ? 4 : 5;
    localparam MTW  = BRW - RW;
    localparam XL   = 8, XLW = 3;
    localparam XAW  = MTW + RW + KW - XLW;                  // 9
    localparam WL   = (COLS > 8) ? COLS : 8;
    localparam WLW  = (COLS > 8) ? ((COLS > 16) ? 5 : 4) : 3;
    localparam WAW  = KW + QW;                              // 8
    localparam DW   = (ROWS * COLS >= KC) ? ROWS * COLS / KC : 1;
    localparam DWW  = (DW == 1) ? 1 : (DW == 2) ? 1 : (DW == 4) ? 2 : (DW == 8) ? 3 : 4;
    localparam GWW  = QW + ((DW == 1) ? 0 : (DW == 2) ? 1 : (DW == 4) ? 2 : (DW == 8) ? 3 : 4);
    localparam NG   = COLS / DW;
    localparam GW   = (NG <= 2) ? 1 : (NG == 4) ? 2 : (NG == 8) ? 3 : 4;
    localparam AW   = MTW + RW + GW + QW - 1;
    localparam RL   = ROWS * LAMBDA;
    localparam TOT  = NG * RL;
    localparam EW   = (TOT <= 64) ? 7 : (TOT <= 128) ? 8 : (TOT <= 256) ? 9 : (TOT <= 512) ? 10 : 11;
    localparam NW   = 7;
    localparam CW   = QW + 3;
    localparam GWD  = DW * LAMBDA;

    // ------------------------------------------------------------------ wires
    wire [ROWS*32-1:0]  a_row;
    wire [COLS*32-1:0]  b_col, dr_out;
    wire [COLS*CW-1:0]  ctl_col;
    wire [ROWS-1:0]     inj_row;
    wire [ROWS*QW-1:0]  inj_q;
    wire [COLS-1:0]     col_en;
    wire                err_pe, err_x, err_w, err_dr, err_ep;
    // reset tree: one registered copy of rst_n per sub-block (+ one for the GELU pool inside the
    // epilogue), each a kept-hierarchy cm_rstbuf instance so that synthesis cannot merge the
    // copies and no single reset net fans out to the whole engine (docs/p3_report.md section 6);
    // all blocks leave reset in the same cycle, one cycle after rst_n
    wire [7:0] rst_r;
    genvar rsti;
    generate
        for (rsti = 0; rsti < 8; rsti = rsti + 1) begin : g_rst
            cm_rstbuf u_rb (.clk(clk), .rst_n(rst_n), .rst_n_q(rst_r[rsti]));
        end
    endgenerate

    gm_array #(.ROWS(ROWS), .COLS(COLS), .LAMBDA(LAMBDA), .QW(QW), .FMA_STAGES(FMA_STAGES)) u_array (
        .clk(clk), .rst_n(rst_r[0]), .a_row(a_row), .b_col(b_col), .ctl_col(ctl_col),
        .inj_row(inj_row), .inj_q(inj_q), .col_en(col_en), .dr_out(dr_out), .err_overrun(err_pe));

    // feeders
    wire                 xw_valid, xw_buf, ww_valid, ww_buf;
    wire [XL*XLW-1:0]    xw_lane;
    wire [XL*XAW-1:0]    xw_addr;
    wire [XL*32-1:0]     xw_data;
    wire [XL-1:0]        xw_we;
    wire [8*WLW-1:0]     ww_lane;
    wire [8*WAW-1:0]     ww_addr;
    wire [8*32-1:0]      ww_data;
    wire [7:0]           ww_we;
    wire                 ps_start, ps_buf, x_idle, w_ready;
    wire [MTW-1:0]       ps_mt;
    wire [KW:0]          ps_kt;
    wire [1:0]           x_busy, w_busy, x_hold, w_hold;
    gm_xfeed #(.ROWS(ROWS), .LAMBDA(LAMBDA), .BR(BR), .KC(KC), .XL(XL), .XLW(XLW), .RW(RW),
               .MTW(MTW), .KW(KW), .AW(XAW)) u_xfeed (
        .clk(clk), .rst_n(rst_r[1]),
        .wr_valid(xw_valid), .wr_buf(xw_buf), .wr_lane(xw_lane), .wr_addr(xw_addr), .wr_data(xw_data), .wr_we(xw_we),
        .ps_start(ps_start), .ps_buf(ps_buf), .ps_mt(ps_mt), .ps_kt(ps_kt), .ps_idle(x_idle), .busy(x_busy),
        .a_row(a_row), .err_conflict(err_x));
    gm_wfeed #(.COLS(COLS), .LAMBDA(LAMBDA), .QW(QW), .KC(KC), .KW(KW), .WL(WL), .WLW(WLW), .AW(WAW)) u_wfeed (
        .clk(clk), .rst_n(rst_r[2]),
        .wr_valid(ww_valid), .wr_buf(ww_buf), .wr_lane(ww_lane), .wr_addr(ww_addr), .wr_data(ww_data), .wr_we(ww_we),
        .mg_en(mg_en), .mg_buf(mg_buf), .mg_we(mg_we), .mg_lane(mg_lane), .mg_addr(mg_addr), .mg_wdata(mg_wdata), .mg_rdata(mg_rdata),
        .ps_start(ps_start), .ps_buf(ps_buf), .ps_kt(ps_kt), .ps_ready(w_ready), .busy(w_busy),
        .b_col(b_col), .ctl_col(ctl_col), .err_conflict(err_w));
    assign arr_active = |w_busy;

    // loader
    wire                 tq_valid, tq_ready, tq_kind, tq_buf, tq_trn, tile_done, tile_done_kind, tile_done_buf;
    wire                 tile_done_l, tile_done_kind_l, tile_done_buf_l;
    wire [31:0]          tq_addr, tq_rs;
    wire [NW-1:0]        tq_n;
    wire [KW:0]          tq_kt;
    gm_loader #(.QW(QW), .KW(KW), .XL(XL), .XLW(XLW), .XAW(XAW), .RW(RW), .MTW(MTW), .WLW(WLW), .WAW(WAW), .NW(NW)) u_loader (
        .clk(clk), .rst_n(rst_r[3]),
        .tq_valid(tq_valid), .tq_ready(tq_ready), .tq_kind(tq_kind), .tq_buf(tq_buf), .tq_trn(tq_trn),
        .tq_addr(tq_addr), .tq_rs(tq_rs), .tq_n(tq_n), .tq_kt(tq_kt),
        .rd_req_valid(rd0_req_valid), .rd_req_ready(rd0_req_ready), .rd_req_addr(rd0_req_addr), .rd_req_len(rd0_req_len),
        .rd_dat_valid(rd0_dat_valid), .rd_dat_ready(rd0_dat_ready), .rd_dat(rd0_dat),
        .x_busy(x_busy | x_hold), .w_busy(w_busy | w_hold),
        .xw_valid(xw_valid), .xw_buf(xw_buf), .xw_lane(xw_lane), .xw_addr(xw_addr), .xw_data(xw_data), .xw_we(xw_we),
        .ww_valid(ww_valid), .ww_buf(ww_buf), .ww_lane(ww_lane), .ww_addr(ww_addr), .ww_data(ww_data), .ww_we(ww_we),
        .tile_done(tile_done_l), .tile_done_kind(tile_done_kind_l), .tile_done_buf(tile_done_buf_l));
    // P8c: the feeders register their write requests (one cycle), so tile_done is delayed by one cycle to
    // keep meaning "the last beat has been written" for gm_ctrl and the merge engine
    reg                  tile_done_q, tile_done_kind_q, tile_done_buf_q;
    always @(posedge clk) begin
        if (!rst_r[3]) tile_done_q <= 1'b0;
        else           tile_done_q <= tile_done_l;
        tile_done_kind_q <= tile_done_kind_l; tile_done_buf_q <= tile_done_buf_l;
    end
    assign tile_done = tile_done_q; assign tile_done_kind = tile_done_kind_q; assign tile_done_buf = tile_done_buf_q;
    assign w_tile_done = tile_done & tile_done_kind;
    assign w_tile_buf  = tile_done_buf;

    // drain + ACC
    wire                 dr_start, dr_half, dr_first, dr_last, dr_free;
    wire [MTW-1:0]       dr_mt;
    wire                 acc_en, acc_half, acc_par, fin_valid, fin_half;
    wire [1:0]           acc_block;
    wire [AW-1:0]        acc_addr;
    wire [DW*32-1:0]     acc_q;
    gm_drain #(.ROWS(ROWS), .COLS(COLS), .LAMBDA(LAMBDA), .QW(QW), .DW(DW), .NG(NG), .GW(GW), .RW(RW),
               .MTW(MTW), .AW(AW), .EW(EW)) u_drain (
        .clk(clk), .rst_n(rst_r[4]),
        .start(dr_start), .s_half(dr_half), .s_mt(dr_mt), .s_first(dr_first), .s_last(dr_last),
        .free(dr_free), .ep_block(acc_block),
        .fin_valid(fin_valid), .fin_half(fin_half),
        .inj_row(inj_row), .inj_q(inj_q), .col_en(col_en), .dr_out(dr_out),
        .ep_en(acc_en), .ep_half(acc_half), .ep_par(acc_par), .ep_addr(acc_addr), .ep_q(acc_q), .err(err_dr));

    // epilogue
    wire                 ob_valid, ob_ready, ob_half, ep_done, ep_done_half, ep_idle;
    wire [NW-1:0]        ob_rows, ob_cols;
    wire [31:0]          ob_c, ob_aux, ob_o2, ob_bias, ob_cidx;
    gm_epilogue #(.ROWS(ROWS), .LAMBDA(LAMBDA), .QW(QW), .DW(DW), .GW(GW), .RW(RW), .MTW(MTW), .AW(AW),
                  .BC(BC), .NW(NW), .NSFU(NSFU), .NROB(NROB), .ROBW(ROBW), .DWW(DWW), .GWW(GWW)) u_ep (
        .clk(clk), .rst_n(rst_r[5]), .rst_n_g(rst_r[7]),
        .ep_bias(d_ep_bias), .ep_scale(d_ep_scale), .ep_sexp(d_ep_sexp), .ep_gelu(d_ep_gelu),
        .ep_gelu_bwd(d_ep_gelu_bwd), .ep_drop(d_ep_drop), .ep_resid(d_ep_resid), .ep_ds(d_ep_ds),
        .ob_valid(ob_valid), .ob_ready(ob_ready), .ob_half(ob_half), .ob_rows(ob_rows), .ob_cols(ob_cols),
        .ob_c(ob_c), .ob_aux(ob_aux), .ob_o2(ob_o2), .ob_bias(ob_bias), .ob_cidx(ob_cidx), .ob_rs(d_c_rs),
        .fin_valid(fin_valid), .fin_half(fin_half), .ep_done(ep_done), .ep_done_half(ep_done_half), .idle(ep_idle),
        .acc_block(acc_block), .acc_en(acc_en), .acc_half(acc_half), .acc_par(acc_par), .acc_addr(acc_addr), .acc_q(acc_q),
        .rd_req_valid(rd1_req_valid), .rd_req_ready(rd1_req_ready), .rd_req_addr(rd1_req_addr), .rd_req_len(rd1_req_len),
        .rd_dat_valid(rd1_dat_valid), .rd_dat_ready(rd1_dat_ready), .rd_dat(rd1_dat),
        .mk_req_valid(mk_req_valid), .mk_req_ready(mk_req_ready), .mk_req_idx(mk_req_idx),
        .mk_rsp_valid(mk_rsp_valid), .mk_rsp_keep(mk_rsp_keep),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_addr(wr_addr), .wr_data(wr_data), .wr_mask(wr_mask),
        .err(err_ep));

    // controller
    gm_ctrl #(.ROWS(ROWS), .COLS(COLS), .LAMBDA(LAMBDA), .QW(QW), .DW(DW), .FMA_STAGES(FMA_STAGES), .BR(BR), .BRW(BRW),
              .BC(BC), .BCW(BCW), .KC(KC), .KW(KW), .XL(XL), .RW(RW), .MTW(MTW), .NW(NW)) u_ctrl (
        .clk(clk), .rst_n(rst_r[6]), .start(start), .busy(busy), .done(done),
        .d_m(d_m), .d_n(d_n), .d_k0(d_k0), .d_k1(d_k1), .d_k2(d_k2), .d_nb1(d_nb1), .d_nb2(d_nb2),
        .d_a_base0(d_a_base0), .d_a_base1(d_a_base1), .d_a_base2(d_a_base2), .d_a_rs(d_a_rs), .d_a_bs1(d_a_bs1), .d_a_bs2(d_a_bs2), .d_a_trn(d_a_trn),
        .d_b_base0(d_b_base0), .d_b_base1(d_b_base1), .d_b_base2(d_b_base2), .d_b_rs(d_b_rs), .d_b_bs1(d_b_bs1), .d_b_bs2(d_b_bs2), .d_b_trn(d_b_trn),
        .d_c_base(d_c_base), .d_c_rs(d_c_rs), .d_c_bs1(d_c_bs1), .d_c_bs2(d_c_bs2),
        .d_bias_base(d_bias_base), .d_aux_base(d_aux_base), .d_o2_base(d_o2_base),
        .tq_valid(tq_valid), .tq_ready(tq_ready), .tq_kind(tq_kind), .tq_buf(tq_buf), .tq_trn(tq_trn),
        .tq_addr(tq_addr), .tq_rs(tq_rs), .tq_n(tq_n), .tq_kt(tq_kt),
        .tile_done(tile_done), .tile_done_kind(tile_done_kind), .tile_done_buf(tile_done_buf), .merge_busy(merge_busy),
        .x_hold(x_hold), .w_hold(w_hold),
        .ps_start(ps_start), .ps_buf(ps_buf), .ps_mt(ps_mt), .ps_kt(ps_kt), .x_idle(x_idle), .w_ready(w_ready),
        .dr_start(dr_start), .dr_half(dr_half), .dr_mt(dr_mt), .dr_first(dr_first), .dr_last(dr_last),
        .dr_free(dr_free),
        .ob_valid(ob_valid), .ob_ready(ob_ready), .ob_half(ob_half), .ob_rows(ob_rows), .ob_cols(ob_cols),
        .ob_c(ob_c), .ob_aux(ob_aux), .ob_o2(ob_o2), .ob_bias(ob_bias), .ob_cidx(ob_cidx),
        .ep_done(ep_done), .ep_done_half(ep_done_half), .ep_idle(ep_idle));

    assign err = {3'd0, err_ep, err_dr, err_w, err_x, err_pe};
    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_p = (GWD == 0);
    /* verilator lint_on UNUSEDSIGNAL */
endmodule
`default_nettype wire
