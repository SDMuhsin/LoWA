`default_nettype none
// top_chip: the LoWA-TA chip (P7a).  Sequencer (ct_seq + ct_dma) + GEMM engine (gm_top, ROWS x COLS) + vector
// unit (vu_top) + LoWA unit (lw_top: sampled gradient, LOWA_SCAL, AdamW, merge-on-load) + memory system (mm_top
// with the on-chip Threefry dropout-mask generators, MK_DRAM = 0).
// Chip boundary (nothing else crosses it): clk, rst_n (synchronous, active low), irq (level, ct_seq), one AXI4-Lite
// slave (host CSRs, top_axil -> ct_seq CSR bus, docs/isa.md section 7) and one AXI4 master (device DRAM, 256-bit,
// PLAN ruling R3).
// mm_top channel map (ARID / AWID = channel):
//   read  0 gm rd0 (tile stream, RFL0)  1 gm rd1 (epilogue)  2 vu rd0  3 vu rd1  4 lw rd0  5 lw rd1  6 lw rd2 (merge)
//         7 ct (descriptor fetch + DMA_COPY)
//   write 0 gm  1 vu  2 lw  3 ct (DMA_COPY)
//   keep  0 gm epilogue (site / layer = GEMM payload word 4 [11:8] / word 5 [3:0], MKNC0 Threefry pipelines)
//         1 vu (VEC payload word 0 [15:12] / word 1 [3:0], MKNC pipelines)
// Engine descriptor fields are the payload words held by ct_seq until retire, in sw/isa.py field order.
module top_chip #(
    parameter ROWS = 16,
    parameter COLS = 16,
    parameter NSFU = 16
) (
    input  wire          clk,
    input  wire          rst_n,
    output wire          irq,
    // AXI4-Lite slave: host CSRs
    input  wire          s_awvalid,
    output wire          s_awready,
    input  wire [11:0]   s_awaddr,
    input  wire          s_wvalid,
    output wire          s_wready,
    input  wire [31:0]   s_wdata,
    input  wire [3:0]    s_wstrb,
    output wire          s_bvalid,
    input  wire          s_bready,
    output wire [1:0]    s_bresp,
    input  wire          s_arvalid,
    output wire          s_arready,
    input  wire [11:0]   s_araddr,
    output wire          s_rvalid,
    input  wire          s_rready,
    output wire [31:0]   s_rdata,
    output wire [1:0]    s_rresp,
    // AXI4 master: device DRAM
    output wire          m_arvalid,
    input  wire          m_arready,
    output wire [3:0]    m_arid,
    output wire [31:0]   m_araddr,
    output wire [7:0]    m_arlen,
    output wire [2:0]    m_arsize,
    output wire [1:0]    m_arburst,
    input  wire          m_rvalid,
    output wire          m_rready,
    input  wire [3:0]    m_rid,
    input  wire [255:0]  m_rdata,
    input  wire [1:0]    m_rresp,
    input  wire          m_rlast,
    output wire          m_awvalid,
    input  wire          m_awready,
    output wire [3:0]    m_awid,
    output wire [31:0]   m_awaddr,
    output wire [7:0]    m_awlen,
    output wire [2:0]    m_awsize,
    output wire [1:0]    m_awburst,
    output wire          m_wvalid,
    input  wire          m_wready,
    output wire [255:0]  m_wdata,
    output wire [31:0]   m_wstrb,
    output wire          m_wlast,
    input  wire          m_bvalid,
    output wire          m_bready,
    input  wire [3:0]    m_bid,
    input  wire [1:0]    m_bresp
);
    localparam GP = 36, VP = 25, LP = 10;
    localparam NR = 8, NM = 2, NW = 4;
    localparam GKW = 4 * ((ROWS * COLS >= 64) ? ROWS * COLS / 64 : 1);   // gm keep bits per response

    // ---------------------------------------------------------------- host CSR bus
    wire        csr_we, csr_re;
    wire [11:0] csr_addr;
    wire [31:0] csr_wdata, csr_rdata;
    top_axil u_axil (
        .clk(clk), .rst_n(rst_n),
        .s_awvalid(s_awvalid), .s_awready(s_awready), .s_awaddr(s_awaddr), .s_wvalid(s_wvalid), .s_wready(s_wready),
        .s_wdata(s_wdata), .s_wstrb(s_wstrb), .s_bvalid(s_bvalid), .s_bready(s_bready), .s_bresp(s_bresp),
        .s_arvalid(s_arvalid), .s_arready(s_arready), .s_araddr(s_araddr), .s_rvalid(s_rvalid), .s_rready(s_rready),
        .s_rdata(s_rdata), .s_rresp(s_rresp),
        .csr_we(csr_we), .csr_re(csr_re), .csr_addr(csr_addr), .csr_wdata(csr_wdata), .csr_rdata(csr_rdata));

    // ---------------------------------------------------------------- memory system channel bundles
    wire [NR-1:0]     rd_req_valid, rd_req_ready, rd_dat_valid, rd_dat_ready;
    wire [32*NR-1:0]  rd_req_addr;
    wire [7*NR-1:0]   rd_req_len;
    wire [256*NR-1:0] rd_dat;
    wire [NM-1:0]     mk_req_valid, mk_req_ready, mk_rsp_valid;
    wire [32*NM-1:0]  mk_req_idx;
    wire [16*NM-1:0]  mk_rsp_keep;
    wire [4*NM-1:0]   mk_site, mk_layer;
    wire [NW-1:0]     wr_valid, wr_ready, wr_idle;
    wire [32*NW-1:0]  wr_addr;
    wire [256*NW-1:0] wr_data;
    wire [8*NW-1:0]   wr_mask;
    wire [63:0]       mk_seed;
    wire [23:0]       mk_step;
    /* verilator lint_off UNUSEDSIGNAL */   // read-idle / AXI error status and engine error flags: visible to the
    wire [NR-1:0]     rd_idle;               // testbench (hierarchical), no CSR in the P6c CSR map
    wire [1:0]        mm_err;
    wire [7:0]        gm_err;
    wire [15:0]       m_stat_tiles;
    wire              gm_busy, gm_arr_active, vu_busy, lw_busy, lw_m_busy;
    wire [3:0]        unit_busy;
    /* verilator lint_on UNUSEDSIGNAL */

    // ---------------------------------------------------------------- sequencer
    wire              gm_start, mg_start, vu_start, lw_start, gm_done, mg_done, vu_done, lw_done;
    wire [32*GP-1:0]  gm_p;
    wire [32*VP-1:0]  vu_p;
    wire [32*LP-1:0]  lw_p;
    ct_seq #(.GP(GP), .VP(VP), .LP(LP)) u_seq (
        .clk(clk), .rst_n(rst_n),
        .csr_we(csr_we), .csr_re(csr_re), .csr_addr(csr_addr), .csr_wdata(csr_wdata), .csr_rdata(csr_rdata), .irq(irq),
        .rd_req_valid(rd_req_valid[7]), .rd_req_ready(rd_req_ready[7]), .rd_req_addr(rd_req_addr[32*7 +: 32]),
        .rd_req_len(rd_req_len[7*7 +: 7]), .rd_dat_valid(rd_dat_valid[7]), .rd_dat_ready(rd_dat_ready[7]),
        .rd_dat(rd_dat[256*7 +: 256]),
        .wr_valid(wr_valid[3]), .wr_ready(wr_ready[3]), .wr_addr(wr_addr[32*3 +: 32]), .wr_data(wr_data[256*3 +: 256]),
        .wr_mask(wr_mask[8*3 +: 8]),
        .wr_idle({wr_idle[2], wr_idle[1], wr_idle[0], wr_idle[3]}),
        .gm_start(gm_start), .gm_p(gm_p), .gm_done(gm_done), .mg_done(mg_done), .mg_start(mg_start),
        .vu_start(vu_start), .vu_p(vu_p), .vu_done(vu_done), .lw_start(lw_start), .lw_p(lw_p), .lw_done(lw_done),
        .mk_seed(mk_seed), .mk_step(mk_step), .unit_busy(unit_busy));

    // ---------------------------------------------------------------- payload words
    wire [31:0] g [0:GP-1];
    wire [31:0] v [0:VP-1];
    wire [31:0] l [0:LP-1];
    genvar gi;
    generate
        for (gi = 0; gi < GP; gi = gi + 1) begin : g_gw
            assign g[gi] = gm_p[32*gi +: 32];
        end
        for (gi = 0; gi < VP; gi = gi + 1) begin : g_vw
            assign v[gi] = vu_p[32*gi +: 32];
        end
        for (gi = 0; gi < LP; gi = gi + 1) begin : g_lw
            assign l[gi] = lw_p[32*gi +: 32];
        end
    endgenerate
    assign mk_site  = {v[0][15:12], g[4][11:8]};
    assign mk_layer = {v[1][3:0],   g[5][3:0]};

    // ---------------------------------------------------------------- GEMM engine + merge port
    wire         mg_en, mg_buf, mg_we, w_tile_done, w_tile_buf;
    wire [3:0]   mg_lane;
    wire [7:0]   mg_addr;
    wire [31:0]  mg_wdata, mg_rdata;
    wire [1:0]   merge_busy;
    wire [GKW-1:0] gm_keep = mk_rsp_keep[GKW-1:0];
    gm_top #(.ROWS(ROWS), .COLS(COLS), .NSFU(NSFU)) u_gm (
        .clk(clk), .rst_n(rst_n), .start(gm_start), .busy(gm_busy), .done(gm_done), .err(gm_err), .arr_active(gm_arr_active),
        .d_m(g[0][15:0]), .d_n(g[0][31:16]), .d_k0(g[1][15:0]), .d_k1(g[1][31:16]), .d_k2(g[2][15:0]),
        .d_nb1(g[2][31:16]), .d_nb2(g[3][15:0]),
        .d_a_base0(g[6]), .d_a_base1(g[7]), .d_a_base2(g[8]), .d_a_rs(g[9]), .d_a_bs1(g[10]), .d_a_bs2(g[11]),
        .d_a_trn(g[3][16]),
        .d_b_base0(g[12]), .d_b_base1(g[13]), .d_b_base2(g[14]), .d_b_rs(g[15]), .d_b_bs1(g[16]), .d_b_bs2(g[17]),
        .d_b_trn(g[3][17]),
        .d_c_base(g[18]), .d_c_rs(g[19]), .d_c_bs1(g[20]), .d_c_bs2(g[21]),
        .d_bias_base(g[22]), .d_aux_base(g[23]), .d_o2_base(g[24]),
        .d_ep_bias(g[3][18]), .d_ep_scale(g[3][19]), .d_ep_gelu(g[3][20]), .d_ep_gelu_bwd(g[3][21]),
        .d_ep_drop(g[3][22]), .d_ep_resid(g[3][23]), .d_ep_sexp(g[4][7:0]), .d_ep_ds(g[25]),
        .rd0_req_valid(rd_req_valid[0]), .rd0_req_ready(rd_req_ready[0]), .rd0_req_addr(rd_req_addr[0 +: 32]),
        .rd0_req_len(rd_req_len[0 +: 7]), .rd0_dat_valid(rd_dat_valid[0]), .rd0_dat_ready(rd_dat_ready[0]),
        .rd0_dat(rd_dat[0 +: 256]),
        .rd1_req_valid(rd_req_valid[1]), .rd1_req_ready(rd_req_ready[1]), .rd1_req_addr(rd_req_addr[32 +: 32]),
        .rd1_req_len(rd_req_len[7 +: 7]), .rd1_dat_valid(rd_dat_valid[1]), .rd1_dat_ready(rd_dat_ready[1]),
        .rd1_dat(rd_dat[256 +: 256]),
        .mk_req_valid(mk_req_valid[0]), .mk_req_ready(mk_req_ready[0]), .mk_req_idx(mk_req_idx[0 +: 32]),
        .mk_rsp_valid(mk_rsp_valid[0]), .mk_rsp_keep(gm_keep),
        .wr_valid(wr_valid[0]), .wr_ready(wr_ready[0]), .wr_addr(wr_addr[0 +: 32]), .wr_data(wr_data[0 +: 256]),
        .wr_mask(wr_mask[0 +: 8]),
        .mg_en(mg_en), .mg_buf(mg_buf), .mg_we(mg_we), .mg_lane(mg_lane), .mg_addr(mg_addr), .mg_wdata(mg_wdata),
        .mg_rdata(mg_rdata), .w_tile_done(w_tile_done), .w_tile_buf(w_tile_buf), .merge_busy(merge_busy));

    // ---------------------------------------------------------------- vector unit
    vu_top u_vu (
        .clk(clk), .rst_n(rst_n), .start(vu_start), .busy(vu_busy), .done(vu_done),
        .d_vop(v[0][4:0]), .d_rows(v[2][15:0]), .d_cols(v[2][31:16]), .d_rows_per_b(v[3][15:0]), .d_emb_l(v[3][31:16]),
        .d_st_en(v[0][8]), .d_x_reset(v[0][9]), .d_g_reset(v[0][10]),
        .d_a_base(v[4]), .d_a_rs(v[5]), .d_b_base(v[6]), .d_b_rs(v[7]), .d_c_base(v[8]), .d_c_rs(v[9]),
        .d_cv0_base(v[10]), .d_cv1_base(v[11]), .d_cv2_base(v[12]), .d_km_rs(v[13]),
        .d_st0_base(v[14]), .d_st1_base(v[15]), .d_o2_base(v[16]), .d_ids_base(v[17]),
        .d_mk_base(v[18]), .d_mk_rs(v[19]), .d_c0(v[20]), .d_c1(v[21]), .d_c2(v[22]), .d_c3(v[23]), .d_ds(v[24]),
        .rd0_req_valid(rd_req_valid[2]), .rd0_req_ready(rd_req_ready[2]), .rd0_req_addr(rd_req_addr[64 +: 32]),
        .rd0_req_len(rd_req_len[14 +: 7]), .rd0_dat_valid(rd_dat_valid[2]), .rd0_dat_ready(rd_dat_ready[2]),
        .rd0_dat(rd_dat[512 +: 256]),
        .rd1_req_valid(rd_req_valid[3]), .rd1_req_ready(rd_req_ready[3]), .rd1_req_addr(rd_req_addr[96 +: 32]),
        .rd1_req_len(rd_req_len[21 +: 7]), .rd1_dat_valid(rd_dat_valid[3]), .rd1_dat_ready(rd_dat_ready[3]),
        .rd1_dat(rd_dat[768 +: 256]),
        .mk_req_valid(mk_req_valid[1]), .mk_req_ready(mk_req_ready[1]), .mk_req_idx(mk_req_idx[32 +: 32]),
        .mk_rsp_valid(mk_rsp_valid[1]), .mk_rsp_keep(mk_rsp_keep[16 +: 16]),
        .wr_valid(wr_valid[1]), .wr_ready(wr_ready[1]), .wr_addr(wr_addr[32 +: 32]), .wr_data(wr_data[256 +: 256]),
        .wr_mask(wr_mask[8 +: 8]));

    // ---------------------------------------------------------------- LoWA unit
    lw_top u_lw (
        .clk(clk), .rst_n(rst_n), .start(lw_start), .busy(lw_busy), .done(lw_done),
        .d_op(l[0][1:0]), .d_flag(l[0][8]),
        .d_a0(l[1]), .d_a1(l[2]), .d_a2(l[3]), .d_a3(l[4]), .d_a4(l[5]), .d_a5(l[6]), .d_a6(l[7]), .d_a7(l[8]), .d_len(l[9]),
        .m_start(mg_start), .m_busy(lw_m_busy), .m_done(mg_done), .m_trn(g[3][25]),
        .m_nb(g[26][7:0]), .m_mb(g[26][15:8]), .m_kc0(g[26][23:16]), .m_kc1(g[26][31:24]), .m_kc2(g[27][7:0]),
        .m_ms0(g[27][9:8]), .m_ms1(g[27][11:10]), .m_ms2(g[27][13:12]),
        .m_hits_a(g[28]), .m_dir_a(g[29]), .m_coef_a(g[30]), .m_hits_b(g[31]), .m_dir_b(g[32]), .m_coef_b(g[33]),
        .m_h_base(g[34]), .m_gamma(g[35]), .m_stat_tiles(m_stat_tiles),
        .rd0_req_valid(rd_req_valid[4]), .rd0_req_ready(rd_req_ready[4]), .rd0_req_addr(rd_req_addr[128 +: 32]),
        .rd0_req_len(rd_req_len[28 +: 7]), .rd0_dat_valid(rd_dat_valid[4]), .rd0_dat_ready(rd_dat_ready[4]),
        .rd0_dat(rd_dat[1024 +: 256]),
        .rd1_req_valid(rd_req_valid[5]), .rd1_req_ready(rd_req_ready[5]), .rd1_req_addr(rd_req_addr[160 +: 32]),
        .rd1_req_len(rd_req_len[35 +: 7]), .rd1_dat_valid(rd_dat_valid[5]), .rd1_dat_ready(rd_dat_ready[5]),
        .rd1_dat(rd_dat[1280 +: 256]),
        .rd2_req_valid(rd_req_valid[6]), .rd2_req_ready(rd_req_ready[6]), .rd2_req_addr(rd_req_addr[192 +: 32]),
        .rd2_req_len(rd_req_len[42 +: 7]), .rd2_dat_valid(rd_dat_valid[6]), .rd2_dat_ready(rd_dat_ready[6]),
        .rd2_dat(rd_dat[1536 +: 256]),
        .wr_valid(wr_valid[2]), .wr_ready(wr_ready[2]), .wr_addr(wr_addr[64 +: 32]), .wr_data(wr_data[512 +: 256]),
        .wr_mask(wr_mask[16 +: 8]),
        .mg_en(mg_en), .mg_buf(mg_buf), .mg_we(mg_we), .mg_lane(mg_lane), .mg_addr(mg_addr), .mg_wdata(mg_wdata),
        .mg_rdata(mg_rdata), .w_tile_done(w_tile_done), .w_tile_buf(w_tile_buf), .merge_busy(merge_busy));

    // ---------------------------------------------------------------- memory system + dropout-mask generators
    mm_top #(.NR(NR), .NM(NM), .NW(NW), .AXL(3), .MK_DRAM(0)) u_mm (
        .clk(clk), .rst_n(rst_n),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready), .rd_req_addr(rd_req_addr), .rd_req_len(rd_req_len),
        .rd_dat_valid(rd_dat_valid), .rd_dat_ready(rd_dat_ready), .rd_dat(rd_dat),
        .mk_base(64'd0), .mk_req_valid(mk_req_valid), .mk_req_ready(mk_req_ready), .mk_req_idx(mk_req_idx),
        .mk_rsp_valid(mk_rsp_valid), .mk_rsp_keep(mk_rsp_keep),
        .mk_seed(mk_seed), .mk_step(mk_step), .mk_site(mk_site), .mk_layer(mk_layer),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_addr(wr_addr), .wr_data(wr_data), .wr_mask(wr_mask),
        .rd_idle(rd_idle), .wr_idle(wr_idle), .err(mm_err),
        .m_arvalid(m_arvalid), .m_arready(m_arready), .m_arid(m_arid), .m_araddr(m_araddr), .m_arlen(m_arlen),
        .m_arsize(m_arsize), .m_arburst(m_arburst),
        .m_rvalid(m_rvalid), .m_rready(m_rready), .m_rid(m_rid), .m_rdata(m_rdata), .m_rresp(m_rresp), .m_rlast(m_rlast),
        .m_awvalid(m_awvalid), .m_awready(m_awready), .m_awid(m_awid), .m_awaddr(m_awaddr), .m_awlen(m_awlen),
        .m_awsize(m_awsize), .m_awburst(m_awburst),
        .m_wvalid(m_wvalid), .m_wready(m_wready), .m_wdata(m_wdata), .m_wstrb(m_wstrb), .m_wlast(m_wlast),
        .m_bvalid(m_bvalid), .m_bready(m_bready), .m_bid(m_bid), .m_bresp(m_bresp));
endmodule
`default_nettype wire
