// P6a integration (included by tb/lowa/tb_lw_top.sv under +define+MM_DRAM, mode 0 cases): lw_top's rd0 / rd1 /
// rd2 / wr channels on hw/mem/mm_top (no keep-bit channel: the generator's inputs are tied off), AXI4 on tb_dram.
    wire         m_arvalid, m_arready, m_rvalid, m_rready, m_rlast, m_awvalid, m_awready, m_wvalid, m_wready, m_wlast, m_bvalid, m_bready;
    wire [3:0]   m_arid, m_rid, m_awid, m_bid;
    wire [31:0]  m_araddr, m_awaddr;
    wire [7:0]   m_arlen, m_awlen;
    wire [2:0]   m_arsize, m_awsize;
    wire [1:0]   m_arburst, m_awburst, m_rresp, m_bresp, mm_err;
    wire [255:0] m_rdata, m_wdata;
    wire [31:0]  m_wstrb;
    wire [2:0]   mm_rd_idle;
    wire         mm_wr_idle;
    mm_top #(.NR(3), .NM(1), .NW(1), .AXL(3), .MK_DRAM(0)) u_mm (
        .clk(clk), .rst_n(rst_n),
        .rd_req_valid({rd2_req_valid, rd1_req_valid, rd0_req_valid}), .rd_req_ready({rd2_req_ready, rd1_req_ready, rd0_req_ready}),
        .rd_req_addr({rd2_req_addr, rd1_req_addr, rd0_req_addr}), .rd_req_len({rd2_req_len, rd1_req_len, rd0_req_len}),
        .rd_dat_valid({rd2_dat_valid, rd1_dat_valid, rd0_dat_valid}), .rd_dat_ready({rd2_dat_ready, rd1_dat_ready, rd0_dat_ready}),
        .rd_dat({rd2_dat, rd1_dat, rd0_dat}),
        .mk_base(32'd0), .mk_req_valid(1'b0), .mk_req_ready(), .mk_req_idx(32'd0), .mk_rsp_valid(), .mk_rsp_keep(),
        .mk_seed(64'd0), .mk_step(24'd0), .mk_site(4'd0), .mk_layer(4'd0),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_addr(wr_addr), .wr_data(wr_data), .wr_mask(wr_mask),
        .rd_idle(mm_rd_idle), .wr_idle(mm_wr_idle), .err(mm_err),
        .m_arvalid(m_arvalid), .m_arready(m_arready), .m_arid(m_arid), .m_araddr(m_araddr), .m_arlen(m_arlen), .m_arsize(m_arsize), .m_arburst(m_arburst),
        .m_rvalid(m_rvalid), .m_rready(m_rready), .m_rid(m_rid), .m_rdata(m_rdata), .m_rresp(m_rresp), .m_rlast(m_rlast),
        .m_awvalid(m_awvalid), .m_awready(m_awready), .m_awid(m_awid), .m_awaddr(m_awaddr), .m_awlen(m_awlen), .m_awsize(m_awsize), .m_awburst(m_awburst),
        .m_wvalid(m_wvalid), .m_wready(m_wready), .m_wdata(m_wdata), .m_wstrb(m_wstrb), .m_wlast(m_wlast),
        .m_bvalid(m_bvalid), .m_bready(m_bready), .m_bid(m_bid), .m_bresp(m_bresp));
    `include "mm_dram_inst.svh"
