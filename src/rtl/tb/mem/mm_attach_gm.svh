// P6a integration (included by tb/gemm/tb_gm_top.sv and tb/vec/tb_vu_top.sv under +define+MM_DRAM):
// the engine's rd0 / rd1 / mk / wr channels on hw/mem/mm_top, its AXI4 port on tb/mem/tb_dram.sv.
// The image (+reg= / +img=, tb/mem/gen_integ.py) holds the case data at memory_map.md addresses and the
// dropout keep bits in the MASK_STANDIN region (debug stand-in, MK_DRAM = 1), or, with +define+MM_MKGEN, the keep
// bits come from mm_top's on-chip Threefry generator (MK_DRAM = 0, P6b) configured by +mkseed= +mkstep= +mksite=
// +mklayer= (hex); MM_MKNC = its Threefry pipelines (default 9, the vector unit's rate).
// Knobs: +lat (tb's lat) +stall (tb's stall, DRAM backpressure) +bw +wlat +reorder +ilv.
    wire         m_arvalid, m_arready, m_rvalid, m_rready, m_rlast, m_awvalid, m_awready, m_wvalid, m_wready, m_wlast, m_bvalid, m_bready;
    wire [3:0]   m_arid, m_rid, m_awid, m_bid;
    wire [31:0]  m_araddr, m_awaddr;
    wire [7:0]   m_arlen, m_awlen;
    wire [2:0]   m_arsize, m_awsize;
    wire [1:0]   m_arburst, m_awburst, m_rresp, m_bresp, mm_err;
    wire [255:0] m_rdata, m_wdata;
    wire [31:0]  m_wstrb;
    wire [1:0]   mm_rd_idle;
    wire         mm_wr_idle;
    wire [15:0]  mm_keep;
`ifndef MM_MAXR
`define MM_MAXR 8
`endif
`ifndef MM_RQL
`define MM_RQL 3
`endif
`ifndef MM_RFL
`define MM_RFL 4
`endif
`ifndef MM_MKNC
`define MM_MKNC 9
`endif
`ifdef MM_MKGEN
    localparam MM_MK_DRAM = 0;
`else
    localparam MM_MK_DRAM = 1;
`endif
    reg  [63:0]  mm_seed = 0;
    reg  [23:0]  mm_step = 0;
    reg  [3:0]   mm_site = 0, mm_layer = 0;
    initial begin
        if ($value$plusargs("mkseed=%h", mm_seed)) ;
        if ($value$plusargs("mkstep=%h", mm_step)) ;
        if ($value$plusargs("mksite=%h", mm_site)) ;
        if ($value$plusargs("mklayer=%h", mm_layer)) ;
    end
    // MM_MAXR / MM_RQL / MM_RFL: outstanding-read experiments (default = the mm_top defaults)
    mm_top #(.NR(2), .NM(1), .NW(1), .AXL(3), .MAXR(`MM_MAXR), .RQL(`MM_RQL), .RFL(`MM_RFL), .MK_DRAM(MM_MK_DRAM),
             .MKNC0(`MM_MKNC)) u_mm (
        .clk(clk), .rst_n(rst_n),
        .rd_req_valid({rd1_req_valid, rd0_req_valid}), .rd_req_ready({rd1_req_ready, rd0_req_ready}),
        .rd_req_addr({rd1_req_addr, rd0_req_addr}), .rd_req_len({rd1_req_len, rd0_req_len}),
        .rd_dat_valid({rd1_dat_valid, rd0_dat_valid}), .rd_dat_ready({rd1_dat_ready, rd0_dat_ready}), .rd_dat({rd1_dat, rd0_dat}),
        .mk_base(32'h8000_0000), .mk_req_valid(mk_req_valid), .mk_req_ready(mk_req_ready), .mk_req_idx(mk_req_idx),
        .mk_rsp_valid(mk_rsp_valid), .mk_rsp_keep(mm_keep),
        .mk_seed(mm_seed), .mk_step(mm_step), .mk_site(mm_site), .mk_layer(mm_layer),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_addr(wr_addr), .wr_data(wr_data), .wr_mask(wr_mask),
        .rd_idle(mm_rd_idle), .wr_idle(mm_wr_idle), .err(mm_err),
        .m_arvalid(m_arvalid), .m_arready(m_arready), .m_arid(m_arid), .m_araddr(m_araddr), .m_arlen(m_arlen), .m_arsize(m_arsize), .m_arburst(m_arburst),
        .m_rvalid(m_rvalid), .m_rready(m_rready), .m_rid(m_rid), .m_rdata(m_rdata), .m_rresp(m_rresp), .m_rlast(m_rlast),
        .m_awvalid(m_awvalid), .m_awready(m_awready), .m_awid(m_awid), .m_awaddr(m_awaddr), .m_awlen(m_awlen), .m_awsize(m_awsize), .m_awburst(m_awburst),
        .m_wvalid(m_wvalid), .m_wready(m_wready), .m_wdata(m_wdata), .m_wstrb(m_wstrb), .m_wlast(m_wlast),
        .m_bvalid(m_bvalid), .m_bready(m_bready), .m_bid(m_bid), .m_bresp(m_bresp));
    assign mk_rsp_keep = mm_keep[$bits(mk_rsp_keep)-1:0];
    localparam MMO_NR = 2, MMO_NW = 1;
    `include "mm_dram_inst.svh"
