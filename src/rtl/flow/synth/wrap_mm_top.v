`default_nettype none
// Synthesis/STA wrapper (not part of the design): mm_top in the full-chip configuration (gm rd0 / rd1, vu rd0 / rd1,
// lw rd0 / rd1 / rd2 = 7 read channels, 2 keep-bit channels (on-chip Threefry generators, MK_DRAM = 0: GEMM 3 pipelines, vector 9), 3 write channels, 256-bit
// AXI4) with every input registered so all paths are register to register.
module wrap_mm_top #(parameter NR = 7, parameter NM = 2, parameter NW = 3, parameter AXL = 3, parameter MK_DRAM = 0) (
    input  wire                clk, rst_n,
    input  wire [NR-1:0]       rd_req_valid, rd_dat_ready,
    input  wire [32*NR-1:0]    rd_req_addr,
    input  wire [7*NR-1:0]     rd_req_len,
    output wire [NR-1:0]       rd_req_ready, rd_dat_valid, rd_idle,
    output wire [256*NR-1:0]   rd_dat,
    input  wire [32*NM-1:0]    mk_base, mk_req_idx,
    input  wire [NM-1:0]       mk_req_valid,
    input  wire [63:0]         mk_seed,
    input  wire [23:0]         mk_step,
    input  wire [4*NM-1:0]     mk_site, mk_layer,
    output wire [NM-1:0]       mk_req_ready, mk_rsp_valid,
    output wire [16*NM-1:0]    mk_rsp_keep,
    input  wire [NW-1:0]       wr_valid,
    input  wire [32*NW-1:0]    wr_addr,
    input  wire [256*NW-1:0]   wr_data,
    input  wire [8*NW-1:0]     wr_mask,
    output wire [NW-1:0]       wr_ready, wr_idle,
    output wire [1:0]          err,
    output wire                m_arvalid, m_rready, m_awvalid, m_wvalid, m_wlast, m_bready,
    output wire [3:0]          m_arid, m_awid,
    output wire [31:0]         m_araddr, m_awaddr,
    output wire [7:0]          m_arlen, m_awlen,
    output wire [2:0]          m_arsize, m_awsize,
    output wire [1:0]          m_arburst, m_awburst,
    output wire [32*(1<<AXL)-1:0] m_wdata,
    output wire [4*(1<<AXL)-1:0]  m_wstrb,
    input  wire                m_arready, m_rvalid, m_rlast, m_awready, m_wready, m_bvalid,
    input  wire [3:0]          m_rid, m_bid,
    input  wire [32*(1<<AXL)-1:0] m_rdata,
    input  wire [1:0]          m_rresp, m_bresp
);
    reg [NR-1:0] q_rv, q_dr; reg [32*NR-1:0] q_ra; reg [7*NR-1:0] q_rl;
    reg [32*NM-1:0] q_mb, q_mi; reg [NM-1:0] q_mv; reg [63:0] q_ms; reg [23:0] q_mt; reg [4*NM-1:0] q_msi, q_ml;
    reg [NW-1:0] q_wv; reg [32*NW-1:0] q_wa; reg [256*NW-1:0] q_wd; reg [8*NW-1:0] q_wm;
    reg q_arr, q_rv2, q_rl2, q_awr, q_wr, q_bv; reg [3:0] q_rid, q_bid; reg [32*(1<<AXL)-1:0] q_rd; reg [1:0] q_rr, q_br;
    always @(posedge clk) begin
        q_rv <= rd_req_valid; q_dr <= rd_dat_ready; q_ra <= rd_req_addr; q_rl <= rd_req_len;
        q_mb <= mk_base; q_mi <= mk_req_idx; q_mv <= mk_req_valid; q_ms <= mk_seed; q_mt <= mk_step; q_msi <= mk_site; q_ml <= mk_layer;
        q_wv <= wr_valid; q_wa <= wr_addr; q_wd <= wr_data; q_wm <= wr_mask;
        q_arr <= m_arready; q_rv2 <= m_rvalid; q_rl2 <= m_rlast; q_awr <= m_awready; q_wr <= m_wready; q_bv <= m_bvalid;
        q_rid <= m_rid; q_bid <= m_bid; q_rd <= m_rdata; q_rr <= m_rresp; q_br <= m_bresp;
    end
    mm_top #(.NR(NR), .NM(NM), .NW(NW), .AXL(AXL), .MK_DRAM(MK_DRAM)) u (
        .clk(clk), .rst_n(rst_n),
        .rd_req_valid(q_rv), .rd_req_ready(rd_req_ready), .rd_req_addr(q_ra), .rd_req_len(q_rl),
        .rd_dat_valid(rd_dat_valid), .rd_dat_ready(q_dr), .rd_dat(rd_dat),
        .mk_base(q_mb), .mk_req_valid(q_mv), .mk_req_ready(mk_req_ready), .mk_req_idx(q_mi),
        .mk_rsp_valid(mk_rsp_valid), .mk_rsp_keep(mk_rsp_keep),
        .mk_seed(q_ms), .mk_step(q_mt), .mk_site(q_msi), .mk_layer(q_ml),
        .wr_valid(q_wv), .wr_ready(wr_ready), .wr_addr(q_wa), .wr_data(q_wd), .wr_mask(q_wm),
        .rd_idle(rd_idle), .wr_idle(wr_idle), .err(err),
        .m_arvalid(m_arvalid), .m_arready(q_arr), .m_arid(m_arid), .m_araddr(m_araddr), .m_arlen(m_arlen), .m_arsize(m_arsize), .m_arburst(m_arburst),
        .m_rvalid(q_rv2), .m_rready(m_rready), .m_rid(q_rid), .m_rdata(q_rd), .m_rresp(q_rr), .m_rlast(q_rl2),
        .m_awvalid(m_awvalid), .m_awready(q_awr), .m_awid(m_awid), .m_awaddr(m_awaddr), .m_awlen(m_awlen), .m_awsize(m_awsize), .m_awburst(m_awburst),
        .m_wvalid(m_wvalid), .m_wready(q_wr), .m_wdata(m_wdata), .m_wstrb(m_wstrb), .m_wlast(m_wlast),
        .m_bvalid(q_bv), .m_bready(m_bready), .m_bid(q_bid), .m_bresp(q_br));
endmodule
`default_nettype wire
