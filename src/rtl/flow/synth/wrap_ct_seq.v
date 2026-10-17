`default_nettype none
// Synthesis/STA wrapper (not part of the design): ct_seq (+ ct_dma) with every input registered so all paths are
// register to register.
module wrap_ct_seq (
    input  wire          clk, rst_n,
    input  wire          csr_we, csr_re,
    input  wire [11:0]   csr_addr,
    input  wire [31:0]   csr_wdata,
    output wire [31:0]   csr_rdata,
    output wire          irq,
    output wire          rd_req_valid,
    input  wire          rd_req_ready,
    output wire [31:0]   rd_req_addr,
    output wire [6:0]    rd_req_len,
    input  wire          rd_dat_valid,
    output wire          rd_dat_ready,
    input  wire [255:0]  rd_dat,
    output wire          wr_valid,
    input  wire          wr_ready,
    output wire [31:0]   wr_addr,
    output wire [255:0]  wr_data,
    output wire [7:0]    wr_mask,
    input  wire [3:0]    wr_idle,
    output wire          gm_start, mg_start, vu_start, lw_start,
    output wire [32*36-1:0] gm_p,
    output wire [32*25-1:0] vu_p,
    output wire [32*10-1:0] lw_p,
    input  wire          gm_done, mg_done, vu_done, lw_done,
    output wire [63:0]   mk_seed,
    output wire [23:0]   mk_step,
    output wire [3:0]    unit_busy
);
    reg q_we, q_re, q_rr, q_dv, q_wr, q_gd, q_md, q_vd, q_ld; reg [11:0] q_a; reg [31:0] q_wd; reg [255:0] q_d; reg [3:0] q_wi;
    always @(posedge clk) begin
        q_we <= csr_we; q_re <= csr_re; q_a <= csr_addr; q_wd <= csr_wdata; q_rr <= rd_req_ready; q_dv <= rd_dat_valid;
        q_d <= rd_dat; q_wr <= wr_ready; q_wi <= wr_idle; q_gd <= gm_done; q_md <= mg_done; q_vd <= vu_done; q_ld <= lw_done;
    end
    ct_seq u (.clk(clk), .rst_n(rst_n), .csr_we(q_we), .csr_re(q_re), .csr_addr(q_a), .csr_wdata(q_wd),
        .csr_rdata(csr_rdata), .irq(irq), .rd_req_valid(rd_req_valid), .rd_req_ready(q_rr), .rd_req_addr(rd_req_addr),
        .rd_req_len(rd_req_len), .rd_dat_valid(q_dv), .rd_dat_ready(rd_dat_ready), .rd_dat(q_d), .wr_valid(wr_valid),
        .wr_ready(q_wr), .wr_addr(wr_addr), .wr_data(wr_data), .wr_mask(wr_mask), .wr_idle(q_wi),
        .gm_start(gm_start), .gm_p(gm_p), .gm_done(q_gd), .mg_done(q_md), .mg_start(mg_start), .vu_start(vu_start),
        .vu_p(vu_p), .vu_done(q_vd), .lw_start(lw_start), .lw_p(lw_p), .lw_done(q_ld), .mk_seed(mk_seed),
        .mk_step(mk_step), .unit_busy(unit_busy));
endmodule
`default_nettype wire
