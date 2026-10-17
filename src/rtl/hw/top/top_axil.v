`default_nettype none
// top_axil: AXI4-Lite slave (32-bit data, 4 KiB CSR window) in front of the sequencer's CSR bus (ct_seq csr_*,
// docs/isa.md section 7).  One transaction at a time: a write needs AW and W together (both accepted in the same
// cycle, one csr_we pulse, then B); a read accepts AR (one csr_re pulse; ct_seq registers csr_rdata on it), and R
// returns that register the next cycle.  A write has priority over a read offered in the same cycle.  Responses are
// always OKAY; WSTRB is ignored (every CSR is a full 32-bit word; partial writes are not supported, docs/p7a_report.md).
module top_axil (
    input  wire          clk,
    input  wire          rst_n,
    // AXI4-Lite slave
    input  wire          s_awvalid,
    output wire          s_awready,
    input  wire [11:0]   s_awaddr,
    input  wire          s_wvalid,
    output wire          s_wready,
    input  wire [31:0]   s_wdata,
    /* verilator lint_off UNUSEDSIGNAL */   // full-word CSRs only: WSTRB is not used
    input  wire [3:0]    s_wstrb,
    /* verilator lint_on UNUSEDSIGNAL */
    output reg           s_bvalid,
    input  wire          s_bready,
    output wire [1:0]    s_bresp,
    input  wire          s_arvalid,
    output wire          s_arready,
    input  wire [11:0]   s_araddr,
    output reg           s_rvalid,
    input  wire          s_rready,
    output wire [31:0]   s_rdata,
    output wire [1:0]    s_rresp,
    // CSR bus (ct_seq)
    output wire          csr_we,
    output wire          csr_re,
    output wire [11:0]   csr_addr,
    output wire [31:0]   csr_wdata,
    input  wire [31:0]   csr_rdata
);
    wire wr_go = s_awvalid & s_wvalid & ~s_bvalid;
    wire rd_go = s_arvalid & ~s_rvalid & ~wr_go;
    assign s_awready = wr_go;
    assign s_wready  = wr_go;
    assign s_arready = rd_go;
    assign csr_we    = wr_go;
    assign csr_re    = rd_go;
    assign csr_addr  = wr_go ? s_awaddr : s_araddr;
    assign csr_wdata = s_wdata;
    assign s_rdata   = csr_rdata;
    assign s_bresp   = 2'b00;
    assign s_rresp   = 2'b00;
    always @(posedge clk) begin
        if (!rst_n) begin
            s_bvalid <= 1'b0;
            s_rvalid <= 1'b0;
        end else begin
            if (wr_go) s_bvalid <= 1'b1;
            else if (s_bready) s_bvalid <= 1'b0;
            if (rd_go) s_rvalid <= 1'b1;
            else if (s_rready) s_rvalid <= 1'b0;
        end
    end
endmodule
`default_nettype wire
