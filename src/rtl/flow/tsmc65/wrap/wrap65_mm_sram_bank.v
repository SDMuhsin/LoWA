`default_nettype none
// Synthesis/STA wrapper (not part of the design): registers every input of mm_sram_bank so all timing
// paths are register to register (as flow/synth/wrap_*.v).  The backend comes from MM_TECH (2 in flow/tsmc65).
module wrap65_mm_sram_bank #(parameter WORDS = 768, parameter WIDTH = 32, parameter AW = 10) (
    input  wire             clk,
    input  wire             en,
    input  wire             we,
    input  wire [AW-1:0]    addr,
    input  wire [WIDTH-1:0] d,
    output wire [WIDTH-1:0] q
);
    reg en_r, we_r; reg [AW-1:0] addr_r; reg [WIDTH-1:0] d_r;
    always @(posedge clk) begin en_r <= en; we_r <= we; addr_r <= addr; d_r <= d; end
    mm_sram_bank #(.WORDS(WORDS), .WIDTH(WIDTH), .AW(AW)) u_dut (.clk(clk), .en(en_r), .we(we_r), .addr(addr_r),
        .d(d_r), .q(q));
endmodule
`default_nettype wire
