`default_nettype none
// mm_bank_generic (mm_sram_bank backend TECH = 0, the simulation default): behavioural WORDS x WIDTH
// single-port memory with exactly the mm_sram_bank port contract: one access per cycle, a write
// stores d at addr, a read loads q from addr at the clock edge (valid the next cycle) and q is held
// until the next read.  No reset (datapath).  Not meant for synthesis (a tool would infer flops).
module mm_bank_generic #(
    parameter WORDS = 256,
    parameter WIDTH = 32,
    parameter AW    = 8
) (
    input  wire             clk,
    input  wire             en,
    input  wire             we,
    input  wire [AW-1:0]    addr,
    input  wire [WIDTH-1:0] d,
    output wire [WIDTH-1:0] q
);
    reg [WIDTH-1:0] mem [0:WORDS-1];
    reg [WIDTH-1:0] q_r;
    always @(posedge clk) begin
        if (en & we) mem[addr] <= d;
        if (en & ~we) q_r <= mem[addr];
    end
    assign q = q_r;
endmodule
`default_nettype wire
