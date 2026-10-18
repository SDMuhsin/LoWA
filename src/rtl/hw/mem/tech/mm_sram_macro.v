`default_nettype none
// mm_sram_macro: behavioural stand-in for one compiled single-port SRAM / register-file macro, WORDS x WIDTH,
// used by the tsmc65_sram backend (mm_bank_t65s) in every simulation and lint run.  For DC and the server-side
// contract simulations it is replaced by the git-ignored site wrapper flow/tsmc65/site/mm_sram_macro_site.v,
// which instantiates the generated macro of the matching shape (PLAN R16; the macro names live only there).
// Interface and behaviour of the compiled macros (as their vendor models show):
//   cen_n = 0 accesses a at the rising clock edge; wen_n selects write (0) or read (1);
//   read : q = word a after the edge;
//   write: the word is stored and q = d after the edge (write-through; the compilers offer no other mode);
//   no access (cen_n = 1): q holds.  No reset.
module mm_sram_macro #(
    parameter WORDS = 256,
    parameter WIDTH = 32,
    parameter AW    = 8
) (
    input  wire             clk,
    input  wire             cen_n,
    input  wire             wen_n,
    input  wire [AW-1:0]    a,
    input  wire [WIDTH-1:0] d,
    output wire [WIDTH-1:0] q
);
    reg [WIDTH-1:0] mem [0:WORDS-1];
    reg [WIDTH-1:0] q_r;
    always @(posedge clk) begin
        if (!cen_n && !wen_n) begin
            mem[a] <= d;
            q_r    <= d;
        end
        if (!cen_n && wen_n) q_r <= mem[a];
    end
    assign q = q_r;
endmodule
`default_nettype wire
