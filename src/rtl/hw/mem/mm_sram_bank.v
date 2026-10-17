`default_nettype none
// mm_sram_bank: a WORDS x WIDTH single-port on-chip memory bank (P3/P6, PLAN R11a).
// Port contract (identical for every technology backend):
//   one access per cycle: en = 1 accesses addr, we selects write (1) or read (0); no byte enables;
//   read data q is valid the cycle after a read access and is held until the next read access
//   (writes never change q); a write is visible to a read in the following cycle.  No reset.
// Backends (hw/mem/tech/), chosen by parameter TECH, whose default is the macro MM_TECH:
//   0 generic : behavioural array (mm_bank_generic), the simulation default
//   1 gf180   : GF180 sram256x8 macro tiles (mm_bank_gf180), the default when SYNTHESIS is defined
//               (the Yosys/OpenROAD GF180 flow in flow/synth, flow/sta)
//   2 tsmc65  : latch register file with clock-gated word writes (mm_bank_t65, flow/tsmc65)
// Select globally with +define+MM_TECH=<n> (simulators) or -define {MM_TECH=<n>} (DC).
`ifndef MM_TECH
`ifdef SYNTHESIS
`define MM_TECH 1
`else
`define MM_TECH 0
`endif
`endif
module mm_sram_bank #(
    parameter WORDS = 256,
    parameter WIDTH = 32,
    parameter AW    = 8,                    // address width, 2**AW >= WORDS
    parameter TECH  = `MM_TECH              // 0 generic, 1 gf180, 2 tsmc65
) (
    input  wire             clk,
    input  wire             en,             // access this cycle
    input  wire             we,             // 1 = write, 0 = read
    input  wire [AW-1:0]    addr,
    input  wire [WIDTH-1:0] d,
    output wire [WIDTH-1:0] q
);
    generate
        if (TECH == 1) begin : g_gf180
            mm_bank_gf180 #(.WORDS(WORDS), .WIDTH(WIDTH), .AW(AW)) u_bank (
                .clk(clk), .en(en), .we(we), .addr(addr), .d(d), .q(q));
        end else if (TECH == 2) begin : g_t65
            mm_bank_t65 #(.WORDS(WORDS), .WIDTH(WIDTH), .AW(AW)) u_bank (
                .clk(clk), .en(en), .we(we), .addr(addr), .d(d), .q(q));
        end else begin : g_gen
            mm_bank_generic #(.WORDS(WORDS), .WIDTH(WIDTH), .AW(AW)) u_bank (
                .clk(clk), .en(en), .we(we), .addr(addr), .d(d), .q(q));
        end
    endgenerate
endmodule
`default_nettype wire
