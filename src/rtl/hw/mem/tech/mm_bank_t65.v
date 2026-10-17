`default_nettype none
// mm_bank_t65 (mm_sram_bank backend TECH = 2): WORDS x WIDTH single-port latch register file for
// TSMC 65 nm, where no SRAM compiler is usable (PLAN R11a; after the template's mq65_latch_rf).
// Same port contract as every mm_sram_bank backend:
//   read : en & ~we at posedge N loads q_r from the addressed word (valid after N, held until the
//          next read; writes never change q).
//   write: en & we at posedge N captures d in wd_r; the addressed word's clock gate (one mm_icg per
//          word, enable = en & we & (addr == w), latched while clk is low) opens that word's
//          latches for the high phase after N; they close at the falling edge holding wd_r.
//          A read at posedge N+1 sees the new word.
// Timing (flow/tsmc65 SDC): wd_r -> word latch D within the high phase (flop -> positive latch is
// the tool default, borrowing up to the closing edge); word latch -> q_r is a full-cycle path
// through the WORDS:1 read mux; the ICG enable is a normal flop-like setup at posedge N.
// Datapath (wd_r, latches, q_r) is not reset.
module mm_bank_t65 #(
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
    wire            wr = en & we;
    reg [WIDTH-1:0] wd_r;                   // write-data capture register
    reg [WIDTH-1:0] q_r;                    // read-data register
    wire [WIDTH-1:0] word [0:WORDS-1];

    always @(posedge clk) begin
        if (wr) wd_r <= d;
        if (en & ~we) q_r <= word[addr];
    end
    assign q = q_r;

    genvar w;
    generate
        for (w = 0; w < WORDS; w = w + 1) begin : g_w
            wire            gclk;
            reg [WIDTH-1:0] lat;
            /* verilator lint_off WIDTHEXPAND */
            mm_icg u_icg (.clk(clk), .en(wr & (addr == w)), .gclk(gclk));
            /* verilator lint_on WIDTHEXPAND */
            /* verilator lint_off COMBDLY */  // latch: non-blocking so posedge samplers never race
            /* verilator lint_off LATCH */    // transparent-high word latch, the storage cell
            always @(gclk or wd_r)
                if (gclk) lat <= wd_r;
            /* verilator lint_on LATCH */
            /* verilator lint_on COMBDLY */
            assign word[w] = lat;
        end
    endgenerate
endmodule
`default_nettype wire
