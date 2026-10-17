`default_nettype none
// mm_icg: integrated clock gate for the tsmc65 latch register file (mm_bank_t65):
// gclk = clk & en_l, en_l = en latched while clk is low (glitch-free positive-edge gate).
// This file is the behavioural model used by every simulation and by the open-source lint.
// The TSMC-65 synthesis flow (flow/tsmc65) does not read it: it reads the git-ignored site
// wrapper flow/tsmc65/site/mm_icg_site.v instead, which maps mm_icg to the library ICG cell
// (template flow/tsmc65/site/mm_icg_site.example.v).  Latch by design (PLAN R11a); the Yosys
// latch check in flow/lint.sh exempts mm_icg and mm_bank_t65 only.
module mm_icg (
    input  wire clk,
    input  wire en,
    output wire gclk
);
    reg en_l;
    /* verilator lint_off COMBDLY */  // latch: non-blocking so posedge samplers never race
    /* verilator lint_off LATCH */    // the enable latch is the point of the cell
    always @(clk or en)
        if (!clk) en_l <= en;
    /* verilator lint_on LATCH */
    /* verilator lint_on COMBDLY */
    assign gclk = clk & en_l;
endmodule
`default_nettype wire
