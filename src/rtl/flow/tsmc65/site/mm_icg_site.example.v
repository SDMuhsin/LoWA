`default_nettype none
// Template: copy to site/mm_icg_site.v (git-ignored) and map mm_icg to the library's integrated
// clock-gating cell (latch-based, positive edge, test enable tied low).
module mm_icg (
    input  wire clk,
    input  wire en,
    output wire gclk
);
    <ICG_CELL> u_cg (.<CLK_PIN>(clk), .<EN_PIN>(en), .<TE_PIN>(1'b0), .<GCLK_PIN>(gclk));
endmodule
`default_nettype wire
