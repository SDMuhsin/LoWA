`default_nettype none
// Template (P8e, PLAN R16): copy to site/mm_sram_macro_site.v (git-ignored).  Maps mm_sram_macro (the stand-in in
// hw/mem/tech/mm_sram_macro.v) to the compiled macro of each bank shape used in full16 (sram/mc_gen.sh gen):
// chip enable and write enable active low, margin-adjust pins tied to the setting the datatable summary uses,
// retention / power-down pins inactive.  The macro's read data is write-through, which mm_bank_t65s handles.
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
    generate
        if (WORDS == 256 && WIDTH == 32) begin : g256
            <MACRO_256x32> u_m (.<Q>(q), .<CLK>(clk), .<CEN>(cen_n), .<WEN>(wen_n), .<A>(a), .<D>(d) <TIE_OFFS>);
        end else if (WORDS == 512 && WIDTH == 32) begin : g512
            <MACRO_512x32> u_m (.<Q>(q), .<CLK>(clk), .<CEN>(cen_n), .<WEN>(wen_n), .<A>(a), .<D>(d) <TIE_OFFS>);
        end else if (WORDS == 768 && WIDTH == 32) begin : g768
            <MACRO_768x32> u_m (.<Q>(q), .<CLK>(clk), .<CEN>(cen_n), .<WEN>(wen_n), .<A>(a), .<D>(d) <TIE_OFFS>);
        end else begin : g_none
            mm_sram_macro_no_such_shape u_m ();   // elaboration error: no macro generated for this shape
        end
    endgenerate
endmodule
`default_nettype wire
