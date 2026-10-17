`default_nettype none
// gf180mcu_ocd_ip_sram__sram256x8m8wm1: the GF180 256 x 8 single-port SRAM macro (P3, hw/mem).
// File/module name = the vendor cell name (the one documented exception to the mm_ prefix rule),
// so that P6/P8 bind the real LEF/GDS/LIB by instance name (llmdocs/asic_flow_tutorial.md section 3).
//   `SYNTHESIS : empty (* blackbox *) module, the macro is kept as a cell (Yosys -DSYNTHESIS).
//   otherwise  : behavioural model, same semantics as GanAccelAsic's signed-off copy
//                (src/rtl/mem/gf180_sram_macro.v): active-low CEN / GWEN / per-bit WEN,
//                Q registered on enabled cycles (read data valid the cycle after the access).
`ifdef SYNTHESIS
(* blackbox *)
module gf180mcu_ocd_ip_sram__sram256x8m8wm1 (
    input  wire       CLK,
    input  wire       CEN,
    input  wire [7:0] A,
    input  wire       GWEN,
    input  wire [7:0] WEN,
    input  wire [7:0] D,
    output wire [7:0] Q
);
endmodule
`else
module gf180mcu_ocd_ip_sram__sram256x8m8wm1 (
    input  wire       CLK,
    input  wire       CEN,
    input  wire [7:0] A,
    input  wire       GWEN,
    input  wire [7:0] WEN,
    input  wire [7:0] D,
    output wire [7:0] Q
);
    reg [7:0] mem [0:255];
    reg [7:0] q_reg;
    assign Q = q_reg;
    integer i;
    always @(posedge CLK) begin
        if (!CEN) begin
            if (!GWEN) begin
                for (i = 0; i < 8; i = i + 1)
                    if (!WEN[i]) mem[A][i] <= D[i];
            end
            q_reg <= mem[A];
        end
    end
endmodule
`endif
`default_nettype wire
