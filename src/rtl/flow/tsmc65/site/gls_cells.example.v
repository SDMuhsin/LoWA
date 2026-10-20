// Template: copy to site/gls_cells.v (git-ignored) and give one line per sequential library cell the chip
// netlists use (P9b gate-level power simulation, sim/run_gls.sh).  The library's own Verilog models clock their
// flops and latches through zero-delay UDPs; with ideal clocks a flop behind a clock gate then samples its data a
// delta cycle after the ungated flops have updated.  These models move every flop / latch output GLS_DQ ns after
// its clock edge / enable, keep the clock gates at zero delay and start all state at 0.  Pin names must match
// the library cell.
`timescale 1ns/10ps
`ifndef GLS_DQ
`define GLS_DQ 0.05
`endif
// D flop (D, CP, Q); with QN; synchronous active-low clear (Q <= D & CN); mux flop (Q <= SA ? DA : DB);
// enable flop; asynchronous active-low clear (with / without QN); high-transparent latch; positive-edge ICG
// (latch transparent while CP low on E | TE, Q = CP & latch)
`define GLS_DFF(name)  module name (D, CP, Q);  input D, CP; output reg Q = 1'b0; always @(posedge CP) Q <= #(`GLS_DQ) D; endmodule
`define GLS_DFFN(name) module name (D, CP, Q, QN); input D, CP; output reg Q = 1'b0; output QN; assign QN = ~Q; always @(posedge CP) Q <= #(`GLS_DQ) D; endmodule
`define GLS_DFFK(name) module name (D, CP, CN, Q); input D, CP, CN; output reg Q = 1'b0; always @(posedge CP) Q <= #(`GLS_DQ) (D & CN); endmodule
`define GLS_DFFX(name) module name (DA, DB, SA, CP, Q); input DA, DB, SA, CP; output reg Q = 1'b0; always @(posedge CP) Q <= #(`GLS_DQ) (SA ? DA : DB); endmodule
`define GLS_DFFE(name) module name (D, E, CP, Q); input D, E, CP; output reg Q = 1'b0; always @(posedge CP) if (E) Q <= #(`GLS_DQ) D; endmodule
`define GLS_DFFC(name) module name (D, CP, CDN, Q); input D, CP, CDN; output reg Q = 1'b0; always @(posedge CP or negedge CDN) if (!CDN) Q <= #(`GLS_DQ) 1'b0; else Q <= #(`GLS_DQ) D; endmodule
`define GLS_DFFCN(name) module name (D, CP, CDN, Q, QN); input D, CP, CDN; output reg Q = 1'b0; output QN; assign QN = ~Q; always @(posedge CP or negedge CDN) if (!CDN) Q <= #(`GLS_DQ) 1'b0; else Q <= #(`GLS_DQ) D; endmodule
`define GLS_LAT(name)  module name (D, E, Q); input D, E; output reg Q = 1'b0; always @(D or E) if (E) Q <= #(`GLS_DQ) D; endmodule
`define GLS_ICG(name)  module name (TE, E, CP, Q); input TE, E, CP; output Q; reg en = 1'b0; always @(CP or E or TE) if (!CP) en = E | TE; assign Q = CP & en; endmodule
`GLS_DFF(<DFF_CELL>)
`GLS_LAT(<LATCH_CELL>)
`GLS_ICG(<ICG_CELL>)
