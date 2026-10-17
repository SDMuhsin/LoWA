`default_nettype none
// cm_rstbuf: one registered copy of the synchronous reset (docs/p3_report.md section 6).
// A reset tree is built from N instances of this module, one per load group, so that no single
// reset net fans out to a whole block.  The module keeps its hierarchy through `synth -flatten`
// ((* keep_hierarchy *)): Yosys opt_merge folds identical flops of one module into a single
// flop, but never merges across a kept module boundary, so every instance stays a separate
// flop in the netlist.  Latency: rst_n_q = rst_n delayed one cycle (all copies release in the
// same cycle).
(* keep_hierarchy *)
module cm_rstbuf (
    input  wire clk,
    input  wire rst_n,
    output reg  rst_n_q
);
    always @(posedge clk) rst_n_q <= rst_n;
endmodule
`default_nettype wire
