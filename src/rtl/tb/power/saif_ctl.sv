`timescale 1ns/1ps
// saif_ctl: second top-level module for the P8d power runs (VCS).  Records the toggles of the DUT of a unit
// testbench for +saifcyc=<n> clock cycles from time 0 (reset, descriptor start, loads and the steady state of
// the op), writes dut.saif in the working directory and ends the simulation.  The testbench itself is unchanged; its own checks do
// not run, because the op is cut off after n cycles.  Select the testbench with +define+SAIF_VU (tb_vu_top),
// default tb_gm_top.
`ifdef SAIF_VU
 `define SAIF_TB tb_vu_top
 `define SAIF_SCOPE "tb_vu_top.dut"
`else
 `define SAIF_TB tb_gm_top
 `define SAIF_SCOPE "tb_gm_top.dut"
`endif
module saif_ctl;
    integer n;
    initial begin
        if (!$value$plusargs("saifcyc=%d", n)) n = 20000;
        $set_toggle_region(`SAIF_TB.dut);
        $toggle_start;
        repeat (n) @(posedge `SAIF_TB.clk);
        $toggle_stop;
        $toggle_report("dut.saif", 1.0e-9, `SAIF_SCOPE);
        $display("SAIF_WRITTEN %0d cycles", n);
        $finish;
    end
endmodule
