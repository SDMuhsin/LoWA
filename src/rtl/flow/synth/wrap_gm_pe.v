`default_nettype none
// Synthesis/STA wrapper (not part of the design): registers every input of gm_pe so all timing
// paths are register to register.  Parameters are overridden by flow/synth/synth.sh.
module wrap_gm_pe #(parameter FMA_STAGES = 2, parameter LAMBDA = 4, parameter QW = 2) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire [31:0]   a_in, b_in, dr_in,
    input  wire [QW+2:0] ctl_in,
    input  wire          inj,
    input  wire [QW-1:0] inj_q,
    output wire [31:0]   a_out, b_out, dr_out,
    output wire [QW+2:0] ctl_out,
    output wire          err_overrun
);
    reg [31:0] a_r, b_r, d_r; reg [QW+2:0] c_r; reg i_r; reg [QW-1:0] q_r;
    always @(posedge clk) begin a_r <= a_in; b_r <= b_in; d_r <= dr_in; c_r <= ctl_in; i_r <= inj; q_r <= inj_q; end
    (* keep_hierarchy *) gm_pe #(.FMA_STAGES(FMA_STAGES), .LAMBDA(LAMBDA), .QW(QW)) u_dut (
        .clk(clk), .rst_n(rst_n), .a_in(a_r), .b_in(b_r), .ctl_in(c_r), .a_out(a_out), .b_out(b_out),
        .ctl_out(ctl_out), .inj(i_r), .inj_q(q_r), .dr_in(d_r), .dr_out(dr_out), .err_overrun(err_overrun));
endmodule
`default_nettype wire
