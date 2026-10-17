`default_nettype none
// Synthesis/STA wrapper (not part of the design): registers every input of fp_i2f so all timing
// paths are register to register.  Parameters are overridden by flow/synth/synth.sh.
module wrap_fp_i2f #(parameter STAGES = 1) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [31:0] x,
    output wire        out_valid,
    output wire [31:0] y
);
    reg v_r; reg [31:0] x_r;
    always @(posedge clk) begin v_r <= in_valid; x_r <= x; end
    (* keep_hierarchy *) fp_i2f #(.STAGES(STAGES)) u_dut (.clk(clk), .rst_n(rst_n), .in_valid(v_r), .x(x_r),
        .out_valid(out_valid), .y(y));
endmodule
`default_nettype wire
