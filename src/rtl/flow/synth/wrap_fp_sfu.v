`default_nettype none
// Synthesis/STA wrapper (not part of the design): registers every input of fp_sfu so all timing
// paths are register to register.  Parameters are overridden by flow/synth/synth.sh.
module wrap_fp_sfu #(parameter LF = 2, parameter NTHR = LF + 2) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    output wire        in_ready,
    input  wire [2:0]  in_fn,
    input  wire [31:0] in_x,
    input  wire [3:0]  in_tag,
    output wire        out_valid,
    output wire [31:0] out_y,
    output wire [3:0]  out_tag
);
    reg v_r; reg [2:0] f_r; reg [31:0] x_r; reg [3:0] t_r;
    always @(posedge clk) begin v_r <= in_valid; f_r <= in_fn; x_r <= in_x; t_r <= in_tag; end
    (* keep_hierarchy *) fp_sfu #(.LF(LF), .NTHR(NTHR), .TAGW(4)) u_dut (.clk(clk), .rst_n(rst_n),
        .in_valid(v_r), .in_ready(in_ready), .in_fn(f_r), .in_x(x_r), .in_tag(t_r),
        .out_valid(out_valid), .out_y(out_y), .out_tag(out_tag));
endmodule
`default_nettype wire
