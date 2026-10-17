`default_nettype none
// Synthesis/STA wrapper (not part of the design): registers every input of fp_cmp so all timing
// paths are register to register.  Parameters are overridden by flow/synth/synth.sh.
module wrap_fp_cmp #(parameter STAGES = 1) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [31:0] a, b,
    output wire        out_valid, lt, eq, gt, un,
    output wire [31:0] mx, mn
);
    reg v_r; reg [31:0] a_r, b_r;
    always @(posedge clk) begin v_r <= in_valid; a_r <= a; b_r <= b; end
    (* keep_hierarchy *) fp_cmp #(.STAGES(STAGES)) u_dut (.clk(clk), .rst_n(rst_n), .in_valid(v_r), .a(a_r), .b(b_r),
        .out_valid(out_valid), .lt(lt), .eq(eq), .gt(gt), .un(un), .mx(mx), .mn(mn));
endmodule
`default_nettype wire
