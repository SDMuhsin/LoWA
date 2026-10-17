`default_nettype none
// fp_i2f: signed 32-bit integer -> binary32, RNE (x86 cvtsi2ss under MXCSR RNE; L1 `i2f`).
// Used for the step-scalar integers (dataflow.md section 6) and the ln exponent.  0 -> +0.
// Fixed-latency valid-only.  Cut 1 (STAGES >= 2) after |x| + normalise; output register always;
// STAGES > 2 adds delay registers after it.
module fp_i2f #(
    parameter STAGES = 1
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [31:0] x,
    output wire        out_valid,
    output wire [31:0] y
);
    localparam C1 = (STAGES >= 2) ? 1 : 0;
    localparam XD = (STAGES > 2) ? STAGES - 2 : 0;

    wire        s   = x[31];
    wire [31:0] mag = s ? (~x + 32'd1) : x;          // -2^31 -> 0x80000000 (unsigned), exact
    wire [31:0] nm;
    wire [4:0]  lz;
    fp_norm #(.W(32), .CW(5)) u_norm (.x(mag), .y(nm), .lz(lz));
    wire        z   = (x == 32'd0);
    wire [10:0] e   = 11'd31 - {6'd0, lz};

    wire        v1, s1, z1;
    wire [10:0] e1;
    wire [31:0] nm1;
    cm_delay #(.W(1), .N(C1), .RST(1)) u_v1 (.clk(clk), .rst_n(rst_n), .d(in_valid), .q(v1));
    cm_delay #(.W(45), .N(C1), .RST(0)) u_d1 (.clk(clk), .rst_n(rst_n),
        .d({s, z, e, nm}), .q({s1, z1, e1, nm1}));

    wire [31:0] rnd;
    fp_round #(.MW(32), .EW(11)) u_rnd (.s(s1), .e(e1), .m(nm1), .st(1'b0), .y(rnd));
    wire [31:0] res = z1 ? 32'd0 : rnd;

    wire        v2;
    wire [31:0] y2;
    cm_delay #(.W(1), .N(1), .RST(1)) u_v2 (.clk(clk), .rst_n(rst_n), .d(v1), .q(v2));
    cm_delay #(.W(32), .N(1), .RST(0)) u_d2 (.clk(clk), .rst_n(rst_n), .d(res), .q(y2));
    cm_delay #(.W(1), .N(XD), .RST(1)) u_vx (.clk(clk), .rst_n(rst_n), .d(v2), .q(out_valid));
    cm_delay #(.W(32), .N(XD), .RST(0)) u_dx (.clk(clk), .rst_n(rst_n), .d(y2), .q(y));
endmodule
`default_nettype wire
