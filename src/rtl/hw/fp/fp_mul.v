`default_nettype none
// fp_mul: y = a * b, binary32, one rounding (RNE), DAZ inputs, FTZ results (tininess after
// rounding), overflow -> +-inf, NaN results = 0x7FC00000, inf * 0 = NaN (docs/arith.md section 1).
// Bit-identical to L1 `mul` (= fma(a, b, -0)).
//
// Fixed-latency valid-only pipeline, latency = STAGES (>= 1).
//   stage A: decode, specials, exponent sum, 24 x 24 significand product
//   cut 1   (enabled when STAGES >= 2)
//   stage B: 1-bit normalise, round, FTZ / overflow, pack, special override
//   output register (always); STAGES > 2 adds STAGES-2 delay registers after it.
module fp_mul #(
    parameter STAGES = 1
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire        out_valid,
    output wire [31:0] y
);
    localparam C1 = (STAGES >= 2) ? 1 : 0;
    localparam XD = (STAGES > 2) ? STAGES - 2 : 0;

    // ------------------------------------------------------------------ stage A
    wire [7:0]  ea = a[30:23], eb = b[30:23];
    wire        za = (ea == 8'd0), zb = (eb == 8'd0);            // DAZ: subnormal reads as zero
    wire        xa = (ea == 8'hFF), xb = (eb == 8'hFF);
    wire        na = xa & (a[22:0] != 23'd0), nb = xb & (b[22:0] != 23'd0);
    wire        ia = xa & ~na, ib = xb & ~nb;
    wire        sp = a[31] ^ b[31];
    wire        is_nan  = na | nb | (ia & zb) | (za & ib);
    wire        is_inf  = (ia | ib) & ~is_nan;
    wire        is_zero = (za | zb) & ~xa & ~xb;
    wire [47:0] prod = {1'b1, a[22:0]} * {1'b1, b[22:0]};
    wire [10:0] esum = {3'd0, ea} + {3'd0, eb} - 11'd254;         // Ea + Eb (unbiased)
    wire [1:0]  cls  = is_nan ? 2'd3 : is_inf ? 2'd2 : is_zero ? 2'd1 : 2'd0;

    wire        v1, sp1;
    wire [1:0]  cls1;
    wire [10:0] esum1;
    wire [47:0] prod1;
    cm_delay #(.W(1), .N(C1), .RST(1)) u_v1 (.clk(clk), .rst_n(rst_n), .d(in_valid), .q(v1));
    cm_delay #(.W(62), .N(C1), .RST(0)) u_d1 (.clk(clk), .rst_n(rst_n),
        .d({sp, cls, esum, prod}), .q({sp1, cls1, esum1, prod1}));

    // ------------------------------------------------------------------ stage B
    wire        hi = prod1[47];
    wire [47:0] mn = hi ? prod1 : {prod1[46:0], 1'b0};
    wire [10:0] en = esum1 + {10'd0, hi};
    wire [31:0] rnd;
    fp_round #(.MW(48), .EW(11)) u_rnd (.s(sp1), .e(en), .m(mn), .st(1'b0), .y(rnd));
    wire [31:0] res = (cls1 == 2'd3) ? 32'h7FC00000 :
                      (cls1 == 2'd2) ? {sp1, 8'hFF, 23'd0} :
                      (cls1 == 2'd1) ? {sp1, 31'd0} : rnd;

    wire        v2;
    wire [31:0] y2;
    cm_delay #(.W(1), .N(1), .RST(1)) u_v2 (.clk(clk), .rst_n(rst_n), .d(v1), .q(v2));
    cm_delay #(.W(32), .N(1), .RST(0)) u_d2 (.clk(clk), .rst_n(rst_n), .d(res), .q(y2));
    cm_delay #(.W(1), .N(XD), .RST(1)) u_vx (.clk(clk), .rst_n(rst_n), .d(v2), .q(out_valid));
    cm_delay #(.W(32), .N(XD), .RST(0)) u_dx (.clk(clk), .rst_n(rst_n), .d(y2), .q(y));
endmodule
`default_nettype wire
