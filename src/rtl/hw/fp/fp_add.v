`default_nettype none
// fp_add: y = a + b (sub = 0) or a - b (sub = 1), binary32, one rounding (RNE), DAZ inputs, FTZ
// results, overflow -> +-inf, NaN = 0x7FC00000, inf - inf = NaN, exact zero sums are +0 except
// (-0) + (-0) = -0 (docs/arith.md section 1).  Bit-identical to L1 `add` / `sub`.
//
// Datapath: swap so |big| >= |small|; window w[27:0] holds big at [27:4]; small is shifted right
// by d = Ebig - Esmall, bits [27:1] are kept exactly and w[0] is the sticky OR of everything
// shifted below bit 1.  Real bits lost implies d >= 2, so the result MSB is at >= 26 and the round
// bit at >= 2: the sticky never sits next to the round bit (exact RNE).
//
// Fixed-latency valid-only pipeline, latency = STAGES (>= 1).
//   stage A: decode, specials, magnitude compare, swap, align        cut 1 (STAGES >= 3)
//   stage B: 29-bit add / subtract                                   cut 2 (STAGES >= 2)
//   stage C: leading-zero count + normalise                          cut 3 (STAGES >= 4)
//   stage D: round, FTZ / overflow, pack, special override           output register
//   STAGES > 4 adds STAGES-4 delay registers after the output register.
module fp_add #(
    parameter STAGES = 2
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire        sub,
    output wire        out_valid,
    output wire [31:0] y
);
    localparam C1 = (STAGES >= 3) ? 1 : 0;
    localparam C2 = (STAGES >= 2) ? 1 : 0;
    localparam C3 = (STAGES >= 4) ? 1 : 0;
    localparam XD = (STAGES > 4) ? STAGES - 4 : 0;

    // ------------------------------------------------------------------ stage A
    wire        sa = a[31], sb = b[31] ^ sub;
    wire [7:0]  ea = a[30:23], eb = b[30:23];
    wire        za = (ea == 8'd0), zb = (eb == 8'd0);
    wire        xa = (ea == 8'hFF), xb = (eb == 8'hFF);
    wire        na = xa & (a[22:0] != 23'd0), nb = xb & (b[22:0] != 23'd0);
    wire        ia = xa & ~na, ib = xb & ~nb;
    wire        is_nan = na | nb | (ia & ib & (sa ^ sb));
    wire        is_inf = (ia | ib) & ~is_nan;
    wire        inf_s  = ia ? sa : sb;
    wire        both0  = za & zb;
    // DAZ'd magnitudes {exp, frac}; a zero operand has magnitude 0
    wire [30:0] ma = za ? 31'd0 : a[30:0];
    wire [30:0] mb = zb ? 31'd0 : b[30:0];
    wire        swp = (mb > ma);
    wire [30:0] mg = swp ? mb : ma;
    wire [30:0] ml = swp ? ma : mb;
    wire        sg = swp ? sb : sa;
    wire        eff_sub = sa ^ sb;
    wire [7:0]  d  = mg[30:23] - ml[30:23];
    wire [23:0] fl = (ml[30:23] == 8'd0) ? 24'd0 : {1'b1, ml[22:0]};
    wire [27:0] xl = {fl, 4'd0};
    wire [4:0]  dsat = (d > 8'd28) ? 5'd28 : d[4:0];
    wire [27:0] shl  = xl >> dsat;
    wire [27:0] lost = xl & ~({28{1'b1}} << dsat);
    wire [27:0] wl = {shl[27:1], shl[0] | (|lost)};
    wire [27:0] wg = {(mg[30:23] == 8'd0) ? 1'b0 : 1'b1, mg[22:0], 4'd0};
    wire [10:0] eg = {3'd0, mg[30:23]} - 11'd127;     // unbiased exponent of w[27]
    wire [1:0]  cls = is_nan ? 2'd3 : is_inf ? 2'd2 : both0 ? 2'd1 : 2'd0;
    wire        zs  = is_inf ? inf_s : (sa & sb);      // sign for inf / both-zero results

    wire        v1, sg1, es1, zs1;
    wire [1:0]  cls1;
    wire [10:0] eg1;
    wire [27:0] wg1, wl1;
    cm_delay #(.W(1), .N(C1), .RST(1)) u_v1 (.clk(clk), .rst_n(rst_n), .d(in_valid), .q(v1));
    cm_delay #(.W(72), .N(C1), .RST(0)) u_d1 (.clk(clk), .rst_n(rst_n),
        .d({sg, eff_sub, zs, cls, eg, wg, wl}), .q({sg1, es1, zs1, cls1, eg1, wg1, wl1}));

    // ------------------------------------------------------------------ stage B
    wire [28:0] sum = es1 ? ({1'b0, wg1} - {1'b0, wl1}) : ({1'b0, wg1} + {1'b0, wl1});

    wire        v2, sg2, zs2;
    wire [1:0]  cls2;
    wire [10:0] eg2;
    wire [28:0] sum2;
    cm_delay #(.W(1), .N(C2), .RST(1)) u_v2 (.clk(clk), .rst_n(rst_n), .d(v1), .q(v2));
    cm_delay #(.W(44), .N(C2), .RST(0)) u_d2 (.clk(clk), .rst_n(rst_n),
        .d({sg1, zs1, cls1, eg1, sum}), .q({sg2, zs2, cls2, eg2, sum2}));

    // ------------------------------------------------------------------ stage C
    wire [28:0] nm;
    wire [4:0]  lz;
    fp_norm #(.W(29), .CW(5)) u_norm (.x(sum2), .y(nm), .lz(lz));
    wire        exz = (sum2 == 29'd0);                 // exact zero (or both operands zero)
    wire [10:0] en  = eg2 + 11'd1 - {6'd0, lz};        // sum[28] has weight 2^(Ebig + 1)

    wire        v3, sg3, zs3, exz3;
    wire [1:0]  cls3;
    wire [10:0] en3;
    wire [28:0] nm3;
    cm_delay #(.W(1), .N(C3), .RST(1)) u_v3 (.clk(clk), .rst_n(rst_n), .d(v2), .q(v3));
    cm_delay #(.W(45), .N(C3), .RST(0)) u_d3 (.clk(clk), .rst_n(rst_n),
        .d({sg2, zs2, exz, cls2, en, nm}), .q({sg3, zs3, exz3, cls3, en3, nm3}));

    // ------------------------------------------------------------------ stage D
    wire [31:0] rnd;
    fp_round #(.MW(29), .EW(11)) u_rnd (.s(sg3), .e(en3), .m(nm3), .st(1'b0), .y(rnd));
    wire [31:0] res = (cls3 == 2'd3) ? 32'h7FC00000 :
                      (cls3 == 2'd2) ? {zs3, 8'hFF, 23'd0} :
                      (cls3 == 2'd1) ? {zs3, 31'd0} :
                      exz3           ? 32'd0 : rnd;

    wire        v4;
    wire [31:0] y4;
    cm_delay #(.W(1), .N(1), .RST(1)) u_v4 (.clk(clk), .rst_n(rst_n), .d(v3), .q(v4));
    cm_delay #(.W(32), .N(1), .RST(0)) u_d4 (.clk(clk), .rst_n(rst_n), .d(res), .q(y4));
    cm_delay #(.W(1), .N(XD), .RST(1)) u_vx (.clk(clk), .rst_n(rst_n), .d(v4), .q(out_valid));
    cm_delay #(.W(32), .N(XD), .RST(0)) u_dx (.clk(clk), .rst_n(rst_n), .d(y4), .q(y));
endmodule
`default_nettype wire
