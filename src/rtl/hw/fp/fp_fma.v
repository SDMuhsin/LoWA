`default_nettype none
// fp_fma: y = a * b + c with the exact product and ONE rounding (RNE), binary32, DAZ inputs, FTZ
// results (tininess after rounding), overflow -> +-inf, NaN = 0x7FC00000, inf * 0 and
// inf - inf = NaN, exact zero results +0 except (-0) + (-0) = -0 (docs/arith.md section 1).
// Bit-identical to L1 `fma`.  mul(a, b) = fma(a, b, -0) and add(a, b) = fma(a, 1, b) exactly.
//
// Datapath (window w[76:0], bit 0 = sticky only):
//   P = ma * mb (48 bits) sits at w[50:3]; its LSB has exponent Ea + Eb - 46.
//   c (24 bits) starts at w[76:53] and is shifted right by sh = 50 - d, d = (Ec - 23) - (Ea + Eb - 46),
//   so its LSB lands at w[3 + d].  Bits that fall below w[1] are OR'ed into w[0].
//   d > 50 (c far above the product) or P = 0: sh = 0.  Then P is placed closer to c than it is,
//   but P < 2^51 < ulp(c) / 4 in window units, so c +- P rounds to c either way (exact RNE).
//   Real bits are lost only when c is far below P (result MSB >= w[47], round bit >= w[23]),
//   so the sticky never touches the round decision.
//   S = c_w +- P_w (78 bits, |.|), normalise (7-level LZC/shift), round with fp_round.
//
// Fixed-latency valid-only pipeline, latency = STAGES (>= 1).
//   stage A: decode, specials, 24 x 24 product, exponent difference, align c    cut 1 (STAGES >= 3)
//   stage B: 78-bit add / subtract and absolute value                            cut 2 (STAGES >= 2)
//   stage C: leading-zero count + normalise                                      cut 3 (STAGES >= 4)
//   stage D: round, FTZ / overflow, pack, special override                       output register
//   STAGES > 4 adds STAGES-4 delay registers after the output register.
// Timing (docs/p2_report.md, Yosys + OpenSTA, ss_125C_3v00, 40 ns): STAGES = 1 fails, 2 closes
// (A+B | C+D), 3 and 4 are limited by stage A.
module fp_fma #(
    parameter STAGES = 2
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    output wire        out_valid,
    output wire [31:0] y
);
    localparam C1 = (STAGES >= 3) ? 1 : 0;
    localparam C2 = (STAGES >= 2) ? 1 : 0;
    localparam C3 = (STAGES >= 4) ? 1 : 0;
    localparam XD = (STAGES > 4) ? STAGES - 4 : 0;

    // ------------------------------------------------------------------ stage A
    wire [7:0]  ea = a[30:23], eb = b[30:23], ec = c[30:23];
    wire        za = (ea == 8'd0), zb = (eb == 8'd0), zc = (ec == 8'd0);   // DAZ
    wire        xa = (ea == 8'hFF), xb = (eb == 8'hFF), xc = (ec == 8'hFF);
    wire        na = xa & (a[22:0] != 23'd0);
    wire        nb = xb & (b[22:0] != 23'd0);
    wire        nc = xc & (c[22:0] != 23'd0);
    wire        ia = xa & ~na, ib = xb & ~nb, ic = xc & ~nc;
    wire        sp = a[31] ^ b[31];
    wire        sc = c[31];
    wire        pinf = ia | ib;
    wire        is_nan = na | nb | nc | (ia & zb) | (za & ib) | (pinf & ic & (sp ^ sc));
    wire        is_inf = (pinf | ic) & ~is_nan;
    wire        inf_s  = pinf ? sp : sc;
    wire        pz     = za | zb;                       // product is an exact zero
    wire        both0  = pz & zc;
    wire [1:0]  cls = is_nan ? 2'd3 : is_inf ? 2'd2 : both0 ? 2'd1 : 2'd0;
    wire        zs  = is_inf ? inf_s : (sp & sc);       // sign for inf / both-zero results

    wire [47:0] prod = pz ? 48'd0 : ({1'b1, a[22:0]} * {1'b1, b[22:0]});
    wire [23:0] mc   = zc ? 24'd0 : {1'b1, c[22:0]};

    // sh = 50 - d = ea + eb - ec - 100 (biased fields); negative or P = 0 -> clamp to 0
    wire [10:0] shr   = {3'd0, ea} + {3'd0, eb} - {3'd0, ec} - 11'd100;
    wire        clamp = (shr[10] | pz) & ~zc;
    wire [6:0]  sh    = clamp ? 7'd0 : (shr[10:7] != 4'd0) ? 7'd77 :
                        (shr[6:0] > 7'd77) ? 7'd77 : shr[6:0];
    wire [75:0] cext  = {mc, 52'd0};                    // c at w[76:53], seen from w[1] up
    wire [75:0] csh   = cext >> sh;                     // = w[76:1] after the shift
    // c bit i (at w[53 + i]) is lost when 53 + i - sh < 1, i.e. i < sh - 52
    wire [4:0]  shm   = (sh <= 7'd52) ? 5'd0 : (sh >= 7'd76) ? 5'd24 : sh[4:0] - 5'd20;  // sh - 52
    wire [23:0] lmask = ~({24{1'b1}} << shm);
    wire        cst   = |(mc & lmask);
    wire [76:0] cw    = {csh, cst};
    // exponent (unbiased) of window bit 0
    wire [10:0] eref  = clamp ? ({3'd0, ec} - 11'd203) : ({3'd0, ea} + {3'd0, eb} - 11'd303);
    wire        esub  = sp ^ sc;

    wire        v1, esub1, sc1, sp1, zs1;
    wire [1:0]  cls1;
    wire [10:0] eref1;
    wire [47:0] prod1;
    wire [76:0] cw1;
    cm_delay #(.W(1), .N(C1), .RST(1)) u_v1 (.clk(clk), .rst_n(rst_n), .d(in_valid), .q(v1));
    cm_delay #(.W(142), .N(C1), .RST(0)) u_d1 (.clk(clk), .rst_n(rst_n),
        .d({esub, sc, sp, zs, cls, eref, prod, cw}),
        .q({esub1, sc1, sp1, zs1, cls1, eref1, prod1, cw1}));

    // ------------------------------------------------------------------ stage B
    wire [77:0] cw78 = {1'b0, cw1};
    wire [77:0] pw78 = {27'd0, prod1, 3'd0};
    wire [77:0] dif  = cw78 - pw78;
    wire        neg  = esub1 & dif[77];
    wire [77:0] mag  = ~esub1 ? (cw78 + pw78) : neg ? (pw78 - cw78) : dif;
    wire        rs   = neg ? sp1 : sc1;                 // sign of a non-zero exact result

    wire        v2, rs2, zs2;
    wire [1:0]  cls2;
    wire [10:0] eref2;
    wire [77:0] mag2;
    cm_delay #(.W(1), .N(C2), .RST(1)) u_v2 (.clk(clk), .rst_n(rst_n), .d(v1), .q(v2));
    cm_delay #(.W(93), .N(C2), .RST(0)) u_d2 (.clk(clk), .rst_n(rst_n),
        .d({rs, zs1, cls1, eref1, mag}), .q({rs2, zs2, cls2, eref2, mag2}));

    // ------------------------------------------------------------------ stage C
    wire [77:0] nm;
    wire [6:0]  lz;
    fp_norm #(.W(78), .CW(7)) u_norm (.x(mag2), .y(nm), .lz(lz));
    wire        exz = (mag2 == 78'd0);
    wire [10:0] en  = eref2 + 11'd77 - {4'd0, lz};

    wire        v3, rs3, zs3, exz3;
    wire [1:0]  cls3;
    wire [10:0] en3;
    wire [77:0] nm3;
    cm_delay #(.W(1), .N(C3), .RST(1)) u_v3 (.clk(clk), .rst_n(rst_n), .d(v2), .q(v3));
    cm_delay #(.W(94), .N(C3), .RST(0)) u_d3 (.clk(clk), .rst_n(rst_n),
        .d({rs2, zs2, exz, cls2, en, nm}), .q({rs3, zs3, exz3, cls3, en3, nm3}));

    // ------------------------------------------------------------------ stage D
    wire [31:0] rnd;
    fp_round #(.MW(78), .EW(11)) u_rnd (.s(rs3), .e(en3), .m(nm3), .st(1'b0), .y(rnd));
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
