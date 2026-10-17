`default_nettype none
// fp_cmp: IEEE comparison, max and min of binary32 operands after DAZ (docs/arith.md section 1:
// "max and compares use the DAZ'd operands").  da, db = a, b with subnormals read as signed zeros.
//   lt/eq/gt = da < db, da == db, da > db (all 0 if unordered); un = a or b is NaN.  +0 == -0.
//   mx = gt ? da : db   (L1 `fmax`: the C expression daz(a) > daz(b) ? daz(a) : daz(b))
//   mn = gt ? db : da   (L1 `fmin`; min(c, 1) is the clip `c > 1 ? 1 : c` of dataflow.md 6)
//   A NaN result is written as 0x7FC00000.  So mx(NaN, x) = x and mx(x, NaN) = NaN, as in L1.
// Fixed-latency valid-only, latency = STAGES (>= 1): combinational logic, then STAGES registers.
module fp_cmp #(
    parameter STAGES = 1
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire        out_valid,
    output wire        lt,
    output wire        eq,
    output wire        gt,
    output wire        un,
    output wire [31:0] mx,
    output wire [31:0] mn
);
    wire        za = (a[30:23] == 8'd0), zb = (b[30:23] == 8'd0);
    wire        na = (a[30:23] == 8'hFF) & (a[22:0] != 23'd0);
    wire        nb = (b[30:23] == 8'hFF) & (b[22:0] != 23'd0);
    wire [31:0] da = za ? {a[31], 31'd0} : a;
    wire [31:0] db = zb ? {b[31], 31'd0} : b;
    wire        sa = da[31], sb = db[31];
    wire        unord = na | nb;
    wire        bothz = za & zb;
    wire        mgt = da[30:0] > db[30:0];
    wire        mlt = da[30:0] < db[30:0];
    // ordered relations on non-NaN operands
    wire        g = ~bothz & ((~sa & sb) | (~sa & ~sb & mgt) | (sa & sb & mlt));
    wire        e = bothz | (da == db);
    wire        r_gt = ~unord & g;
    wire        r_eq = ~unord & e;
    wire        r_lt = ~unord & ~g & ~e;
    wire [31:0] xm = r_gt ? da : db;
    wire [31:0] nm = r_gt ? db : da;
    wire        xn = (xm[30:23] == 8'hFF) & (xm[22:0] != 23'd0);
    wire        nn = (nm[30:23] == 8'hFF) & (nm[22:0] != 23'd0);
    wire [31:0] r_mx = xn ? 32'h7FC00000 : xm;
    wire [31:0] r_mn = nn ? 32'h7FC00000 : nm;

    cm_delay #(.W(1), .N(STAGES), .RST(1)) u_v (.clk(clk), .rst_n(rst_n), .d(in_valid), .q(out_valid));
    cm_delay #(.W(68), .N(STAGES), .RST(0)) u_d (.clk(clk), .rst_n(rst_n),
        .d({r_lt, r_eq, r_gt, unord, r_mx, r_mn}), .q({lt, eq, gt, un, mx, mn}));
endmodule
`default_nettype wire
