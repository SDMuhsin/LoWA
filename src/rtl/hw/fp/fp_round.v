`default_nettype none
// fp_round: round a normalised magnitude to binary32, RNE, unbounded exponent, then FTZ and
// overflow (docs/arith.md section 1).  Combinational.
//   m[MW-1] = 1 (normalised), value = 1.m[MW-2:0] x 2^e, plus `st` = OR of all bits below m.
//   e: EW-bit two's complement, the unbiased exponent of m[MW-1].
// Result: |r| < 2^-126 after rounding -> +-0 (sign s); > max -> +-inf; else the packed number.
module fp_round #(
    parameter MW = 48,
    parameter EW = 11
) (
    input  wire          s,
    input  wire [EW-1:0] e,
    input  wire [MW-1:0] m,
    input  wire          st,
    output wire [31:0]   y
);
    wire [23:0] q   = m[MW-1 -: 24];
    wire        rb  = m[MW-25];
    wire        sb;
    generate
        if (MW > 25) begin : g_st
            assign sb = (|m[MW-26:0]) | st;
        end else begin : g_nost
            assign sb = st;
        end
    endgenerate
    wire        inc = rb & (sb | q[0]);
    /* verilator lint_off UNUSEDSIGNAL */      // q1[23] is the hidden 1
    wire [24:0] q1  = {1'b0, q} + {24'd0, inc};
    /* verilator lint_on UNUSEDSIGNAL */
    wire        cy  = q1[24];                      // 1.11..1 + ulp = 10.0..0
    wire [EW-1:0] ef  = e + {{(EW-1){1'b0}}, cy};
    localparam [EW-1:0] K128 = 128, K126 = 126;
    wire [EW-1:0] eo  = ef - K128;                 // >= 0  <=>  ef > 127
    wire [EW-1:0] eu  = ef + K126;                 // <  0  <=>  ef < -126
    wire        ovf = ~eo[EW-1];
    wire        unf = eu[EW-1];
    wire [7:0]  eb  = ef[7:0] + 8'd127;            // biased exponent when in range
    wire [22:0] fr  = cy ? 23'd0 : q1[22:0];
    assign y = unf ? {s, 31'd0} :
               ovf ? {s, 8'hFF, 23'd0} :
                     {s, eb, fr};
endmodule
`default_nettype wire
