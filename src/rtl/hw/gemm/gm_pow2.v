`default_nettype none
// gm_pow2: y = mul(x, 2^e) for a signed 8-bit e, bit-identical to L1 `mul` / `scale` with a
// power-of-two constant (the only scales isa.md allows: 0.125 for the attention scores, 0.5 in
// GELU).  The product of a normal x by 2^e is exact, so RNE is the identity and only the
// exponent moves: DAZ (subnormal x -> +-0), FTZ (new exponent <= 0 -> +-0, sign kept),
// overflow (new exponent >= 255 -> +-inf), NaN -> 0x7FC00000, +-inf -> +-inf, +-0 -> +-0.
// Combinational (used inside registered epilogue stages).
module gm_pow2 (
    input  wire [31:0] x,
    input  wire [7:0]  e,                   // two's complement
    output wire [31:0] y
);
    wire        s  = x[31];
    wire [7:0]  ex = x[30:23];
    wire        z  = (ex == 8'd0);
    wire        xx = (ex == 8'hFF);
    wire        nan = xx & (x[22:0] != 23'd0);
    wire [9:0]  en = {2'b00, ex} + {{2{e[7]}}, e};      // signed sum, 10 bits
    wire        ovf = ~en[9] & (en[8] | (en[7:0] == 8'hFF));
    wire        udf = en[9] | (en[8:0] == 9'd0);
    assign y = nan ? 32'h7FC00000 :
               xx  ? {s, 8'hFF, 23'd0} :
               z   ? {s, 31'd0} :
               ovf ? {s, 8'hFF, 23'd0} :
               udf ? {s, 31'd0} : {s, en[7:0], x[22:0]};
endmodule
`default_nettype wire
