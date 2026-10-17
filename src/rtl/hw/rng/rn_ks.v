`default_nettype none
// rn_ks: Threefry-2x32 key schedule and counter word 1 for one dropout mask stream (model/rng.py,
// docs/dataflow.md section 7), combinational:
//   key (k0, k1) = (seed[31:0], seed[63:32]), ks2 = 0x1BD11BDA ^ k0 ^ k1
//   counter word 1 c1 = step << 8 | site << 4 | layer  (step < 2^24, site / layer < 16)
//   x1i = c1 + k1 (the initial x1), injections j = 1, 2, 3: ija = ks[j mod 3], ijb = ks[(j+1) mod 3] + j
module rn_ks (
    input  wire [63:0] seed,
    input  wire [23:0] step,
    input  wire [3:0]  site,
    input  wire [3:0]  layer,
    output wire [31:0] k0,
    output wire [31:0] x1i,
    output wire [31:0] i1a, i1b, i2a, i2b, i3a, i3b
);
    wire [31:0] k1  = seed[63:32];
    assign k0       = seed[31:0];
    wire [31:0] ks2 = 32'h1BD11BDA ^ k0 ^ k1;
    assign x1i = {step, site, layer} + k1;
    assign i1a = k1;
    assign i1b = ks2 + 32'd1;
    assign i2a = ks2;
    assign i2b = k0 + 32'd2;
    assign i3a = k0;
    assign i3b = k1 + 32'd3;
endmodule
`default_nettype wire
