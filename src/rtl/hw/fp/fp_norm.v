`default_nettype none
// fp_norm: leading-zero count and left normalisation of a W-bit magnitude (combinational).
// y = x << lz with y[W-1] = 1 for x != 0; lz = number of leading zeros of x.  For x = 0, y = 0 and
// lz is meaningless (callers handle zero separately).  CW must satisfy 2^CW >= W.  Log shifter:
// stage j tests the top 2^j bits and shifts by 2^j if they are all zero (CW stages, MSB first).
module fp_norm #(
    parameter W  = 32,
    parameter CW = 5
) (
    input  wire [W-1:0]  x,
    output wire [W-1:0]  y,
    output wire [CW-1:0] lz
);
    localparam P = 1 << CW;                      // padded width
    wire [P-1:0] st [0:CW];
    generate
        if (P > W) begin : g_pad
            assign st[0] = {x, {(P - W){1'b0}}};
        end else begin : g_nopad
            assign st[0] = x;
        end
        genvar i;
        for (i = 0; i < CW; i = i + 1) begin : g_stage
            localparam integer S = 1 << (CW - 1 - i);
            wire z = ~|st[i][P-1 -: S];
            assign st[i+1] = z ? (st[i] << S) : st[i];
            assign lz[CW-1-i] = z;
        end
    endgenerate
    assign y = st[CW][P-1 -: W];
endmodule
`default_nettype wire
