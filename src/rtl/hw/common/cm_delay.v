`default_nettype none
// cm_delay: N-cycle delay line (N = 0: wire).  RST = 1 resets the stages to 0 on rst_n = 0
// (synchronous, active low; used for valid bits), RST = 0 leaves them unreset (datapath).
module cm_delay #(
    parameter W   = 1,
    parameter N   = 1,
    parameter RST = 0
) (
    /* verilator lint_off UNUSEDSIGNAL */   // clk unused when N = 0, rst_n when RST = 0 or N = 0
    input  wire         clk,
    input  wire         rst_n,
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire [W-1:0] d,
    output wire [W-1:0] q
);
    generate
        if (N == 0) begin : g_wire
            assign q = d;
        end else begin : g_reg
            reg [W*N-1:0] r;                   // stage k = r[W*k +: W]; stage N-1 is the output
            wire [W*N-1:0] nxt;
            if (N == 1) begin : g_one
                assign nxt = d;
            end else begin : g_many
                assign nxt = {r[W*(N-1)-1:0], d};
            end
            if (RST != 0) begin : g_rst
                always @(posedge clk)
                    if (!rst_n) r <= {(W*N){1'b0}};
                    else        r <= nxt;
            end else begin : g_nrst
                always @(posedge clk) r <= nxt;
            end
            assign q = r[W*N-1 -: W];
        end
    endgenerate
endmodule
`default_nettype wire
