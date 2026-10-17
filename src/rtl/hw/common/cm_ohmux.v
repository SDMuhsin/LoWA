`default_nettype none
// cm_ohmux: combinational one-hot AND-OR read mux, y = d[i] for the i with sel[i] = 1 (0 if none).
// Written for fanout control (docs/p3_report.md section 6): a binary read pointer of an N-entry,
// W-bit register file drives ~N*W/2 mux selects from a single flop and Yosys / ABC insert no buffer
// tree.  Here every one-hot select bit drives only W/REP AND gates: the output is cut into REP
// slices of SW = ceil(W / REP) bits and slice r uses its own copy of the one-hot vector,
// sel[r*N +: N].  The parent keeps the REP copies in `(* keep *)` registers so that opt_merge does
// not fold them back into one flop.  The OR reduction is two levels (groups of 8 then the groups),
// so the logic depth is ~log2(N) + 3 gates for every N <= 64.
module cm_ohmux #(
    parameter W   = 32,                     // output width
    parameter N   = 4,                      // entries
    parameter REP = 1                       // copies of the one-hot select
) (
    input  wire [REP*N-1:0] sel,
    input  wire [N*W-1:0]   d,
    output wire [W-1:0]     y
);
    localparam SW = (W + REP - 1) / REP;    // slice width
    localparam WP = SW * REP;               // padded width
    localparam G  = 8;                      // OR group size
    localparam NG = (N + G - 1) / G;

    wire [N*WP-1:0] dp;                     // zero-padded entries
    /* verilator lint_off UNUSEDSIGNAL */   // padding bits WP-1:W are zero and unused
    wire [WP-1:0]   yp;
    /* verilator lint_on UNUSEDSIGNAL */
    genvar i, r, g, j;
    generate
        for (i = 0; i < N; i = i + 1) begin : g_pad
            if (WP == W) begin : g_eq
                assign dp[i*WP +: WP] = d[i*W +: W];
            end else begin : g_ne
                assign dp[i*WP +: WP] = {{(WP-W){1'b0}}, d[i*W +: W]};
            end
        end
        for (r = 0; r < REP; r = r + 1) begin : g_slice
            wire [NG*SW-1:0] grp;
            for (g = 0; g < NG; g = g + 1) begin : g_grp
                wire [G*SW-1:0] term;
                for (j = 0; j < G; j = j + 1) begin : g_term
                    if (g*G + j < N) begin : g_in
                        assign term[j*SW +: SW] = dp[(g*G + j)*WP + r*SW +: SW] & {SW{sel[r*N + g*G + j]}};
                    end else begin : g_out
                        assign term[j*SW +: SW] = {SW{1'b0}};
                    end
                end
                assign grp[g*SW +: SW] = term[0*SW +: SW] | term[1*SW +: SW] | term[2*SW +: SW] | term[3*SW +: SW]
                                       | term[4*SW +: SW] | term[5*SW +: SW] | term[6*SW +: SW] | term[7*SW +: SW];
            end
            reg [SW-1:0] acc;
            integer k;
            always @* begin
                acc = {SW{1'b0}};
                for (k = 0; k < NG; k = k + 1) acc = acc | grp[k*SW +: SW];
            end
            assign yp[r*SW +: SW] = acc;
        end
    endgenerate
    assign y = yp[W-1:0];
endmodule
`default_nettype wire
