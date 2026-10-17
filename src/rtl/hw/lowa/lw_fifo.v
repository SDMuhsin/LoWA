`default_nettype none
// lw_fifo: flop FIFO, DEPTH = 2**LOG2 entries of W bits, valid/ready on both sides (one push and one
// pop per cycle, pop of a just-pushed entry the cycle after).  The read side is a one-hot select
// through cm_ohmux and the write side uses per-entry one-hot enables (no binary-pointer register
// file reads: docs/p3_report.md section 6).  `count` = entries held, `free` = DEPTH - count.
module lw_fifo #(
    parameter W    = 32,
    parameter LOG2 = 4
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          in_valid,
    output wire          in_ready,
    input  wire [W-1:0]  in_d,
    output wire          out_valid,
    input  wire          out_ready,
    output wire [W-1:0]  out_d,
    output reg  [LOG2:0] count
);
    localparam N = 1 << LOG2;
    reg  [N-1:0]  wp_oh, rp_oh;             // one-hot pointers
    reg  [N*W-1:0] mem;
    wire push = in_valid & in_ready;
    wire pop  = out_valid & out_ready;
    assign in_ready  = (count != N[LOG2:0]);
    assign out_valid = (count != {(LOG2 + 1){1'b0}});
    always @(posedge clk)
        if (!rst_n) begin
            wp_oh <= {{(N - 1){1'b0}}, 1'b1};
            rp_oh <= {{(N - 1){1'b0}}, 1'b1};
            count <= {(LOG2 + 1){1'b0}};
        end else begin
            if (push) wp_oh <= {wp_oh[N-2:0], wp_oh[N-1]};
            if (pop)  rp_oh <= {rp_oh[N-2:0], rp_oh[N-1]};
            count <= count + {{LOG2{1'b0}}, push} - {{LOG2{1'b0}}, pop};
        end
    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : g_e
            always @(posedge clk) if (push & wp_oh[i]) mem[i*W +: W] <= in_d;
        end
    endgenerate
    cm_ohmux #(.W(W), .N(N), .REP(1)) u_rd (.sel(rp_oh), .d(mem), .y(out_d));
endmodule
`default_nettype wire
