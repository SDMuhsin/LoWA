`default_nettype none
// lw_unpack: 256-bit memory beat -> one 32-bit word per cycle (word 0 first).  `beat_nw` (1..8) is
// the number of valid words of the beat (a request of len words delivers ceil(len/8) beats; the
// surplus words of the last beat are dropped here).  Valid/ready on both sides; a beat is accepted
// when the holding register is free or its last word is leaving.
module lw_unpack (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         beat_valid,
    output wire         beat_ready,
    input  wire [255:0] beat_d,
    input  wire [3:0]   beat_nw,
    output wire         w_valid,
    input  wire         w_ready,
    output wire [31:0]  w_d,
    output wire         w_last
);
    reg          full;
    reg  [255:0] d;
    reg  [3:0]   nw;
    reg  [2:0]   idx;
    wire fire = w_valid & w_ready;
    assign w_last = ({1'b0, idx} == nw - 4'd1);
    assign beat_ready = ~full | (fire & w_last);
    assign w_valid = full;
    wire [7:0] sel_oh = (idx == 3'd0) ? 8'h01 : (idx == 3'd1) ? 8'h02 : (idx == 3'd2) ? 8'h04 : (idx == 3'd3) ? 8'h08 :
                        (idx == 3'd4) ? 8'h10 : (idx == 3'd5) ? 8'h20 : (idx == 3'd6) ? 8'h40 : 8'h80;
    cm_ohmux #(.W(32), .N(8), .REP(1)) u_sel (.sel(sel_oh), .d(d), .y(w_d));
    always @(posedge clk)
        if (!rst_n) begin
            full <= 1'b0; idx <= 3'd0;
        end else begin
            if (beat_valid & beat_ready) begin
                full <= 1'b1; idx <= 3'd0;
            end else if (fire) begin
                idx <= idx + 3'd1;
                if (w_last) full <= 1'b0;
            end
        end
    always @(posedge clk) if (beat_valid & beat_ready) begin d <= beat_d; nw <= beat_nw; end
endmodule
`default_nettype wire
