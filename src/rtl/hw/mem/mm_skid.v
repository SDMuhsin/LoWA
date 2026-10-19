`default_nettype none
// mm_skid: two-entry skid buffer with a registered ready, in front of every mm_top engine request port (P8f, PLAN R17).
// in_ready is a flop (= skid entry empty), so no combinational path runs from an engine's request through mm_top's
// ordering checks back to its ready.  Full throughput: an accepted request goes to the head when the head is free or
// leaves this cycle, otherwise to the skid entry; the head leaves on h_go (mm_top: the port takes it).  Latency from
// in_valid to the head: one cycle (the head register is the old request register / stage-A input).
// Each entry carries NC 2-bit age counters (mm_top's handshake-order bookkeeping: per other channel, how many of its
// buffered requests are older than this entry, 0..2); dec[i] decrements counter i of both entries (saturating at 0),
// in_c is the new entry's value (already net of this cycle's departures, so it is not decremented).
module mm_skid #(
    parameter W  = 32,                       // payload bits
    parameter NC = 1                         // age counters per entry
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              in_valid,
    output reg               in_ready,       // registered: the skid entry is empty
    input  wire [W-1:0]      in_d,
    input  wire [2*NC-1:0]   in_c,
    input  wire [NC-1:0]     dec,
    output reg               h_v,            // head (oldest entry)
    output reg  [W-1:0]      h_d,
    output reg  [2*NC-1:0]   h_c,
    input  wire              h_go,           // the head leaves this cycle (only with h_v)
    output reg               s_v,            // skid entry (younger; s_v implies h_v)
    output reg  [W-1:0]      s_d,
    output reg  [2*NC-1:0]   s_c
);
    wire hfree = !h_v || h_go;
    wire acc   = in_valid && in_ready;
    reg  [2*NC-1:0] hc_d, sc_d;              // counters after this cycle's decrements
    integer i;
    always @* begin
        for (i = 0; i < NC; i = i + 1) begin
            hc_d[2*i +: 2] = h_c[2*i +: 2] - {1'b0, dec[i] && (h_c[2*i +: 2] != 2'd0)};
            sc_d[2*i +: 2] = s_c[2*i +: 2] - {1'b0, dec[i] && (s_c[2*i +: 2] != 2'd0)};
        end
    end
    always @(posedge clk) begin
        if (!rst_n) begin
            h_v <= 1'b0; s_v <= 1'b0; in_ready <= 1'b0;
        end else if (hfree) begin
            h_v <= s_v || acc; s_v <= 1'b0; in_ready <= 1'b1;
        end else begin
            if (acc) s_v <= 1'b1;
            in_ready <= !(s_v || acc);
        end
    end
    // payload and counters (no reset; read only while the entry's valid bit is set)
    always @(posedge clk) begin
        if (hfree) begin
            if (s_v) begin h_d <= s_d; h_c <= sc_d; end
            else if (acc) begin h_d <= in_d; h_c <= in_c; end
        end else begin
            h_c <= hc_d;
            if (acc) begin s_d <= in_d; s_c <= in_c; end
            else s_c <= sc_d;
        end
    end
endmodule
`default_nettype wire
