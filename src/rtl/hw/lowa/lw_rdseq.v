`default_nettype none
// lw_rdseq: sequential read of `nwords` consecutive fp32 words from byte address `base` through one
// memory read channel (requests of <= 64 words, in-order 256-bit beats as in the GEMM engine's
// rd channels, up to 8 requests outstanding) delivered as one word per cycle with its index.
// `go` (one cycle, when idle) starts a transfer; `done` pulses with the last word delivered.
module lw_rdseq (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         go,
    input  wire [31:0]  base,
    input  wire [15:0]  nwords,
    output wire         busy,
    output wire         done,
    output reg          req_valid,
    input  wire         req_ready,
    output reg  [31:0]  req_addr,
    output reg  [6:0]   req_len,
    input  wire         dat_valid,
    output wire         dat_ready,
    input  wire [255:0] dat,
    output wire         w_valid,
    input  wire         w_ready,
    output wire [31:0]  w_d,
    output reg  [15:0]  w_idx
);
    reg         active;
    reg  [15:0] rem_req;                        // words still to request
    reg  [15:0] rem_dat;                        // words still to deliver
    wire [6:0]  this_len = (rem_req > 16'd64) ? 7'd64 : rem_req[6:0];
    // outstanding request lengths (in-order beats)
    wire        lq_in_ready, lq_out_valid;
    wire [6:0]  lq_len;
    wire        lq_pop;
    wire [3:0]  lq_count;
    lw_fifo #(.W(7), .LOG2(3)) u_lq (.clk(clk), .rst_n(rst_n), .in_valid(req_valid & req_ready), .in_ready(lq_in_ready),
                                      .in_d(req_len), .out_valid(lq_out_valid), .out_ready(lq_pop), .out_d(lq_len), .count(lq_count));
    // beats of the head request
    reg  [6:0]  bt_done;                        // words of the head request already taken
    wire [6:0]  bt_left = lq_len - bt_done;
    wire [3:0]  beat_nw = (bt_left > 7'd8) ? 4'd8 : bt_left[3:0];
    wire        beat_last = (bt_left <= 7'd8);
    wire        beat_ready;
    wire        beat_fire = dat_valid & dat_ready;
    assign dat_ready = beat_ready & lq_out_valid;
    assign lq_pop = beat_fire & beat_last;
    wire        w_last;
    lw_unpack u_un (.clk(clk), .rst_n(rst_n), .beat_valid(dat_valid & lq_out_valid), .beat_ready(beat_ready), .beat_d(dat),
                    .beat_nw(beat_nw), .w_valid(w_valid), .w_ready(w_ready), .w_d(w_d), .w_last(w_last));
    wire w_fire = w_valid & w_ready;
    assign busy = active;
    wire unused_r = (|lq_count) | w_last;
    assign done = active & w_fire & (rem_dat == 16'd1);
    always @(posedge clk)
        if (!rst_n) begin
            active <= 1'b0; req_valid <= 1'b0; rem_req <= 16'd0; rem_dat <= 16'd0; bt_done <= 7'd0; w_idx <= 16'd0;
        end else begin
            if (go & ~active) begin
                active <= 1'b1; rem_req <= nwords; rem_dat <= nwords; req_addr <= base; w_idx <= 16'd0; bt_done <= 7'd0;
            end
            if (req_valid & req_ready) begin
                req_valid <= 1'b0;
                req_addr <= req_addr + {23'd0, req_len, 2'b00};
                rem_req <= rem_req - {9'd0, req_len};
            end else if (active & ~req_valid & (rem_req != 16'd0) & lq_in_ready) begin
                req_valid <= 1'b1; req_len <= this_len;
            end
            if (beat_fire) bt_done <= beat_last ? 7'd0 : bt_done + 7'd8;
            if (w_fire) begin
                w_idx <= w_idx + 16'd1;
                rem_dat <= rem_dat - 16'd1;
                if (rem_dat == 16'd1) active <= 1'b0;
            end
        end
endmodule
`default_nettype wire
