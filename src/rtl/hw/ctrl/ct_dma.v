`default_nettype none
// ct_dma: DMA_COPY (ISA 0x30) engine of the sequencer (P6c).  Copies n2 x n1 rows of n0 words:
//   for i2 < n2, i1 < n1: words [src + i2*s_s2 + i1*s_s1, + 4*n0) -> [dst + i2*d_s2 + i1*d_s1, ...)
// in chunks of <= 64 words (one read request each); every 8-word read beat is held in one register and written
// as one write beat (word j -> dst + 4j, word mask for the chunk tail).  One chunk at a time (a chunk's writes
// are issued before the next chunk is read), so a copy whose source and destination overlap behaves as a
// sequential word copy chunk by chunk.  The read channel is shared with the sequencer's fetch (rd_own).
module ct_dma (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire [32*9-1:0] p,          // src, dst, n0, n1, s_s1, d_s1, n2, s_s2, d_s2 (isa.md DMA_COPY)
    output reg          busy,
    output reg          done,
    output wire         rd_own,        // the read channel is in use by the DMA
    output reg          rd_req_valid,
    input  wire         rd_req_ready,
    output reg  [31:0]  rd_req_addr,
    output reg  [6:0]   rd_req_len,
    input  wire         rd_dat_valid,
    output wire         rd_dat_ready,
    input  wire [255:0] rd_dat,
    output reg          wr_valid,
    input  wire         wr_ready,
    output reg  [31:0]  wr_addr,
    output reg  [255:0] wr_data,
    output reg  [7:0]   wr_mask
);
    localparam S_IDLE = 3'd0, S_REQ = 3'd1, S_DAT = 3'd2, S_NEXT = 3'd3;
    reg  [2:0]  st;
    reg  [31:0] s_s1, d_s1, s_s2, d_s2, n0;
    reg  [31:0] row_s, row_d, pl_s, pl_d;     // current row / plane bases
    reg  [31:0] i1, i2, off;                  // row, plane counters, word offset in the row
    reg  [31:0] n1, n2;
    reg  [6:0]  clen, bleft;                  // chunk length (words), words of the chunk not yet received
    reg  [31:0] wa;
    wire [31:0] rem = n0 - off;
    wire [6:0]  nxt_len = (rem > 32'd64) ? 7'd64 : rem[6:0];
    assign rd_own = (st != S_IDLE);
    assign rd_dat_ready = (st == S_DAT) && !wr_valid;
    integer k;
    always @(posedge clk) begin
        if (!rst_n) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0; rd_req_valid <= 1'b0; wr_valid <= 1'b0;
        end else begin
            done <= 1'b0;
            if (wr_valid && wr_ready) wr_valid <= 1'b0;
            case (st)
                S_IDLE: if (start) begin
                    n0 <= p[64 +: 32]; n1 <= p[96 +: 32];
                    s_s1 <= p[128 +: 32]; d_s1 <= p[160 +: 32]; n2 <= p[192 +: 32]; s_s2 <= p[224 +: 32];
                    d_s2 <= p[256 +: 32];
                    row_s <= p[0 +: 32]; row_d <= p[32 +: 32]; pl_s <= p[0 +: 32]; pl_d <= p[32 +: 32];
                    i1 <= 32'd0; i2 <= 32'd0; off <= 32'd0; busy <= 1'b1; st <= S_REQ;
                end
                S_REQ: begin
                    if (!rd_req_valid) begin
                        rd_req_valid <= 1'b1; rd_req_addr <= row_s + {off[29:0], 2'b00}; rd_req_len <= nxt_len;
                        clen <= nxt_len; bleft <= nxt_len; wa <= row_d + {off[29:0], 2'b00};
                    end else if (rd_req_ready) begin
                        rd_req_valid <= 1'b0; st <= S_DAT;
                    end
                end
                S_DAT: if (rd_dat_valid && !wr_valid) begin
                    wr_valid <= 1'b1; wr_addr <= wa; wr_data <= rd_dat;
                    for (k = 0; k < 8; k = k + 1) wr_mask[k] <= (bleft > k[6:0]);
                    wa <= wa + 32'd32;
                    if (bleft <= 7'd8) begin bleft <= 7'd0; st <= S_NEXT; end
                    else bleft <= bleft - 7'd8;
                end
                S_NEXT: if (!wr_valid) begin                          // chunk written: next chunk / row / plane
                    if (off + {25'd0, clen} < n0) begin off <= off + {25'd0, clen}; st <= S_REQ; end
                    else if (i1 + 32'd1 < n1) begin
                        i1 <= i1 + 32'd1; off <= 32'd0; row_s <= row_s + s_s1; row_d <= row_d + d_s1; st <= S_REQ;
                    end else if (i2 + 32'd1 < n2) begin
                        i2 <= i2 + 32'd1; i1 <= 32'd0; off <= 32'd0;
                        pl_s <= pl_s + s_s2; pl_d <= pl_d + d_s2; row_s <= pl_s + s_s2; row_d <= pl_d + d_s2; st <= S_REQ;
                    end else begin busy <= 1'b0; done <= 1'b1; st <= S_IDLE; end
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
`default_nettype wire
