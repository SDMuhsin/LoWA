`default_nettype none
// vu_mask: dropout keep-bit fetcher for the vector unit.  For one row-segment step it requests,
// per 16-column group and per row, the 16 keep bits at element index idx0 + r * mk_rs + 16 g on
// the mask channel (same channel as the GEMM engine: index request, MKW-bit response, in order)
// and queues them per lane (DEPTH entries of 16 bits).  The sequencer consumes one bit per lane
// per element (mk_pop with the issue), the entry is dropped after its 16th bit or at the row end.
module vu_mask #(
    parameter V     = 16,
    parameter VW    = 4,
    parameter RBW   = 8,
    parameter DEPTH = 4,
    parameter MKW   = 16
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         mk_start,
    input  wire [31:0]  mk_idx0,
    input  wire [31:0]  mk_rs,
    /* verilator lint_off UNUSEDSIGNAL */   // rows >= nrows get harmless requests (lockstep lanes)
    input  wire [VW:0]  mk_nrows,
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire [RBW:0] mk_len,
    input  wire         mk_pop,
    output wire [2:0]   mk_cnt_min,          // entries available in every lane FIFO
    output wire [V-1:0] keep_bit,
    // mask channel
    output reg          mk_req_valid,
    input  wire         mk_req_ready,
    output reg  [31:0]  mk_req_idx,
    input  wire         mk_rsp_valid,
    input  wire [MKW-1:0] mk_rsp_keep
);
    // request side: groups g = 0 .. ngrp-1, rows r = 0 .. V-1 (rows >= nrows get harmless requests)
    reg  [RBW-4:0] ngrp, g_q, grp_done;
    reg  [VW:0]  r_q;
    reg          req_run;
    reg  [31:0]  row_idx;                      // idx0 + r * mk_rs, incremented per row
    reg  [31:0]  grp_idx;
    /* verilator lint_off UNUSEDSIGNAL */   // lenp15[3:0] is shifted out
    wire [RBW:0] lenp15 = mk_len + 15;
    /* verilator lint_on UNUSEDSIGNAL */
    wire [RBW-4:0] ngrp_w = lenp15[RBW:4];
    always @(posedge clk) begin
        if (!rst_n) begin
            req_run <= 1'b0; mk_req_valid <= 1'b0; g_q <= {(RBW-3){1'b0}}; r_q <= {(VW+1){1'b0}}; ngrp <= {(RBW-3){1'b0}};
            row_idx <= 32'd0; grp_idx <= 32'd0;
        end else begin
            if (mk_start) begin
                req_run <= 1'b1; g_q <= {(RBW-3){1'b0}}; r_q <= {(VW+1){1'b0}}; ngrp <= ngrp_w; row_idx <= mk_idx0; grp_idx <= mk_idx0;
            end else if (req_run & (~mk_req_valid | mk_req_ready) & (g_q < grp_done + DEPTH)) begin
                mk_req_valid <= 1'b1; mk_req_idx <= row_idx;
                if (&r_q[VW-1:0]) begin                           // r_q == V - 1 (V a power of two)
                    r_q <= {(VW+1){1'b0}}; row_idx <= grp_idx + 32'd16; grp_idx <= grp_idx + 32'd16;
                    if (g_q == ngrp - 1) req_run <= 1'b0; else g_q <= g_q + 1'b1;
                end else begin
                    r_q <= r_q + 1'b1; row_idx <= row_idx + mk_rs;
                end
            end else if (mk_req_ready) mk_req_valid <= 1'b0;
        end
    end
    // response side: lane FIFOs
    reg  [MKW-1:0] fq [0:V*DEPTH-1];
    reg  [V*3-1:0] wp, rp;                     // per-lane FIFO pointers (packed: Yosys-friendly resets)
    reg  [3:0]   bitpos;                       // bit index within the entry
    reg  [RBW:0] jcnt;                         // element index within the row
    reg  [VW:0]  r_b;                          // lane of the next response
    reg  [RBW:0] len_q;
    reg  [2:0]   cmin;
    reg  [2:0]   cl;
    integer k, ko;
    always @* begin
        cmin = 3'd7;
        for (ko = 0; ko < V; ko = ko + 1) begin
            cl = wp[ko*3 +: 3] - rp[ko*3 +: 3];
            if (cl < cmin) cmin = cl;
        end
    end
    assign mk_cnt_min = cmin;
    wire last_bit = (bitpos == 4'd15) | (jcnt == len_q - 1);
    genvar gv;
    generate
        for (gv = 0; gv < V; gv = gv + 1) begin : g_bit
            wire [VW+1:0] ridx = {gv[VW-1:0], rp[gv*3 +: 2]};
            assign keep_bit[gv] = fq[ridx][bitpos];
        end
    endgenerate
    always @(posedge clk) begin
        if (!rst_n) begin
            wp <= {(V*3){1'b0}}; rp <= {(V*3){1'b0}};
            bitpos <= 4'd0; jcnt <= {(RBW+1){1'b0}}; r_b <= {(VW+1){1'b0}}; len_q <= {(RBW+1){1'b0}}; grp_done <= {(RBW-3){1'b0}};
        end else begin
            if (mk_start) begin
                bitpos <= 4'd0; jcnt <= {(RBW+1){1'b0}}; r_b <= {(VW+1){1'b0}}; len_q <= mk_len; grp_done <= {(RBW-3){1'b0}};
                wp <= {(V*3){1'b0}}; rp <= {(V*3){1'b0}};
            end
            if (mk_rsp_valid) begin
                fq[{r_b[VW-1:0], wp[r_b[VW-1:0]*3 +: 2]}] <= mk_rsp_keep;
                wp[r_b[VW-1:0]*3 +: 3] <= wp[r_b[VW-1:0]*3 +: 3] + 3'd1;
                r_b <= (&r_b[VW-1:0]) ? {(VW+1){1'b0}} : r_b + 1'b1;
            end
            if (mk_pop) begin
                jcnt <= jcnt + 1'b1;
                if (last_bit) begin
                    bitpos <= 4'd0; grp_done <= grp_done + 1'b1;
                    for (k = 0; k < V; k = k + 1) rp[k*3 +: 3] <= rp[k*3 +: 3] + 3'd1;
                end else bitpos <= bitpos + 4'd1;
            end
        end
    end
endmodule
`default_nettype wire
