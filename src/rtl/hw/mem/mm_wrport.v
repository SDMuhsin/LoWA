`default_nettype none
// mm_wrport: one engine write channel (wr of gm_top, vu_top, lw_top) on the AXI4 master.
// Client side (P8f: the head of mm_top's request skid buffer, mm_skid; wr_ready is mm_top internal): a beat = byte
// address (4-byte aligned, any 32-byte offset) + 8 words + an 8-bit word mask;
// word j goes to addr / 4 + j when mask[j].  Writes are posted: the beat is accepted into stage A.
// Stage A splits the beat into the AXW-word AXI lines it touches (1 or 2 for 256 bit), one per cycle,
// lines with no masked word are dropped.  Stage B keeps one pending line (later beats to the same line
// merge into it, later data wins) and an open INCR burst of consecutive lines; the burst is closed at 16
// beats, at a 4 KiB boundary, at an address discontinuity, after TO idle cycles, or on `flush` (a read
// is waiting for this data).  W beats wait in the W FIFO (2^WFL >= 16 beats); a closed burst sits in
// the burst queue (2^WQL entries) until its AW is issued and stays there until its B response (same
// AWID per port, so B returns in order) -- that is the pending-write set mm_top orders reads against.
module mm_wrport #(
    parameter AXL = 3,                       // log2 words per AXI beat (1, 2, 3)
    parameter WFL = 4,                       // log2 W FIFO depth (>= 4: one 16-beat burst)
    parameter WQL = 2,                       // log2 burst queue depth (closed bursts until B)
    parameter TO  = 3                        // idle cycles before an open burst is closed
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               wr_valid,
    output wire               wr_ready,
    /* verilator lint_off UNUSEDSIGNAL */   // [1:0]: word-aligned addresses
    input  wire [31:0]        wr_addr,
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire [255:0]       wr_data,
    input  wire [7:0]         wr_mask,
    input  wire               blk,           // the offered beat (head of mm_top's skid buffer) must wait (mm_top)
    input  wire               flush,
    output wire               aw_valid,
    input  wire               aw_ready,
    output wire [29-AXL:0]    aw_line,
    output wire [3:0]         aw_len,
    output wire               w_valid,
    input  wire               w_ready,
    output wire [32*(1<<AXL)-1:0] w_data,
    output wire [(1<<AXL)-1:0]    w_wm,      // word mask (WSTRB = each bit x4)
    input  wire               b_valid,       // routed B response of this port
    output wire [3+(1<<WQL)-1:0]       pend_v,   // stage A, pending line, open burst, queue
    output wire [30*(3+(1<<WQL))-1:0]  pend_lo,
    output wire [30*(3+(1<<WQL))-1:0]  pend_hi,
    output wire               idle
);
    localparam AXW = 1 << AXL;
    localparam LW  = 30 - AXL;
    localparam WQ  = 1 << WQL;
    localparam PGB = 10 - AXL;

    // ---------------------------------------------------------------- stage A: beat -> lines
    reg             a_v;
    reg  [29:0]     a_a;
    reg  [255:0]    a_d;
    reg  [7:0]      a_m;
    reg  [2:0]      a_i;                     // line of the beat being emitted
    wire [AXL-1:0]  a_o = a_a[AXL-1:0];
    wire [3:0]      a_ow = {{(4-AXL){1'b0}}, a_o} + 4'd7;
    /* verilator lint_off UNUSEDSIGNAL */   // a_nlw[3] is 0 (at most 5 lines)
    wire [3:0]      a_nlw = a_ow >> AXL;
    wire [2:0]      a_nl = a_nlw[2:0];      // lines - 1
    /* verilator lint_on UNUSEDSIGNAL */
    wire            a_lastl = (a_i == a_nl);
    wire            b_rdy;
    wire            a_step = a_v && b_rdy;
    assign wr_ready = !blk && (!a_v || (a_step && a_lastl));
    wire            a_acc = wr_valid && wr_ready;
    // the current line's words and word mask
    reg  [32*AXW-1:0] x_d;
    reg  [AXW-1:0]    x_m;
    reg  [4:0]        jx;
    integer w, j;
    always @* begin
        x_d = {32*AXW{1'b0}}; x_m = {AXW{1'b0}};
        for (w = 0; w < AXW; w = w + 1) begin
            jx = ({2'b0, a_i} << AXL) + w[4:0] - {{(5-AXL){1'b0}}, a_o};
            for (j = 0; j < 8; j = j + 1)
                if (jx == j[4:0]) begin x_d[32*w +: 32] = a_d[32*j +: 32]; x_m[w] = a_m[j]; end
        end
    end
    wire [LW-1:0]   x_line = a_a[29:AXL] + {{(LW-3){1'b0}}, a_i};
    wire            xv = a_step && (x_m != {AXW{1'b0}});
    // first / last masked word of the line (word address) for the pending ranges
    reg  [AXL-1:0]  x_f, x_l;
    always @* begin
        x_f = {AXL{1'b0}}; x_l = {AXL{1'b0}};
        for (w = AXW - 1; w >= 0; w = w - 1) if (x_m[w]) x_f = w[AXL-1:0];
        for (w = 0; w < AXW; w = w + 1) if (x_m[w]) x_l = w[AXL-1:0];
    end
    always @(posedge clk) begin
        if (!rst_n) a_v <= 1'b0;
        else begin
            if (a_acc) begin
                a_v <= 1'b1; a_a <= wr_addr[31:2]; a_d <= wr_data; a_m <= wr_mask; a_i <= 3'd0;
            end else if (a_step) begin
                if (a_lastl) a_v <= 1'b0;
                else a_i <= a_i + 3'd1;
            end
        end
    end

    // ---------------------------------------------------------------- stage B: pending line + open burst
    reg             pl_v;
    reg  [LW-1:0]   pl_line;
    reg  [32*AXW-1:0] pl_d;
    reg  [AXW-1:0]  pl_m;
    reg  [29:0]     pl_lo, pl_hi;
    reg  [4:0]      bn;                      // beats of the open burst (0 = none)
    reg  [LW-1:0]   bs;
    reg  [29:0]     b_lo, b_hi;
    reg  [WQL:0]    q_cnt, q_nis;
    reg  [LW-1:0]   q_line [0:WQ-1];
    reg  [3:0]      q_len  [0:WQ-1];
    reg  [29:0]     q_lo   [0:WQ-1];
    reg  [29:0]     q_hi   [0:WQ-1];
    reg  [3:0]      idle_c;
    wire            wf_inr;
    wire            q_room = (q_cnt != WQ[WQL:0]);
    wire            q_pop = b_valid;
    wire [WQL:0]    q_tail = q_cnt - {{WQL{1'b0}}, q_pop};
    assign b_rdy = wf_inr && q_room;
    wire            same = pl_v && xv && (x_line == pl_line);
    wire            fl_go = (flush || idle_c >= TO[3:0]) && !xv && b_rdy;
    wire            push_pl = pl_v && b_rdy && ((xv && !same) || fl_go);
    wire            close_only = !pl_v && (bn != 5'd0) && fl_go;
    wire            contig = (bn != 5'd0) && (pl_line == bs + {{(LW-5){1'b0}}, bn}) && (pl_line[PGB-1:0] != {PGB{1'b0}});
    // a push closes the open burst when it is not contiguous, or after the push it holds 16 beats
    wire            cl_old = push_pl && (bn != 5'd0) && !contig;
    wire            cl_16  = push_pl && contig && (bn == 5'd15);
    wire            q_push = cl_old || cl_16 || close_only;
    wire [LW-1:0]   qp_line = bs;
    wire [3:0]      qp_len  = cl_16 ? 4'd15 : bn[3:0] - 4'd1;
    wire [29:0]     qp_lo   = cl_16 ? ((pl_lo < b_lo) ? pl_lo : b_lo) : b_lo;
    wire [29:0]     qp_hi   = cl_16 ? ((pl_hi > b_hi) ? pl_hi : b_hi) : b_hi;
    wire [29:0]     x_lo = {x_line, x_f};
    wire [29:0]     x_hi = {x_line, x_l};
    integer k, ww;
    always @(posedge clk) begin
        if (!rst_n) begin
            pl_v <= 1'b0; bn <= 5'd0; q_cnt <= {(WQL+1){1'b0}}; q_nis <= {(WQL+1){1'b0}}; idle_c <= 4'd0;
        end else begin
            idle_c <= (a_v || wr_valid) ? 4'd0 : (idle_c == 4'd15 ? idle_c : idle_c + 4'd1);
            if (same) begin
                for (ww = 0; ww < AXW; ww = ww + 1) if (x_m[ww]) pl_d[32*ww +: 32] <= x_d[32*ww +: 32];
                pl_m <= pl_m | x_m;
                if (x_lo < pl_lo) pl_lo <= x_lo;
                if (x_hi > pl_hi) pl_hi <= x_hi;
            end else if (xv && (!pl_v || push_pl)) begin
                pl_v <= 1'b1; pl_line <= x_line; pl_d <= x_d; pl_m <= x_m; pl_lo <= x_lo; pl_hi <= x_hi;
            end else if (push_pl) pl_v <= 1'b0;
            if (push_pl) begin
                if (cl_16) bn <= 5'd0;
                else if (contig) begin
                    bn <= bn + 5'd1;
                    if (pl_lo < b_lo) b_lo <= pl_lo;
                    if (pl_hi > b_hi) b_hi <= pl_hi;
                end else begin
                    bn <= 5'd1; bs <= pl_line; b_lo <= pl_lo; b_hi <= pl_hi;
                end
            end else if (close_only) bn <= 5'd0;
            // burst queue: pop the head on B, push a closed burst at the tail
            for (k = 0; k < WQ; k = k + 1) begin
                if (q_pop && k < WQ - 1) begin
                    q_line[k] <= q_line[k+1]; q_len[k] <= q_len[k+1]; q_lo[k] <= q_lo[k+1]; q_hi[k] <= q_hi[k+1];
                end
                if (q_push && (q_tail == k[WQL:0])) begin
                    q_line[k] <= qp_line; q_len[k] <= qp_len; q_lo[k] <= qp_lo; q_hi[k] <= qp_hi;
                end
            end
            q_cnt <= q_cnt + (q_push ? 1 : 0) - (q_pop ? 1 : 0);
            q_nis <= q_nis + ((aw_valid && aw_ready) ? 1 : 0) - (q_pop ? 1 : 0);
        end
    end
    assign aw_valid = (q_nis < q_cnt);
    assign aw_line  = q_line[q_nis[WQL-1:0]];
    assign aw_len   = q_len[q_nis[WQL-1:0]];

    // ---------------------------------------------------------------- W FIFO
    /* verilator lint_off UNUSEDSIGNAL */
    wire [WFL:0]    wf_cnt;
    /* verilator lint_on UNUSEDSIGNAL */
    lw_fifo #(.W(32*AXW+AXW), .LOG2(WFL)) u_wf (.clk(clk), .rst_n(rst_n), .in_valid(push_pl), .in_ready(wf_inr),
        .in_d({pl_m, pl_d}), .out_valid(w_valid), .out_ready(w_ready), .out_d({w_wm, w_data}), .count(wf_cnt));

    // ---------------------------------------------------------------- pending ranges (word addresses)
    assign pend_v[0] = a_v;          assign pend_lo[29:0] = a_a;  assign pend_hi[29:0] = a_a + 30'd7;
    assign pend_v[1] = pl_v;         assign pend_lo[59:30] = pl_lo; assign pend_hi[59:30] = pl_hi;
    assign pend_v[2] = (bn != 5'd0); assign pend_lo[89:60] = b_lo;  assign pend_hi[89:60] = b_hi;
    genvar gi;
    generate
        for (gi = 0; gi < WQ; gi = gi + 1) begin : g_pq
            assign pend_v[3+gi] = (gi < q_cnt);
            assign pend_lo[30*(3+gi) +: 30] = q_lo[gi];
            assign pend_hi[30*(3+gi) +: 30] = q_hi[gi];
        end
    endgenerate
    assign idle = !a_v && !pl_v && (bn == 5'd0) && (q_cnt == {(WQL+1){1'b0}});
endmodule
`default_nettype wire
