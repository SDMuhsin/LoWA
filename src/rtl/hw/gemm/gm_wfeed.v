`default_nettype none
// gm_wfeed: the W (B-operand) tile buffers and the column feeders of the array (docs/p3_report.md
// section 2).  Two buffers (double-buffered block of BC = COLS*LAMBDA W rows x KC words), each
// WL = max(COLS, 8) lanes of 32-bit mm_sram_bank (LAMBDA*KC = 256 words).  W row n = c*LAMBDA + q
// (column c of the array, sub-row q), word k lives in
//   lane (c + k + (WL/LAMBDA)*q) mod WL, address {k, q}
// so that a ROW-mode DMA beat (one W row, 8 consecutive k) and a TRN-mode beat (8 consecutive W
// rows, one k: two c's x LAMBDA q's) each hit 8 distinct lanes.  All lanes of the active buffer read
// the same address {k, q} once per cycle (the pass counter), a rotator by (k + (WL/LAMBDA)*q) mod WL
// puts lane data on its column (registered), and column c is then delayed by c more cycles in
// the skew chain, so PE(0, c) consumes (k, q) c cycles after PE(0, 0), which consumes it 3 cycles
// after `ps_start` (read issued the cycle after, SRAM Q the next, rotator register the next).
// The skew cannot be done by addressing on the W side (docs/p3_report.md section 2), hence the
// COLS*(COLS-1)/2 skew registers.  The control word {valid, first, last, q} travels with the data.
// The next pass may start in the cycle of the current pass's last read (ps_ready), so full chunks
// stream back to back with no bubble.  busy[b] = buffer b is being read.
// Merge-engine port (P5, LoWA merge-on-load): single-word read-modify-write access to one lane of
// a buffer, read data valid the next cycle; it has priority over a DMA beat on the same lane
// (err_conflict flags the dropped beat word; by protocol the merge only runs after the W tile has
// been written and before the buffer is read).
module gm_wfeed #(
    parameter COLS   = 16,
    parameter LAMBDA = 4,
    parameter QW     = 2,
    parameter KC     = 64,
    parameter KW     = 6,                   // log2 KC
    parameter WL     = 16,                  // max(COLS, 8)
    parameter WLW    = 4,                   // log2 WL
    parameter AW     = 8                    // KW + QW
) (
    input  wire                 clk,
    input  wire                 rst_n,
    // write port (DMA beats, 8 words)
    input  wire                 wr_valid,
    input  wire                 wr_buf,
    input  wire [8*WLW-1:0]     wr_lane,
    input  wire [8*AW-1:0]      wr_addr,
    input  wire [8*32-1:0]      wr_data,
    input  wire [7:0]           wr_we,
    // merge RMW port
    input  wire                 mg_en,
    input  wire                 mg_buf,
    input  wire                 mg_we,
    input  wire [WLW-1:0]       mg_lane,
    input  wire [AW-1:0]        mg_addr,
    input  wire [31:0]          mg_wdata,
    output wire [31:0]          mg_rdata,
    // pass control
    input  wire                 ps_start,
    input  wire                 ps_buf,
    input  wire [KW:0]          ps_kt,      // 1 .. KC
    output wire                 ps_ready,   // a new pass may start this cycle
    output wire [1:0]           busy,
    // to the array
    output wire [COLS*32-1:0]       b_col,
    output wire [COLS*(QW+3)-1:0]   ctl_col,
    output reg                  err_conflict
);
    localparam CW = QW + 3;
    localparam WORDS = LAMBDA * KC;

    // ------------------------------------------------------------------ pass counter
    reg           run, rbuf;
    reg [KW:0]    kt;
    reg [KW-1:0]  k;
    reg [QW-1:0]  q;
    /* verilator lint_off WIDTHEXPAND */
    wire lastk = (k == kt - 1'b1);
    wire lastq = (q == LAMBDA - 1);
    /* verilator lint_on WIDTHEXPAND */
    wire last_cycle = run & lastk & lastq;
    assign ps_ready = ~run | last_cycle;
    always @(posedge clk) begin
        if (!rst_n) begin
            run <= 1'b0; k <= {KW{1'b0}}; q <= {QW{1'b0}};
        end else if (ps_start & ps_ready) begin
            run <= 1'b1; rbuf <= ps_buf; kt <= ps_kt; k <= {KW{1'b0}}; q <= {QW{1'b0}};
        end else if (run) begin
            if (lastq) begin
                q <= {QW{1'b0}};
                k <= k + 1'b1;
            end else begin
                q <= q + 1'b1;
            end
            if (last_cycle) run <= 1'b0;
        end
    end
    wire          rd_en  = run;
    wire          rd_buf = rbuf;
    wire [AW-1:0] rd_addr = {k, q};
    assign busy[0] = run & ~rbuf;
    assign busy[1] = run &  rbuf;

    // control word for the data read this cycle, aligned to the data one cycle later
    reg  [CW-1:0] ctl_s1;
    // rotation amount (k + (WL/LAMBDA)*q) modulo WL (a power of two): only the low WLW bits matter
    wire [KW+QW-1:0] qsh = {{KW{1'b0}}, q} << (WLW - QW);
    /* verilator lint_off UNUSEDSIGNAL */
    wire [KW+QW-1:0] rotw = {{QW{1'b0}}, k} + qsh;
    /* verilator lint_on UNUSEDSIGNAL */
    wire [WLW-1:0] rot = rotw[WLW-1:0];
    always @(posedge clk) begin
        if (!rst_n) ctl_s1 <= {CW{1'b0}};
        else        ctl_s1 <= {run, (k == {KW{1'b0}}), lastk, q};
    end
    // per-column one-hot {buffer, lane} select, registered with the data: column c reads lane
    // (rot + c) mod WL of buffer rbuf.  One flop per (column, buffer, lane) so that no select net
    // fans out beyond the 32 AND gates of its column (docs/p3_report.md section 6).
    reg  [COLS*2*WL-1:0] sel_s1;
    (* keep *) reg [COLS-1:0] rbuf_c;       // per-column copies of rbuf (fanout control)
    always @(posedge clk) if (ps_start & ps_ready) rbuf_c <= {COLS{ps_buf}};
    genvar sc, sl;
    generate
        for (sc = 0; sc < COLS; sc = sc + 1) begin : g_sel
            /* verilator lint_off WIDTHTRUNC */   // (c + rot) mod WL
            wire [WLW-1:0] rc = rot + sc;
            /* verilator lint_on WIDTHTRUNC */
            for (sl = 0; sl < WL; sl = sl + 1) begin : g_lane
                /* verilator lint_off WIDTHEXPAND */
                always @(posedge clk) begin
                    sel_s1[sc*2*WL + sl]      <= ~rbuf_c[sc] & (rc == sl);
                    sel_s1[sc*2*WL + WL + sl] <=  rbuf_c[sc] & (rc == sl);
                end
                /* verilator lint_on WIDTHEXPAND */
            end
        end
    endgenerate

    // ------------------------------------------------------------------ lanes
    wire [31:0] q0 [0:WL-1];
    wire [31:0] q1 [0:WL-1];
    wire [2*WL-1:0] conf;                   // per (buffer, lane) access collision
    genvar b, l, c;
    generate
        for (b = 0; b < 2; b = b + 1) begin : g_buf
            for (l = 0; l < WL; l = l + 1) begin : g_lane
                reg          wen;
                reg [AW-1:0] waddr;
                reg [31:0]   wdata;
                integer j;
                always @* begin
                    wen = 1'b0; waddr = {AW{1'b0}}; wdata = 32'd0;
                    for (j = 0; j < 8; j = j + 1)
                        /* verilator lint_off WIDTHEXPAND */
                        if (wr_valid && wr_we[j] && wr_buf == b && wr_lane[j*WLW +: WLW] == l) begin
                        /* verilator lint_on WIDTHEXPAND */
                            wen = 1'b1;
                            waddr = waddr | wr_addr[j*AW +: AW];
                            wdata = wdata | wr_data[j*32 +: 32];
                        end
                end
                /* verilator lint_off WIDTHEXPAND */
                wire mg_hit = mg_en && mg_buf == b && mg_lane == l;
                wire rd_hit = rd_en && rd_buf == b;
                /* verilator lint_on WIDTHEXPAND */
                wire          en   = mg_hit | wen | rd_hit;
                wire          we   = mg_hit ? mg_we : wen;
                wire [AW-1:0] addr = mg_hit ? mg_addr : wen ? waddr : rd_addr;
                wire [31:0]   d    = mg_hit ? mg_wdata : wdata;
                wire [31:0]   qq;
                mm_sram_bank #(.WORDS(WORDS), .WIDTH(32), .AW(AW)) u_bank (
                    .clk(clk), .en(en), .we(we), .addr(addr), .d(d), .q(qq));
                if (b == 0) begin : g_q0
                    assign q0[l] = qq;
                end else begin : g_q1
                    assign q1[l] = qq;
                end
                assign conf[b*WL + l] = (mg_hit & wen) | (rd_hit & (wen | mg_hit));
            end
        end
    endgenerate
    always @(posedge clk)
        if (!rst_n) err_conflict <= 1'b0;
        else if (|conf) err_conflict <= 1'b1;
    // merge read data: the lane accessed last cycle
    reg            mg_buf_s1;
    reg [WLW-1:0]  mg_lane_s1;
    always @(posedge clk) begin mg_buf_s1 <= mg_buf; mg_lane_s1 <= mg_lane; end
    assign mg_rdata = mg_buf_s1 ? q1[mg_lane_s1] : q0[mg_lane_s1];

    // ------------------------------------------------------------------ rotator + skew chains
    wire [2*WL*32-1:0] qall;                // entry l = buffer 0 lane l, entry WL + l = buffer 1 lane l
    generate
        for (l = 0; l < WL; l = l + 1) begin : g_qall
            assign qall[l*32 +: 32]      = q0[l];
            assign qall[(WL+l)*32 +: 32] = q1[l];
        end
        for (c = 0; c < COLS; c = c + 1) begin : g_col
            wire [31:0] w;
            cm_ohmux #(.W(32), .N(2*WL), .REP(1)) u_rot (.sel(sel_s1[c*2*WL +: 2*WL]), .d(qall), .y(w));
            wire [31:0]   bd;
            wire [CW-1:0] cd;
            cm_delay #(.W(32), .N(c + 1), .RST(0)) u_bd (.clk(clk), .rst_n(rst_n), .d(w), .q(bd));
            cm_delay #(.W(CW), .N(c + 1), .RST(1)) u_cd (.clk(clk), .rst_n(rst_n), .d(ctl_s1), .q(cd));
            assign b_col[c*32 +: 32]  = bd;
            assign ctl_col[c*CW +: CW] = cd;
        end
    endgenerate
endmodule
`default_nettype wire
