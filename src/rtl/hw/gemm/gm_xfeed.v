`default_nettype none
// gm_xfeed: the X (A-operand) tile buffers and the row feeders of the array (docs/p3_report.md
// section 2).  Two buffers (double-buffered block of BR rows x KC words), each XL = 8 lanes of
// 32-bit mm_sram_bank (BR*KC/XL words).  Word (r, k) of the block (r = mt*ROWS + rr, the block
// row) lives in
//   lane (k + r) mod XL, address {mt, rr, k / XL}
// so that (a) a ROW-mode DMA beat (one row, 8 consecutive k) and (b) a TRN-mode beat (8 consecutive
// rows, one k) each hit 8 distinct lanes (one write per lane per cycle), and (c) the ROWS/LAMBDA rows
// that need a new word in the same cycle of a pass read distinct lanes (rows phi + LAMBDA*i read
// k = K - i: lanes K + phi + (LAMBDA-1)*i, distinct because gcd(LAMBDA-1, XL) = 1 and ROWS/LAMBDA <= XL).
// The operand skew is done by addressing: row r starts its pass r cycles after `ps_start` (a token
// shifts down the rows), issues the read of word k at local cycle LAMBDA*k + 1, and presents
// A[r, k] on a_row[r] from two cycles later for LAMBDA cycles, which is exactly when PE(r, 0)
// consumes (k, 0..LAMBDA-1): 3 + r cycles after ps_start, the W feeder's timing.
// Passes may overlap: the next pass may start LAMBDA*KT cycles after this one; on the SAME buffer
// the lane sets of the two passes are disjoint only when KT mod XL == 0 (ps_idle otherwise; the
// controller waits).  busy[b] = buffer b has a pass in flight (the loader may not write it).
// Write port: one DMA beat per cycle, word j to lane wr_lane[j] at wr_addr[j] (lanes distinct by
// the layout above; wr_we[j] masks words beyond the tile).  A write and a read of the same lane in
// one cycle never happen by protocol (err_conflict is a sticky flag if they do; the write wins).
// P8c timing (PLAN R13): (a) the write request is registered per lane after the lane decode (wen_s / waddr_s /
// wdata_s), so a beat reaches its banks one cycle after wr_valid (gm_top delays tile_done to match); (b) the
// lane read request is registered too (ren_s / raddr_s), computed from the rows' next state, so the read is
// issued in the same cycle as before (no latency change).  The bank port then sees two registers and one mux.
module gm_xfeed #(
    parameter ROWS   = 16,
    parameter LAMBDA = 4,
    parameter BR     = 64,
    parameter KC     = 64,
    parameter XL     = 8,
    parameter XLW    = 3,                   // log2 XL
    parameter RW     = 4,                   // log2 ROWS
    parameter MTW    = 2,                   // log2 (BR / ROWS)
    parameter KW     = 6,                   // log2 KC
    parameter AW     = 9                    // MTW + RW + KW - XLW
) (
    input  wire                 clk,
    input  wire                 rst_n,
    // write port (DMA beats)
    input  wire                 wr_valid,
    input  wire                 wr_buf,
    input  wire [XL*XLW-1:0]    wr_lane,
    input  wire [XL*AW-1:0]     wr_addr,
    input  wire [XL*32-1:0]     wr_data,
    input  wire [XL-1:0]        wr_we,
    // pass control
    input  wire                 ps_start,
    input  wire                 ps_buf,
    input  wire [MTW-1:0]       ps_mt,
    input  wire [KW:0]          ps_kt,      // 1 .. KC
    output wire                 ps_idle,    // no row active, no start token in flight
    output wire [1:0]           busy,
    // to the array
    output wire [ROWS*32-1:0]   a_row,
    output reg                  err_conflict
);
    localparam WORDS = BR * KC / XL;
    localparam TW = 1 + 1 + MTW + KW + 1;   // token {valid, buf, mt, kt}

    // ------------------------------------------------------------------ start token chain
    wire [TW-1:0] tok [0:ROWS];
    assign tok[0] = {ps_start, ps_buf, ps_mt, ps_kt};
    genvar r;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : g_tok
            reg [TW-1:0] t_s;
            always @(posedge clk)
                if (!rst_n) t_s <= {TW{1'b0}};
                else        t_s <= tok[r];
            assign tok[r+1] = t_s;
        end
    endgenerate
    reg [ROWS-1:0] tok_v, tok_b;
    integer ti;
    always @* for (ti = 0; ti < ROWS; ti = ti + 1) begin
        tok_v[ti] = tok[ti][TW-1];
        tok_b[ti] = tok[ti][TW-2];
    end

    wire [31:0] q0 [0:XL-1];                // lane read data of buffer 0 / 1
    wire [31:0] q1 [0:XL-1];
    wire [2*XL*32-1:0] qall;                // entry l = buffer 0 lane l, entry XL + l = buffer 1 lane l
    genvar ql;
    generate
        for (ql = 0; ql < XL; ql = ql + 1) begin : g_qall
            assign qall[ql*32 +: 32]      = q0[ql];
            assign qall[(XL+ql)*32 +: 32] = q1[ql];
        end
    endgenerate

    // ------------------------------------------------------------------ per-row readers
    // row r: active after its token; issues the read of word k at phase 0 of k; holds (buf, lane)
    // for the data capture one cycle after the issue; the last read is at k = kt - 1.
    wire [ROWS-1:0]     rd_req;             // read issue NEXT cycle (next-state of the rows, P8c)
    wire [ROWS-1:0]     rd_buf;
    wire [ROWS*XLW-1:0] rd_lane;
    wire [ROWS*AW-1:0]  rd_addr;
    wire [ROWS-1:0]     row_act;
    wire [ROWS-1:0]     row_buf;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : g_row
            wire            tv  = tok[r][TW-1];
            wire            tb  = tok[r][TW-2];
            wire [MTW-1:0]  tmt = tok[r][KW+1 +: MTW];
            wire [KW:0]     tkt = tok[r][KW:0];
            reg             act;
            reg             bsel;
            reg  [MTW-1:0]  mt;
            reg  [KW:0]     kt;
            reg  [KW-1:0]   k;
            reg  [1:0]      ph;             // phase 0 .. LAMBDA-1 (LAMBDA <= 4)
            reg             cap, cap_buf;
            reg  [2*XL-1:0] cap_sel;        // one-hot {buffer, lane} of the word captured this cycle
            integer         cl;
            /* verilator lint_off WIDTHEXPAND */
            wire            lastk = (k == kt - 1'b1);
            wire            ph_last = (ph == LAMBDA - 1);
            /* verilator lint_on WIDTHEXPAND */
            wire            issue = act & (ph == 2'd0);
            wire [RW-1:0]   rr = r;
            /* verilator lint_off UNUSEDSIGNAL */   // block row mt*ROWS + r, only mod XL is used
            wire [MTW+RW-1:0] rblk = {mt, rr};
            /* verilator lint_on UNUSEDSIGNAL */
            wire [XLW-1:0]  lane = k[XLW-1:0] + rblk[XLW-1:0];
            always @(posedge clk) begin
                if (!rst_n) begin
                    act <= 1'b0; ph <= 2'd0; k <= {KW{1'b0}}; cap <= 1'b0;
                end else begin
                    cap <= issue;
                    cap_buf <= bsel;
                    for (cl = 0; cl < XL; cl = cl + 1) begin
                        /* verilator lint_off WIDTHEXPAND */
                        cap_sel[cl]      <= ~bsel & (lane == cl);
                        cap_sel[XL + cl] <=  bsel & (lane == cl);
                        /* verilator lint_on WIDTHEXPAND */
                    end
                    if (tv) begin
                        act <= 1'b1; bsel <= tb; mt <= tmt; kt <= tkt; k <= {KW{1'b0}}; ph <= 2'd0;
                    end else if (act) begin
                        if (ph_last) begin
                            ph <= 2'd0;
                            k <= k + 1'b1;
                        end else begin
                            ph <= ph + 1'b1;
                        end
                        // the row is done once the last word has been issued (data captured
                        // one cycle later, which cap/cap_buf still cover)
                        if (issue & lastk) act <= 1'b0;
                    end
                end
            end
            // next-state copy of the issue (for the registered lane read request): mirrors the update above
            reg             act_n, bsel_n;
            reg  [MTW-1:0]  mt_n;
            reg  [KW-1:0]   k_n;
            reg  [1:0]      ph_n;
            always @* begin
                act_n = act; bsel_n = bsel; mt_n = mt; k_n = k; ph_n = ph;
                if (!rst_n) begin
                    act_n = 1'b0; ph_n = 2'd0; k_n = {KW{1'b0}};
                end else if (tv) begin
                    act_n = 1'b1; bsel_n = tb; mt_n = tmt; k_n = {KW{1'b0}}; ph_n = 2'd0;
                end else if (act) begin
                    if (ph_last) begin ph_n = 2'd0; k_n = k + 1'b1; end
                    else         ph_n = ph + 1'b1;
                    if (issue & lastk) act_n = 1'b0;
                end
            end
            /* verilator lint_off UNUSEDSIGNAL */
            wire [MTW+RW-1:0] rblk_n = {mt_n, rr};
            /* verilator lint_on UNUSEDSIGNAL */
            assign rd_req[r]  = act_n & (ph_n == 2'd0);
            assign rd_buf[r]  = bsel_n;
            assign rd_lane[r*XLW +: XLW] = k_n[XLW-1:0] + rblk_n[XLW-1:0];
            assign rd_addr[r*AW +: AW] = {mt_n, rr, k_n[KW-1:XLW]};
            assign row_act[r] = act | cap;
            assign row_buf[r] = act ? bsel : cap_buf;
            // data capture
            wire [31:0] qsel;
            cm_ohmux #(.W(32), .N(2*XL), .REP(1)) u_cap (.sel(cap_sel), .d(qall), .y(qsel));
            reg [31:0] a_s;
            always @(posedge clk) if (cap) a_s <= qsel;
            assign a_row[r*32 +: 32] = a_s;
        end
    endgenerate

    // ------------------------------------------------------------------ lane access muxes
    wire [2*XL-1:0] conf;                   // per (buffer, lane) read/write collision
    genvar b, l;
    generate
        for (b = 0; b < 2; b = b + 1) begin : g_buf
            for (l = 0; l < XL; l = l + 1) begin : g_lane
                // read request to (b, l) in the next cycle: at most one row by construction
                reg           ren;
                reg [AW-1:0]  raddr;
                reg           ren_s;
                reg [AW-1:0]  raddr_s;
                integer i;
                always @* begin
                    ren = 1'b0; raddr = {AW{1'b0}};
                    for (i = 0; i < ROWS; i = i + 1)
                        /* verilator lint_off WIDTHEXPAND */
                        if (rd_req[i] && rd_buf[i] == b && rd_lane[i*XLW +: XLW] == l) begin
                        /* verilator lint_on WIDTHEXPAND */
                            ren = 1'b1;
                            raddr = raddr | rd_addr[i*AW +: AW];
                        end
                end
                // write word to (b, l): at most one beat word by the layout
                reg           wen;
                reg [AW-1:0]  waddr;
                reg [31:0]    wdata;
                integer j;
                always @* begin
                    wen = 1'b0; waddr = {AW{1'b0}}; wdata = 32'd0;
                    for (j = 0; j < XL; j = j + 1)
                        /* verilator lint_off WIDTHEXPAND */
                        if (wr_valid && wr_we[j] && wr_buf == b && wr_lane[j*XLW +: XLW] == l) begin
                        /* verilator lint_on WIDTHEXPAND */
                            wen = 1'b1;
                            waddr = waddr | wr_addr[j*AW +: AW];
                            wdata = wdata | wr_data[j*32 +: 32];
                        end
                end
                // registered lane requests (P8c): the write one cycle after the beat, the read in its issue cycle
                reg           wen_s;
                reg [AW-1:0]  waddr_s;
                reg [31:0]    wdata_s;
                always @(posedge clk) begin
                    if (!rst_n) begin wen_s <= 1'b0; ren_s <= 1'b0; end
                    else        begin wen_s <= wen;  ren_s <= ren;  end
                    waddr_s <= waddr; wdata_s <= wdata; raddr_s <= raddr;
                end
                wire [31:0] q;
                mm_sram_bank #(.WORDS(WORDS), .WIDTH(32), .AW(AW)) u_bank (
                    .clk(clk), .en(wen_s | ren_s), .we(wen_s), .addr(wen_s ? waddr_s : raddr_s),
                    .d(wdata_s), .q(q));
                if (b == 0) begin : g_q0
                    assign q0[l] = q;
                end else begin : g_q1
                    assign q1[l] = q;
                end
                assign conf[b*XL + l] = wen_s & ren_s;
            end
        end
    endgenerate
    always @(posedge clk)
        if (!rst_n) err_conflict <= 1'b0;
        else if (|conf) err_conflict <= 1'b1;

    // ------------------------------------------------------------------ status
    assign busy[0] = |(row_act & ~row_buf) | |(tok_v & ~tok_b);
    assign busy[1] = |(row_act &  row_buf) | |(tok_v &  tok_b);
    assign ps_idle = ~(|row_act) & ~(|tok_v);
endmodule
`default_nettype wire
