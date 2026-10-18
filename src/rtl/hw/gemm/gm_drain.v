`default_nettype none
// gm_drain: the C / accumulator path (docs/dataflow.md section 2, docs/p3_report.md section 3).
// After every K chunk the PEs hold their chunk partials p in res[q]; this engine moves them out
// through the column drain chains and does the contract's chunk sum
//     ACC = add(ACC, p),  ACC starting at +0 for the first chunk (old = +0, no preload)
// with DW fp_add lanes, and keeps ACC in SRAM: 2 halves (output blocks alternate halves, so the
// epilogue of block n reads half n&1 while block n+1 accumulates in the other) x 2 parities
// (q odd / even: a read-modify-write stream alternates parity every cycle, and the RMW latency is
// odd, so reads and writes never meet on one lane) x DW lanes (j) of BR*BC/(4*DW) words,
// address {mt, r, g, q >> 1} for output column LAMBDA*(DW*g + j) + q of micro-tile mt, row r.
// Schedule (engine-local cycle e, start = the cycle `start` is accepted): the column groups
// g = 0 .. NG-1 (DW columns each) are drained one after the other, RL = ROWS*LAMBDA cycles each,
// in the order row R-1 .. 0, q = 0 .. LAMBDA-1; word d of group g (r = R-1 - d/LAMBDA,
// q = d mod LAMBDA):  ACC read at e = g*RL + d, PE(r, c) injects res[q] at e + 1 - (R-1-r) (it then
// travels R-1-r hops down the chain, no two words collide), the word is at the chain bottom at
// e + 2, the adder takes it with the registered ACC word, y at e + 4, written back at e + 5.
// (e = 0 is the cycle after `start` is accepted; the injection origin is start + 2.)
// P8c2 (PLAN R14a): with DRS > 0 the chain has XR(r) = floor((R-1)/DRS) - floor(r/DRS) extra registers below row r
// (gm_array); row r then injects XR(r) cycles earlier (offset (LAMBDA-1)*(R-1-r) - XR(r) >= 0), so every word
// still reaches the chain bottom at e + 2 and nothing else in the schedule moves.  Rows above the last inject
// earlier than before, but their captures precede the last row's by one cycle per row while their injections
// follow it by >= LAMBDA-2 cycles per row, so the last row (XR = 0, unchanged) stays the binding case of the
// controller's capture-before-injection rule (gm_ctrl DOFF), and the controller's last-injection bound only gets
// more conservative.
// The controller schedules `start` so that the array's captures precede the injections and
// the engine is free (docs/p3_report.md section 3); err flags a start while busy.
// The epilogue reads the other half through ep_* (DW words per access, data the next cycle).
module gm_drain #(
    parameter ROWS   = 16,
    parameter COLS   = 16,
    parameter LAMBDA = 4,
    parameter QW     = 2,
    parameter DW     = 4,                   // drain words per cycle (fp_add lanes)
    parameter NG     = 4,                   // COLS / DW column groups
    parameter GW     = 2,                   // log2 NG (>= 1)
    parameter RW     = 4,                   // log2 ROWS
    parameter MTW    = 2,                   // log2 micro-tiles per block
    parameter AW     = 9,                   // MTW + RW + GW + QW - 1
    parameter EW     = 9,                   // log2(NG*RL) + 1 (engine counter width)
    parameter DRS    = 1                    // drain chain register spacing of gm_array (0 = none)
) (
    input  wire                 clk,
    input  wire                 rst_n,
    // schedule
    input  wire                 start,
    input  wire                 s_half,
    input  wire [MTW-1:0]       s_mt,
    input  wire                 s_first,    // first K chunk of the block: old = +0
    input  wire                 s_last,     // last K chunk: the written sums are final
    output wire                 fin_valid,  // the last final word of this drain was written
    output wire                 fin_half,
    output wire                 free,       // start may be accepted this cycle
    output wire [1:0]           ep_block,   // half h is (or is about to be) accessed by a drain
    // array
    output wire [ROWS-1:0]      inj_row,
    output wire [ROWS*QW-1:0]   inj_q,
    output wire [COLS-1:0]      col_en,
    input  wire [COLS*32-1:0]   dr_out,
    // epilogue read port
    input  wire                 ep_en,
    input  wire                 ep_half,
    input  wire                 ep_par,
    input  wire [AW-1:0]        ep_addr,
    output wire [DW*32-1:0]     ep_q,
    output reg                  err
);
    localparam RL  = ROWS * LAMBDA;
    localparam TOT = NG * RL;
    localparam WORDS = (1 << AW);

    // ------------------------------------------------------------------ engine counter
    reg          run, half, first, last;
    reg [MTW-1:0] mt;
    reg [EW-1:0] e;                         // 0 .. TOT-1
    /* verilator lint_off WIDTHEXPAND */
    wire last_e = (e == TOT - 1);
    /* verilator lint_on WIDTHEXPAND */
    assign free = ~run | last_e;
    wire accept = start & free;
    always @(posedge clk) begin
        if (!rst_n) begin
            run <= 1'b0; e <= {EW{1'b0}};
        end else if (accept) begin
            run <= 1'b1; e <= {EW{1'b0}}; half <= s_half; mt <= s_mt; first <= s_first; last <= s_last;
        end else if (run) begin
            e <= e + 1'b1;
            if (last_e) run <= 1'b0;
        end
    end
    // activity window of a drain on its half: reads from e = 0, writes until e = TOT - 1 + 5;
    // the epilogue decides its read one cycle ahead, so the accept cycle is included.  One window counter per
    // half (P7a fix): a drain accepted on the other half within 6 cycles of the previous drain's end must not
    // end the previous half's window, whose RMW write tail is still running (short-K blocks, docs/p7a_report.md)
    reg  [EW:0] act_cnt0, act_cnt1;
    /* verilator lint_off WIDTHTRUNC */
    localparam [EW:0] TOT6 = TOT + 6;
    /* verilator lint_on WIDTHTRUNC */
    always @(posedge clk) begin
        if (!rst_n) begin
            act_cnt0 <= {(EW+1){1'b0}};
            act_cnt1 <= {(EW+1){1'b0}};
        end else begin
            if (accept & ~s_half) act_cnt0 <= TOT6;
            else if (act_cnt0 != {(EW+1){1'b0}}) act_cnt0 <= act_cnt0 - 1'b1;
            if (accept & s_half) act_cnt1 <= TOT6;
            else if (act_cnt1 != {(EW+1){1'b0}}) act_cnt1 <= act_cnt1 - 1'b1;
        end
    end
    assign ep_block[0] = (accept & ~s_half) | (act_cnt0 != {(EW+1){1'b0}});
    assign ep_block[1] = (accept &  s_half) | (act_cnt1 != {(EW+1){1'b0}});
    wire [4*DW-1:0] lconf;                  // per-lane port collisions
    always @(posedge clk)
        if (!rst_n) err <= 1'b0;
        else if ((start & ~free) | (|lconf)) err <= 1'b1;
    // the last word of the drain is written at e = TOT-1 + 5 (RMW pipeline below)
    wire fin_h;
    cm_delay #(.W(1), .N(5), .RST(1)) u_fv (.clk(clk), .rst_n(rst_n), .d(run & last_e & last), .q(fin_valid));
    cm_delay #(.W(1), .N(5), .RST(0)) u_fh (.clk(clk), .rst_n(rst_n), .d(half), .q(fin_h));
    assign fin_half = fin_h;

    // group / word within group (from e): g = e / RL, d = e mod RL (RL is a power of two)
    localparam RLW = RW + QW;
    wire [RLW-1:0] d = e[RLW-1:0];
    wire [GW-1:0]  g;
    generate
        if (NG > 1) begin : g_g
            assign g = e[RLW +: GW];
        end else begin : g_g1
            assign g = {GW{1'b0}};
        end
    endgenerate
    wire [RW-1:0] rr = ~d[RLW-1:QW];        // R-1 - d/LAMBDA
    wire [QW-1:0] qq = d[QW-1:0];
    wire [AW-1:0] rd_addr = {mt, rr, g, qq[QW-1:1]};
    wire          rd_par  = qq[0];
    wire          rd_en   = run;

    // ------------------------------------------------------------------ injection control (e + 1)
    // and the e + 2 copies used by the RMW pipeline
    wire            run1, run2, half2, first2;
    wire [RLW-1:0]  d1;
    wire [GW-1:0]   g1, g2;
    wire [AW-1:0]   addr2;
    wire            par2;
    cm_delay #(.W(1), .N(1), .RST(1)) u_r1 (.clk(clk), .rst_n(rst_n), .d(run), .q(run1));
    cm_delay #(.W(RLW+GW), .N(1), .RST(0)) u_d1 (.clk(clk), .rst_n(rst_n), .d({d, g}), .q({d1, g1}));
    cm_delay #(.W(1), .N(2), .RST(1)) u_r2 (.clk(clk), .rst_n(rst_n), .d(run), .q(run2));
    cm_delay #(.W(GW+AW+3), .N(2), .RST(0)) u_d2 (.clk(clk), .rst_n(rst_n),
        .d({g, rd_addr, rd_par, half, first}), .q({g2, addr2, par2, half2, first2}));
    genvar r, c, j;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : g_inj
            localparam XR  = (DRS > 0) ? ((ROWS - 1) / (DRS > 0 ? DRS : 1) - r / (DRS > 0 ? DRS : 1)) : 0;
            localparam OFF = (LAMBDA - 1) * (ROWS - 1 - r) - XR;
            /* verilator lint_off WIDTHEXPAND */
            /* verilator lint_off UNSIGNED */    // OFF = 0 for the last row
            wire hit = run1 && (d1 >= OFF) && (d1 < OFF + LAMBDA);
            /* verilator lint_on UNSIGNED */
            /* verilator lint_off UNUSEDSIGNAL */  // only the low QW bits (0 .. LAMBDA-1 when hit)
            /* verilator lint_off WIDTHTRUNC */
            wire [RLW-1:0] dq = d1 - OFF;
            /* verilator lint_on WIDTHTRUNC */
            /* verilator lint_on UNUSEDSIGNAL */
            /* verilator lint_on WIDTHEXPAND */
            assign inj_row[r] = hit;
            assign inj_q[r*QW +: QW] = dq[QW-1:0];
        end
        for (c = 0; c < COLS; c = c + 1) begin : g_ce
            /* verilator lint_off WIDTHEXPAND */
            assign col_en[c] = run1 && (g1 == c / DW);
            /* verilator lint_on WIDTHEXPAND */
        end
    endgenerate

    // ------------------------------------------------------------------ ACC lanes
    // lane (h, p, j): h = half, p = parity (q & 1), j = word within the DW-wide drain word
    wire [4*DW*32-1:0] lq;                  // lane (h, p, j) at slice (h*2 + p)*DW + j
    reg         ep_half_s, ep_par_s;
    always @(posedge clk) begin ep_half_s <= ep_half; ep_par_s <= ep_par; end
    // write data pipeline (see below)
    wire          wr_v;
    wire          wr_half, wr_par;
    wire [AW-1:0] wr_addr;
    wire [DW*32-1:0] wr_data;
    genvar h, p;
    generate
        for (h = 0; h < 2; h = h + 1) begin : g_h
            for (p = 0; p < 2; p = p + 1) begin : g_p
                for (j = 0; j < DW; j = j + 1) begin : g_j
                    /* verilator lint_off WIDTHEXPAND */
                    wire drd = rd_en && half == h && rd_par == p;
                    wire dwr = wr_v && wr_half == h && wr_par == p;
                    wire erd = ep_en && ep_half == h && ep_par == p;
                    /* verilator lint_on WIDTHEXPAND */
                    wire          en   = drd | dwr | erd;
                    wire [AW-1:0] addr = dwr ? wr_addr : drd ? rd_addr : ep_addr;
                    mm_sram_bank #(.WORDS(WORDS), .WIDTH(32), .AW(AW)) u_bank (
                        .clk(clk), .en(en), .we(dwr), .addr(addr), .d(wr_data[j*32 +: 32]), .q(lq[((h*2+p)*DW+j)*32 +: 32]));
                    assign lconf[(h*2 + p)*DW + j] = (drd & dwr) | ((drd | dwr) & erd);
                end
            end
        end
    endgenerate

    wire [1:0] ep_sel = {ep_half_s, ep_par_s};
    assign ep_q = lq[ep_sel * (DW*32) +: DW*32];

    // ------------------------------------------------------------------ RMW pipeline
    // e + 1: ACC Q valid -> register (acc_old_s valid at e + 2, with the chain word)
    reg  [DW*32-1:0] acc_old_s;
    wire [DW*32-1:0] acc_old_s_d;
    reg              half_s1, par_s1;
    always @(posedge clk) begin half_s1 <= half; par_s1 <= rd_par; end
    wire [1:0] old_sel = {half_s1, par_s1};
    assign acc_old_s_d = lq[old_sel * (DW*32) +: DW*32];
    always @(posedge clk) acc_old_s <= acc_old_s_d;
    // chain words of group g2 at e + 2
    wire [DW*32-1:0] pw = dr_out[g2 * (DW*32) +: DW*32];
    wire [DW*32-1:0] ysum;
    wire [DW-1:0]    yv;
    generate
        for (j = 0; j < DW; j = j + 1) begin : g_add
            wire [31:0] old = first2 ? 32'd0 : acc_old_s[j*32 +: 32];
            fp_add #(.STAGES(2)) u_add (.clk(clk), .rst_n(rst_n), .in_valid(run2),
                .a(old), .b(pw[j*32 +: 32]), .sub(1'b0), .out_valid(yv[j]), .y(ysum[j*32 +: 32]));
        end
    endgenerate
    // e + 4: y valid; register once more -> write at e + 5 (read-to-write latency 5, odd)
    wire            half4, par4;
    wire [AW-1:0]   addr4;
    cm_delay #(.W(AW+2), .N(2), .RST(0)) u_a4 (.clk(clk), .rst_n(rst_n),
        .d({addr2, half2, par2}), .q({addr4, half4, par4}));
    reg              wv_s5, half5, par5;
    reg  [AW-1:0]    addr5;
    reg  [DW*32-1:0] y5;
    always @(posedge clk) begin
        if (!rst_n) wv_s5 <= 1'b0;
        else        wv_s5 <= yv[0];
        half5 <= half4; par5 <= par4; addr5 <= addr4; y5 <= ysum;
    end
    assign wr_v = wv_s5;
    assign wr_half = half5;
    assign wr_par = par5;
    assign wr_addr = addr5;
    assign wr_data = y5;
    /* verilator lint_off UNUSEDSIGNAL */   // the lanes' out_valid are identical; only lane 0 is used
    wire unused_yv = |yv;
    /* verilator lint_on UNUSEDSIGNAL */
endmodule
`default_nettype wire
