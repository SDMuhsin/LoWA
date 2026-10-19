`default_nettype none
// mm_rdport: one engine read channel (rd0 / rd1 / rd2 of gm_top, vu_top, lw_top) on the AXI4 master.
// Client side (docs/p3_report.md 2.5): a request = byte address (4-byte aligned, any 32-byte offset) and
// a length of 1..64 words; the answer is ceil(len / 8) in-order 256-bit beats, word j of beat b = the
// word at addr / 4 + 8 b + j (words past len are don't-care).
// AXI side: the request is split into INCR bursts of AXW = 2^AXL words per beat, at most 16 beats each
// and never across a 4 KiB boundary; the bursts of this port carry one ARID, so they return in order.
// A burst is offered to the arbiter only when the return FIFO (2^RFL beats) has room for all of its
// beats (credit reserved at grant), so the master can hold RREADY = 1 and one port can never block the
// R channel of another.  The packer drops the leading address offset, packs AXW-word AXI beats into
// 8-word client beats (one per cycle in steady state) and drops the tail of the last beat.
// Ordering (docs/p6a_report.md section 2): every accepted request stays in the pending queue (2^RQL
// entries, word range [lo, hi]) until its last beat has arrived; mm_top blocks overlapping writes
// against these entries and blocks this port's next request with `blk` while it overlaps a pending write.
// P8c timing (PLAN R13) / P8f (PLAN R17): the request register is mm_top's skid buffer (mm_skid, registered
// req_ready toward the engine); this port receives the buffer's head (req_*, registers) and takes it into the
// pending queue when there is room and the RAW check (`blk`, mm_top) passes; req_ready here is that take signal
// (combinational, mm_top internal).  The order point of a read is its acceptance into the skid buffer.
module mm_rdport #(
    parameter AXL = 3,                       // log2 words per AXI beat (1, 2, 3 = 64, 128, 256 bit)
    parameter RFL = 5,                       // log2 return FIFO depth in AXI beats (>= 4)
    parameter RQL = 3                        // log2 pending-request queue depth
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               req_valid,     // head of mm_top's request skid buffer (registered)
    output wire               req_ready,     // the head is taken into the pending queue this cycle
    input  wire [29:0]        req_lo,        // word address
    input  wire [6:0]         req_len,
    output reg                dat_valid,
    input  wire               dat_ready,
    output reg  [255:0]       dat,
    input  wire               blk,           // the request overlaps a pending or older buffered write (mm_top)
    output wire               ar_valid,
    input  wire               ar_ready,
    output wire [29-AXL:0]    ar_line,       // AXI beat index (byte address >> (AXL + 2))
    output wire [3:0]         ar_len,        // beats - 1
    input  wire               r_valid,       // routed R beat of this port (always accepted)
    input  wire [32*(1<<AXL)-1:0] r_data,
    output wire [(1<<RQL)-1:0]    pend_v,
    output wire [30*(1<<RQL)-1:0] pend_lo,
    output wire [30*(1<<RQL)-1:0] pend_hi,
    output wire               idle
);
    localparam AXW = 1 << AXL;
    localparam LW  = 30 - AXL;               // line index width
    localparam RQ  = 1 << RQL;
    localparam RFD = 1 << RFL;
    localparam PGB = 10 - AXL;               // log2 lines per 4 KiB page

    // ---------------------------------------------------------------- pending-request queue
    reg  [RQ-1:0]   q_v;
    reg  [29:0]     q_a  [0:RQ-1];
    reg  [6:0]      q_n  [0:RQ-1];
    reg  [5:0]      q_br [0:RQ-1];           // beats still to arrive
    reg  [RQL:0]    q_cnt, q_sp;             // entries, index of the entry being split
    wire [29:0]     in_a  = req_lo;
    wire [6:0]      in_n  = req_len;
    wire [AXL-1:0]  in_o  = in_a[AXL-1:0];
    /* verilator lint_off UNUSEDSIGNAL */   // low bits: offset within the last beat
    wire [7:0]      in_ow = {{(8-AXL){1'b0}}, in_o} + {1'b0, in_n} - 8'd1;
    wire [7:0]      in_nbw = (in_ow >> AXL) + 8'd1;
    wire [5:0]      in_nb = in_nbw[5:0];    // AXI beats of the request (<= 33)
    /* verilator lint_on UNUSEDSIGNAL */
    wire [RQL:0]    q_tail;
    wire            meta_rdy;
    assign req_ready = (q_cnt != RQ[RQL:0]) && meta_rdy && !blk;
    wire            acc = req_valid && req_ready;                                // skid head -> queue
    wire            arrive_last = r_valid && (q_br[0] == 6'd1);
    assign q_tail = q_cnt - {{RQL{1'b0}}, arrive_last};
    genvar gi;
    generate
        for (gi = 0; gi < RQ; gi = gi + 1) begin : g_pend
            assign pend_v[gi] = q_v[gi];
            assign pend_lo[30*gi +: 30] = q_a[gi];
            assign pend_hi[30*gi +: 30] = q_a[gi] + {23'd0, q_n[gi]} - 30'd1;
        end
    endgenerate

    // ---------------------------------------------------------------- splitter
    reg             sp_act;
    reg  [LW-1:0]   sp_line;
    reg  [5:0]      sp_left;
    reg  [7:0]      cred;                    // free, unreserved return-FIFO beats
    wire [10:0]     pg_l = (11'd1 << PGB) - {{(11-PGB){1'b0}}, sp_line[PGB-1:0]};   // beats to the 4 KiB page end
    wire [5:0]      b_a = (sp_left > 6'd16) ? 6'd16 : sp_left;
    wire [5:0]      blen = (pg_l < {5'd0, b_a}) ? pg_l[5:0] : b_a;
    assign ar_valid = sp_act && (cred >= {2'b0, blen});
    assign ar_line  = sp_line;
    assign ar_len   = blen[3:0] - 4'd1;
    wire            ar_fire = ar_valid && ar_ready;
    wire            sp_done = ar_fire && (sp_left == blen);
    wire [RQL:0]    sp_idx = q_sp;
    wire            rf_pop;

    integer k;
    always @(posedge clk) begin
        if (!rst_n) begin
            q_v <= {RQ{1'b0}}; q_cnt <= {(RQL+1){1'b0}}; q_sp <= {(RQL+1){1'b0}};
            sp_act <= 1'b0; cred <= RFD[7:0];
        end else begin
            // queue: pop (shift) on the last arriving beat of the head, push at the tail
            for (k = 0; k < RQ; k = k + 1) begin
                if (arrive_last) begin
                    if (k < RQ - 1) begin
                        q_v[k] <= q_v[k+1]; q_a[k] <= q_a[k+1]; q_n[k] <= q_n[k+1]; q_br[k] <= q_br[k+1];
                    end else q_v[k] <= 1'b0;
                end else if (r_valid && k == 0) q_br[0] <= q_br[0] - 6'd1;
                if (acc && (q_tail == k[RQL:0])) begin
                    q_v[k] <= 1'b1; q_a[k] <= in_a; q_n[k] <= in_n; q_br[k] <= in_nb;
                end
            end
            q_cnt <= q_cnt + (acc ? 1 : 0) - (arrive_last ? 1 : 0);
            // splitter: one burst per grant, next request when the current one is fully issued
            if (!sp_act) begin
                if (sp_idx < q_cnt) begin
                    sp_act  <= 1'b1;
                    sp_line <= q_a[sp_idx[RQL-1:0]][29:AXL];
                    sp_left <= q_br[sp_idx[RQL-1:0]];
                end
            end else if (ar_fire) begin
                sp_line <= sp_line + {{(LW-6){1'b0}}, blen};
                sp_left <= sp_left - blen;
                if (sp_done) sp_act <= 1'b0;
            end
            q_sp <= q_sp + ((!sp_act && sp_idx < q_cnt) ? 1 : 0) - (arrive_last ? 1 : 0);
            cred <= cred + {7'd0, rf_pop} - (ar_fire ? {2'b0, blen} : 8'd0);
        end
    end

    // ---------------------------------------------------------------- return FIFO + packer metadata
    wire            rf_v;
    wire [32*AXW-1:0] rf_d;
    /* verilator lint_off UNUSEDSIGNAL */   // occupancy is tracked by the credit counter
    wire [RFL:0]    rf_cnt;
    wire            rf_inr;
    wire [RFL:0]    mf_cnt;
    /* verilator lint_on UNUSEDSIGNAL */
    lw_fifo #(.W(32*AXW), .LOG2(RFL)) u_rf (.clk(clk), .rst_n(rst_n), .in_valid(r_valid), .in_ready(rf_inr),
        .in_d(r_data), .out_valid(rf_v), .out_ready(rf_pop), .out_d(rf_d), .count(rf_cnt));
    wire            mf_v, mf_pop;
    wire [AXL+6:0]  mf_d;
    lw_fifo #(.W(AXL+7), .LOG2(RFL)) u_mf (.clk(clk), .rst_n(rst_n), .in_valid(acc), .in_ready(meta_rdy),
        .in_d({in_o, in_n}), .out_valid(mf_v), .out_ready(mf_pop), .out_d(mf_d), .count(mf_cnt));
    wire [AXL-1:0]  m_o = mf_d[AXL+6:7];
    wire [6:0]      m_n = mf_d[6:0];

    // ---------------------------------------------------------------- packer
    reg  [31:0]     hold [0:7];              // < 8 words carried to the next client beat
    reg  [3:0]      hc;
    reg             pk_first;                // next input beat is the first of the request
    reg  [6:0]      rem_in;                  // words of the request not yet taken from the FIFO
    wire [6:0]      rem_cur = pk_first ? m_n : rem_in;
    wire [AXL:0]    s_in = pk_first ? {1'b0, m_o} : {(AXL+1){1'b0}};
    wire [AXL:0]    av_in = AXW[AXL:0] - s_in;
    wire [6:0]      c_in = ({{(6-AXL){1'b0}}, av_in} < rem_cur) ? {{(6-AXL){1'b0}}, av_in} : rem_cur;
    wire            last_in = (c_in == rem_cur);
    wire [4:0]      tot = {1'b0, hc} + c_in[4:0];
    wire            flush = mf_v && !pk_first && (rem_in == 7'd0) && (hc != 4'd0);
    wire            o_free = !dat_valid || dat_ready;
    wire            take = mf_v && rf_v && o_free && !flush;
    wire            emit = take && (tot >= 5'd8 || last_in);
    assign rf_pop = take;
    assign mf_pop = (take && last_in && tot <= 5'd8) || (flush && o_free);
    reg  [31:0]     comb [0:15];
    reg  [4:0]      ix;
    integer m, i, md;
    always @* begin
        for (m = 0; m < 16; m = m + 1) begin
            comb[m] = 32'd0;
            ix = 5'd0; i = 0;
            if (m < 8 && m < hc) comb[m] = hold[m];
            else begin
                ix = m[4:0] - {1'b0, hc} + {{(4-AXL){1'b0}}, s_in};
                for (i = 0; i < AXW; i = i + 1)
                    if (ix == i[4:0]) comb[m] = rf_d[32*i +: 32];
            end
        end
    end
    always @(posedge clk) begin
        if (!rst_n) begin
            dat_valid <= 1'b0; hc <= 4'd0; pk_first <= 1'b1; rem_in <= 7'd0;
        end else begin
            if (dat_valid && dat_ready) dat_valid <= 1'b0;
            if (flush && o_free) begin
                dat_valid <= 1'b1;
                hc <= 4'd0; pk_first <= 1'b1;
            end else if (take) begin
                rem_in   <= rem_cur - c_in;
                pk_first <= last_in && tot <= 5'd8;
                if (emit) begin
                    dat_valid <= 1'b1;
                    hc <= (tot > 5'd8) ? tot[3:0] - 4'd8 : 4'd0;
                end else hc <= tot[3:0];
            end
        end
    end
    // datapath (no reset).  A flush emits comb too: with pk_first = 0 its words 0 .. hc-1 are hold[0 .. hc-1]
    // and the rest are past the request's length (don't-care), so `dat` has one data source and one load
    // enable (fanout: the enable and the hold select each drive 256 flops, docs/p6a_report.md section 5)
    wire ld_dat = emit || (flush && o_free);
    always @(posedge clk) begin
        if (ld_dat)
            for (md = 0; md < 8; md = md + 1) dat[32*md +: 32] <= comb[md];
        if (take)
            for (md = 0; md < 8; md = md + 1) hold[md] <= emit ? comb[md+8] : comb[md];
    end
    assign idle = (q_cnt == {(RQL+1){1'b0}}) && !mf_v && !dat_valid;
endmodule
`default_nettype wire
