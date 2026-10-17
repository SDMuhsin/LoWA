`default_nettype none
// mm_mkport: DRAM *test stand-in* for the dropout keep-bit source of one engine's mk channel.
// PLAN ruling R2 / spec D1: the real keep bits come from the on-chip Threefry generator (P6b) and are
// never stored.  For P6a the bits are read from a bit-packed DRAM region (bit i of the stream = bit
// (i mod 8) of byte base + i / 8), so that the engines' real-data dropout cases run end to end.  The mk
// interface is the engines' plain keep-bit stream (request = element index, response = MKW keep bits
// for index .. index + MKW - 1, in order, no ready); mm_top selects this module or an external source
// with the parameter MK_DRAM.  The traffic this stand-in adds is not part of the design's budget and
// is reported separately (docs/p6a_report.md).
// Per request the module fetches the 1 or 2 AXW-word lines that hold its bits (a line equal to the last
// fetched one is not fetched again), as single-beat bursts, with credit for the line FIFO reserved
// before the AR; the response side walks the fetched lines in order.
module mm_mkport #(
    parameter AXL = 3,
    parameter MKW = 16,                      // keep bits per response (<= 32 * AXW)
    parameter FL  = 2                        // log2 line FIFO depth
) (
    input  wire               clk,
    input  wire               rst_n,
    /* verilator lint_off UNUSEDSIGNAL */   // line aligned: low bits unused
    input  wire [31:0]        base,          // byte address of keep bit 0 (line aligned)
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire               mk_req_valid,
    output wire               mk_req_ready,
    input  wire [31:0]        mk_req_idx,
    output reg                mk_rsp_valid,
    output reg  [MKW-1:0]     mk_rsp_keep,
    output wire               ar_valid,
    input  wire               ar_ready,
    output wire [29-AXL:0]    ar_line,
    output wire [3:0]         ar_len,
    input  wire               r_valid,
    input  wire [32*(1<<AXL)-1:0] r_data,
    output wire               idle
);
    localparam AXW = 1 << AXL;
    localparam LB  = 32 * AXW;               // bits per line
    localparam LBL = 5 + AXL;
    localparam TW  = 32 - LBL;               // line index width (index >> LBL)
    localparam LW  = 30 - AXL;

    // ---------------------------------------------------------------- fetch side
    wire [TW-1:0]   l0 = mk_req_idx[31:LBL];
    /* verilator lint_off UNUSEDSIGNAL */   // only the line index of the last bit is used
    wire [31:0]     ie = mk_req_idx + MKW - 1;
    /* verilator lint_on UNUSEDSIGNAL */
    wire [TW-1:0]   l1 = ie[31:LBL];
    wire            two = (l1 != l0);
    reg             lf_v;
    reg  [TW-1:0]   lf;
    reg             ph;                      // 0: first line of the request, 1: second
    reg             n0_q;                    // the first line was fetched (latched when ph -> 1)
    reg  [FL+1:0]   cred;
    wire            mq_rdy;
    wire            need0 = !(lf_v && lf == l0);
    wire            f_want = mk_req_valid && mq_rdy && (ph ? 1'b1 : need0);
    wire [TW-1:0]   f_tag = ph ? l1 : l0;
    assign ar_valid = f_want && (cred != {(FL+2){1'b0}});
    assign ar_line  = base[31:AXL+2] + {{(LW-TW){1'b0}}, f_tag};
    assign ar_len   = 4'd0;
    wire            ar_fire = ar_valid && ar_ready;
    // the request is accepted when its last needed line has been issued (or none was needed)
    assign mk_req_ready = mq_rdy && (ph ? ar_fire : (need0 ? (ar_fire && !two) : !two));
    wire            go2 = !ph && mk_req_valid && mq_rdy && two && (need0 ? ar_fire : 1'b1);
    wire            mreq = mk_req_valid && mk_req_ready;
    wire            lq_pop;
    always @(posedge clk) begin
        if (!rst_n) begin
            lf_v <= 1'b0; ph <= 1'b0; cred <= {1'b0, 1'b1, {FL{1'b0}}};
        end else begin
            if (ar_fire) begin lf_v <= 1'b1; lf <= f_tag; end
            if (go2) begin ph <= 1'b1; n0_q <= need0; end
            else if (mreq) ph <= 1'b0;
            cred <= cred + {{(FL+1){1'b0}}, lq_pop} - {{(FL+1){1'b0}}, ar_fire};
        end
    end
    // line FIFO (data of every issued line, in issue order)
    wire            lq_v;
    wire [LB-1:0]   lq_d;
    /* verilator lint_off UNUSEDSIGNAL */
    wire            lq_inr;
    wire [FL:0]     lq_cnt;
    wire [3:0]      mq_cnt;
    /* verilator lint_on UNUSEDSIGNAL */
    lw_fifo #(.W(LB), .LOG2(FL)) u_lq (.clk(clk), .rst_n(rst_n), .in_valid(r_valid), .in_ready(lq_inr),
        .in_d(r_data), .out_valid(lq_v), .out_ready(lq_pop), .out_d(lq_d), .count(lq_cnt));
    // request metadata: first line fetched for it, second line fetched, bit offset in the first line
    wire            mq_v, mq_pop;
    wire [LBL+1:0]  mq_d;
    lw_fifo #(.W(LBL+2), .LOG2(3)) u_mq (.clk(clk), .rst_n(rst_n), .in_valid(mreq), .in_ready(mq_rdy),
        .in_d({ph ? n0_q : need0, two, mk_req_idx[LBL-1:0]}), .out_valid(mq_v), .out_ready(mq_pop), .out_d(mq_d), .count(mq_cnt));

    // ---------------------------------------------------------------- response side
    // invariant: between requests `cu_d` holds the last line fetched before the next request, so a request
    // whose first line was not fetched finds it in cu_d; a fetched first line is popped into cu_d first,
    // a second line is used from the FIFO head and then popped into cu_d.
    wire            m_n0  = mq_d[LBL+1];
    wire            m_two = mq_d[LBL];
    wire [LBL-1:0]  m_off = mq_d[LBL-1:0];
    reg             r0;                      // first line of the head request already in cu_d
    reg  [LB-1:0]   cu_d;
    wire            ld0 = mq_v && m_n0 && !r0;
    wire            can = mq_v && !ld0 && (!m_two || lq_v);
    assign mq_pop = can;
    assign lq_pop = (ld0 && lq_v) || (can && m_two);
    wire [2*LB-1:0] cat = {lq_d, cu_d};
    /* verilator lint_off UNUSEDSIGNAL */   // the MKW low bits are the response
    wire [2*LB-1:0] sh = cat >> m_off;
    /* verilator lint_on UNUSEDSIGNAL */
    always @(posedge clk) begin
        if (!rst_n) begin
            r0 <= 1'b0; mk_rsp_valid <= 1'b0;
        end else begin
            mk_rsp_valid <= can;
            if (can) mk_rsp_keep <= sh[MKW-1:0];
            if (lq_pop) cu_d <= lq_d;
            r0 <= ld0 ? lq_v : (can ? 1'b0 : r0);
        end
    end
    assign idle = !mq_v && !ph && (cred + {1'b0, lq_cnt} == {1'b0, 1'b1, {FL{1'b0}}});
endmodule
`default_nettype wire
