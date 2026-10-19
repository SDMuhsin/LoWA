`default_nettype none
// mm_top: the LoWA-TA memory system (P6a): NR engine read channels (mm_rdport), NM keep-bit channels
// (MK_DRAM = 0, default: the on-chip Threefry dropout-mask generator rn_mkgen per channel (P6b, ruling R7);
// MK_DRAM = 1: mm_mkport, the DRAM test stand-in, kept for debug), NW engine
// write channels (mm_wrport), and one AXI4 master port (the chip boundary, PLAN ruling R3).
// AXI4: data 32 * 2^AXL bit (default 256), INCR bursts of <= 16 beats, full-width beats only (partial
// words by WSTRB), never across 4 KiB; ARID = read channel (0 .. NR-1 engine reads, NR .. NR+NM-1 keep
// bits), AWID = write channel; per ID in order, different IDs may complete in any order (and R beats of
// different IDs may interleave); RREADY = BREADY = 1 (every R beat has reserved space).
// AR / AW: one registered slot each, filled by a round-robin arbiter over the channels that have a burst
// (no starvation: a requesting channel waits at most NR+NM-1 (NW-1) grants); at most MAXR read and MAXW
// write bursts outstanding.  W follows AW order (W-order FIFO of (channel, length)).
// Same-address ordering (word granular, docs/p6a_report.md section 2): accesses are ordered by their handshake on the
// engine port.  P8f (PLAN R17): every engine request port (read, write) has a two-entry skid buffer (mm_skid) with a
// registered ready, so no combinational path crosses an mm_top port; a buffered request is ordered from its handshake,
// and each buffered entry carries, per channel of the other kind (and per other write channel), the number of that
// channel's buffered entries that are older than it.  A read leaves its buffer (into the port's pending queue) only
// when it overlaps no pending write (in a write port: accepted from the buffer, not yet B-acknowledged) and no older
// buffered write; a write beat leaves its buffer (into the write port) only when it overlaps no pending read
// (accepted from the buffer, last beat not yet arrived), no older buffered read, no pending write of another write
// channel (different AWIDs are unordered in AXI) and no older buffered write of another channel.  A younger
// overlapping access never passes an older one, and an older one never waits for a younger one (no deadlock).
// Same-cycle handshakes: a write beat and a read accepted in the same cycle are ordered write first (as since P8c);
// two write beats of different channels in the same cycle are unordered (as before P8f).  The compiler / sequencer
// never issue an overlapping read / write pair in one cycle (docs/p8c2_report.md section 3; checked in every
// integration and e2e test by tb/mem/mm_ovl_chk.svh).  So every read returns the data of exactly the writes accepted
// before it; a waiting access makes the write channels close their open bursts (flush).
module mm_top #(
    parameter NR      = 2,                   // engine read channels (>= 1)
    parameter NM      = 1,                   // keep-bit channels
    parameter NW      = 1,                   // engine write channels (>= 1)
    parameter AXL     = 3,                   // log2 words per AXI beat: 1, 2, 3 (64, 128, 256 bit)
    parameter RFL0    = 6,                   // log2 return FIFO beats, read channel 0 (the GEMM tile stream)
    parameter RFL     = 4,                   // log2 return FIFO beats, other read channels
    parameter RQL     = 3,                   // log2 pending requests per read channel
    parameter WFL     = 4,                   // log2 W FIFO beats per write channel
    parameter WQL     = 2,                   // log2 queued bursts per write channel
    parameter MAXR    = 8,                   // outstanding read bursts (all IDs)
    parameter MAXW    = 4,                   // outstanding write bursts (all IDs)
    parameter MK_DRAM = 0,                   // 0: on-chip generator (rn_mkgen), 1: keep bits from DRAM (debug stand-in)
    parameter MKW     = 16,
    parameter MKNC0   = 3,                   // Threefry pipelines, keep-bit channel 0 (GEMM epilogue: 4 bits / cycle)
    parameter MKNC    = 9,                   // Threefry pipelines, other keep-bit channels (vector unit: 16 bits / cycle)
    parameter MKST    = 3                    // rn_tf pipeline stages
) (
    input  wire                 clk,
    input  wire                 rst_n,
    // engine read channels
    input  wire [NR-1:0]        rd_req_valid,
    output wire [NR-1:0]        rd_req_ready,
    /* verilator lint_off UNUSEDSIGNAL */   // [1:0] of each address: word-aligned byte addresses
    input  wire [32*NR-1:0]     rd_req_addr,
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire [7*NR-1:0]      rd_req_len,
    output wire [NR-1:0]        rd_dat_valid,
    input  wire [NR-1:0]        rd_dat_ready,
    output wire [256*NR-1:0]    rd_dat,
    // keep-bit channels
    input  wire [32*NM-1:0]     mk_base,
    input  wire [NM-1:0]        mk_req_valid,
    output wire [NM-1:0]        mk_req_ready,
    input  wire [32*NM-1:0]     mk_req_idx,
    output wire [NM-1:0]        mk_rsp_valid,
    output wire [MKW*NM-1:0]    mk_rsp_keep,
    input  wire [63:0]          mk_seed,            // generator configuration (MK_DRAM = 0): seed (all channels),
    input  wire [23:0]          mk_step,            // step, and per channel site / layer of the mask being served
    input  wire [4*NM-1:0]      mk_site,
    input  wire [4*NM-1:0]      mk_layer,
    // engine write channels
    input  wire [NW-1:0]        wr_valid,
    output wire [NW-1:0]        wr_ready,
    /* verilator lint_off UNUSEDSIGNAL */   // [1:0] of each address: word-aligned byte addresses
    input  wire [32*NW-1:0]     wr_addr,
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire [256*NW-1:0]    wr_data,
    input  wire [8*NW-1:0]      wr_mask,
    // status
    output reg  [NR-1:0]        rd_idle,
    output reg  [NW-1:0]        wr_idle,        // all writes of the channel B-acknowledged (sequencer flag)
    output reg  [1:0]           err,            // sticky: [0] RRESP != OKAY, [1] BRESP != OKAY
    // AXI4 master
    output reg                  m_arvalid,
    input  wire                 m_arready,
    output reg  [3:0]           m_arid,
    output wire [31:0]          m_araddr,
    output wire [7:0]           m_arlen,
    output wire [2:0]           m_arsize,
    output wire [1:0]           m_arburst,
    input  wire                 m_rvalid,
    output wire                 m_rready,
    input  wire [3:0]           m_rid,
    input  wire [32*(1<<AXL)-1:0] m_rdata,
    input  wire [1:0]           m_rresp,
    input  wire                 m_rlast,
    output reg                  m_awvalid,
    input  wire                 m_awready,
    output reg  [3:0]           m_awid,
    output wire [31:0]          m_awaddr,
    output wire [7:0]           m_awlen,
    output wire [2:0]           m_awsize,
    output wire [1:0]           m_awburst,
    output wire                 m_wvalid,
    input  wire                 m_wready,
    output wire [32*(1<<AXL)-1:0] m_wdata,
    output wire [4*(1<<AXL)-1:0]  m_wstrb,
    output wire                 m_wlast,
    input  wire                 m_bvalid,
    output wire                 m_bready,
    input  wire [3:0]           m_bid,
    input  wire [1:0]           m_bresp
);
    localparam AXW = 1 << AXL;
    localparam LW  = 30 - AXL;
    localparam NMD = (MK_DRAM != 0) ? NM : 0;          // keep-bit channels on the AXI port
    localparam NA  = NR + NMD;                         // AR requesters
    localparam RQ  = 1 << RQL;
    localparam WE  = 3 + (1 << WQL);                   // pending-write entries per write channel

    // ---------------------------------------------------------------- read channels
    wire [NA-1:0]     a_v, a_g;
    wire [LW*NA-1:0]  a_line;
    wire [4*NA-1:0]   a_len;
    wire [NA-1:0]     r_v;
    wire [NR-1:0]     r_blk;
    // request skid buffers (P8f): read buffer r = {len, word address}; write buffer w = {mask, data, word address}
    localparam RSW = 37;
    localparam WSW = 30 + 256 + 8;
    localparam WNC = NR + NW;                  // write-entry age counters: per read port, per write channel
    wire [NR-1:0]     rh_v, rs_v, r_go, r_in, rp_go;
    wire [RSW*NR-1:0] rh_d, rs_d;
    wire [2*NW*NR-1:0] rh_c, rin_c;
    wire [NW-1:0]     wh_v, ws_v, w_go, w_in, wp_go;
    wire [WSW*NW-1:0] wh_d;
    wire [2*WNC*NW-1:0] wh_c, win_c;
    /* verilator lint_off UNUSEDSIGNAL */   // skid entries: only the address is compared here; their counters
    wire [WSW*NW-1:0] ws_d;                 // and payload move to the head inside mm_skid
    wire [2*NW*NR-1:0] rs_c;
    wire [2*WNC*NW-1:0] ws_c;
    /* verilator lint_on UNUSEDSIGNAL */
    wire [NR-1:0]     rp_idle;
    wire [NW-1:0]     wp_idle;
    // P8c timing (PLAN R13): the seed is a per-step CSR value, registered here (one cycle) so that the input
    // port does not drive the key schedule of rn_mkgen directly
    reg  [63:0]       mk_seed_q;
    always @(posedge clk) mk_seed_q <= mk_seed;
    wire [RQ*NR-1:0]  rp_v;
    wire [30*RQ*NR-1:0] rp_lo, rp_hi;
    wire [WE*NW-1:0]  wp_v;
    wire [30*WE*NW-1:0] wp_lo, wp_hi;
    genvar gi, ge;
    // reset tree (docs/p3_report.md section 6): one kept-hierarchy cm_rstbuf per channel port and one for
    // the arbiters / W-order FIFO, so that no single reset net fans out to the whole memory system; all
    // blocks leave reset in the same cycle, one cycle after rst_n
    localparam NRB = NR + NM + NW + 1;
    wire [NRB-1:0] rst_r;
    generate
        for (gi = 0; gi < NRB; gi = gi + 1) begin : g_rst
            cm_rstbuf u_rb (.clk(clk), .rst_n(rst_n), .rst_n_q(rst_r[gi]));
        end
    endgenerate
    wire rst_c = rst_r[NRB-1];
    /* verilator lint_off UNUSEDSIGNAL */   // keep-bit source idle (debug); the unused source's configuration
    wire [NM-1:0]     mk_idle;
    wire              mk_unused = (MK_DRAM != 0) ? (^mk_seed_q ^ ^mk_step ^ ^mk_site ^ ^mk_layer) : ^mk_base;
    /* verilator lint_on UNUSEDSIGNAL */
    generate
        for (gi = 0; gi < NA; gi = gi + 1) begin : g_rv
            assign r_v[gi] = m_rvalid && (m_rid == gi);
        end
        for (gi = 0; gi < NR; gi = gi + 1) begin : g_rd
            // new entry's age counters: per write channel, its buffered beats older than this read after this
            // cycle (a beat accepted in the same cycle counts as older: write first)
            for (ge = 0; ge < NW; ge = ge + 1) begin : g_rc
                assign rin_c[2*(NW*gi+ge) +: 2] = {1'b0, wh_v[ge]} + {1'b0, ws_v[ge]} + {1'b0, w_in[ge]} - {1'b0, w_go[ge]};
            end
            assign r_in[gi] = rd_req_valid[gi] && rd_req_ready[gi];
            mm_skid #(.W(RSW), .NC(NW)) u_rs (
                .clk(clk), .rst_n(rst_r[gi]), .in_valid(rd_req_valid[gi]), .in_ready(rd_req_ready[gi]),
                .in_d({rd_req_len[7*gi +: 7], rd_req_addr[32*gi+2 +: 30]}), .in_c(rin_c[2*NW*gi +: 2*NW]), .dec(w_go),
                .h_v(rh_v[gi]), .h_d(rh_d[RSW*gi +: RSW]), .h_c(rh_c[2*NW*gi +: 2*NW]), .h_go(r_go[gi]),
                .s_v(rs_v[gi]), .s_d(rs_d[RSW*gi +: RSW]), .s_c(rs_c[2*NW*gi +: 2*NW]));
            mm_rdport #(.AXL(AXL), .RFL(gi == 0 ? RFL0 : RFL), .RQL(RQL)) u_rd (
                .clk(clk), .rst_n(rst_r[gi]),
                .req_valid(rh_v[gi]), .req_ready(rp_go[gi]),
                .req_lo(rh_d[RSW*gi +: 30]), .req_len(rh_d[RSW*gi+30 +: 7]),
                .dat_valid(rd_dat_valid[gi]), .dat_ready(rd_dat_ready[gi]), .dat(rd_dat[256*gi +: 256]),
                .blk(r_blk[gi]),
                .ar_valid(a_v[gi]), .ar_ready(a_g[gi]), .ar_line(a_line[LW*gi +: LW]), .ar_len(a_len[4*gi +: 4]),
                .r_valid(r_v[gi]), .r_data(m_rdata),
                .pend_v(rp_v[RQ*gi +: RQ]), .pend_lo(rp_lo[30*RQ*gi +: 30*RQ]), .pend_hi(rp_hi[30*RQ*gi +: 30*RQ]),
                .idle(rp_idle[gi]));
            assign r_go[gi] = rh_v[gi] && rp_go[gi];
            // RAW: the buffer head against every pending write and every older buffered write beat
            wire [29:0] lo = rh_d[RSW*gi +: 30];
            wire [29:0] hi = lo + {23'd0, rh_d[RSW*gi+30 +: 7]} - 30'd1;
            wire [WE*NW-1:0] ov;
            wire [NW-1:0]    ob;
            for (ge = 0; ge < WE * NW; ge = ge + 1) begin : g_ov
                assign ov[ge] = wp_v[ge] && (lo <= wp_hi[30*ge +: 30]) && (wp_lo[30*ge +: 30] <= hi);
            end
            for (ge = 0; ge < NW; ge = ge + 1) begin : g_ob
                wire [1:0]  c   = rh_c[2*(NW*gi+ge) +: 2];
                wire [29:0] wl0 = wh_d[WSW*ge +: 30];
                wire [29:0] wl1 = ws_d[WSW*ge +: 30];
                assign ob[ge] = ((c != 2'd0) && (lo <= wl0 + 30'd7) && (wl0 <= hi))
                             || ((c == 2'd2) && (lo <= wl1 + 30'd7) && (wl1 <= hi));
            end
            assign r_blk[gi] = (|ov) || (|ob);
            // read-idle flag, registered (P8f): rises one cycle after the channel is empty, falls with the handshake
            always @(posedge clk)
                if (!rst_r[gi]) rd_idle[gi] <= 1'b1;
                else            rd_idle[gi] <= rp_idle[gi] && !rh_v[gi] && !r_in[gi];
        end
        for (gi = 0; gi < NM; gi = gi + 1) begin : g_mk
            if (MK_DRAM != 0) begin : g_dram
                mm_mkport #(.AXL(AXL), .MKW(MKW)) u_mk (
                    .clk(clk), .rst_n(rst_r[NR+gi]), .base(mk_base[32*gi +: 32]),
                    .mk_req_valid(mk_req_valid[gi]), .mk_req_ready(mk_req_ready[gi]), .mk_req_idx(mk_req_idx[32*gi +: 32]),
                    .mk_rsp_valid(mk_rsp_valid[gi]), .mk_rsp_keep(mk_rsp_keep[MKW*gi +: MKW]),
                    .ar_valid(a_v[NR+gi]), .ar_ready(a_g[NR+gi]), .ar_line(a_line[LW*(NR+gi) +: LW]),
                    .ar_len(a_len[4*(NR+gi) +: 4]), .r_valid(r_v[NR+gi]), .r_data(m_rdata), .idle(mk_idle[gi]));
            end else begin : g_gen
                rn_mkgen #(.MKW(MKW), .NCORE(gi == 0 ? MKNC0 : MKNC), .STAGES(MKST)) u_mk (
                    .clk(clk), .rst_n(rst_r[NR+gi]), .seed(mk_seed_q), .step(mk_step),
                    .site(mk_site[4*gi +: 4]), .layer(mk_layer[4*gi +: 4]),
                    .mk_req_valid(mk_req_valid[gi]), .mk_req_ready(mk_req_ready[gi]), .mk_req_idx(mk_req_idx[32*gi +: 32]),
                    .mk_rsp_valid(mk_rsp_valid[gi]), .mk_rsp_keep(mk_rsp_keep[MKW*gi +: MKW]), .idle(mk_idle[gi]));
            end
        end
    endgenerate

    // AR arbiter (round robin) and slot
    reg  [3:0]  ar_last;                     // last granted requester
    reg  [7:0]  rd_out;
    reg  [3:0]  ar_sel;
    reg         ar_any;
    /* verilator lint_off UNUSEDSIGNAL */   // loop index, only [3:0] is a port number
    integer i, n, i2, n2;
    /* verilator lint_on UNUSEDSIGNAL */
    always @* begin
        ar_any = 1'b0; ar_sel = 4'd0;
        for (n = 2 * NA - 1; n >= 1; n = n - 1) begin
            i = ({28'd0, ar_last} + n) % NA;
            if (n <= NA && a_v[i]) begin ar_any = 1'b1; ar_sel = i[3:0]; end
        end
    end
    wire ar_free = !m_arvalid || m_arready;
    wire ar_go = ar_any && ar_free && (rd_out < MAXR);
    generate
        for (gi = 0; gi < NA; gi = gi + 1) begin : g_ag
            assign a_g[gi] = ar_go && (ar_sel == gi);
        end
    endgenerate
    reg [LW-1:0] ar_l;
    reg [3:0]    ar_n;
    assign m_araddr  = {ar_l, {(AXL+2){1'b0}}};
    assign m_arlen   = {4'd0, ar_n};
    assign m_arsize  = AXL[2:0] + 3'd2;
    assign m_arburst = 2'b01;
    assign m_rready  = 1'b1;
    wire   rd_done = m_rvalid && m_rlast;
    always @(posedge clk) begin
        if (!rst_c) begin
            m_arvalid <= 1'b0; ar_last <= 4'd0; rd_out <= 8'd0;
        end else begin
            if (ar_go) begin
                m_arvalid <= 1'b1; m_arid <= ar_sel; ar_last <= ar_sel;
                ar_l <= a_line[LW*ar_sel +: LW]; ar_n <= a_len[4*ar_sel +: 4];
            end else if (m_arready) m_arvalid <= 1'b0;
            rd_out <= rd_out + (ar_go ? 8'd1 : 8'd0) - (rd_done ? 8'd1 : 8'd0);
        end
    end

    // ---------------------------------------------------------------- write channels
    wire [NW-1:0]      w_av, w_ag, w_wv, w_wr, w_bv, w_blk;
    wire [LW*NW-1:0]   w_line;
    wire [4*NW-1:0]    w_len;
    wire [32*AXW*NW-1:0] w_d;
    wire [AXW*NW-1:0]  w_m;
    wire [NW-1:0]      w_waw;
    wire               rd_wait = |(rh_v & r_blk) || |w_waw;   // a waiting read or write: close open bursts
    // P6c timing: the flush request is registered (one cycle later).  It only closes open write bursts early
    // so that a blocked access makes progress (the blocked access keeps it asserted); the ordering rules
    // (RAW / WAR / WAW at acceptance) are unchanged.  Removes the offered-request compare -> flush -> write
    // queue enable path (P6b: -3.79 ns at ss).
    reg                rd_wait_q;
    always @(posedge clk)
        if (!rst_c) rd_wait_q <= 1'b0;
        else        rd_wait_q <= rd_wait;
    generate
        for (gi = 0; gi < NW; gi = gi + 1) begin : g_wr
            // new entry's age counters: per read port, its buffered reads older than this beat after this cycle
            // (a read accepted in the same cycle is younger); per other write channel, its older buffered beats
            // (a beat accepted there in the same cycle is unordered: not counted)
            for (ge = 0; ge < NR; ge = ge + 1) begin : g_cr
                assign win_c[2*(WNC*gi+ge) +: 2] = {1'b0, rh_v[ge]} + {1'b0, rs_v[ge]} - {1'b0, r_go[ge]};
            end
            for (ge = 0; ge < NW; ge = ge + 1) begin : g_cw
                if (ge == gi) begin : g_self
                    assign win_c[2*(WNC*gi+NR+ge) +: 2] = 2'd0;
                end else begin : g_oth
                    assign win_c[2*(WNC*gi+NR+ge) +: 2] = {1'b0, wh_v[ge]} + {1'b0, ws_v[ge]} - {1'b0, w_go[ge]};
                end
            end
            assign w_in[gi] = wr_valid[gi] && wr_ready[gi];
            mm_skid #(.W(WSW), .NC(WNC)) u_ws (
                .clk(clk), .rst_n(rst_r[NR+NM+gi]), .in_valid(wr_valid[gi]), .in_ready(wr_ready[gi]),
                .in_d({wr_mask[8*gi +: 8], wr_data[256*gi +: 256], wr_addr[32*gi+2 +: 30]}),
                .in_c(win_c[2*WNC*gi +: 2*WNC]), .dec({w_go, r_go}),
                .h_v(wh_v[gi]), .h_d(wh_d[WSW*gi +: WSW]), .h_c(wh_c[2*WNC*gi +: 2*WNC]), .h_go(w_go[gi]),
                .s_v(ws_v[gi]), .s_d(ws_d[WSW*gi +: WSW]), .s_c(ws_c[2*WNC*gi +: 2*WNC]));
            mm_wrport #(.AXL(AXL), .WFL(WFL), .WQL(WQL)) u_wr (
                .clk(clk), .rst_n(rst_r[NR+NM+gi]),
                .wr_valid(wh_v[gi]), .wr_ready(wp_go[gi]), .wr_addr({wh_d[WSW*gi +: 30], 2'b00}),
                .wr_data(wh_d[WSW*gi+30 +: 256]), .wr_mask(wh_d[WSW*gi+286 +: 8]),
                .blk(w_blk[gi]), .flush(rd_wait_q),
                .aw_valid(w_av[gi]), .aw_ready(w_ag[gi]), .aw_line(w_line[LW*gi +: LW]), .aw_len(w_len[4*gi +: 4]),
                .w_valid(w_wv[gi]), .w_ready(w_wr[gi]), .w_data(w_d[32*AXW*gi +: 32*AXW]), .w_wm(w_m[AXW*gi +: AXW]),
                .b_valid(w_bv[gi]),
                .pend_v(wp_v[WE*gi +: WE]), .pend_lo(wp_lo[30*WE*gi +: 30*WE]), .pend_hi(wp_hi[30*WE*gi +: 30*WE]),
                .idle(wp_idle[gi]));
            assign w_go[gi] = wh_v[gi] && wp_go[gi];
            assign w_bv[gi] = m_bvalid && (m_bid == gi);
            // write-idle flag (sequencer), registered (P8f): rises one cycle after the channel has nothing buffered
            // or pending, falls with the handshake
            always @(posedge clk)
                if (!rst_r[NR+NM+gi]) wr_idle[gi] <= 1'b1;
                else                  wr_idle[gi] <= wp_idle[gi] && !wh_v[gi] && !w_in[gi];
            // WAR: the buffer head against every pending read and every older buffered read
            wire [29:0] lo = wh_d[WSW*gi +: 30];
            wire [29:0] hi = lo + 30'd7;
            wire [RQ*NR-1:0] ov;
            wire [NR-1:0]    oi;
            for (ge = 0; ge < RQ * NR; ge = ge + 1) begin : g_ov
                assign ov[ge] = rp_v[ge] && (lo <= rp_hi[30*ge +: 30]) && (rp_lo[30*ge +: 30] <= hi);
            end
            for (ge = 0; ge < NR; ge = ge + 1) begin : g_oi
                // [lo, lo + 7] overlaps [rlo, rlo + len - 1] (len >= 1), with one subtract instead of an add
                // followed by a compare (P6b timing): dd = lo - rlo; lo < rlo: overlap iff rlo - lo <= 7, i.e.
                // dd in [-7, -1]; lo >= rlo: overlap iff dd < len (<= 64).  Older buffered reads: the first c
                // entries of the read buffer (head, then skid entry)
                wire [1:0]  c    = wh_c[2*(WNC*gi+ge) +: 2];
                wire [29:0] rlo0 = rh_d[RSW*ge +: 30];
                wire [6:0]  rln0 = rh_d[RSW*ge+30 +: 7];
                wire [29:0] rlo1 = rs_d[RSW*ge +: 30];
                wire [6:0]  rln1 = rs_d[RSW*ge+30 +: 7];
                wire [30:0] dd0  = {1'b0, lo} - {1'b0, rlo0};
                wire [30:0] dd1  = {1'b0, lo} - {1'b0, rlo1};
                wire        o0   = dd0[30] ? ((&dd0[29:3]) && (|dd0[2:0])) : (!(|dd0[29:7]) && (dd0[6:0] < rln0));
                wire        o1   = dd1[30] ? ((&dd1[29:3]) && (|dd1[2:0])) : (!(|dd1[29:7]) && (dd1[6:0] < rln1));
                assign oi[ge] = ((c != 2'd0) && o0) || ((c == 2'd2) && o1);
            end
            // WAW across channels (different AWIDs are not ordered by AXI): the beat waits for every
            // overlapping pending write and older buffered beat of the other write channels
            wire [WE*NW-1:0] ow;
            wire [NW-1:0]    obw;
            for (ge = 0; ge < WE * NW; ge = ge + 1) begin : g_ow
                assign ow[ge] = (ge / WE != gi) && wp_v[ge] && (lo <= wp_hi[30*ge +: 30]) && (wp_lo[30*ge +: 30] <= hi);
            end
            for (ge = 0; ge < NW; ge = ge + 1) begin : g_obw
                wire [1:0]  c   = wh_c[2*(WNC*gi+NR+ge) +: 2];
                wire [29:0] wl0 = wh_d[WSW*ge +: 30];
                wire [29:0] wl1 = ws_d[WSW*ge +: 30];
                assign obw[ge] = (ge != gi) && (((c != 2'd0) && (lo <= wl0 + 30'd7) && (wl0 <= hi))
                                             || ((c == 2'd2) && (lo <= wl1 + 30'd7) && (wl1 <= hi)));
            end
            assign w_blk[gi] = (|ov) || (|oi) || (|ow) || (|obw);
            assign w_waw[gi] = wh_v[gi] && ((|ow) || (|obw));
        end
    endgenerate

    // AW arbiter (round robin), slot, W-order FIFO
    reg  [3:0]  aw_last, aw_sel;
    reg         aw_any;
    reg  [7:0]  wr_out;
    always @* begin
        aw_any = 1'b0; aw_sel = 4'd0;
        for (n2 = 2 * NW - 1; n2 >= 1; n2 = n2 - 1) begin
            i2 = ({28'd0, aw_last} + n2) % NW;
            if (n2 <= NW && w_av[i2]) begin aw_any = 1'b1; aw_sel = i2[3:0]; end
        end
    end
    wire        wo_inr, wo_v;
    wire [7:0]  wo_d;
    wire        aw_go = aw_any && (!m_awvalid || m_awready) && (wr_out < MAXW) && wo_inr;
    generate
        for (gi = 0; gi < NW; gi = gi + 1) begin : g_wg
            assign w_ag[gi] = aw_go && (aw_sel == gi);
        end
    endgenerate
    reg [LW-1:0] aw_l;
    reg [3:0]    aw_n;
    assign m_awaddr  = {aw_l, {(AXL+2){1'b0}}};
    assign m_awlen   = {4'd0, aw_n};
    assign m_awsize  = AXL[2:0] + 3'd2;
    assign m_awburst = 2'b01;
    assign m_bready  = 1'b1;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [3:0]   wo_cnt;
    /* verilator lint_on UNUSEDSIGNAL */
    reg  [3:0]   wcnt;
    wire [3:0]   wo_p = wo_d[7:4];
    wire [3:0]   wo_n = wo_d[3:0];
    assign m_wlast = (wcnt == wo_n);
    wire w_fire = m_wvalid && m_wready;
    lw_fifo #(.W(8), .LOG2(3)) u_wo (.clk(clk), .rst_n(rst_c), .in_valid(aw_go), .in_ready(wo_inr),
        .in_d({aw_sel, w_len[4*aw_sel +: 4]}), .out_valid(wo_v), .out_ready(w_fire && m_wlast), .out_d(wo_d), .count(wo_cnt));
    wire [NW-1:0] wsel;
    generate
        for (gi = 0; gi < NW; gi = gi + 1) begin : g_ws
            assign wsel[gi] = (wo_p == gi);
        end
    endgenerate
    assign m_wvalid = wo_v && |(w_wv & wsel);
    assign m_wdata  = w_d[32*AXW*wo_p +: 32*AXW];
    generate
        for (gi = 0; gi < AXW; gi = gi + 1) begin : g_strb
            assign m_wstrb[4*gi +: 4] = {4{w_m[AXW*wo_p + gi]}};
        end
        for (gi = 0; gi < NW; gi = gi + 1) begin : g_wrd
            assign w_wr[gi] = wo_v && m_wready && (wo_p == gi);
        end
    endgenerate
    always @(posedge clk) begin
        if (!rst_c) begin
            m_awvalid <= 1'b0; aw_last <= 4'd0; wr_out <= 8'd0; wcnt <= 4'd0; err <= 2'b00;
        end else begin
            if (aw_go) begin
                m_awvalid <= 1'b1; m_awid <= aw_sel; aw_last <= aw_sel;
                aw_l <= w_line[LW*aw_sel +: LW]; aw_n <= w_len[4*aw_sel +: 4];
            end else if (m_awready) m_awvalid <= 1'b0;
            wr_out <= wr_out + (aw_go ? 8'd1 : 8'd0) - (m_bvalid ? 8'd1 : 8'd0);
            if (w_fire) wcnt <= m_wlast ? 4'd0 : wcnt + 4'd1;
            if (m_rvalid && m_rresp != 2'b00) err[0] <= 1'b1;
            if (m_bvalid && m_bresp != 2'b00) err[1] <= 1'b1;
        end
    end
endmodule
`default_nettype wire
