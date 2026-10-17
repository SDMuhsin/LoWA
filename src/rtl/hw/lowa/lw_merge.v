`default_nettype none
// lw_merge: LoWA merge-on-load (docs/dataflow.md section 3, docs/p5_report.md).  While the GEMM
// engine streams W0 tiles into its W buffers, this engine adds gamma * c_j * h_tap at every cell of
// the tile hit by an atom tap, through gm_wfeed's single-word RMW port (mg_*), in the hit-list order
// (block, then entry e = 25 j + tap ascending), one fp_add per hit:
//     v = mul(gamma, c_j); val = mul(v, h_tap); W[cell] = add(W[cell], val)
// Tables (sw/lowa_tables.py): per module a hit list (6400 x 32 bit, sorted by 64 x 64 W block), a
// 256-word block directory (first index, count) and the coefficients (PARAM).  The engine replays the
// loader's tile order (nb, mb, seg, kc; docs/p3_report.md) so that it knows the block of every
// w_tile_done: ROW loads use block (nb, kc), TRN loads (kc, nb) of the same directory.  Up to two
// module slots (A, B) per descriptor, one per K segment (GEMM_ACC2: Wq in segment 0, Wv in 2).
// Per tile: the hits are fetched into a FIFO (one tile ahead), and from w_tile_done until the last
// write merge_busy[buf] holds the GEMM controller off the buffer.  Pipeline per hit (issued every
// other cycle; hits on a cell still in flight wait):
//   s0 pop, lane/address, coef read | s1 mul(gamma, c) | s3 mul(v, h), mg read | s5 add | s8 mg write
// Only COLS = 16, LAMBDA = 4 (64-row W blocks, lane (c + k + 4q) mod 16, address {k, q}).
module lw_merge (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    output wire         busy,
    output reg          done,
    // descriptor
    input  wire         d_trn,
    input  wire [7:0]   d_nb, d_mb, d_kc0, d_kc1, d_kc2,
    input  wire [1:0]   d_ms0, d_ms1, d_ms2,          // module slot per K segment: 0 none, 1 A, 2 B
    input  wire [31:0]  d_hits_a, d_dir_a, d_coef_a, d_hits_b, d_dir_b, d_coef_b, d_h_base, d_gamma,
    // memory read channel
    output wire         rd_req_valid,
    input  wire         rd_req_ready,
    output wire [31:0]  rd_req_addr,
    output wire [6:0]   rd_req_len,
    input  wire         rd_dat_valid,
    output wire         rd_dat_ready,
    input  wire [255:0] rd_dat,
    // GEMM engine W-buffer merge port
    output reg          mg_en,
    output reg          mg_buf,
    output reg          mg_we,
    output reg  [3:0]   mg_lane,
    output reg  [7:0]   mg_addr,
    output reg  [31:0]  mg_wdata,
    input  wire [31:0]  mg_rdata,
    input  wire         w_tile_done,
    input  wire         w_tile_buf,
    output reg  [1:0]   merge_busy,
    output reg  [15:0]  stat_tiles                      // tiles merged by this descriptor
);
    // ------------------------------------------------------------------ main FSM
    localparam [3:0] S_IDLE = 4'd0, S_H = 4'd1, S_CA = 4'd2, S_DA = 4'd3, S_CB = 4'd4, S_DB = 4'd5,
                     S_TILE = 4'd6, S_DIR = 4'd7, S_DIR2 = 4'd8, S_DIR3 = 4'd12, S_HITS = 4'd9, S_MARK = 4'd10, S_WAIT = 4'd11;
    reg  [3:0]  st;
    reg         use_b;
    assign busy = (st != S_IDLE);
    // sequential reader
    reg         rq_go;
    reg  [31:0] rq_base;
    reg  [15:0] rq_n;
    wire        rq_busy, rq_done, rw_valid, rw_ready;
    wire [31:0] rw_d;
    wire [15:0] rw_idx;
    lw_rdseq u_rd (.clk(clk), .rst_n(rst_n), .go(rq_go), .base(rq_base), .nwords(rq_n), .busy(rq_busy), .done(rq_done),
                   .req_valid(rd_req_valid), .req_ready(rd_req_ready), .req_addr(rd_req_addr), .req_len(rd_req_len),
                   .dat_valid(rd_dat_valid), .dat_ready(rd_dat_ready), .dat(rd_dat),
                   .w_valid(rw_valid), .w_ready(rw_ready), .w_d(rw_d), .w_idx(rw_idx));
    // tile sequence (loader order: nb, mb, seg, kc)
    reg  [7:0]  t_nb, t_mb, t_kc;
    reg  [1:0]  t_seg;
    wire [7:0]  seg_kc = (t_seg == 2'd0) ? d_kc0 : (t_seg == 2'd1) ? d_kc1 : d_kc2;
    wire [1:0]  seg_ms = (t_seg == 2'd0) ? d_ms0 : (t_seg == 2'd1) ? d_ms1 : d_ms2;
    wire [1:0]  last_seg = (d_kc2 != 8'd0) ? 2'd2 : (d_kc1 != 8'd0) ? 2'd1 : 2'd0;
    wire        t_last = (t_kc == seg_kc - 8'd1) & (t_seg == last_seg) & (t_mb == d_mb - 8'd1) & (t_nb == d_nb - 8'd1);
    reg         t_more;                                 // tiles left to fetch
    wire [3:0]  blk_a = d_trn ? t_kc[3:0] : t_nb[3:0];
    wire [3:0]  blk_b = d_trn ? t_nb[3:0] : t_kc[3:0];
    reg         t_slot;                                 // 0 = A, 1 = B of the tile being fetched
    // directory / coefficient banks: 2 slots x 256 words
    reg         dir_en, dir_we;
    reg  [8:0]  dir_addr;
    reg  [31:0] ld_d;                                   // word registered with the bank controls
    wire [31:0] dir_q;
    always @(posedge clk) ld_d <= rw_d;
    mm_sram_bank #(.WORDS(512), .WIDTH(32), .AW(9)) u_dir (.clk(clk), .en(dir_en), .we(dir_we), .addr(dir_addr), .d(ld_d), .q(dir_q));
    reg         cf_en, cf_we;
    reg  [8:0]  cf_addr;
    wire [31:0] cf_q;
    mm_sram_bank #(.WORDS(512), .WIDTH(32), .AW(9)) u_cf (.clk(clk), .en(cf_en), .we(cf_we), .addr(cf_addr), .d(ld_d), .q(cf_q));
    // taps
    reg  [25*32-1:0] hreg;
    genvar gi;
    generate
        for (gi = 0; gi < 25; gi = gi + 1) begin : g_h
            localparam [4:0] GI = gi;
            always @(posedge clk) if ((st == S_H) & rw_valid & (rw_idx[4:0] == GI)) hreg[gi*32 +: 32] <= rw_d;
        end
    endgenerate
    // hit FIFO: {marker, slot, hit[24:0]}
    wire        hf_in_valid, hf_in_ready, hf_out_valid, hf_out_ready;
    wire [26:0] hf_in_d, hf_out_d;
    wire [6:0]  hf_count;
    lw_fifo #(.W(27), .LOG2(6)) u_hf (.clk(clk), .rst_n(rst_n), .in_valid(hf_in_valid), .in_ready(hf_in_ready), .in_d(hf_in_d),
                                       .out_valid(hf_out_valid), .out_ready(hf_out_ready), .out_d(hf_out_d), .count(hf_count));
    wire        push_mark = (st == S_MARK) & hf_in_ready;
    assign hf_in_valid = ((st == S_HITS) & rw_valid) | push_mark;
    assign hf_in_d = (st == S_MARK) ? {1'b1, t_slot, 25'd0} : {1'b0, t_slot, rw_d[24:0]};
    assign rw_ready = (st == S_HITS) ? hf_in_ready : 1'b1;
    wire        ld_word = rw_valid & rw_ready;
    wire        p_issue;
    always @(posedge clk) begin
        dir_en <= 1'b0; dir_we <= 1'b0; cf_en <= 1'b0; cf_we <= 1'b0;
        if ((st == S_DA || st == S_DB) & ld_word) begin dir_en <= 1'b1; dir_we <= 1'b1; dir_addr <= {st == S_DB, rw_idx[7:0]}; end
        if ((st == S_CA || st == S_CB) & ld_word) begin cf_en <= 1'b1; cf_we <= 1'b1; cf_addr <= {st == S_CB, rw_idx[7:0]}; end
        if (st == S_DIR) begin dir_en <= 1'b1; dir_addr <= {t_slot, blk_a, blk_b}; end
        if (p_issue) begin cf_en <= 1'b1; cf_addr <= {hf_out_d[25], hf_out_d[24:17]}; end
    end
    // ------------------------------------------------------------------ consumer: tile events + RMW pipeline
    wire        ev_in_ready, ev_out_valid, ev_pop;
    wire        ev_buf;
    wire [2:0]  ev_count;
    lw_fifo #(.W(1), .LOG2(2)) u_ev (.clk(clk), .rst_n(rst_n), .in_valid(w_tile_done & busy), .in_ready(ev_in_ready), .in_d(w_tile_buf),
                                      .out_valid(ev_out_valid), .out_ready(ev_pop), .out_d(ev_buf), .count(ev_count));
    reg         phase;                                  // hits issue on phase 0 only
    reg         draining;                               // marker seen, waiting for the pipeline
    reg  [8:0]  pv;                                     // stage valid bits s0..s8
    reg  [11:0] pc [0:8];                               // {lane, addr} per stage
    reg         pbuf [0:8];
    reg  [24:0] p_oh_tap_s0, p_oh_tap_s1, p_oh_tap_s2, p_oh_tap_s3;   // one-hot tap (25), s0 .. s3
    wire        head_mark = hf_out_valid & hf_out_d[26];
    wire [5:0]  h_r = hf_out_d[5:0], h_c = hf_out_d[11:6];
    wire [5:0]  h_n = d_trn ? h_c : h_r;
    wire [5:0]  h_k = d_trn ? h_r : h_c;
    wire [3:0]  h_lane = h_n[5:2] + h_k[3:0] + {h_n[1:0], 2'b00};
    wire [11:0] h_cell = {h_lane, h_k, h_n[1:0]};
    wire        hazard = (pv[0] & (pc[0] == h_cell)) | (pv[1] & (pc[1] == h_cell)) | (pv[2] & (pc[2] == h_cell)) |
                         (pv[3] & (pc[3] == h_cell)) | (pv[4] & (pc[4] == h_cell)) | (pv[5] & (pc[5] == h_cell));
    assign      p_issue = ev_out_valid & ~draining & hf_out_valid & ~hf_out_d[26] & ~phase & ~hazard;
    wire        take_mark = ev_out_valid & ~draining & head_mark;
    assign hf_out_ready = p_issue | take_mark;
    assign ev_pop = draining & (pv == 9'd0);
    // fp units
    wire        m1_ov, m2_ov, ad_ov;
    wire [31:0] m1_y, m2_y, ad_y, h_sel;
    fp_mul #(.STAGES(2)) u_m1 (.clk(clk), .rst_n(rst_n), .in_valid(pv[1]), .a(d_gamma), .b(cf_q), .out_valid(m1_ov), .y(m1_y));
    cm_ohmux #(.W(32), .N(25), .REP(1)) u_hsel (.sel(p_oh_tap_s3), .d(hreg), .y(h_sel));
    fp_mul #(.STAGES(2)) u_m2 (.clk(clk), .rst_n(rst_n), .in_valid(pv[3]), .a(m1_y), .b(h_sel), .out_valid(m2_ov), .y(m2_y));
    reg  [31:0] w_s4;
    fp_add #(.STAGES(2)) u_ad (.clk(clk), .rst_n(rst_n), .in_valid(pv[5]), .a(w_s4), .b(m2_y), .sub(1'b0), .out_valid(ad_ov), .y(ad_y));
    wire unused_ov = m1_ov | m2_ov | ad_ov;
    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            pv <= 9'd0; phase <= 1'b0; draining <= 1'b0; merge_busy <= 2'b00; mg_en <= 1'b0; mg_we <= 1'b0;
        end else begin
            mg_en <= pv[2] | pv[7];
            mg_we <= pv[7];
            phase <= ~phase & busy;
            pv <= {pv[7:0], p_issue};
            if (w_tile_done & busy) merge_busy[w_tile_buf] <= 1'b1;
            if (take_mark) draining <= 1'b1;
            if (ev_pop) begin draining <= 1'b0; merge_busy[ev_buf] <= 1'b0; end
        end
        for (i = 8; i > 0; i = i - 1) begin pc[i] <= pc[i-1]; pbuf[i] <= pbuf[i-1]; end
        if (p_issue) begin pc[0] <= h_cell; pbuf[0] <= ev_buf; end
        p_oh_tap_s0 <= 25'd1 << hf_out_d[16:12];
        p_oh_tap_s1 <= p_oh_tap_s0; p_oh_tap_s2 <= p_oh_tap_s1; p_oh_tap_s3 <= p_oh_tap_s2;
        if (pv[4]) w_s4 <= mg_rdata;
        mg_buf <= pv[7] ? pbuf[7] : pbuf[2];
        mg_lane <= pv[7] ? pc[7][11:8] : pc[2][11:8];
        mg_addr <= pv[7] ? pc[7][7:0] : pc[2][7:0];
        mg_wdata <= ad_y;
    end
    // ------------------------------------------------------------------ FSM
    reg  [15:0] t_cnt_total;
    always @(posedge clk)
        if (!rst_n) begin
            st <= S_IDLE; rq_go <= 1'b0; done <= 1'b0; t_more <= 1'b0; stat_tiles <= 16'd0;
        end else begin
            rq_go <= 1'b0; done <= 1'b0;
            case (st)
                S_IDLE: if (start) begin
                    st <= S_H; rq_go <= 1'b1; rq_base <= d_h_base; rq_n <= 16'd25;
                    use_b <= (d_ms0 == 2'd2) | (d_ms1 == 2'd2) | (d_ms2 == 2'd2);
                    t_nb <= 8'd0; t_mb <= 8'd0; t_seg <= 2'd0; t_kc <= 8'd0; t_more <= 1'b1; stat_tiles <= 16'd0; t_cnt_total <= 16'd0;
                end
                S_H:  if (rq_done) begin st <= S_CA; rq_go <= 1'b1; rq_base <= d_coef_a; rq_n <= 16'd256; end
                S_CA: if (rq_done) begin st <= S_DA; rq_go <= 1'b1; rq_base <= d_dir_a; rq_n <= 16'd256; end
                S_DA: if (rq_done) begin
                    if (use_b) begin st <= S_CB; rq_go <= 1'b1; rq_base <= d_coef_b; rq_n <= 16'd256; end
                    else st <= S_TILE;
                end
                S_CB: if (rq_done) begin st <= S_DB; rq_go <= 1'b1; rq_base <= d_dir_b; rq_n <= 16'd256; end
                S_DB: if (rq_done) st <= S_TILE;
                S_TILE: begin                           // classify the next tile
                    if (!t_more) st <= S_WAIT;
                    else if (seg_ms == 2'd0) st <= S_MARK;
                    else begin t_slot <= seg_ms[1]; st <= S_DIR; end
                end
                S_DIR:  st <= S_DIR2;                   // dir_en registered
                S_DIR2: st <= S_DIR3;                   // bank access
                S_DIR3: begin                           // dir_q valid
                    if (dir_q[23:16] == 8'd0) st <= S_MARK;
                    else begin
                        st <= S_HITS; rq_go <= 1'b1; rq_n <= {8'd0, dir_q[23:16]};
                        rq_base <= (t_slot ? d_hits_b : d_hits_a) + {14'd0, dir_q[15:0], 2'b00};
                    end
                end
                S_HITS: if (rq_done) st <= S_MARK;
                S_MARK: if (hf_in_ready) begin          // marker pushed; advance the tile counters
                    st <= S_TILE;
                    t_cnt_total <= t_cnt_total + 16'd1;
                    if (t_last) t_more <= 1'b0;
                    else if (t_kc != seg_kc - 8'd1) t_kc <= t_kc + 8'd1;
                    else begin
                        t_kc <= 8'd0;
                        if (t_seg != last_seg) t_seg <= t_seg + 2'd1;
                        else begin
                            t_seg <= 2'd0;
                            if (t_mb != d_mb - 8'd1) t_mb <= t_mb + 8'd1;
                            else begin t_mb <= 8'd0; t_nb <= t_nb + 8'd1; end
                        end
                    end
                end
                S_WAIT: if ((stat_tiles == t_cnt_total) & ~ev_out_valid & (hf_count == 7'd0)) begin st <= S_IDLE; done <= 1'b1; end
                default: st <= S_IDLE;
            endcase
            if (ev_pop) stat_tiles <= stat_tiles + 16'd1;
        end
    wire unused_m = (|ev_count) | ev_in_ready | (|rw_idx[15:8]) | (|dir_q[31:24]) | rq_busy;
endmodule
`default_nettype wire
