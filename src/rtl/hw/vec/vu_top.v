`default_nettype none
// vu_top: the LoWA-TA vector / SFU unit (P4).  V lanes (vu_lane: tile buffers, one fp_fma under
// micro-ops, accumulator, max / min, fp_sfu), the sequencer (vu_seq, descriptor -> tiles -> phases ->
// row segments -> elements), the loader (vu_loader, rd0 tiles / rd1 scalars, ids, column vectors),
// the dropout keep-bit fetcher (vu_mask) and the writer (vu_writer).  Executes one vector
// descriptor (d_*; held stable from `start` until `done`): LN_FWD, LN_BWD, SMAX_FWD, SMAX_REC,
// SMAX_BWD, EMB, CE, MSE, GN_ROWS, GN_FIN and the element-wise ops (flow/gen/vu_ucode.py VOPS).
// Memory side = the GEMM engine's (docs/p3_report.md 2.5): rd0 / rd1 requests (addr, len <= 64
// words) + 256-bit beats in order, wr beats + word mask, mk keep-bit requests (index) + MKW-bit
// responses; P6 attaches the same DMA.
// Lane = row for every row reduction (reduction order independent of V, docs/dataflow.md 5).
module vu_top #(
    parameter V    = 16,                // lanes (power of two, 4 .. 16)
    parameter VW   = 4,                 // log2 V
    parameter NS   = 8,                 // barrel slots per lane
    parameter SW   = 3,
    parameter RB   = 256,               // words per tile buffer (row segment)
    parameter RBW  = 8,
    parameter NOB  = 2,
    parameter HBL  = 2,                 // log2 beats per rd0 row request (vu_loader, P6b ruling R8)
    parameter WK   = 1,                 // vu_writer groups per row visit (<= NOB); 1 = row-interleaved (P6b: 2 measured slower)
    parameter MKW  = 16,
    parameter LF   = 2
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    output wire         busy,
    output wire         done,
    // descriptor
    input  wire [4:0]   d_vop,
    input  wire [15:0]  d_rows, d_cols, d_rows_per_b, d_emb_l,
    input  wire         d_st_en, d_x_reset, d_g_reset,
    input  wire [31:0]  d_a_base, d_a_rs, d_b_base, d_b_rs, d_c_base, d_c_rs,
    input  wire [31:0]  d_cv0_base, d_cv1_base, d_cv2_base, d_km_rs,
    input  wire [31:0]  d_st0_base, d_st1_base, d_o2_base, d_ids_base,
    input  wire [31:0]  d_mk_base, d_mk_rs,
    input  wire [31:0]  d_c0, d_c1, d_c2, d_c3, d_ds,
    // memory read channel 0 (tiles)
    output wire         rd0_req_valid,
    input  wire         rd0_req_ready,
    output wire [31:0]  rd0_req_addr,
    output wire [6:0]   rd0_req_len,
    input  wire         rd0_dat_valid,
    output wire         rd0_dat_ready,
    input  wire [255:0] rd0_dat,
    // memory read channel 1 (scalars, ids, column vectors)
    output wire         rd1_req_valid,
    input  wire         rd1_req_ready,
    output wire [31:0]  rd1_req_addr,
    output wire [6:0]   rd1_req_len,
    input  wire         rd1_dat_valid,
    output wire         rd1_dat_ready,
    input  wire [255:0] rd1_dat,
    // mask channel
    output wire         mk_req_valid,
    input  wire         mk_req_ready,
    output wire [31:0]  mk_req_idx,
    input  wire         mk_rsp_valid,
    input  wire [MKW-1:0] mk_rsp_keep,
    // write channel
    output wire         wr_valid,
    input  wire         wr_ready,
    output wire [31:0]  wr_addr,
    output wire [255:0] wr_data,
    output wire [7:0]   wr_mask
);
    // ------------------------------------------------------------------ reset tree
    wire [3:0] rst_r;
    genvar gi;
    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : g_rst
            cm_rstbuf u_rb (.clk(clk), .rst_n(rst_n), .rst_n_q(rst_r[gi]));
        end
    endgenerate
    wire [V-1:0] rst_l;
    generate
        for (gi = 0; gi < V; gi = gi + 1) begin : g_rstl
            cm_rstbuf u_rb (.clk(clk), .rst_n(rst_n), .rst_n_q(rst_l[gi]));
        end
    endgenerate

    // ------------------------------------------------------------------ sequencer
    wire        iss_v, iss_abuf, cv_par, km_mode, ex_v, acc_start, acc_chunk, acc_flush, max_start, ob_flush, cur_abuf;
    wire [RBW-1:0] iss_addr;
    wire [SW-1:0] iss_slot, ex_slot;
    wire [35:0] ex_uop;
    wire        xl_sel, sreg_clr;
    wire [VW-1:0] xl_row;
    wire [31:0] xtot;
    wire        ld_start, ld_done, ld_a, ld_b, ld_abuf, ld_gather, ld_km, ld_cvpar;
    wire [2:0]  ld_cv;
    wire [1:0]  ld_sreg;
    wire [15:0] ld_row0, ld_seg_off, ld_b_idx;
    wire [VW:0] ld_nrows;
    wire [RBW:0] ld_seg_len;
    wire        mk_start, mk_pop;
    wire [31:0] mk_idx0;
    wire [VW:0] mk_nrows;
    wire [RBW:0] mk_len;
    wire        wr_req, wr_room, wr_idle;
    wire [1:0]  wr_kind;
    wire [15:0] wr_row0, wr_seg_off;
    wire [VW:0] wr_nrows;
    wire [RBW:0] wr_seg_len;
    // lane status
    wire [V-1:0] l_acc_idle, l_st_busy, l_ob_rdy, l_acc_room, l_st_ok, l_ob_room;
    wire [V*4-1:0] l_sfu_cnt;
    wire [V*4-1:0] l_acc_cnt, l_ob_nw;
    wire [V*256-1:0] l_ob_data;
    wire [V*32-1:0] l_xl, l_s0, l_s3, l_s1;
    wire [V-1:0] ob_pop;
    wire        acc_idle_all = &l_acc_idle, acc_room_all = &l_acc_room,
                st_busy_any = |l_st_busy, st_ok_all = &l_st_ok, ob_room_all = &l_ob_room;
    reg  [3:0]  sfu_cnt_min;
    integer mi;
    always @* begin
        sfu_cnt_min = 4'hF;
        for (mi = 0; mi < V; mi = mi + 1) if (l_sfu_cnt[mi*4 +: 4] < sfu_cnt_min) sfu_cnt_min = l_sfu_cnt[mi*4 +: 4];
    end
    wire [2:0]  mk_cnt_min;
    wire [31:0] xl_lane_val = l_xl[xl_row * 32 +: 32];
    vu_seq #(.V(V), .VW(VW), .NS(NS), .SW(SW), .RB(RB), .RBW(RBW)) u_seq (
        .clk(clk), .rst_n(rst_r[0]), .start(start), .busy(busy), .done(done),
        .d_vop(d_vop), .d_rows(d_rows), .d_cols(d_cols), .d_rows_per_b(d_rows_per_b),
        .d_st_en(d_st_en), .d_x_reset(d_x_reset), .d_g_reset(d_g_reset), .d_mk_base(d_mk_base), .d_mk_rs(d_mk_rs),
        .acc_idle_all(acc_idle_all), .acc_room_all(acc_room_all), .sfu_cnt_min(sfu_cnt_min), .st_busy_any(st_busy_any),
        .st_ok_all(st_ok_all), .ob_room_all(ob_room_all), .xl_lane_val(xl_lane_val),
        .iss_v(iss_v), .iss_abuf(iss_abuf), .iss_addr(iss_addr), .iss_slot(iss_slot), .cv_par(cv_par), .km_mode(km_mode),
        .ex_v(ex_v), .ex_slot(ex_slot), .ex_uop(ex_uop), .acc_start(acc_start), .acc_chunk(acc_chunk), .acc_flush(acc_flush),
        .max_start(max_start), .ob_flush(ob_flush), .cur_abuf(cur_abuf), .xl_sel(xl_sel), .xl_row(xl_row), .xtot(xtot),
        .sreg_clr(sreg_clr),
        .ld_start(ld_start), .ld_done(ld_done), .ld_a(ld_a), .ld_b(ld_b), .ld_abuf(ld_abuf), .ld_gather(ld_gather),
        .ld_km(ld_km), .ld_cvpar(ld_cvpar), .ld_cv(ld_cv), .ld_sreg(ld_sreg), .ld_row0(ld_row0), .ld_seg_off(ld_seg_off),
        .ld_b_idx(ld_b_idx), .ld_nrows(ld_nrows), .ld_seg_len(ld_seg_len),
        .mk_start(mk_start), .mk_idx0(mk_idx0), .mk_nrows(mk_nrows), .mk_len(mk_len), .mk_cnt_min(mk_cnt_min), .mk_pop(mk_pop),
        .wr_req(wr_req), .wr_kind(wr_kind), .wr_row0(wr_row0), .wr_nrows(wr_nrows), .wr_seg_off(wr_seg_off),
        .wr_seg_len(wr_seg_len), .wr_room(wr_room), .wr_idle(wr_idle));

    // ------------------------------------------------------------------ loader
    wire [V-1:0] lt_we, lt_buf, sld_v;
    wire [V*RBW-1:0] lt_addr;
    wire [V*32-1:0] lt_d, sld_d;
    wire [1:0]  sld_sel;
    wire [2:0]  cv_we;
    wire        cv_wpar;
    wire [RBW-1:0] cv_waddr;
    wire [31:0] cv_wd;
    vu_loader #(.V(V), .VW(VW), .RBW(RBW), .HBL(HBL)) u_ld (
        .clk(clk), .rst_n(rst_r[1]), .unit_start(start & ~busy), .ld_start(ld_start), .ld_done(ld_done),
        .ld_a(ld_a), .ld_b(ld_b), .ld_abuf(ld_abuf), .ld_gather(ld_gather), .ld_km(ld_km), .ld_cvpar(ld_cvpar),
        .ld_cv(ld_cv), .ld_sreg(ld_sreg), .ld_row0(ld_row0), .ld_seg_off(ld_seg_off), .ld_b_idx(ld_b_idx),
        .ld_nrows(ld_nrows), .ld_seg_len(ld_seg_len),
        .d_a_base(d_a_base), .d_a_rs(d_a_rs), .d_b_base(d_b_base), .d_b_rs(d_b_rs),
        .d_cv0_base(d_cv0_base), .d_cv1_base(d_cv1_base), .d_cv2_base(d_cv2_base), .d_km_rs(d_km_rs),
        .d_st0_base(d_st0_base), .d_st1_base(d_st1_base), .d_ids_base(d_ids_base), .d_cols(d_cols), .d_emb_l(d_emb_l),
        .rd0_req_valid(rd0_req_valid), .rd0_req_ready(rd0_req_ready), .rd0_req_addr(rd0_req_addr), .rd0_req_len(rd0_req_len),
        .rd0_dat_valid(rd0_dat_valid), .rd0_dat_ready(rd0_dat_ready), .rd0_dat(rd0_dat),
        .rd1_req_valid(rd1_req_valid), .rd1_req_ready(rd1_req_ready), .rd1_req_addr(rd1_req_addr), .rd1_req_len(rd1_req_len),
        .rd1_dat_valid(rd1_dat_valid), .rd1_dat_ready(rd1_dat_ready), .rd1_dat(rd1_dat),
        .lt_we(lt_we), .lt_buf(lt_buf), .lt_addr(lt_addr), .lt_d(lt_d),
        .sld_v(sld_v), .sld_sel(sld_sel), .sld_d(sld_d),
        .cv_we(cv_we), .cv_wpar(cv_wpar), .cv_waddr(cv_waddr), .cv_wd(cv_wd));

    // ------------------------------------------------------------------ column-vector banks (3 x 2 parities)
    // read with the element issue (address j, or (seg_off + j) >> 5 in key-mask mode); data to the
    // lanes one cycle later
    wire [RBW-1:0] cv_raddr = km_mode ? {5'd0, iss_addr[RBW-1:5]} : iss_addr;   // softmax rows fit one segment
    wire [31:0] cvq [0:5];
    generate
        for (gi = 0; gi < 6; gi = gi + 1) begin : g_cv
            wire par_i = ((gi % 2) != 0);
            wire we_i = cv_we[gi / 2] & (cv_wpar == par_i);
            wire rd_i = iss_v & (cv_par == par_i);
            mm_sram_bank #(.WORDS(RB), .WIDTH(32), .AW(RBW)) u_cv (
                .clk(clk), .en(we_i | rd_i), .we(we_i), .addr(we_i ? cv_waddr : cv_raddr), .d(cv_wd), .q(cvq[gi]));
        end
    endgenerate
    reg  cv_par_q, km_mode_q;
    reg  [4:0] km_bit_q;
    always @(posedge clk) begin cv_par_q <= cv_par; km_mode_q <= km_mode; km_bit_q <= iss_addr[4:0]; end
    wire [31:0] cv0_d = cv_par_q ? cvq[1] : cvq[0];
    wire [31:0] cv1_d = cv_par_q ? cvq[3] : cvq[2];
    wire [31:0] cv2_d = cv_par_q ? cvq[5] : cvq[4];
    wire        km_d = ~km_mode_q | cv0_d[km_bit_q];

    // ------------------------------------------------------------------ keep bits and labels
    wire [V-1:0] keep_bit;
    vu_mask #(.V(V), .VW(VW), .RBW(RBW), .MKW(MKW)) u_mk (
        .clk(clk), .rst_n(rst_r[2]), .mk_start(mk_start), .mk_idx0(mk_idx0), .mk_rs(d_mk_rs), .mk_nrows(mk_nrows),
        .mk_len(mk_len), .mk_pop(mk_pop), .mk_cnt_min(mk_cnt_min), .keep_bit(keep_bit),
        .mk_req_valid(mk_req_valid), .mk_req_ready(mk_req_ready), .mk_req_idx(mk_req_idx),
        .mk_rsp_valid(mk_rsp_valid), .mk_rsp_keep(mk_rsp_keep));

    // ------------------------------------------------------------------ lanes
    generate
        for (gi = 0; gi < V; gi = gi + 1) begin : g_lane
            wire isy = (l_s0[gi*32 +: 32] == {{(32-RBW){1'b0}}, iss_addr});
            assign l_acc_room[gi] = (l_acc_cnt[gi*4 +: 4] < 4'd4);     // up to 3 pushes in flight (depth 8)
            assign l_st_ok[gi] = ~l_st_busy[gi];
            vu_lane #(.NS(NS), .SW(SW), .RB(RB), .RBW(RBW), .NOB(NOB), .LF(LF)) u_lane (
                .clk(clk), .rst_n(rst_l[gi]),
                .ld_we(lt_we[gi]), .ld_buf(lt_buf[gi]), .ld_addr(lt_addr[gi*RBW +: RBW]), .ld_d(lt_d[gi*32 +: 32]),
                .sld_v(sld_v[gi] | sreg_clr), .sld_sel(sreg_clr ? 2'd2 : sld_sel), .sld_d(sreg_clr ? 32'd0 : sld_d[gi*32 +: 32]),
                .iss_v(iss_v), .iss_abuf(iss_abuf), .iss_addr(iss_addr), .iss_slot(iss_slot),
                .iss_keep(keep_bit[gi]), .iss_isy(isy), .km_d(km_d), .cv0_d(cv0_d), .cv1_d(cv1_d), .cv2_d(cv2_d),
                .ex_v(ex_v), .ex_slot(ex_slot), .ex_uop(ex_uop), .c0(d_c0), .c1(d_c1), .c2(d_c2), .c3(d_c3), .ds(d_ds),
                .acc_start(acc_start), .acc_chunk(acc_chunk), .acc_flush(acc_flush), .acc_idle(l_acc_idle[gi]), .acc_cnt(l_acc_cnt[gi*4 +: 4]),
                .max_start(max_start), .sfu_out_cnt(l_sfu_cnt[gi*4 +: 4]), .st_busy(l_st_busy[gi]), .cur_abuf(cur_abuf),
                .ob_flush(ob_flush), .ob_room(l_ob_room[gi]), .ob_rdy(l_ob_rdy[gi]), .ob_nw(l_ob_nw[gi*4 +: 4]),
                .ob_data(l_ob_data[gi*256 +: 256]), .ob_pop(ob_pop[gi]),
                .xl_sel(xl_sel), .xtot(xtot), .xl_val(l_xl[gi*32 +: 32]), .s0_o(l_s0[gi*32 +: 32]), .s1_o(l_s1[gi*32 +: 32]),
                .s3_o(l_s3[gi*32 +: 32]));
        end
    endgenerate

    // ------------------------------------------------------------------ writer
    vu_writer #(.V(V), .VW(VW), .RBW(RBW), .WK(WK)) u_wr (
        .clk(clk), .rst_n(rst_r[3]),
        .d_c_base(d_c_base), .d_c_rs(d_c_rs), .d_st0_base(d_st0_base), .d_st1_base(d_st1_base), .d_o2_base(d_o2_base),
        .wr_req(wr_req), .wr_kind(wr_kind), .wr_row0(wr_row0), .wr_nrows(wr_nrows), .wr_seg_off(wr_seg_off),
        .wr_seg_len(wr_seg_len), .wr_room(wr_room), .wr_idle(wr_idle),
        .ob_rdy(l_ob_rdy), .ob_nw(l_ob_nw), .ob_data(l_ob_data), .ob_pop(ob_pop),
        .lane_s0(l_s0), .lane_s1(l_s1), .lane_s3(l_s3),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_addr(wr_addr), .wr_data(wr_data), .wr_mask(wr_mask));
endmodule
`default_nettype wire
