`timescale 1ns/1ps
// tb_lw_gemm: the GEMM engine (gm_top) with the LoWA unit's merge engine (lw_top, merge queue) attached to
// its W-buffer RMW port: one GEMM descriptor runs on W0 streamed from memory while lw_merge adds the atoms to
// every W tile; every word the engine writes is compared with L1's GEMM on L1's merged weight
// (tb/lowa/gen_lowa.py merge_gemm / nomerge_gemm).  Same plusargs and memory model as tb_gm_top (two GEMM
// read channels + one for the merge tables, one write channel).  Prints CYCLES, MERGE (tiles, mg accesses,
// max / total cycles merge_busy held a buffer) and one PASS / FAIL line.
module tb_lw_gemm;
    parameter ROWS = 16;
    parameter COLS = 16;
    parameter NSFU = 16;
    parameter MEMW = 23;
    localparam DWP = (ROWS * COLS >= 64) ? ROWS * COLS / 64 : 1;
    localparam GWD = 4 * DWP;

    reg clk = 1'b0, rst_n = 1'b0;
    always #20 clk = ~clk;
    integer cyc = 0;
    always @(posedge clk) cyc <= cyc + 1;

    // ------------------------------------------------------------------ descriptors
    reg [15:0] d_m = 0, d_n = 0, d_k0 = 0, d_k1 = 0, d_k2 = 0, d_nb1 = 0, d_nb2 = 0;
    reg [31:0] d_a_base0 = 0, d_a_base1 = 0, d_a_base2 = 0, d_a_rs = 0, d_a_bs1 = 0, d_a_bs2 = 0;
    reg [31:0] d_b_base0 = 0, d_b_base1 = 0, d_b_base2 = 0, d_b_rs = 0, d_b_bs1 = 0, d_b_bs2 = 0;
    reg [31:0] d_c_base = 0, d_c_rs = 0, d_c_bs1 = 0, d_c_bs2 = 0, d_bias_base = 0, d_aux_base = 0, d_o2_base = 0, d_ep_ds = 0;
    reg        d_a_trn = 0, d_b_trn = 0, d_ep_bias = 0, d_ep_scale = 0, d_ep_gelu = 0, d_ep_gelu_bwd = 0, d_ep_drop = 0, d_ep_resid = 0;
    reg [7:0]  d_ep_sexp = 0;
    reg        m_en = 0, m_trn = 0;
    reg [1:0]  m_ms0 = 0, m_ms1 = 0, m_ms2 = 0;
    reg [7:0]  m_nb = 0, m_mb = 0, m_kc0 = 0, m_kc1 = 0, m_kc2 = 0;
    reg [31:0] m_hits_a = 0, m_dir_a = 0, m_coef_a = 0, m_hits_b = 0, m_dir_b = 0, m_coef_b = 0, m_h_base = 0, m_gamma = 0;
    integer    exp_n = 0;

    // ------------------------------------------------------------------ DUTs
    reg          start = 1'b0, m_start = 1'b0;
    // done / m_done are one-cycle pulses at different cycles: latch each (cleared at the starts)
    reg          done_seen = 1'b0, m_done_seen = 1'b0;
    always @(posedge clk) begin
        if (start) done_seen <= 1'b0; else if (done) done_seen <= 1'b1;
        if (m_start) m_done_seen <= 1'b0; else if (m_done) m_done_seen <= 1'b1;
    end
    wire         busy, done, arr_active, m_busy, m_done;
    wire [7:0]   err;
    wire [15:0]  m_stat_tiles;
    wire         rd0_req_valid, rd0_dat_ready, rd1_req_valid, rd1_dat_ready, rd2_req_valid, rd2_dat_ready, mk_req_valid, wr_valid;
    wire [31:0]  rd0_req_addr, rd1_req_addr, rd2_req_addr, mk_req_idx, wr_addr;
    wire [6:0]   rd0_req_len, rd1_req_len, rd2_req_len;
    wire [255:0] wr_data;
    wire [7:0]   wr_mask;
    reg          rd0_req_ready, rd0_dat_valid, rd1_req_ready, rd1_dat_valid, rd2_req_ready, rd2_dat_valid, mk_req_ready, mk_rsp_valid, wr_ready;
    reg  [255:0] rd0_dat, rd1_dat, rd2_dat;
    reg  [GWD-1:0] mk_rsp_keep;
    wire         mg_en, mg_buf, mg_we, w_tile_done, w_tile_buf;
    wire [3:0]   mg_lane;
    wire [7:0]   mg_addr;
    wire [31:0]  mg_wdata, mg_rdata;
    wire [1:0]   merge_busy;
    wire         lw_wr_valid, lw_rd0_req_valid, lw_rd1_req_valid;
    gm_top #(.ROWS(ROWS), .COLS(COLS), .NSFU(NSFU)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done), .err(err), .arr_active(arr_active),
        .d_m(d_m), .d_n(d_n), .d_k0(d_k0), .d_k1(d_k1), .d_k2(d_k2), .d_nb1(d_nb1), .d_nb2(d_nb2),
        .d_a_base0(d_a_base0), .d_a_base1(d_a_base1), .d_a_base2(d_a_base2), .d_a_rs(d_a_rs), .d_a_bs1(d_a_bs1), .d_a_bs2(d_a_bs2), .d_a_trn(d_a_trn),
        .d_b_base0(d_b_base0), .d_b_base1(d_b_base1), .d_b_base2(d_b_base2), .d_b_rs(d_b_rs), .d_b_bs1(d_b_bs1), .d_b_bs2(d_b_bs2), .d_b_trn(d_b_trn),
        .d_c_base(d_c_base), .d_c_rs(d_c_rs), .d_c_bs1(d_c_bs1), .d_c_bs2(d_c_bs2),
        .d_bias_base(d_bias_base), .d_aux_base(d_aux_base), .d_o2_base(d_o2_base),
        .d_ep_bias(d_ep_bias), .d_ep_scale(d_ep_scale), .d_ep_gelu(d_ep_gelu), .d_ep_gelu_bwd(d_ep_gelu_bwd),
        .d_ep_drop(d_ep_drop), .d_ep_resid(d_ep_resid), .d_ep_sexp(d_ep_sexp), .d_ep_ds(d_ep_ds),
        .rd0_req_valid(rd0_req_valid), .rd0_req_ready(rd0_req_ready), .rd0_req_addr(rd0_req_addr), .rd0_req_len(rd0_req_len),
        .rd0_dat_valid(rd0_dat_valid), .rd0_dat_ready(rd0_dat_ready), .rd0_dat(rd0_dat),
        .rd1_req_valid(rd1_req_valid), .rd1_req_ready(rd1_req_ready), .rd1_req_addr(rd1_req_addr), .rd1_req_len(rd1_req_len),
        .rd1_dat_valid(rd1_dat_valid), .rd1_dat_ready(rd1_dat_ready), .rd1_dat(rd1_dat),
        .mk_req_valid(mk_req_valid), .mk_req_ready(mk_req_ready), .mk_req_idx(mk_req_idx),
        .mk_rsp_valid(mk_rsp_valid), .mk_rsp_keep(mk_rsp_keep),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_addr(wr_addr), .wr_data(wr_data), .wr_mask(wr_mask),
        .mg_en(mg_en), .mg_buf(mg_buf), .mg_we(mg_we), .mg_lane(mg_lane), .mg_addr(mg_addr), .mg_wdata(mg_wdata),
        .mg_rdata(mg_rdata), .w_tile_done(w_tile_done), .w_tile_buf(w_tile_buf), .merge_busy(merge_busy));
    lw_top lw (
        .clk(clk), .rst_n(rst_n), .start(1'b0), .busy(), .done(), .d_op(2'd0),
        .d_a0(32'd0), .d_a1(32'd0), .d_a2(32'd0), .d_a3(32'd0), .d_a4(32'd0), .d_a5(32'd0), .d_a6(32'd0), .d_a7(32'd0), .d_len(32'd0), .d_flag(1'b0),
        .m_start(m_start), .m_busy(m_busy), .m_done(m_done), .m_trn(m_trn), .m_nb(m_nb), .m_mb(m_mb), .m_kc0(m_kc0), .m_kc1(m_kc1), .m_kc2(m_kc2),
        .m_ms0(m_ms0), .m_ms1(m_ms1), .m_ms2(m_ms2), .m_hits_a(m_hits_a), .m_dir_a(m_dir_a), .m_coef_a(m_coef_a),
        .m_hits_b(m_hits_b), .m_dir_b(m_dir_b), .m_coef_b(m_coef_b), .m_h_base(m_h_base), .m_gamma(m_gamma), .m_stat_tiles(m_stat_tiles),
        .rd0_req_valid(lw_rd0_req_valid), .rd0_req_ready(1'b1), .rd0_req_addr(), .rd0_req_len(), .rd0_dat_valid(1'b0), .rd0_dat_ready(), .rd0_dat(256'd0),
        .rd1_req_valid(lw_rd1_req_valid), .rd1_req_ready(1'b1), .rd1_req_addr(), .rd1_req_len(), .rd1_dat_valid(1'b0), .rd1_dat_ready(), .rd1_dat(256'd0),
        .rd2_req_valid(rd2_req_valid), .rd2_req_ready(rd2_req_ready), .rd2_req_addr(rd2_req_addr), .rd2_req_len(rd2_req_len),
        .rd2_dat_valid(rd2_dat_valid), .rd2_dat_ready(rd2_dat_ready), .rd2_dat(rd2_dat),
        .wr_valid(lw_wr_valid), .wr_ready(1'b1), .wr_addr(), .wr_data(), .wr_mask(),
        .mg_en(mg_en), .mg_buf(mg_buf), .mg_we(mg_we), .mg_lane(mg_lane), .mg_addr(mg_addr), .mg_wdata(mg_wdata), .mg_rdata(mg_rdata),
        .w_tile_done(w_tile_done), .w_tile_buf(w_tile_buf), .merge_busy(merge_busy));

    // ------------------------------------------------------------------ memory model (as tb_gm_top + a third read channel)
    reg [31:0] mem [0:(1 << MEMW) - 1];
    integer lat = 24, stall = 0, maxcyc = 400000000, trace = 0;
    function stall_now; input integer dummy; begin stall_now = (stall > 0) && (($urandom % 100) < stall); end endfunction
    reg [31:0] q0_a [0:63]; reg [6:0] q0_l [0:63]; integer q0_t [0:63]; integer q0_w = 0, q0_r = 0, q0_b = 0;
    reg [31:0] q1_a [0:63]; reg [6:0] q1_l [0:63]; integer q1_t [0:63]; integer q1_w = 0, q1_r = 0, q1_b = 0;
    reg [31:0] q2_a [0:63]; reg [6:0] q2_l [0:63]; integer q2_t [0:63]; integer q2_w = 0, q2_r = 0, q2_b = 0;
    reg st0 = 0, st1 = 0, st2 = 0, stw = 0;
    integer j;
    always @* begin
        rd0_req_ready = ((q0_w - q0_r) < 32) && !st0;
        rd0_dat_valid = (q0_r < q0_w) && (cyc >= q0_t[q0_r % 64]) && !st0;
        for (j = 0; j < 8; j = j + 1) rd0_dat[j*32 +: 32] = mem[((q0_a[q0_r % 64] >> 2) + q0_b * 8 + j) & ((1 << MEMW) - 1)];
        rd1_req_ready = ((q1_w - q1_r) < 32) && !st1;
        rd1_dat_valid = (q1_r < q1_w) && (cyc >= q1_t[q1_r % 64]) && !st1;
        for (j = 0; j < 8; j = j + 1) rd1_dat[j*32 +: 32] = mem[((q1_a[q1_r % 64] >> 2) + q1_b * 8 + j) & ((1 << MEMW) - 1)];
        rd2_req_ready = ((q2_w - q2_r) < 32) && !st2;
        rd2_dat_valid = (q2_r < q2_w) && (cyc >= q2_t[q2_r % 64]) && !st2;
        for (j = 0; j < 8; j = j + 1) rd2_dat[j*32 +: 32] = mem[((q2_a[q2_r % 64] >> 2) + q2_b * 8 + j) & ((1 << MEMW) - 1)];
        wr_ready = !stw;
        mk_req_ready = 1'b1; mk_rsp_valid = 1'b0; mk_rsp_keep = {GWD{1'b0}};
    end
    integer nb0, nb1, nb2;
    always @(posedge clk) begin
        st0 <= stall_now(0); st1 <= stall_now(0); st2 <= stall_now(0); stw <= stall_now(0);
        if (rd0_req_valid && rd0_req_ready) begin q0_a[q0_w % 64] <= rd0_req_addr; q0_l[q0_w % 64] <= rd0_req_len; q0_t[q0_w % 64] <= cyc + lat; q0_w <= q0_w + 1; end
        if (rd0_dat_valid && rd0_dat_ready) begin nb0 = (q0_l[q0_r % 64] + 7) / 8; if (q0_b == nb0 - 1) begin q0_b <= 0; q0_r <= q0_r + 1; end else q0_b <= q0_b + 1; end
        if (rd1_req_valid && rd1_req_ready) begin q1_a[q1_w % 64] <= rd1_req_addr; q1_l[q1_w % 64] <= rd1_req_len; q1_t[q1_w % 64] <= cyc + lat; q1_w <= q1_w + 1; end
        if (rd1_dat_valid && rd1_dat_ready) begin nb1 = (q1_l[q1_r % 64] + 7) / 8; if (q1_b == nb1 - 1) begin q1_b <= 0; q1_r <= q1_r + 1; end else q1_b <= q1_b + 1; end
        if (rd2_req_valid && rd2_req_ready) begin q2_a[q2_w % 64] <= rd2_req_addr; q2_l[q2_w % 64] <= rd2_req_len; q2_t[q2_w % 64] <= cyc + lat; q2_w <= q2_w + 1; end
        if (rd2_dat_valid && rd2_dat_ready) begin nb2 = (q2_l[q2_r % 64] + 7) / 8; if (q2_b == nb2 - 1) begin q2_b <= 0; q2_r <= q2_r + 1; end else q2_b <= q2_b + 1; end
        if (wr_valid && wr_ready) for (j = 0; j < 8; j = j + 1) if (wr_mask[j]) mem[((wr_addr >> 2) + j) & ((1 << MEMW) - 1)] <= wr_data[j*32 +: 32];
        if (mk_req_valid || lw_wr_valid || lw_rd0_req_valid || lw_rd1_req_valid) begin $display("FAIL lw_gemm unexpected channel use"); $finish; end
    end
    // merge statistics: cycles merge_busy holds each buffer (from w_tile_done to the last write), mg accesses
    integer mg_cnt = 0, tiles = 0, hold_max = 0, hold_tot = 0, held0 = 0, held1 = 0, gate_cyc = 0;
    always @(posedge clk) begin
        if (mg_en) mg_cnt <= mg_cnt + 1;
        if (w_tile_done) tiles <= tiles + 1;
        if (merge_busy[0]) held0 <= held0 + 1; else if (held0 != 0) begin hold_tot = hold_tot + held0; if (held0 > hold_max) hold_max = held0; held0 <= 0; end
        if (merge_busy[1]) held1 <= held1 + 1; else if (held1 != 0) begin hold_tot = hold_tot + held1; if (held1 > hold_max) hold_max = held1; held1 <= 0; end
        // cycles the controller's next block is ready except for the merge (dut.u_ctrl.tiles_ok gated by merge_busy)
        if (dut.u_ctrl.x_ld[dut.u_ctrl.ct_buf] & dut.u_ctrl.w_ld[dut.u_ctrl.ct_buf] & merge_busy[dut.u_ctrl.ct_buf]) gate_cyc <= gate_cyc + 1;
        if (trace && w_tile_done) $display("T%0d WTILE buf %0d", cyc, w_tile_buf);
        if (trace > 1 && mg_en) $display("T%0d MG %s buf %0d lane %0d addr %0d d %h", cyc, mg_we ? "wr" : "rd", mg_buf, mg_lane, mg_addr, mg_wdata);
    end

    // ------------------------------------------------------------------ run
    reg [1023:0] fdesc, fmem, fexp;
    reg [255:0] key;
    reg [31:0] val;
    integer fd, rc, n_bad = 0, n_chk = 0, t0, t1, act = 0;
    reg [31:0] ea, ev;
    always @(posedge clk) if (busy && arr_active) act <= act + 1;
    task set_key;
        begin
            case (key)
                "m": d_m = val[15:0];  "n": d_n = val[15:0];  "k0": d_k0 = val[15:0];  "k1": d_k1 = val[15:0];
                "k2": d_k2 = val[15:0];  "nb1": d_nb1 = val[15:0];  "nb2": d_nb2 = val[15:0];
                "a_base0": d_a_base0 = val;  "a_base1": d_a_base1 = val;  "a_base2": d_a_base2 = val;
                "a_rs": d_a_rs = val;  "a_bs1": d_a_bs1 = val;  "a_bs2": d_a_bs2 = val;  "a_trn": d_a_trn = val[0];
                "b_base0": d_b_base0 = val;  "b_base1": d_b_base1 = val;  "b_base2": d_b_base2 = val;
                "b_rs": d_b_rs = val;  "b_bs1": d_b_bs1 = val;  "b_bs2": d_b_bs2 = val;  "b_trn": d_b_trn = val[0];
                "c_base": d_c_base = val;  "c_rs": d_c_rs = val;  "c_bs1": d_c_bs1 = val;  "c_bs2": d_c_bs2 = val;
                "bias_base": d_bias_base = val;  "ep_bias": d_ep_bias = val[0];
                "m_en": m_en = val[0];  "m_trn": m_trn = val[0];  "m_nb": m_nb = val[7:0];  "m_mb": m_mb = val[7:0];
                "m_kc0": m_kc0 = val[7:0];  "m_kc1": m_kc1 = val[7:0];  "m_kc2": m_kc2 = val[7:0];
                "m_ms0": m_ms0 = val[1:0];  "m_ms1": m_ms1 = val[1:0];  "m_ms2": m_ms2 = val[1:0];
                "m_hits_a": m_hits_a = val;  "m_dir_a": m_dir_a = val;  "m_coef_a": m_coef_a = val;
                "m_hits_b": m_hits_b = val;  "m_dir_b": m_dir_b = val;  "m_coef_b": m_coef_b = val;
                "m_h_base": m_h_base = val;  "m_gamma": m_gamma = val;
                "mode": ;  "exp_n": exp_n = val;
                default: begin $display("FAIL lw_gemm unknown descriptor key %0s", key); $finish; end
            endcase
        end
    endtask

    initial begin
        if (!$value$plusargs("desc=%s", fdesc) || !$value$plusargs("mem=%s", fmem) || !$value$plusargs("exp=%s", fexp)) begin
            $display("FAIL lw_gemm need +desc +mem +exp"); $finish;
        end
        if ($value$plusargs("lat=%d", lat)) ;
        if ($value$plusargs("stall=%d", stall)) ;
        if ($value$plusargs("maxcyc=%d", maxcyc)) ;
        if ($value$plusargs("trace=%d", trace)) ;
        fd = $fopen(fdesc, "r");
        if (fd == 0) begin $display("FAIL lw_gemm cannot open desc"); $finish; end
        while (!$feof(fd)) begin rc = $fscanf(fd, "%s %h\n", key, val); if (rc == 2) set_key; end
        $fclose(fd);
        $readmemh(fmem, mem);
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);
        t0 = cyc;
        if (m_en) begin m_start = 1'b1; @(negedge clk); m_start = 1'b0; end     // merge descriptor first (events are latched while it is busy)
        start = 1'b1; @(negedge clk); start = 1'b0;
        while (!(done | done_seen) || (m_en && !(m_done | m_done_seen))) begin
            @(negedge clk);
            if (cyc - t0 > maxcyc) begin $display("FAIL lw_gemm watchdog (%0d cycles) done %b m_done %b err %b", cyc - t0, done, m_done, err); $finish; end
        end
        t1 = cyc;
        repeat (4) @(negedge clk);
        fd = $fopen(fexp, "r");
        if (fd == 0) begin $display("FAIL lw_gemm cannot open exp"); $finish; end
        rc = $fscanf(fd, "%h %h\n", ea, ev);
        while (rc == 2) begin
            n_chk = n_chk + 1;
            if (mem[ea & ((1 << MEMW) - 1)] !== ev) begin
                if (n_bad < 10) $display("MISMATCH word %h got %h want %h", ea, mem[ea & ((1 << MEMW) - 1)], ev);
                n_bad = n_bad + 1;
            end
            rc = $fscanf(fd, "%h %h\n", ea, ev);
        end
        $fclose(fd);
        if (err != 8'd0) begin $display("ERROR flags %b", err); n_bad = n_bad + 1; end
        if (n_chk != exp_n) begin $display("expected %0d words, read %0d", exp_n, n_chk); n_bad = n_bad + 1; end
        $display("CYCLES %0d ACTIVE %0d", t1 - t0, act);
        $display("MERGE tiles %0d stat_tiles %0d mg_accesses %0d hold_max %0d hold_total %0d gate_cycles %0d", tiles, m_stat_tiles, mg_cnt, hold_max, hold_tot, gate_cyc);
        if (n_chk == 0) $display("FAIL lw_gemm 0/0 no expected words");
        else if (n_bad == 0) $display("PASS lw_gemm %0d vectors", n_chk);
        else $display("FAIL lw_gemm %0d/%0d mismatches", n_bad, n_chk);
        $finish;
    end
endmodule
