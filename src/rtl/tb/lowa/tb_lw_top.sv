// tb_lw_top: LoWA unit testbench (P5).  Plusargs: +desc=<file> ("key hex" lines) +mem=<file> ($readmemh
// image) +exp=<file> (+lat=<n> +stall=<pct> +maxcyc=<n>).  Memory model: three in-order read channels
// with a programmable latency and random stalls, one write channel (as tb_gm_top).
// mode 0: one main descriptor (SG / SCAL / ADAMW); exp = "wordaddr value" lines, compared after done.
// mode 1: merge descriptor.  The testbench models the GEMM engine's W buffers (2 x 16 lanes x 256
// words, the P3 layout: cell (n, k) in lane (n/4 + k + 4 (n%4)) mod 16, address {k, n%4}) and its
// loader: it writes W0 block (nb, kc) (ROW: rows 64nb.., cols 64kc..; TRN: rows 64kc.., cols 64nb..)
// into the next buffer in loader order, pulses w_tile_done, waits until merge_busy[buf] drops and
// compares all 4096 cells with the expected merged block (exp = "idx value", idx = tile*4096 + n*64 + k).
// Golden values come only from L1 (tb/lowa/gen_lowa.py).  Prints one PASS / FAIL line.
module tb_lw_top;
    parameter MEMW = 22;
    parameter NL = 4, NSQ = 2, NRC = 1;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    integer cyc = 0;
    always @(posedge clk) cyc <= cyc + 1;
    reg start = 0, m_start = 0, d_flag = 0, m_trn = 0;
    reg [1:0] d_op = 0, m_ms0 = 0, m_ms1 = 0, m_ms2 = 0;
    reg [31:0] d_a0 = 0, d_a1 = 0, d_a2 = 0, d_a3 = 0, d_a4 = 0, d_a5 = 0, d_a6 = 0, d_a7 = 0, d_len = 0;
    reg [7:0] m_nb = 0, m_mb = 0, m_kc0 = 0, m_kc1 = 0, m_kc2 = 0;
    reg [31:0] m_hits_a = 0, m_dir_a = 0, m_coef_a = 0, m_hits_b = 0, m_dir_b = 0, m_coef_b = 0, m_h_base = 0, m_gamma = 0;
    wire busy, done, m_busy, m_done;
    wire [15:0] m_stat_tiles;
    wire rd0_req_valid, rd1_req_valid, rd2_req_valid, rd0_dat_ready, rd1_dat_ready, rd2_dat_ready, wr_valid;
    wire [31:0] rd0_req_addr, rd1_req_addr, rd2_req_addr, wr_addr;
    wire [6:0] rd0_req_len, rd1_req_len, rd2_req_len;
    wire [255:0] wr_data;
    wire [7:0] wr_mask;
`ifndef MM_DRAM
    reg rd0_req_ready, rd1_req_ready, rd2_req_ready, rd0_dat_valid, rd1_dat_valid, rd2_dat_valid, wr_ready;
    reg [255:0] rd0_dat, rd1_dat, rd2_dat;
`else
    wire rd0_req_ready, rd1_req_ready, rd2_req_ready, rd0_dat_valid, rd1_dat_valid, rd2_dat_valid, wr_ready;
    wire [255:0] rd0_dat, rd1_dat, rd2_dat;
`endif
    wire mg_en, mg_buf, mg_we;
    wire [3:0] mg_lane;
    wire [7:0] mg_addr;
    wire [31:0] mg_wdata;
    reg [31:0] mg_rdata;
    reg w_tile_done = 0, w_tile_buf = 0;
    wire [1:0] merge_busy;
    lw_top #(.NL(NL), .NSQ(NSQ), .NRC(NRC)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done), .d_op(d_op),
        .d_a0(d_a0), .d_a1(d_a1), .d_a2(d_a2), .d_a3(d_a3), .d_a4(d_a4), .d_a5(d_a5), .d_a6(d_a6), .d_a7(d_a7), .d_len(d_len), .d_flag(d_flag),
        .m_start(m_start), .m_busy(m_busy), .m_done(m_done), .m_trn(m_trn), .m_nb(m_nb), .m_mb(m_mb), .m_kc0(m_kc0), .m_kc1(m_kc1), .m_kc2(m_kc2),
        .m_ms0(m_ms0), .m_ms1(m_ms1), .m_ms2(m_ms2), .m_hits_a(m_hits_a), .m_dir_a(m_dir_a), .m_coef_a(m_coef_a),
        .m_hits_b(m_hits_b), .m_dir_b(m_dir_b), .m_coef_b(m_coef_b), .m_h_base(m_h_base), .m_gamma(m_gamma), .m_stat_tiles(m_stat_tiles),
        .rd0_req_valid(rd0_req_valid), .rd0_req_ready(rd0_req_ready), .rd0_req_addr(rd0_req_addr), .rd0_req_len(rd0_req_len),
        .rd0_dat_valid(rd0_dat_valid), .rd0_dat_ready(rd0_dat_ready), .rd0_dat(rd0_dat),
        .rd1_req_valid(rd1_req_valid), .rd1_req_ready(rd1_req_ready), .rd1_req_addr(rd1_req_addr), .rd1_req_len(rd1_req_len),
        .rd1_dat_valid(rd1_dat_valid), .rd1_dat_ready(rd1_dat_ready), .rd1_dat(rd1_dat),
        .rd2_req_valid(rd2_req_valid), .rd2_req_ready(rd2_req_ready), .rd2_req_addr(rd2_req_addr), .rd2_req_len(rd2_req_len),
        .rd2_dat_valid(rd2_dat_valid), .rd2_dat_ready(rd2_dat_ready), .rd2_dat(rd2_dat),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_addr(wr_addr), .wr_data(wr_data), .wr_mask(wr_mask),
        .mg_en(mg_en), .mg_buf(mg_buf), .mg_we(mg_we), .mg_lane(mg_lane), .mg_addr(mg_addr), .mg_wdata(mg_wdata), .mg_rdata(mg_rdata),
        .w_tile_done(w_tile_done), .w_tile_buf(w_tile_buf), .merge_busy(merge_busy));

    integer trace = 0;
    always @(posedge clk) if (trace) begin
        if (dut.u_mg.p_issue) $display("T%0d MG issue hit %h cell %h slot %0d", cyc, dut.u_mg.hf_out_d[24:0], dut.u_mg.h_cell, dut.u_mg.hf_out_d[25]);
        if (dut.u_mg.take_mark) $display("T%0d MG marker", cyc);
        if (trace > 2 && dut.u_mg.rw_valid && dut.u_mg.st == 3) $display("T%0d MG rw idx %0d d %h rem %0d beat_idx %0d", cyc, dut.u_mg.rw_idx, dut.u_mg.rw_d, dut.u_mg.u_rd.rem_dat, dut.u_mg.u_rd.u_un.idx);
        if (trace > 2 && dut.u_mg.dir_en && dut.u_mg.dir_we) $display("T%0d MG dirw addr %0d d %h", cyc, dut.u_mg.dir_addr, dut.u_mg.u_dir.d);
        if (dut.u_mg.ev_pop) $display("T%0d MG tile done buf %0d", cyc, dut.u_mg.ev_buf);
        if (dut.u_mg.rq_go) $display("T%0d MG rdseq go base %h n %0d st %0d", cyc, dut.u_mg.rq_base, dut.u_mg.rq_n, dut.u_mg.st);
        if (dut.u_mg.st == 12) $display("T%0d MG dir_q %h blk %0d %0d", cyc, dut.u_mg.dir_q, dut.u_mg.blk_a, dut.u_mg.blk_b);
        if (dut.u_sg.disp_start) $display("T%0d SG disp j %0d slot %0d ci %0d cj %0d", cyc, dut.u_sg.j, dut.u_sg.rr, dut.u_sg.ci_n, dut.u_sg.cj_n);
        if (dut.u_sg.w_s2_v && dut.u_sg.w_s2 == 6) $display("T%0d SG fill slot %0d yw6 %h xw6 %h", cyc, dut.u_sg.tgt_s2, dut.u_sg.dy_q, dut.u_sg.x_q);
        if (trace > 1 && dut.u_sg.g_lane[0].f_issue) $display("T%0d SG L0 fma tf %0d h %h p %h c %h", cyc, dut.u_sg.g_lane[0].tf, dut.u_sg.g_lane[0].h_q, dut.u_sg.g_lane[0].m_y, dut.u_sg.g_lane[0].c_in);
        if (dut.u_sg.res_any) $display("T%0d SG acc write j %0d val %h (take %b)", cyc, dut.u_sg.res_j[0], dut.u_sg.res_d[0], dut.u_sg.res_take);
        if (trace > 1 && dut.u_sg.init_s2) $display("T%0d SG acc init slot %0d q %h", cyc, dut.u_sg.tgt, dut.u_sg.ac_q);
    end
    // ------------------------------------------------------------------ memory model
    reg [31:0] mem [0:(1 << MEMW) - 1];
    integer lat = 24, stall = 0, maxcyc = 200000000;
    function stall_now; input integer dummy; begin stall_now = (stall > 0) && (($urandom % 100) < stall); end endfunction
`ifndef MM_DRAM
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
    end
`else
    integer j;
    `include "mm_attach_lw.svh"
`endif
    // ------------------------------------------------------------------ W buffer model (merge mode)
    reg [31:0] wbuf [0:2*16*256-1];
    integer mg_cnt = 0;
    always @(posedge clk) begin
        if (mg_en) begin
            if (mg_we) wbuf[{mg_buf, mg_lane, mg_addr}] <= mg_wdata;
            else mg_rdata <= wbuf[{mg_buf, mg_lane, mg_addr}];
            mg_cnt <= mg_cnt + 1;
        end
    end
    function [12:0] wcell;
        input b_; input [31:0] n_, k_;
        reg [31:0] ln;
        begin
            ln = (n_ / 4 + k_ + 4 * (n_ % 4)) % 16;
            wcell = {b_, ln[3:0], k_[5:0], n_[1:0]};
        end
    endfunction
    reg bufsel = 0;

    // ------------------------------------------------------------------ run
    reg [1023:0] fdesc, fmem, fexp;
    reg [255:0] key;
    reg [31:0] val;
    integer fd, rc, n_bad = 0, n_chk = 0, t0, t1, i, mode = 0, exp_n = 0, w0_base = 0, w0_rs = 0, nb_i, mb_i, seg, kc_i, tile = 0, n, k, o, c, kcs, stall_cyc = 0, max_tile = 0, tile_cyc, t2;
    reg [31:0] ea, ev;
    task set_key;
        begin
            case (key)
                "op": d_op = val[1:0];  "a0": d_a0 = val;  "a1": d_a1 = val;  "a2": d_a2 = val;  "a3": d_a3 = val;
                "a4": d_a4 = val;  "a5": d_a5 = val;  "a6": d_a6 = val;  "a7": d_a7 = val;  "len": d_len = val;  "flag": d_flag = val[0];
                "mode": mode = val;  "exp_n": exp_n = val;
                "m_trn": m_trn = val[0];  "m_nb": m_nb = val[7:0];  "m_mb": m_mb = val[7:0];  "m_kc0": m_kc0 = val[7:0];  "m_kc1": m_kc1 = val[7:0];
                "m_kc2": m_kc2 = val[7:0];  "m_ms0": m_ms0 = val[1:0];  "m_ms1": m_ms1 = val[1:0];  "m_ms2": m_ms2 = val[1:0];
                "m_hits_a": m_hits_a = val;  "m_dir_a": m_dir_a = val;  "m_coef_a": m_coef_a = val;  "m_hits_b": m_hits_b = val;
                "m_dir_b": m_dir_b = val;  "m_coef_b": m_coef_b = val;  "m_h_base": m_h_base = val;  "m_gamma": m_gamma = val;
                "w0_base": w0_base = val;  "w0_rs": w0_rs = val;
                default: begin $display("FAIL lw_top unknown descriptor key %0s", key); $finish; end
            endcase
        end
    endtask
    task do_tile;                           // write W0 block (nb_i, kc_i of segment seg) into the next buffer, merge, compare
        begin
            for (n = 0; n < 64; n = n + 1) for (k = 0; k < 64; k = k + 1) begin
                if (m_trn) begin o = 64 * (kc_i + kcs) + k; c = 64 * nb_i + n; end
                else begin o = 64 * nb_i + n; c = 64 * (kc_i + kcs) + k; end
                wbuf[wcell(bufsel, n, k)] = mem[(w0_base >> 2) + o * (w0_rs >> 2) + c];
            end
            @(negedge clk); w_tile_done = 1; w_tile_buf = bufsel; @(negedge clk); w_tile_done = 0;
            t2 = cyc; tile_cyc = 0;
            @(negedge clk);
            while (merge_busy[bufsel]) begin @(negedge clk); if (cyc - t2 > 100000) begin $display("FAIL lw_top merge tile %0d hangs", tile); $finish; end end
            tile_cyc = cyc - t2;
            if (tile_cyc > max_tile) max_tile = tile_cyc;
            if (tile_cyc > 512) stall_cyc = stall_cyc + tile_cyc - 512;
            for (n = 0; n < 64; n = n + 1) for (k = 0; k < 64; k = k + 1) begin
                rc = $fscanf(fd, "%h %h\n", ea, ev);
                n_chk = n_chk + 1;
                if (rc != 2 || ea != tile * 4096 + n * 64 + k) begin $display("FAIL lw_top exp file out of order at tile %0d n %0d k %0d", tile, n, k); $finish; end
                if (wbuf[wcell(bufsel, n, k)] !== ev) begin
                    if (n_bad < 10) $display("MISMATCH tile %0d n %0d k %0d got %h want %h", tile, n, k, wbuf[wcell(bufsel, n, k)], ev);
                    n_bad = n_bad + 1;
                end
            end
            tile = tile + 1; bufsel = ~bufsel;
        end
    endtask
    initial begin
        if (!$value$plusargs("desc=%s", fdesc) || !$value$plusargs("mem=%s", fmem) || !$value$plusargs("exp=%s", fexp)) begin
            $display("FAIL lw_top need +desc +mem +exp"); $finish;
        end
        if ($value$plusargs("lat=%d", lat)) ;
        if ($value$plusargs("stall=%d", stall)) ;
        if ($value$plusargs("maxcyc=%d", maxcyc)) ;
        if ($value$plusargs("trace=%d", trace)) ;
        fd = $fopen(fdesc, "r");
        if (fd == 0) begin $display("FAIL lw_top cannot open desc"); $finish; end
        while (!$feof(fd)) begin rc = $fscanf(fd, "%s %h\n", key, val); if (rc == 2) set_key; end
        $fclose(fd);
`ifndef MM_DRAM
        $readmemh(fmem, mem);
`else
        mm_load;
`endif
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (4) @(negedge clk);
        fd = $fopen(fexp, "r");
        if (fd == 0) begin $display("FAIL lw_top cannot open exp"); $finish; end
        t0 = cyc;
        if (mode == 0) begin
            start = 1'b1; @(negedge clk); start = 1'b0;
            while (!done) begin @(negedge clk); if (cyc - t0 > maxcyc) begin $display("FAIL lw_top watchdog (%0d cycles)", cyc - t0); $finish; end end
            t1 = cyc;
`ifdef MM_DRAM
            mm_drain;
`endif
            repeat (4) @(negedge clk);
            rc = $fscanf(fd, "%h %h\n", ea, ev);
            while (rc == 2) begin
                n_chk = n_chk + 1;
`ifndef MM_DRAM
                if (mem[ea & ((1 << MEMW) - 1)] !== ev) begin
                    if (n_bad < 10) $display("MISMATCH word %h got %h want %h", ea, mem[ea & ((1 << MEMW) - 1)], ev);
`else
                if (dram.peek(ea) !== ev) begin
                    if (n_bad < 10) $display("MISMATCH word %h got %h want %h", ea, dram.peek(ea), ev);
`endif
                    n_bad = n_bad + 1;
                end
                rc = $fscanf(fd, "%h %h\n", ea, ev);
            end
        end else begin
            m_start = 1'b1; @(negedge clk); m_start = 1'b0;
            // loader order: nb, mb, seg, kc  (kcs = first kc of the segment within the concatenated K)
            for (nb_i = 0; nb_i < m_nb; nb_i = nb_i + 1)
                for (mb_i = 0; mb_i < m_mb; mb_i = mb_i + 1) begin
                    kcs = 0;
                    for (seg = 0; seg < 3; seg = seg + 1) begin
                        for (kc_i = 0; kc_i < ((seg == 0) ? m_kc0 : (seg == 1) ? m_kc1 : m_kc2); kc_i = kc_i + 1) do_tile;
                        kcs = kcs + ((seg == 0) ? m_kc0 : (seg == 1) ? m_kc1 : m_kc2);
                    end
                end
            while (!m_done) begin @(negedge clk); if (cyc - t0 > maxcyc) begin $display("FAIL lw_top merge watchdog"); $finish; end end
            t1 = cyc;
            $display("MERGE tiles %0d mg_accesses %0d max_tile_cycles %0d stall_over_512 %0d", m_stat_tiles, mg_cnt, max_tile, stall_cyc);
        end
        $fclose(fd);
        if (n_chk != exp_n) begin $display("expected %0d words, read %0d", exp_n, n_chk); n_bad = n_bad + 1; end
        $display("CYCLES %0d", t1 - t0);
        if (n_chk == 0) $display("FAIL lw_top 0/0 no expected words");
        else if (n_bad == 0) $display("PASS lw_top %0d vectors", n_chk);
        else $display("FAIL lw_top %0d/%0d mismatches", n_bad, n_chk);
        $finish;
    end
endmodule
