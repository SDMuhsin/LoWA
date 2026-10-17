`timescale 1ns/1ps
// tb_vu_top: runs a sequence of vector descriptors on vu_top against a memory model and compares
// every expected word with the L1 golden (tb/vec/gen_vec.py), bit for bit.
//   +desc=<file>  "key hex" lines; a "run" line starts the unit with the fields set so far
//                 (fields persist between runs); exp_n / mask_n / elements / model are bookkeeping
//   +mem=<file>   $readmemh word image     +exp=<file> "wordaddr value" expected words
//   +mask=<file>  keep bits (one per line, indexed by the mask element index)
//   +lat=<n>      memory latency (default 24)   +stall=<pct> random stalls   +maxcyc=<n> watchdog per run
// Memory model: word-addressed, two in-order read channels (request queue, LAT cycles, then one
// 256-bit beat per cycle), one write channel with a word mask, a keep-bit responder (MKW bits).
// Prints "CYCLES <total> RUNS <n> ELEMENTS <e> MODEL <m>" and one final line:
//   PASS vu_top <n> vectors | FAIL vu_top <m>/<n> mismatches
module tb_vu_top;
    parameter V = 16;
    parameter VW = 4;
    parameter NS = 8;
    parameter SW = 3;
    parameter RB = 256;
    parameter HBL = 2;                  // vu_loader rd0 beats per row request (P6b)
    parameter NOB = 2;                  // lane output registers (P6b sweeps)
    parameter WK = 1;                   // vu_writer groups per row visit (P6b)
    parameter RBW = 8;
    parameter MEMW = 22;                    // log2 words of the memory model
    parameter MASKW = 24;                   // log2 entries of the keep-bit array (Icarus: small)
    localparam MKW = 16;
    reg clk = 1'b0, rst_n = 1'b0, start = 1'b0;
    always #20 clk = ~clk;
    integer cyc = 0;
    always @(posedge clk) cyc <= cyc + 1;
    wire busy, done;
    reg  [4:0]  d_vop = 0;
    reg  [15:0] d_rows = 0, d_cols = 0, d_rows_per_b = 16'hFFF0, d_emb_l = 16;
    reg         d_st_en = 0, d_x_reset = 0, d_g_reset = 0;
    reg  [31:0] d_a_base = 0, d_a_rs = 0, d_b_base = 0, d_b_rs = 0, d_c_base = 0, d_c_rs = 0;
    reg  [31:0] d_cv0_base = 0, d_cv1_base = 0, d_cv2_base = 0, d_km_rs = 0;
    reg  [31:0] d_st0_base = 0, d_st1_base = 0, d_o2_base = 0, d_ids_base = 0, d_mk_base = 0, d_mk_rs = 0;
    reg  [31:0] d_c0 = 0, d_c1 = 0, d_c2 = 0, d_c3 = 0, d_ds = 32'h3F8E38E4;
    wire        rd0_req_valid, rd1_req_valid, mk_req_valid, wr_valid, rd0_dat_ready, rd1_dat_ready;
    wire [31:0] rd0_req_addr, rd1_req_addr, mk_req_idx, wr_addr;
    wire [6:0]  rd0_req_len, rd1_req_len;
    wire [255:0] wr_data;
    wire [7:0]  wr_mask;
`ifndef MM_DRAM
    reg         rd0_req_ready, rd0_dat_valid, rd1_req_ready, rd1_dat_valid, mk_req_ready, mk_rsp_valid, wr_ready;
    reg  [255:0] rd0_dat, rd1_dat;
    reg  [MKW-1:0] mk_rsp_keep;
`else
    wire        rd0_req_ready, rd0_dat_valid, rd1_req_ready, rd1_dat_valid, mk_req_ready, mk_rsp_valid, wr_ready;
    wire [255:0] rd0_dat, rd1_dat;
    wire [MKW-1:0] mk_rsp_keep;
`endif
    reg  [31:0] mem [0:(1 << MEMW) - 1];
    reg         mask [0:(1 << MASKW) - 1];
    reg  [31:0] exp_n = 0, mask_n = 0, elements = 0, model_cyc = 0;
    vu_top #(.V(V), .VW(VW), .NS(NS), .SW(SW), .RB(RB), .RBW(RBW), .MKW(MKW), .HBL(HBL), .NOB(NOB), .WK(WK)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done),
        .d_vop(d_vop), .d_rows(d_rows), .d_cols(d_cols), .d_rows_per_b(d_rows_per_b), .d_emb_l(d_emb_l),
        .d_st_en(d_st_en), .d_x_reset(d_x_reset), .d_g_reset(d_g_reset),
        .d_a_base(d_a_base), .d_a_rs(d_a_rs), .d_b_base(d_b_base), .d_b_rs(d_b_rs), .d_c_base(d_c_base), .d_c_rs(d_c_rs),
        .d_cv0_base(d_cv0_base), .d_cv1_base(d_cv1_base), .d_cv2_base(d_cv2_base), .d_km_rs(d_km_rs),
        .d_st0_base(d_st0_base), .d_st1_base(d_st1_base), .d_o2_base(d_o2_base), .d_ids_base(d_ids_base),
        .d_mk_base(d_mk_base), .d_mk_rs(d_mk_rs), .d_c0(d_c0), .d_c1(d_c1), .d_c2(d_c2), .d_c3(d_c3), .d_ds(d_ds),
        .rd0_req_valid(rd0_req_valid), .rd0_req_ready(rd0_req_ready), .rd0_req_addr(rd0_req_addr), .rd0_req_len(rd0_req_len),
        .rd0_dat_valid(rd0_dat_valid), .rd0_dat_ready(rd0_dat_ready), .rd0_dat(rd0_dat),
        .rd1_req_valid(rd1_req_valid), .rd1_req_ready(rd1_req_ready), .rd1_req_addr(rd1_req_addr), .rd1_req_len(rd1_req_len),
        .rd1_dat_valid(rd1_dat_valid), .rd1_dat_ready(rd1_dat_ready), .rd1_dat(rd1_dat),
        .mk_req_valid(mk_req_valid), .mk_req_ready(mk_req_ready), .mk_req_idx(mk_req_idx),
        .mk_rsp_valid(mk_rsp_valid), .mk_rsp_keep(mk_rsp_keep),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_addr(wr_addr), .wr_data(wr_data), .wr_mask(wr_mask));

    // ------------------------------------------------------------------ memory model
    integer lat = 24, stall = 0, maxcyc = 100000000, trace = 0;
    function stall_now;
        input integer dummy;
        begin
            stall_now = (stall > 0) && (($urandom % 100) < stall);
        end
    endfunction
`ifndef MM_DRAM
    reg [31:0] qm_i [0:63]; integer qm_t [0:63]; integer qm_w = 0, qm_r = 0;
`ifdef MK_GEN
    localparam MKG_W = MKW;
`include "mk_gen_tb.svh"
`endif
    reg [31:0] q0_a [0:63]; reg [6:0] q0_l [0:63]; integer q0_t [0:63]; integer q0_w = 0, q0_r = 0, q0_b = 0;
    reg [31:0] q1_a [0:63]; reg [6:0] q1_l [0:63]; integer q1_t [0:63]; integer q1_w = 0, q1_r = 0, q1_b = 0;
    reg st0 = 0, st1 = 0, stw = 0, stm = 0;
    integer j;
    always @* begin
        rd0_req_ready = ((q0_w - q0_r) < 32) && !st0;
        rd0_dat_valid = (q0_r < q0_w) && (cyc >= q0_t[q0_r % 64]) && !st0;
        for (j = 0; j < 8; j = j + 1) rd0_dat[j*32 +: 32] = mem[((q0_a[q0_r % 64] >> 2) + q0_b * 8 + j) & ((1 << MEMW) - 1)];
        rd1_req_ready = ((q1_w - q1_r) < 32) && !st1;
        rd1_dat_valid = (q1_r < q1_w) && (cyc >= q1_t[q1_r % 64]) && !st1;
        for (j = 0; j < 8; j = j + 1) rd1_dat[j*32 +: 32] = mem[((q1_a[q1_r % 64] >> 2) + q1_b * 8 + j) & ((1 << MEMW) - 1)];
        wr_ready = !stw;
`ifdef MK_GEN
        mk_req_ready = mg_ready; mk_rsp_valid = mg_valid; mk_rsp_keep = mg_keep;
`else
        mk_req_ready = ((qm_w - qm_r) < 32) && !stm;
        mk_rsp_valid = (qm_r < qm_w) && (cyc >= qm_t[qm_r % 64]);
        for (j = 0; j < MKW; j = j + 1)
            mk_rsp_keep[j] = ((qm_i[qm_r % 64] + j) < mask_n) ? mask[qm_i[qm_r % 64] + j] : 1'b0;
`endif
    end
    integer nbeats0, nbeats1;
    always @(posedge clk) begin
        st0 <= stall_now(0); st1 <= stall_now(0); stw <= stall_now(0); stm <= stall_now(0);
        if (rd0_req_valid && rd0_req_ready) begin
            q0_a[q0_w % 64] <= rd0_req_addr; q0_l[q0_w % 64] <= rd0_req_len; q0_t[q0_w % 64] <= cyc + lat; q0_w <= q0_w + 1;
        end
        if (rd0_dat_valid && rd0_dat_ready) begin
            nbeats0 = (q0_l[q0_r % 64] + 7) / 8;
            if (q0_b == nbeats0 - 1) begin q0_b <= 0; q0_r <= q0_r + 1; end
            else q0_b <= q0_b + 1;
        end
        if (rd1_req_valid && rd1_req_ready) begin
            q1_a[q1_w % 64] <= rd1_req_addr; q1_l[q1_w % 64] <= rd1_req_len; q1_t[q1_w % 64] <= cyc + lat; q1_w <= q1_w + 1;
        end
        if (rd1_dat_valid && rd1_dat_ready) begin
            nbeats1 = (q1_l[q1_r % 64] + 7) / 8;
            if (q1_b == nbeats1 - 1) begin q1_b <= 0; q1_r <= q1_r + 1; end
            else q1_b <= q1_b + 1;
        end
        if (mk_req_valid && mk_req_ready) begin
            qm_i[qm_w % 64] <= mk_req_idx; qm_t[qm_w % 64] <= cyc + lat; qm_w <= qm_w + 1;
        end
        if (mk_rsp_valid) qm_r <= qm_r + 1;
        if (wr_valid && wr_ready) begin
            for (j = 0; j < 8; j = j + 1)
                if (wr_mask[j]) mem[((wr_addr >> 2) + j) & ((1 << MEMW) - 1)] <= wr_data[j*32 +: 32];
        end
    end
`else
    // P6a integration: the engine's channels on the real memory system (hw/mem/mm_top) + DRAM model
    integer q0_w = 0, q0_r = 0;
    `include "mm_attach_gm.svh"
`endif
    reg [3:0] cs_q;
    always @(posedge clk) cs_q <= dut.u_seq.cs;
    always @(posedge clk) if (trace) begin
        if (wr_valid && wr_ready) $display("T%0d WR addr %h mask %b w0 %h", cyc, wr_addr, wr_mask, wr_data[31:0]);
        if (rd0_req_valid && rd0_req_ready) $display("T%0d RD0 addr %h len %0d", cyc, rd0_req_addr, rd0_req_len);
        if (rd1_req_valid && rd1_req_ready) $display("T%0d RD1 addr %h len %0d", cyc, rd1_req_addr, rd1_req_len);
        if (mk_req_valid && mk_req_ready) $display("T%0d MKREQ idx %0d", cyc, mk_req_idx);
        if (mk_rsp_valid) $display("T%0d MKRSP keep %b", cyc, mk_rsp_keep);
        if (trace > 1 && dut.u_seq.mk_pop) $display("T%0d MKPOP bits %b", cyc, dut.keep_bit);
        if (trace > 1 && dut.u_seq.iss_v) $display("T%0d ISS addr %0d slot %0d abuf %0d", cyc, dut.u_seq.iss_addr, dut.u_seq.iss_slot, dut.u_seq.iss_abuf);
        if (trace > 1 && dut.g_lane[0].u_lane.iss_v_q) $display("T%0d CAP km_d %b cv0_d %h bit %0d kmode %b par %b keep0 %b", cyc, dut.km_d, dut.cv0_d, dut.km_bit_q, dut.km_mode_q, dut.cv_par_q, dut.g_lane[0].u_lane.iss_keep_q);
        if (trace > 2 && dut.g_lane[0].u_lane.max_iss) $display("T%0d MAX a %h x %h maxr %h ov %b", cyc, dut.g_lane[0].u_lane.max_a, dut.g_lane[0].u_lane.opa, dut.g_lane[0].u_lane.maxr, dut.g_lane[0].u_lane.cmp_ov);
        if (trace > 2 && dut.g_lane[0].u_lane.cmp_ov) $display("T%0d CMPOV mx %h maxr %h", cyc, dut.g_lane[0].u_lane.cmp_mx, dut.g_lane[0].u_lane.maxr);
        if (trace > 2 && dut.g_lane[0].u_lane.wb_v2) $display("T%0d WB slot %0d dst %0d val %h", cyc, dut.g_lane[0].u_lane.wb_s2, dut.g_lane[0].u_lane.wb_d2, dut.g_lane[0].u_lane.wb_valz);
        if (trace > 1 && dut.u_seq.ex_v) $display("T%0d EX slot %0d uop %h", cyc, dut.u_seq.ex_slot, dut.u_seq.ex_uop);
        if (dut.u_seq.ld_start) $display("T%0d LD a %0d b %0d abuf %0d row0 %0d nrows %0d seg %0d len %0d cv %b sreg %b km %0d", cyc,
            dut.u_seq.ld_a, dut.u_seq.ld_b, dut.u_seq.ld_abuf, dut.u_seq.ld_row0, dut.u_seq.ld_nrows, dut.u_seq.ld_seg_off, dut.u_seq.ld_seg_len, dut.u_seq.ld_cv, dut.u_seq.ld_sreg, dut.u_seq.ld_km);
        if (dut.u_seq.cs != cs_q) $display("T%0d CS %0d cp %0d phase %h", cyc, dut.u_seq.cs, dut.u_seq.cp, dut.u_seq.q_phase[dut.u_seq.cp]);
    end

    // ------------------------------------------------------------------ run
    reg [1023:0] fdesc, fmem, fexp, fmask;
    reg [255:0] key;
    reg [31:0] val;
    integer fd, rc, n_bad = 0, n_chk = 0, t0, tot = 0, runs = 0, i;
    reg [31:0] ea, ev;
    task set_key;
        begin
            case (key)
                "vop": d_vop = val[4:0];  "rows": d_rows = val[15:0];  "cols": d_cols = val[15:0];
                "rows_per_b": d_rows_per_b = val[15:0];  "emb_l": d_emb_l = val[15:0];
                "st_en": d_st_en = val[0];  "x_reset": d_x_reset = val[0];  "g_reset": d_g_reset = val[0];
                "a_base": d_a_base = val;  "a_rs": d_a_rs = val;  "b_base": d_b_base = val;  "b_rs": d_b_rs = val;
                "c_base": d_c_base = val;  "c_rs": d_c_rs = val;
                "cv0_base": d_cv0_base = val;  "cv1_base": d_cv1_base = val;  "cv2_base": d_cv2_base = val;  "km_rs": d_km_rs = val;
                "st0_base": d_st0_base = val;  "st1_base": d_st1_base = val;  "o2_base": d_o2_base = val;  "ids_base": d_ids_base = val;
                "mk_base": d_mk_base = val;  "mk_rs": d_mk_rs = val;
                "c0": d_c0 = val;  "c1": d_c1 = val;  "c2": d_c2 = val;  "c3": d_c3 = val;  "ds": d_ds = val;
                "exp_n": exp_n = val;  "mask_n": mask_n = val;  "elements": elements = val;  "model": model_cyc = val;
                "run": run_one;
                default: begin $display("FAIL vu_top unknown descriptor key %0s", key); $finish; end
            endcase
        end
    endtask
    task run_one;
        begin
            @(negedge clk);
            start = 1'b1; t0 = cyc;
            @(negedge clk);
            start = 1'b0;
            while (!done) begin
                @(negedge clk);
                if (cyc - t0 > maxcyc) begin
                    $display("FAIL vu_top watchdog (%0d cycles) run %0d vop %0d: cs %0d cp %0d qv %b ql %b lp %0d pp %0d l_busy %b p_done %b e_run %b e_j %0d sl_busy %b rd0 q %0d/%0d wr_idle %b ld_run %b r0st %0d b0_act %b r1st %0d b1_act %b jb %0d c_n %0d acc_idle %b sfu_out %b xst %0d",
                        cyc - t0, runs, d_vop, dut.u_seq.cs, dut.u_seq.cp, dut.u_seq.q_valid, dut.u_seq.q_loaded, dut.u_seq.lp, dut.u_seq.pp,
                        dut.u_seq.l_busy, dut.u_seq.p_done, dut.u_seq.e_run, dut.u_seq.e_j, dut.u_seq.sl_busy, q0_w, q0_r, dut.u_wr.wr_idle,
                        dut.u_ld.ld_run, dut.u_ld.r0st, dut.u_ld.b0_active, dut.u_ld.r1st, dut.u_ld.b1_active, dut.u_ld.jb, dut.u_ld.c_n,
                        dut.acc_idle_all, dut.sfu_cnt_min != 0, dut.u_seq.x_st);
                    $finish;
                end
            end
            tot = tot + (cyc - t0);
            runs = runs + 1;
            repeat (2) @(negedge clk);
        end
    endtask
    initial begin
        if (!$value$plusargs("desc=%s", fdesc) || !$value$plusargs("mem=%s", fmem) || !$value$plusargs("exp=%s", fexp)) begin
            $display("FAIL vu_top need +desc +mem +exp"); $finish;
        end
        if ($value$plusargs("lat=%d", lat)) ;
        if ($value$plusargs("stall=%d", stall)) ;
        if ($value$plusargs("maxcyc=%d", maxcyc)) ;
        if ($value$plusargs("trace=%d", trace)) ;
`ifndef MM_DRAM
        $readmemh(fmem, mem);
        if ($value$plusargs("mask=%s", fmask)) $readmemh(fmask, mask);
`else
        mm_load;
`endif
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (3) @(negedge clk);
        fd = $fopen(fdesc, "r");
        if (fd == 0) begin $display("FAIL vu_top cannot open desc"); $finish; end
        while (!$feof(fd)) begin
            rc = $fscanf(fd, "%s %h\n", key, val);
            if (rc == 2) set_key;
        end
        $fclose(fd);
`ifdef MM_DRAM
        mm_drain;
`endif
        repeat (4) @(negedge clk);
        fd = $fopen(fexp, "r");
        if (fd == 0) begin $display("FAIL vu_top cannot open exp"); $finish; end
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
        $fclose(fd);
`ifdef MK_GEN
        $display("MKCHECK %0d responses %0d bit mismatches vs L1", mg_chk, mg_bad);
        if (mg_bad != 0 || mg_chk == 0) n_bad = n_bad + 1;
`endif
        if (n_chk != exp_n) begin $display("expected %0d words, read %0d", exp_n, n_chk); n_bad = n_bad + 1; end
        $display("CYCLES %0d RUNS %0d ELEMENTS %0d MODEL %0d", tot, runs, elements, model_cyc);
        if (n_chk == 0) $display("FAIL vu_top 0/0 no expected words");
        else if (n_bad == 0) $display("PASS vu_top %0d vectors", n_chk);
        else $display("FAIL vu_top %0d/%0d mismatches", n_bad, n_chk);
        $finish;
    end
endmodule
