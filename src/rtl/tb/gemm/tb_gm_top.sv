`timescale 1ns/1ps
// tb_gm_top: runs one GEMM descriptor on gm_top against a memory model and compares every word
// the engine writes with the L1 expectation (tb/gemm/gen_gemm.py), bit for bit.
//   +desc=<file>  descriptor ("key hex" lines)      +mem=<file>  $readmemh memory image
//   +exp=<file>   "wordaddr value" expected words   +mask=<file> keep bits (dropout chains)
//   +lat=<n>      memory latency in cycles (default 24)   +stall=<pct>  random stalls on the
//   memory channels (default 0)   +maxcyc=<n> watchdog
// Memory model: word-addressed, two in-order read channels (request queue, LAT cycles, then one
// 256-bit beat per cycle), one write channel with a word mask, a mask responder.
// Prints CYCLES / ACTIVE lines for the driver and one final line:
//   PASS gm_top <n> vectors | FAIL gm_top <m>/<n> mismatches
module tb_gm_top;
    parameter ROWS = 16;
    parameter COLS = 16;
    parameter NSFU = 16;
    parameter MEMW = 23;                    // log2 words of the memory model
    parameter MASKW = 24;                   // log2 entries of the dropout keep-bit array (Icarus: 16;
                                            // ivl's per-word elaboration of a 2^24 array takes hours)
    localparam DWP = (ROWS * COLS >= 64) ? ROWS * COLS / 64 : 1;
    localparam GWD = 4 * DWP;
    localparam MGW = (COLS > 8) ? (COLS > 16 ? 5 : 4) : 3;
    localparam OWP = (DWP == 1) ? 2 : (DWP == 2) ? 3 : (DWP == 4) ? 4 : 5;   // log2 GWD (epilogue OW)
    localparam GMWP = OWP + 3 + 64;                                           // epilogue GMW

    reg clk = 1'b0, rst_n = 1'b0;
    always #20 clk = ~clk;
    integer cyc = 0;
    always @(posedge clk) cyc <= cyc + 1;
`ifdef GM_DRDBG
    always @(posedge clk) begin
        if (dut.u_drain.start & ~dut.u_drain.free) $display("DRDBG %0d start while busy", cyc);
        if (|dut.u_drain.lconf) $display("DRDBG %0d lconf %h", cyc, dut.u_drain.lconf);
        if (dut.u_drain.start) $display("DRDBG %0d drain start", cyc);
        if (dut.u_ctrl.ps_start) $display("DRDBG %0d pass start kt %0d mt %0d", cyc, dut.u_ctrl.ps_kt, dut.u_ctrl.ps_mt);
    end
`endif

    // ------------------------------------------------------------------ descriptor
    reg [15:0] d_m, d_n, d_k0, d_k1, d_k2, d_nb1, d_nb2;
    reg [31:0] d_a_base0, d_a_base1, d_a_base2, d_a_rs, d_a_bs1, d_a_bs2;
    reg [31:0] d_b_base0, d_b_base1, d_b_base2, d_b_rs, d_b_bs1, d_b_bs2;
    reg [31:0] d_c_base, d_c_rs, d_c_bs1, d_c_bs2, d_bias_base, d_aux_base, d_o2_base, d_ep_ds;
    reg        d_a_trn, d_b_trn, d_ep_bias, d_ep_scale, d_ep_gelu, d_ep_gelu_bwd, d_ep_drop, d_ep_resid;
    reg [7:0]  d_ep_sexp;
    integer    exp_n = 0, mask_n = 0, model_m = 0, model_n = 0, model_k = 0, model_nb = 0;
    reg [63:0] macs = 0;

    // ------------------------------------------------------------------ DUT
    reg          start = 1'b0;
    wire         busy, done, arr_active;
    wire [7:0]   err;
    wire         rd0_req_valid, rd0_dat_ready, rd1_req_valid, rd1_dat_ready, mk_req_valid, wr_valid;
    wire [31:0]  rd0_req_addr, rd1_req_addr, mk_req_idx, wr_addr;
    wire [6:0]   rd0_req_len, rd1_req_len;
    wire [255:0] wr_data;
    wire [7:0]   wr_mask;
`ifndef MM_DRAM
    reg          rd0_req_ready, rd0_dat_valid, rd1_req_ready, rd1_dat_valid, mk_req_ready, mk_rsp_valid, wr_ready;
    reg  [255:0] rd0_dat, rd1_dat;
    reg  [GWD-1:0] mk_rsp_keep;
`else
    wire         rd0_req_ready, rd0_dat_valid, rd1_req_ready, rd1_dat_valid, mk_req_ready, mk_rsp_valid, wr_ready;
    wire [255:0] rd0_dat, rd1_dat;
    wire [GWD-1:0] mk_rsp_keep;
`endif
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
        .mg_en(1'b0), .mg_buf(1'b0), .mg_we(1'b0), .mg_lane({MGW{1'b0}}), .mg_addr(8'd0), .mg_wdata(32'd0),
        .mg_rdata(), .w_tile_done(), .w_tile_buf(), .merge_busy(2'b00));

    // ------------------------------------------------------------------ memory model
    reg [31:0] mem [0:(1 << MEMW) - 1];
    reg        mask [0:(1 << MASKW) - 1];
    integer lat = 24, stall = 0, maxcyc = 400000000, trace = 0;
    always @(posedge clk) if (trace) begin
        if (wr_valid && wr_ready) $display("T%0d WR addr %h mask %b w0 %h", cyc, wr_addr, wr_mask, wr_data[31:0]);
        if (rd0_req_valid && rd0_req_ready) $display("T%0d RD0 addr %h len %0d", cyc, rd0_req_addr, rd0_req_len);
        if (dut.ps_start) $display("T%0d PASS buf %0d mt %0d kt %0d", cyc, dut.ps_buf, dut.ps_mt, dut.ps_kt);
        if (dut.dr_start) $display("T%0d DRAIN half %0d mt %0d first %0d last %0d", cyc, dut.dr_half, dut.dr_mt, dut.dr_first, dut.dr_last);
        if (dut.ob_valid && dut.ob_ready) $display("T%0d OB half %0d rows %0d cols %0d c %h", cyc, dut.ob_half, dut.ob_rows, dut.ob_cols, dut.ob_c);
        if (dut.ep_done) $display("T%0d EPDONE half %0d", cyc, dut.ep_done_half);
        if (dut.fin_valid) $display("T%0d FIN half %0d", cyc, dut.fin_half);
        if (trace > 2 && dut.u_loader.c_fire) $display("T%0d BEAT kind %0d buf %0d i %0d b %0d nreq %0d nb %0d len %0d", cyc, dut.u_loader.c_kind, dut.u_loader.c_buf, dut.u_loader.c_i, dut.u_loader.c_b, dut.u_loader.c_nreq, dut.u_loader.c_nb, dut.u_loader.c_len);
        if (trace > 2 && dut.u_ctrl.tq_valid && dut.u_ctrl.tq_ready) $display("T%0d TQ kind %0d buf %0d trn %0d n %0d kt %0d addr %h", cyc, dut.u_ctrl.tq_kind, dut.u_ctrl.tq_buf, dut.u_ctrl.tq_trn, dut.u_ctrl.tq_n, dut.u_ctrl.tq_kt, dut.u_ctrl.tq_addr);
        if (dut.tile_done) $display("T%0d TILEDONE kind %0d buf %0d x_ld %b w_ld %b inuse %b", cyc, dut.tile_done_kind, dut.tile_done_buf, dut.u_ctrl.x_ld, dut.u_ctrl.w_ld, dut.u_ctrl.buf_inuse);
        if (trace > 1 && dut.u_ep.go_valid) $display("T%0d GO off %0d we %b last %b y %h wp %0d", cyc, dut.u_ep.go_meta[GMWP-1 -: OWP], dut.u_ep.go_meta[66], dut.u_ep.go_meta[65], dut.u_ep.go_y, dut.u_ep.as_wp);
        if (trace > 1 && dut.u_ep.g_issue) $display("T%0d GI j %0d q %0d we %b u %h dg %h ad %h rp %0d wp %0d", cyc, dut.u_ep.gs_j, dut.u_ep.gq, dut.u_ep.g_we, dut.u_ep.u_gelu.in_u, dut.u_ep.u_gelu.in_dg, dut.u_ep.gh_ad[63:32], dut.u_ep.gf_rp, dut.u_ep.gf_wp);
        if (trace > 1 && dut.u_ep.vv5 && dut.u_ep.gelu_mode) $display("T%0d GF push ad %h v %h wp %0d", cyc, dut.u_ep.ad5[63:32], dut.u_ep.v5[31:0], dut.u_ep.gf_wp);
    end
    function stall_now;
        input integer dummy;
        begin
            stall_now = (stall > 0) && (($urandom % 100) < stall);
        end
    endfunction

`ifndef MM_DRAM
    // read channel 0 / 1: queues of (addr, len, ready time)
    reg [31:0] qm_i [0:63]; integer qm_t [0:63]; integer qm_w = 0, qm_r = 0;
`ifdef MK_GEN
    localparam MKG_W = GWD;
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
        for (j = 0; j < GWD; j = j + 1)
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

    // ------------------------------------------------------------------ run
    reg [1023:0] fdesc, fmem, fexp, fmask;
    reg [255:0] key;
    reg [31:0] val;
    integer fd, rc, n_bad = 0, n_chk = 0, t0, t1, act = 0, i;
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
                "bias_base": d_bias_base = val;  "aux_base": d_aux_base = val;  "o2_base": d_o2_base = val;
                "ep_bias": d_ep_bias = val[0];  "ep_scale": d_ep_scale = val[0];  "ep_gelu": d_ep_gelu = val[0];
                "ep_gelu_bwd": d_ep_gelu_bwd = val[0];  "ep_drop": d_ep_drop = val[0];  "ep_resid": d_ep_resid = val[0];
                "ep_sexp": d_ep_sexp = val[7:0];  "ep_ds": d_ep_ds = val;
                "exp_n": exp_n = val;  "mask_n": mask_n = val;
                "model_m": model_m = val;  "model_n": model_n = val;  "model_k": model_k = val;  "model_nb": model_nb = val;
                "macs": macs = val;
                default: begin $display("FAIL gm_top unknown descriptor key %0s", key); $finish; end
            endcase
        end
    endtask

    initial begin
        if (!$value$plusargs("desc=%s", fdesc) || !$value$plusargs("mem=%s", fmem) || !$value$plusargs("exp=%s", fexp)) begin
            $display("FAIL gm_top need +desc +mem +exp"); $finish;
        end
        if ($value$plusargs("lat=%d", lat)) ;
        if ($value$plusargs("stall=%d", stall)) ;
        if ($value$plusargs("maxcyc=%d", maxcyc)) ;
        if ($value$plusargs("trace=%d", trace)) ;
        fd = $fopen(fdesc, "r");
        if (fd == 0) begin $display("FAIL gm_top cannot open desc"); $finish; end
        while (!$feof(fd)) begin
            rc = $fscanf(fd, "%s %h\n", key, val);
            if (rc == 2) set_key;
        end
        $fclose(fd);
`ifndef MM_DRAM
        $readmemh(fmem, mem);
        if (mask_n > 0) begin
            if (!$value$plusargs("mask=%s", fmask)) begin $display("FAIL gm_top need +mask"); $finish; end
            $readmemh(fmask, mask);
`else
        mm_load;
        if (0) begin
`endif
        end
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);
        start = 1'b1; t0 = cyc;
        @(negedge clk);
        start = 1'b0;
        while (!done) begin
            @(negedge clk);
            if (cyc - t0 > maxcyc) begin
                $display("FAIL gm_top watchdog (%0d cycles), err=%b", cyc - t0, err);
                $display("  idle: g_act %b bf %0d/%0d/%0d inblk %b tq_valid %b sf_empty %b dr_free %b ob_valid %b ep_idle %b ep_run %b ep_m_act %b ep_s_act %b asm_done %b wr_valid %b g_inflight %0d",
                    dut.u_ctrl.g_act, dut.u_ctrl.bf_wp, dut.u_ctrl.bf_lp, dut.u_ctrl.bf_cp, dut.u_ctrl.ct_inblk, dut.u_ctrl.tq_valid,
                    dut.u_ctrl.sf_empty, dut.u_ctrl.dr_free, dut.u_ctrl.ob_valid, dut.u_ctrl.ep_idle, dut.u_ep.run, dut.u_ep.m_act, dut.u_ep.s_act,
                    dut.u_ep.asm_done, dut.u_ep.wr_valid, dut.u_ep.g_inflight);
                $display("  ld: x_busy %b w_busy %b x_ld %b w_ld %b ct_buf %b lf_buf %b tf %0d/%0d/%0d c_i %0d c_b %0d rd0 q %0d/%0d dat_v %b dat_r %b tiles_ok %b half_busy %b ct_half %b first %b inblk %b",
                    dut.x_busy, dut.w_busy, dut.u_ctrl.x_ld, dut.u_ctrl.w_ld, dut.u_ctrl.ct_buf, dut.u_ctrl.lf_buf,
                    dut.u_loader.tf_wp, dut.u_loader.tf_rp, dut.u_loader.tf_cp, dut.u_loader.c_i, dut.u_loader.c_b, q0_w, q0_r,
                    rd0_dat_valid, rd0_dat_ready, dut.u_ctrl.tiles_ok, dut.u_ctrl.half_busy, dut.u_ctrl.ct_half, dut.u_ctrl.ct_first, dut.u_ctrl.ct_inblk);
                $finish;
            end
        end
        t1 = cyc;
`ifdef MM_DRAM
        mm_drain;
`endif
        repeat (4) @(negedge clk);
        // compare
        fd = $fopen(fexp, "r");
        if (fd == 0) begin $display("FAIL gm_top cannot open exp"); $finish; end
        rc = $fscanf(fd, "%h %h\n", ea, ev);
        while (rc == 2) begin
            begin
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
            end
            rc = $fscanf(fd, "%h %h\n", ea, ev);
        end
        $fclose(fd);
        if (err != 8'd0) begin $display("ERROR flags %b", err); n_bad = n_bad + 1; end
`ifdef MK_GEN
        $display("MKCHECK %0d responses %0d bit mismatches vs L1", mg_chk, mg_bad);
        if (mg_bad != 0 || mg_chk == 0) n_bad = n_bad + 1;
`endif
        if (n_chk != exp_n) begin $display("expected %0d words, read %0d", exp_n, n_chk); n_bad = n_bad + 1; end
        $display("CYCLES %0d ACTIVE %0d MACS %0d MODEL %0d %0d %0d %0d", t1 - t0, act, macs, model_m, model_n, model_k, model_nb);
        if (n_chk == 0) $display("FAIL gm_top 0/0 no expected words");
        else if (n_bad == 0) $display("PASS gm_top %0d vectors", n_chk);
        else $display("FAIL gm_top %0d/%0d mismatches", n_bad, n_chk);
        $finish;
    end
endmodule
