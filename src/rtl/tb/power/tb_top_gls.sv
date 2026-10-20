// tb_top_gls: the chip testbench of tb/top/tb_top.sv for gate-level (and RTL) power runs (P9b).  Same chip
// hookup, DRAM model (tb/mem/tb_dram.sv, nominal knobs) and host script (+reg / +img / +host, ops 1-7 of
// tb_top), but it only probes nets that exist in both the RTL and the synthesized top_chip netlist (top_chip
// has no logic of its own, so its unit-to-unit wires keep their RTL names): no sequencer internals, no
// mm overlap checker.  The clock period is the chip clock (+define+GLS_HALF=<half period ns>, default 2.75 =
// 5.5 ns), so the recorded SAIF durations need no rescaling.
// Switching activity: +saifwin=<file>, lines "name first_cycle end_cycle", cycles counted from the start of the
// first unit op (t0 = the first cycle with a unit start: gm / vu / lw start or rising DMA busy), increasing and
// non-overlapping, so a program cut at a later op (tb/power/mk_cut_stim.py) uses the same op-relative windows.
// For each window the toggles of u_chip are recorded from t0 + first_cycle to t0 + end_cycle and written to
// <name>.saif (then <name>.saif.ok); the run ends after the last window (or at +stopcyc / +maxcyc).
// Op log (+oplog): S/D/R <unit> <cycle> as tb_top (dma start = rising unit_busy[0]); heartbeat +hb=<cycles>
// prints the cycle and the wall clock (date +%s) for the simulation speed.
`timescale 1ns/10ps
`ifndef MM_DMEMW
`define MM_DMEMW 24
`endif
`ifndef GLS_HALF
`define GLS_HALF 2.75
`endif
module tb_top_gls;
    reg clk = 0, rst_n = 0;
    always #(`GLS_HALF) clk = ~clk;
    integer cyc = 0;
    longint maxcyc = 64'd100000000, stopcyc = 0;
    always @(posedge clk) cyc <= cyc + 1;

    reg          s_awvalid = 0, s_wvalid = 0, s_bready = 1, s_arvalid = 0, s_rready = 1;
    reg  [11:0]  s_awaddr = 0, s_araddr = 0;
    reg  [31:0]  s_wdata = 0;
    wire         s_awready, s_wready, s_bvalid, s_arready, s_rvalid, irq;
    wire [1:0]   s_bresp, s_rresp;
    wire [31:0]  s_rdata;
    wire         m_arvalid, m_arready, m_rvalid, m_rready, m_rlast, m_awvalid, m_awready, m_wvalid, m_wready, m_wlast, m_bvalid, m_bready;
    wire [3:0]   m_arid, m_rid, m_awid, m_bid;
    wire [31:0]  m_araddr, m_awaddr;
    wire [7:0]   m_arlen, m_awlen;
    wire [2:0]   m_arsize, m_awsize;
    wire [1:0]   m_arburst, m_awburst, m_rresp, m_bresp;
    wire [255:0] m_rdata, m_wdata;
    wire [31:0]  m_wstrb;
    top_chip u_chip (
        .clk(clk), .rst_n(rst_n), .irq(irq),
        .s_awvalid(s_awvalid), .s_awready(s_awready), .s_awaddr(s_awaddr), .s_wvalid(s_wvalid), .s_wready(s_wready),
        .s_wdata(s_wdata), .s_wstrb(4'hF), .s_bvalid(s_bvalid), .s_bready(s_bready), .s_bresp(s_bresp),
        .s_arvalid(s_arvalid), .s_arready(s_arready), .s_araddr(s_araddr), .s_rvalid(s_rvalid), .s_rready(s_rready),
        .s_rdata(s_rdata), .s_rresp(s_rresp),
        .m_arvalid(m_arvalid), .m_arready(m_arready), .m_arid(m_arid), .m_araddr(m_araddr), .m_arlen(m_arlen), .m_arsize(m_arsize), .m_arburst(m_arburst),
        .m_rvalid(m_rvalid), .m_rready(m_rready), .m_rid(m_rid), .m_rdata(m_rdata), .m_rresp(m_rresp), .m_rlast(m_rlast),
        .m_awvalid(m_awvalid), .m_awready(m_awready), .m_awid(m_awid), .m_awaddr(m_awaddr), .m_awlen(m_awlen), .m_awsize(m_awsize), .m_awburst(m_awburst),
        .m_wvalid(m_wvalid), .m_wready(m_wready), .m_wdata(m_wdata), .m_wstrb(m_wstrb), .m_wlast(m_wlast),
        .m_bvalid(m_bvalid), .m_bready(m_bready), .m_bid(m_bid), .m_bresp(m_bresp));
    tb_dram #(.AXL(3), .MEMW(`MM_DMEMW)) dram (
        .clk(clk), .rst_n(rst_n),
        .arvalid(m_arvalid), .arready(m_arready), .arid(m_arid), .araddr(m_araddr), .arlen(m_arlen), .arsize(m_arsize), .arburst(m_arburst),
        .rvalid(m_rvalid), .rready(m_rready), .rid(m_rid), .rdata(m_rdata), .rresp(m_rresp), .rlast(m_rlast),
        .awvalid(m_awvalid), .awready(m_awready), .awid(m_awid), .awaddr(m_awaddr), .awlen(m_awlen), .awsize(m_awsize), .awburst(m_awburst),
        .wvalid(m_wvalid), .wready(m_wready), .wdata(m_wdata), .wstrb(m_wstrb), .wlast(m_wlast),
        .bvalid(m_bvalid), .bready(m_bready), .bid(m_bid), .bresp(m_bresp));

    // ------------------------------------------------------------------ host model (as tb_top)
    longint hb_in = 0, hb_out = 0, hb_lite = 0, hb_dram = 0;
    task axil_w(input [11:0] a, input [31:0] d);
        begin
            @(negedge clk); s_awvalid = 1; s_awaddr = a; s_wvalid = 1; s_wdata = d;
            @(posedge clk); while (!s_awready) @(posedge clk);
            @(negedge clk); s_awvalid = 0; s_wvalid = 0;
            while (!s_bvalid) @(negedge clk);
            if (s_bresp != 2'b00) begin $display("FAIL tb_top_gls BRESP %b", s_bresp); $finish; end
            hb_in = hb_in + 4; hb_lite = hb_lite + 4;
        end
    endtask
    task axil_r(input [11:0] a, output [31:0] d);
        begin
            @(negedge clk); s_arvalid = 1; s_araddr = a;
            @(posedge clk); while (!s_arready) @(posedge clk);
            @(negedge clk); s_arvalid = 0;
            while (!s_rvalid) @(negedge clk);
            d = s_rdata;
            hb_out = hb_out + 4; hb_lite = hb_lite + 4;
        end
    endtask
    task host_dram_w(input [31:0] w, input [31:0] d);
        integer x;
        begin
            x = dram.xlat(w);
            if (x < 0) begin $display("FAIL tb_top_gls host write to unmapped word %h", w); $finish; end
            dram.mem[x] = d;
            hb_in = hb_in + 4; hb_dram = hb_dram + 4;
        end
    endtask

    // ------------------------------------------------------------------ accounting and op log (top_chip nets only)
    longint a_cyc, a_gm, a_arr, a_vu, a_lw, a_mg, a_dma, a_act, a_rbeat, a_wbeat, a_c0;
    reg [3:0] ub_q = 0;
    integer fop = 0;
    reg in_step = 0;
    longint t0 = -1;
    always @(posedge clk) if (t0 < 0 && (u_chip.gm_start || u_chip.vu_start || u_chip.lw_start || (u_chip.unit_busy[0] && !ub_q[0]))) begin
        t0 = cyc;
        $display("T0 first unit start at cycle %0d", cyc); $fflush();
    end
    always @(posedge clk) begin
        if (in_step) begin
            a_cyc <= a_cyc + 1;
            if (u_chip.gm_busy) a_gm <= a_gm + 1;
            if (u_chip.gm_arr_active) a_arr <= a_arr + 1;
            if (u_chip.vu_busy) a_vu <= a_vu + 1;
            if (u_chip.lw_busy) a_lw <= a_lw + 1;
            if (u_chip.lw_m_busy) a_mg <= a_mg + 1;
            if (u_chip.unit_busy[0]) a_dma <= a_dma + 1;
            if (u_chip.unit_busy != 4'd0) a_act <= a_act + 1;
            if (m_rvalid && m_rready) a_rbeat <= a_rbeat + 1;
            if (m_wvalid && m_wready) a_wbeat <= a_wbeat + 1;
        end
        if (fop != 0) begin
            if (u_chip.gm_start) $fwrite(fop, "S 1 %0d\n", cyc);
            if (u_chip.vu_start) $fwrite(fop, "S 2 %0d\n", cyc);
            if (u_chip.lw_start) $fwrite(fop, "S 3 %0d\n", cyc);
            if (u_chip.unit_busy[0] && !ub_q[0]) $fwrite(fop, "S 0 %0d\n", cyc);
            if (u_chip.gm_done) $fwrite(fop, "D 1 %0d\n", cyc);
            if (u_chip.vu_done) $fwrite(fop, "D 2 %0d\n", cyc);
            if (u_chip.lw_done) $fwrite(fop, "D 3 %0d\n", cyc);
            for (int u = 0; u < 4; u++) if (ub_q[u] && !u_chip.unit_busy[u]) $fwrite(fop, "R %0d %0d\n", u, cyc);
            $fflush(fop);
        end
        ub_q <= u_chip.unit_busy;
    end
    task acct_reset;
        begin
            a_cyc = 0; a_gm = 0; a_arr = 0; a_vu = 0; a_lw = 0; a_mg = 0; a_dma = 0; a_act = 0;
            a_rbeat = 0; a_wbeat = 0; a_c0 = cyc;
        end
    endtask
    // signature of the DRAM write stream (every accepted W beat: data and strobes), for the RTL / netlist comparison
    reg [63:0] wsig = 64'd0;
    longint nwb = 0;
    always @(posedge clk) if (m_wvalid && m_wready) begin
        wsig <= {wsig[62:0], wsig[63]} ^ m_wdata[63:0] ^ m_wdata[127:64] ^ m_wdata[191:128] ^ m_wdata[255:192] ^ {32'd0, m_wstrb} ^ {32'd0, m_awaddr};
        nwb <= nwb + 1;
    end
    reg [7:0] gm_err_q = 0;
    reg [1:0] mm_err_q = 0;
    always @(posedge clk) begin
        gm_err_q <= u_chip.gm_err;
        mm_err_q <= u_chip.mm_err;
        if (u_chip.gm_err !== gm_err_q) $display("GMERR cyc %0d err %b", cyc, u_chip.gm_err);
        if (u_chip.mm_err !== mm_err_q) $display("MMERR cyc %0d err %b", cyc, u_chip.mm_err);
    end
    integer hb = 100000;
    always @(posedge clk) if (hb > 0 && cyc % hb == 0 && cyc > 0) begin
        $display("HB cyc %0d busy %b wbeats %0d wsig %h", cyc, u_chip.unit_busy, nwb, wsig);
        $system("date +HBWALL_%s");
        $fflush();
    end

    // ------------------------------------------------------------------ SAIF windows
    reg [1023:0] fwin;
    reg [8*256-1:0] wname;
    longint w0, w1;
    integer fw, rcw, nwin = 0, fchk;
    string sname, scmd;           // $toggle_report takes the file name as a string (a reg vector is rejected)
`ifndef VERILATOR                 // (a Verilator build of this testbench is the RTL reference; no SAIF there)
    initial begin
        if ($value$plusargs("saifwin=%s", fwin)) begin
            $set_gate_level_monitoring("on");
            $set_toggle_region(tb_top_gls.u_chip);
            fw = $fopen(fwin, "r");
            rcw = $fscanf(fw, "%s %d %d\n", wname, w0, w1);
            while (rcw == 3) begin
                while (t0 < 0 || cyc < t0 + w0) @(posedge clk);
                $toggle_start;
                $display("SAIF_START %0s cyc %0d", wname, cyc); $fflush();
                while (cyc < t0 + w1) @(posedge clk);
                $toggle_stop;
                sname = $sformatf("%0s.saif", wname);
                $toggle_report(sname, 1.0e-9, "tb_top_gls.u_chip");
                $toggle_reset;
                fchk = $fopen(sname, "r");
                if (fchk == 0) $display("SAIF_FAILED %0s not written", sname);
                else begin $fclose(fchk); scmd = $sformatf("touch %0s.saif.ok", wname); $system(scmd); end
                nwin = nwin + 1;
                $display("SAIF_WRITTEN %0s cycles %0d %0d", wname, w0, w1); $system("date +HBWALL_%s"); $fflush();
                rcw = $fscanf(fw, "%s %d %d\n", wname, w0, w1);
            end
            $fclose(fw);
            if (fop != 0) $fclose(fop);
            $display("SAIF_ALL_DONE %0d windows at cycle %0d", nwin, cyc);
            $finish;
        end
    end
`endif

    // ------------------------------------------------------------------ main (host script, as tb_top)
    always @(posedge clk) if (cyc >= maxcyc || (stopcyc > 0 && cyc >= stopcyc)) begin
        $display("STOP tb_top_gls cycle %0d (busy %b) wbeats %0d wsig %h", cyc, u_chip.unit_busy, nwb, wsig);
        if (fop != 0) $fclose(fop);
        $finish;
    end
    reg [1023:0] freg, fimg, fhost, foplog;
    integer fh, rc, nstep = 0, t;
    reg [31:0] op, a, d, rd;
    initial begin
        if (!$value$plusargs("reg=%s", freg) || !$value$plusargs("img=%s", fimg) || !$value$plusargs("host=%s", fhost)) begin
            $display("FAIL tb_top_gls need +reg +img +host"); $finish;
        end
        if ($value$plusargs("maxcyc=%d", maxcyc)) ;
        if ($value$plusargs("stopcyc=%d", stopcyc)) ;
        if ($value$plusargs("hb=%d", hb)) ;
        if ($value$plusargs("oplog=%s", foplog)) fop = $fopen(foplog, "w");
        dram.wfair = 1;
        if ($value$plusargs("lat=%d", t)) dram.lat = t;
        if ($value$plusargs("bw=%d", t)) dram.bw = t;
        if ($value$plusargs("wfair=%d", t)) dram.wfair = t;
        dram.load_bin(freg, fimg);
        $display("LOADED %0d regions", dram.n_rg); $system("date +HBWALL_%s"); $fflush();
        acct_reset;
        repeat (4) @(negedge clk);
        rst_n = 1;
        repeat (4) @(negedge clk);
        fh = $fopen(fhost, "r");
        rc = $fscanf(fh, "%h %h %h\n", op, a, d);
        while (rc == 3) begin
            case (op)
                1: begin axil_w(a[11:0], d); if (a[11:0] == 12'h010) in_step = 1; end
                2: begin axil_r(a[11:0], rd); $display("CSR %h = %h", a[11:0], rd); end
                3: host_dram_w(a, d);
                4: begin
                    wait (irq);
                    @(negedge clk);
                    in_step = 0;
                    nstep = nstep + 1;
                    $display("STEP %0d IRQ at cycle %0d", nstep, cyc);
                    $display("ACCT %0d cyc %0d gm_busy %0d gm_arr %0d vu_busy %0d lw_busy %0d merge_busy %0d dma_busy %0d any_busy %0d rbeats %0d wbeats %0d wall %0d",
                             nstep, a_cyc, a_gm, a_arr, a_vu, a_lw, a_mg, a_dma, a_act, a_rbeat, a_wbeat, cyc - a_c0);
                    $fflush();
                    acct_reset;
                end
                5: ;                                  // dumps: not used in the power runs
                7: begin $display("HOSTBYTES %0d in %0d out %0d axilite %0d dram_window %0d", nstep, hb_in, hb_out, hb_lite, hb_dram); $fflush(); end
                6: begin hb_in = 0; hb_out = 0; hb_lite = 0; hb_dram = 0; end
                default: begin $display("FAIL tb_top_gls bad host op %h", op); $finish; end
            endcase
            rc = $fscanf(fh, "%h %h %h\n", op, a, d);
        end
        $fclose(fh);
        if (fop != 0) $fclose(fop);
        repeat (8) @(negedge clk);
        if (dram.n_err != 0 || u_chip.mm_err != 0 || u_chip.gm_err != 0)
            $display("FAIL tb_top_gls errors: dram %0d mm %b gemm %b", dram.n_err, u_chip.mm_err, u_chip.gm_err);
        else
            $display("PASS tb_top_gls %0d steps (protocol)", nstep);
        $display("DRAMSTATS rd_beats %0d wr_beats %0d ar %0d aw %0d", dram.rd_beats, dram.wr_beats, dram.ar_n, dram.aw_n);
        $finish;
    end
endmodule
