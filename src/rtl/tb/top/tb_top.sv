// tb_top: the LoWA-TA chip (hw/top/top_chip) end to end (P7a).  top_chip's AXI4 master on the DRAM model
// (tb/mem/tb_dram.sv, nominal knobs: lat 24, wlat 8, bw 3200 = 32 B/cycle, wfair 1 per ruling R9), its AXI4-Lite
// slave driven by a host model.  The host runs a script (+host=<file>, one command per line, hex fields):
//   1 a d   AXI4-Lite write CSR a := d                (4 B host -> device)
//   2 a 0   AXI4-Lite read CSR a, printed             (4 B device -> host)
//   3 w d   host DRAM-window write of word address w  (4 B host -> device; ruling R3 host load path)
//   4 0 0   wait for IRQ (cycle limit +maxcyc), print the step's cycle accounting
//   5 k 0   dump the regions of +dump=<file> ("base_word nwords" lines) to <+out>_<k>.hex
//   6 0 0   reset the per-step host byte counters
//   7 0 0   print the host byte counters (HOSTBYTES)
// The initial image (+reg / +img, tb/mem/mm_image.py format) is the "host once" load; it is not counted.
// Op log (+oplog=<file>): S/D/R <unit> <cycle> for every unit start, engine done and retire, for the cycle
// accounting in tb/top/run_e2e.py.  Expected values are never produced here: run_e2e.py compares the dumps with
// the interpreter / L1.  Prints HOSTBYTES / ACCT lines per step and one final PASS / FAIL line (protocol only).
`timescale 1ns/1ps
`ifndef MM_DMEMW
`define MM_DMEMW 24
`endif
module tb_top;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    integer cyc = 0;
    longint maxcyc = 64'd100000000;
    always @(posedge clk) cyc <= cyc + 1;

    // ------------------------------------------------------------------ chip
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

    // ------------------------------------------------------------------ host model
    longint hb_in = 0, hb_out = 0, hb_lite = 0, hb_dram = 0;
    task axil_w(input [11:0] a, input [31:0] d);
        begin
            @(negedge clk); s_awvalid = 1; s_awaddr = a; s_wvalid = 1; s_wdata = d;
            @(posedge clk); while (!s_awready) @(posedge clk);
            @(negedge clk); s_awvalid = 0; s_wvalid = 0;
            while (!s_bvalid) @(negedge clk);
            if (s_bresp != 2'b00) begin $display("FAIL tb_top BRESP %b", s_bresp); $finish; end
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
            if (x < 0) begin $display("FAIL tb_top host write to unmapped word %h", w); $finish; end
            dram.mem[x] = d;
            hb_in = hb_in + 4; hb_dram = hb_dram + 4;
        end
    endtask

    // ------------------------------------------------------------------ accounting (per step, reset at each wait)
    longint a_cyc, a_gm, a_arr, a_vu, a_lw, a_mg, a_dma, a_act, a_none, a_rbeat, a_wbeat, a_dq, a_fetch, a_c0;
    reg [3:0] ub_q = 0;
    integer fop = 0;
    wire running = u_chip.u_seq.running;
    always @(posedge clk) begin
        if (running) begin
            a_cyc <= a_cyc + 1;
            if (u_chip.gm_busy) a_gm <= a_gm + 1;
            if (u_chip.gm_arr_active) a_arr <= a_arr + 1;
            if (u_chip.vu_busy) a_vu <= a_vu + 1;
            if (u_chip.lw_busy) a_lw <= a_lw + 1;
            if (u_chip.lw_m_busy) a_mg <= a_mg + 1;
            if (u_chip.unit_busy[0]) a_dma <= a_dma + 1;
            if (u_chip.unit_busy != 4'd0) a_act <= a_act + 1;
            if (u_chip.unit_busy == 4'd0) a_none <= a_none + 1;
            if (m_rvalid && m_rready) a_rbeat <= a_rbeat + 1;
            if (m_wvalid && m_wready) a_wbeat <= a_wbeat + 1;
            if (!(m_rvalid && m_rready) && !(m_wvalid && m_wready)) a_dq <= a_dq + 1;
            if (u_chip.u_seq.st == 3'd1 || u_chip.u_seq.st == 3'd2 || u_chip.u_seq.st == 3'd3) a_fetch <= a_fetch + 1;
        end
        if (fop != 0) begin
            if (u_chip.gm_start) $fwrite(fop, "S 1 %0d\n", cyc);
            if (u_chip.vu_start) $fwrite(fop, "S 2 %0d\n", cyc);
            if (u_chip.lw_start) $fwrite(fop, "S 3 %0d\n", cyc);
            if (u_chip.u_seq.dma_start) $fwrite(fop, "S 0 %0d\n", cyc);
            if (u_chip.gm_done) $fwrite(fop, "D 1 %0d\n", cyc);
            if (u_chip.vu_done) $fwrite(fop, "D 2 %0d\n", cyc);
            if (u_chip.lw_done) $fwrite(fop, "D 3 %0d\n", cyc);
            for (int u = 0; u < 4; u++) if (ub_q[u] && !u_chip.unit_busy[u]) $fwrite(fop, "R %0d %0d\n", u, cyc);
        end
        ub_q <= u_chip.unit_busy;
    end
    task acct_reset;
        begin
            a_cyc = 0; a_gm = 0; a_arr = 0; a_vu = 0; a_lw = 0; a_mg = 0; a_dma = 0; a_act = 0; a_none = 0;
            a_rbeat = 0; a_wbeat = 0; a_dq = 0; a_fetch = 0; a_c0 = cyc;
        end
    endtask

    // ------------------------------------------------------------------ dumps
    reg [1023:0] fdump, fout;
    task dump(input integer k);
        integer fd, fo, rc, i;
        reg [31:0] b, n;
        reg [8*1024-1:0] nm;
        begin
            fd = $fopen(fdump, "r");
            $sformat(nm, "%0s_%0d.hex", fout, k);
            fo = $fopen(nm, "w");
            rc = $fscanf(fd, "%h %h\n", b, n);
            while (rc == 2) begin
                for (i = 0; i < n; i = i + 1) $fwrite(fo, "%h\n", dram.peek(b + i));
                rc = $fscanf(fd, "%h %h\n", b, n);
            end
            $fclose(fd); $fclose(fo);
            $display("DUMP %0d written", k);
        end
    endtask

    // same-cycle overlapping read / write pairs on the memory system's engine ports (P8c2, R14c)
    localparam MMO_NR = 8, MMO_NW = 4;
`define MMO_INST u_chip.u_mm
    `include "mm_ovl_chk.svh"
    // first cycle of every sticky engine / memory error (debug aid)
    reg [7:0] gm_err_q = 0;
    reg [1:0] mm_err_q = 0;
    always @(posedge clk) begin
        gm_err_q <= u_chip.gm_err;
        mm_err_q <= u_chip.u_mm.err;
        if (u_chip.gm_err != gm_err_q) $display("GMERR cyc %0d err %b pc %h desc %0d", cyc, u_chip.gm_err, u_chip.u_seq.pc, u_chip.u_seq.perf_desc);
        if (u_chip.u_mm.err != mm_err_q) $display("MMERR cyc %0d err %b", cyc, u_chip.u_mm.err);
    end
    // heartbeat (+hb=N cycles, default 1 M): progress and simulation speed
    integer hb = 1000000;
    always @(posedge clk) if (hb > 0 && cyc % hb == 0 && cyc > 0) begin
        $display("HB cyc %0d pc %h st %0d busy %b desc %0d", cyc, u_chip.u_seq.pc, u_chip.u_seq.st, u_chip.unit_busy,
                 u_chip.u_seq.perf_desc);
        $fflush;
    end
    // ------------------------------------------------------------------ main
    always @(posedge clk) if (cyc >= maxcyc) begin
        $display("FAIL tb_top cycle limit %0d reached (pc %h st %0d busy %b)", maxcyc, u_chip.u_seq.pc, u_chip.u_seq.st, u_chip.unit_busy);
        $finish;
    end
    reg [1023:0] freg, fimg, fhost, foplog;
    integer fh, rc, nstep = 0, t;
    reg [31:0] op, a, d, rd;
    initial begin
        if (!$value$plusargs("reg=%s", freg) || !$value$plusargs("img=%s", fimg) || !$value$plusargs("host=%s", fhost)) begin
            $display("FAIL tb_top need +reg +img +host"); $finish;
        end
        if (!$value$plusargs("dump=%s", fdump)) fdump = "";
        if (!$value$plusargs("out=%s", fout)) fout = "dump";
        if ($value$plusargs("maxcyc=%d", maxcyc)) ;
        if ($value$plusargs("hb=%d", hb)) ;
        if ($value$plusargs("oplog=%s", foplog)) fop = $fopen(foplog, "w");
        dram.wfair = 1;
        if ($value$plusargs("lat=%d", t)) dram.lat = t;
        if ($value$plusargs("bw=%d", t)) dram.bw = t;
        if ($value$plusargs("wfair=%d", t)) dram.wfair = t;
        dram.load_bin(freg, fimg);
        $display("LOADED %0d regions", dram.n_rg); $fflush;
        acct_reset;
        repeat (4) @(negedge clk);
        rst_n = 1;
        repeat (4) @(negedge clk);
        fh = $fopen(fhost, "r");
        rc = $fscanf(fh, "%h %h %h\n", op, a, d);
        while (rc == 3) begin
            case (op)
                1: axil_w(a[11:0], d);
                2: begin axil_r(a[11:0], rd); $display("CSR %h = %h", a[11:0], rd); end
                3: host_dram_w(a, d);
                4: begin
                    wait (irq);
                    @(negedge clk);
                    nstep = nstep + 1;
                    $display("STEP %0d IRQ at cycle %0d", nstep, cyc);
                    $display("ACCT %0d cyc %0d gm_busy %0d gm_arr %0d vu_busy %0d lw_busy %0d merge_busy %0d dma_busy %0d any_busy %0d none_busy %0d fetch %0d rbeats %0d wbeats %0d dram_idle %0d wall %0d",
                             nstep, a_cyc, a_gm, a_arr, a_vu, a_lw, a_mg, a_dma, a_act, a_none, a_fetch, a_rbeat, a_wbeat, a_dq, cyc - a_c0);
                    $fflush;
                    acct_reset;
                end
                5: dump(a);
                7: begin $display("HOSTBYTES %0d in %0d out %0d axilite %0d dram_window %0d", nstep, hb_in, hb_out, hb_lite, hb_dram); $fflush; end
                6: begin hb_in = 0; hb_out = 0; hb_lite = 0; hb_dram = 0; end
                default: begin $display("FAIL tb_top bad host op %h", op); $finish; end
            endcase
            rc = $fscanf(fh, "%h %h %h\n", op, a, d);
        end
        $fclose(fh);
        if (fop != 0) $fclose(fop);
        repeat (8) @(negedge clk);
        $display("MMOVL_SUMMARY same-cycle rd/wr pairs %0d range-overlapping %0d word-overlapping %0d", mmo_pairs, mmo_rng, mmo_ovl);
        if (dram.n_err != 0 || u_chip.u_mm.err != 0 || u_chip.u_seq.err != 0 || u_chip.gm_err != 0 || mmo_ovl != 0)
            $display("FAIL tb_top errors: dram %0d axi %b seq %b gemm %b rd/wr overlap %0d", dram.n_err, u_chip.u_mm.err,
                     u_chip.u_seq.err, u_chip.gm_err, mmo_ovl);
        else
            $display("PASS tb_top %0d steps (protocol; data checked by run_e2e.py)", nstep);
        $display("DRAMSTATS rd_beats %0d wr_beats %0d ar %0d aw %0d", dram.rd_beats, dram.wr_beats, dram.ar_n, dram.aw_n);
        $finish;
    end
endmodule
