`timescale 1ns/1ps
// tb_mm_unit: mm_top (2 read channels, 1 keep-bit channel, 2 write channels) against tb_dram.
// Commands (tb/mem/gen_mm.py, +cmd=) are presented strictly in file order, one at a time, each to its
// channel (a command is presented after the previous one's handshake), so the program order of every
// read / write to the same address is the file order; the expected file (+exp=) is the independent
// python reference evaluated in that order on the memory image (+reg= / +img=):
//   cmd  "0 p addr len"                    read on channel p
//        "1 p addr mask d0 .. d7"          write beat on channel p
//        "2 idx"                           keep-bit request
//        "3 idaddr stride base len"        gather: read the id word on channel 1, then read
//                                          base + id * stride (len words) on channel 0
//   exp  "R p g nv w0 .. w7"               next beat of channel p (nv valid words; g = gather id beat)
//        "K keep"                          next keep-bit response
//        "M wordaddr value"                memory word after the run (read through tb_dram.peek)
// Plusargs: +lat +wlat +bw +stall +reorder +ilv +wfair (tb_dram), +rstall (percent of cycles a client holds
// dat_ready low), +maxcyc.  Prints DRAM statistics and one line PASS mm_unit <n> vectors / FAIL ...
module tb_mm_unit;
    parameter AXL  = 3;
    parameter MAXR = 8;
    parameter MAXW = 4;
    parameter NC   = 16384;                  // max commands / expected lines
    localparam AXW = 1 << AXL;
    reg clk = 1'b0, rst_n = 1'b0;
    always #20 clk = ~clk;
    integer cyc = 0;
    always @(posedge clk) cyc <= cyc + 1;

    // ---------------------------------------------------------------- DUT + DRAM
    reg  [1:0]   rd_req_valid = 0;
    wire [1:0]   rd_req_ready, rd_dat_valid, rd_idle, wr_ready, wr_idle;
    reg  [63:0]  rd_req_addr = 0;
    reg  [13:0]  rd_req_len = 0;
    reg  [1:0]   rd_dat_ready = 0;
    wire [511:0] rd_dat;
    reg          mk_req_valid = 0;
    wire         mk_req_ready, mk_rsp_valid;
    reg  [31:0]  mk_req_idx = 0;
    wire [15:0]  mk_rsp_keep;
    reg  [1:0]   wr_valid = 0;
    reg  [63:0]  wr_addr = 0;
    reg  [511:0] wr_data = 0;
    reg  [15:0]  wr_mask = 0;
    wire [1:0]   err;
    wire         arvalid, arready, rvalid, rready, rlast, awvalid, awready, wvalid, wready, wlast, bvalid, bready;
    wire [3:0]   arid, rid, awid, bid;
    wire [31:0]  araddr, awaddr;
    wire [7:0]   arlen, awlen;
    wire [2:0]   arsize, awsize;
    wire [1:0]   arburst, awburst, rresp, bresp;
    wire [32*AXW-1:0] rdata, wdata;
    wire [4*AXW-1:0]  wstrb;
    /* verilator lint_off PINMISSING */
    mm_top #(.NR(2), .NM(1), .NW(2), .AXL(AXL), .MAXR(MAXR), .MAXW(MAXW), .RFL0(5), .RFL(4), .MK_DRAM(1)) dut (
        .clk(clk), .rst_n(rst_n),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready), .rd_req_addr(rd_req_addr), .rd_req_len(rd_req_len),
        .rd_dat_valid(rd_dat_valid), .rd_dat_ready(rd_dat_ready), .rd_dat(rd_dat),
        .mk_base(32'h8000_0000), .mk_req_valid(mk_req_valid), .mk_req_ready(mk_req_ready), .mk_req_idx(mk_req_idx),
        .mk_rsp_valid(mk_rsp_valid), .mk_rsp_keep(mk_rsp_keep),
        .mk_seed(64'd0), .mk_step(24'd0), .mk_site(4'd0), .mk_layer(4'd0),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_addr(wr_addr), .wr_data(wr_data), .wr_mask(wr_mask),
        .rd_idle(rd_idle), .wr_idle(wr_idle), .err(err),
        .m_arvalid(arvalid), .m_arready(arready), .m_arid(arid), .m_araddr(araddr), .m_arlen(arlen), .m_arsize(arsize), .m_arburst(arburst),
        .m_rvalid(rvalid), .m_rready(rready), .m_rid(rid), .m_rdata(rdata), .m_rresp(rresp), .m_rlast(rlast),
        .m_awvalid(awvalid), .m_awready(awready), .m_awid(awid), .m_awaddr(awaddr), .m_awlen(awlen), .m_awsize(awsize), .m_awburst(awburst),
        .m_wvalid(wvalid), .m_wready(wready), .m_wdata(wdata), .m_wstrb(wstrb), .m_wlast(wlast),
        .m_bvalid(bvalid), .m_bready(bready), .m_bid(bid), .m_bresp(bresp));
    /* verilator lint_on PINMISSING */
    tb_dram #(.AXL(AXL), .MEMW(18)) dram (
        .clk(clk), .rst_n(rst_n),
        .arvalid(arvalid), .arready(arready), .arid(arid), .araddr(araddr), .arlen(arlen), .arsize(arsize), .arburst(arburst),
        .rvalid(rvalid), .rready(rready), .rid(rid), .rdata(rdata), .rresp(rresp), .rlast(rlast),
        .awvalid(awvalid), .awready(awready), .awid(awid), .awaddr(awaddr), .awlen(awlen), .awsize(awsize), .awburst(awburst),
        .wvalid(wvalid), .wready(wready), .wdata(wdata), .wstrb(wstrb), .wlast(wlast),
        .bvalid(bvalid), .bready(bready), .bid(bid), .bresp(bresp));

    // ---------------------------------------------------------------- files
    reg  [31:0] c_f [0:NC-1][0:11];          // command fields
    integer     n_cmd = 0;
    reg  [31:0] e_r [0:1][0:NC-1][0:9];      // expected beats per channel: g, nv, w0..w7
    integer     n_er [0:1];
    reg  [15:0] e_k [0:NC-1];
    integer     n_ek = 0;
    reg  [31:0] e_ma [0:NC-1], e_mv [0:NC-1];
    integer     n_em = 0;
    integer     rstall = 0, maxcyc = 2000000;
    reg  [1023:0] fcmd, fexp, freg, fimg;
    integer fd, rc, t, p, k, nf;
    reg  [8*4-1:0] tag;
    reg  [31:0] v [0:11];

    // ---------------------------------------------------------------- checkers
    integer r_i [0:1];
    integer k_i = 0, n_bad = 0, n_chk = 0;
    reg         gid_v = 0;
    reg  [31:0] gid;
    integer q;
    always @(posedge clk) begin
        for (q = 0; q < 2; q = q + 1) begin
            if (rd_dat_valid[q] && rd_dat_ready[q]) begin
                if (r_i[q] >= n_er[q]) begin n_bad = n_bad + 1; $display("extra beat on channel %0d", q); end
                else begin
                    for (k = 0; k < e_r[q][r_i[q]][1]; k = k + 1) begin
                        n_chk = n_chk + 1;
                        if (rd_dat[256*q + 32*k +: 32] !== e_r[q][r_i[q]][2+k]) begin
                            n_bad = n_bad + 1;
                            if (n_bad < 8) $display("MISMATCH ch %0d beat %0d word %0d got %h want %h", q, r_i[q], k, rd_dat[256*q + 32*k +: 32], e_r[q][r_i[q]][2+k]);
                        end
                    end
                    if (e_r[q][r_i[q]][0] != 0) begin gid_v <= 1'b1; gid <= rd_dat[256*q +: 32]; end
                    r_i[q] = r_i[q] + 1;
                end
            end
        end
        if (mk_rsp_valid) begin
            n_chk = n_chk + 1;
            if (k_i >= n_ek || mk_rsp_keep !== e_k[k_i]) begin
                n_bad = n_bad + 1;
                if (n_bad < 8) $display("MISMATCH keep %0d got %h want %h", k_i, mk_rsp_keep, e_k[k_i]);
            end
            k_i = k_i + 1;
        end
        rd_dat_ready <= {($urandom % 100) >= rstall, ($urandom % 100) >= rstall};
    end

    // ---------------------------------------------------------------- watchdog
    integer cmd_i = 0;
    always @(posedge clk) if (cyc > maxcyc + 1000) begin
        $display("TIMEOUT at command %0d/%0d r %0d/%0d %0d/%0d k %0d/%0d", cmd_i, n_cmd, r_i[0], n_er[0], r_i[1], n_er[1], k_i, n_ek);
        $display("FAIL mm_unit watchdog"); $finish;
    end
    // ---------------------------------------------------------------- run
    initial begin
        n_er[0] = 0; n_er[1] = 0; r_i[0] = 0; r_i[1] = 0;
        if (!$value$plusargs("cmd=%s", fcmd) || !$value$plusargs("exp=%s", fexp) || !$value$plusargs("reg=%s", freg)
            || !$value$plusargs("img=%s", fimg)) begin $display("FAIL mm_unit need +cmd +exp +reg +img"); $finish; end
        if ($value$plusargs("lat=%d", t)) dram.lat = t;
        if ($value$plusargs("wlat=%d", t)) dram.wlat = t;
        if ($value$plusargs("bw=%d", t)) dram.bw = t;
        if ($value$plusargs("stall=%d", t)) dram.stall = t;
        if ($value$plusargs("reorder=%d", t)) dram.reorder = t;
        if ($value$plusargs("ilv=%d", t)) dram.ilv = t;
        if ($value$plusargs("wfair=%d", t)) dram.wfair = t;
        if ($value$plusargs("rstall=%d", t)) rstall = t;
        if ($value$plusargs("maxcyc=%d", t)) maxcyc = t;
        dram.load(freg, fimg);
        fd = $fopen(fcmd, "r");
        rc = $fscanf(fd, "%h", t);
        while (rc == 1) begin
            nf = (t == 0) ? 3 : (t == 1) ? 11 : (t == 2) ? 1 : 4;
            c_f[n_cmd][0] = t;
            for (k = 0; k < nf; k = k + 1) rc = $fscanf(fd, "%h", c_f[n_cmd][1+k]);
            n_cmd = n_cmd + 1;
            rc = $fscanf(fd, "%h", t);
        end
        $fclose(fd);
        fd = $fopen(fexp, "r");
        rc = $fscanf(fd, "%s", tag);
        while (rc == 1) begin
            if (tag == "R") begin
                rc = $fscanf(fd, "%h", p);
                for (k = 0; k < 10; k = k + 1) rc = $fscanf(fd, "%h", e_r[p][n_er[p]][k]);
                n_er[p] = n_er[p] + 1;
            end else if (tag == "K") begin rc = $fscanf(fd, "%h", e_k[n_ek]); n_ek = n_ek + 1; end
            else begin rc = $fscanf(fd, "%h %h", e_ma[n_em], e_mv[n_em]); n_em = n_em + 1; end
            rc = $fscanf(fd, "%s", tag);
        end
        $fclose(fd);
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);
        for (t = 0; t < n_cmd; t = t + 1) begin
            p = c_f[t][1]; cmd_i = t;
            case (c_f[t][0])
                0: begin
                    rd_req_addr[32*p +: 32] = c_f[t][2]; rd_req_len[7*p +: 7] = c_f[t][3][6:0]; rd_req_valid[p] = 1'b1;
                    @(posedge clk); while (!rd_req_ready[p]) @(posedge clk);
                    #1 rd_req_valid[p] = 1'b0;
                end
                1: begin
                    wr_addr[32*p +: 32] = c_f[t][2]; wr_mask[8*p +: 8] = c_f[t][3][7:0];
                    for (k = 0; k < 8; k = k + 1) wr_data[256*p + 32*k +: 32] = c_f[t][4+k];
                    wr_valid[p] = 1'b1;
                    @(posedge clk); while (!wr_ready[p]) @(posedge clk);
                    #1 wr_valid[p] = 1'b0;
                end
                2: begin
                    mk_req_idx = c_f[t][1]; mk_req_valid = 1'b1;
                    @(posedge clk); while (!mk_req_ready) @(posedge clk);
                    #1 mk_req_valid = 1'b0;
                end
                default: begin
                    gid_v = 1'b0;
                    rd_req_addr[63:32] = c_f[t][1]; rd_req_len[13:7] = 7'd1; rd_req_valid[1] = 1'b1;
                    @(posedge clk); while (!rd_req_ready[1]) @(posedge clk);
                    #1 rd_req_valid[1] = 1'b0;
                    while (!gid_v) @(posedge clk);
                    #1 rd_req_addr[31:0] = c_f[t][3] + gid * c_f[t][2]; rd_req_len[6:0] = c_f[t][4][6:0]; rd_req_valid[0] = 1'b1;
                    @(posedge clk); while (!rd_req_ready[0]) @(posedge clk);
                    #1 rd_req_valid[0] = 1'b0;
                end
            endcase
        end
        // drain
        while ((r_i[0] < n_er[0] || r_i[1] < n_er[1] || k_i < n_ek || wr_idle != 2'b11 || rd_idle != 2'b11) && cyc < maxcyc)
            @(posedge clk);
        repeat (20) @(posedge clk);
        for (k = 0; k < n_em; k = k + 1) begin
            n_chk = n_chk + 1;
            if (dram.peek(e_ma[k]) !== e_mv[k]) begin
                n_bad = n_bad + 1;
                if (n_bad < 8) $display("MISMATCH mem word %h got %h want %h", e_ma[k], dram.peek(e_ma[k]), e_mv[k]);
            end
        end
        if (cyc >= maxcyc) begin $display("TIMEOUT r %0d/%0d %0d/%0d k %0d/%0d", r_i[0], n_er[0], r_i[1], n_er[1], k_i, n_ek); n_bad = n_bad + 1; end
        if (err != 2'b00) begin $display("AXI error response %b", err); n_bad = n_bad + 1; end
        if (dram.n_err != 0) begin $display("DRAM protocol errors %0d", dram.n_err); n_bad = n_bad + 1; end
        $display("STATS cycles %0d rd_beats %0d wr_beats %0d ar %0d aw %0d max_rd_out %0d max_wr_out %0d first %0d last %0d",
                 cyc, dram.rd_beats, dram.wr_beats, dram.ar_n, dram.aw_n, dram.max_rd_out, dram.max_wr_out, dram.first_cyc, dram.last_cyc);
        if (n_chk == 0) $display("FAIL mm_unit 0/0 nothing checked");
        else if (n_bad == 0) $display("PASS mm_unit %0d vectors", n_chk);
        else $display("FAIL mm_unit %0d/%0d mismatches", n_bad, n_chk);
        $finish;
    end
endmodule
