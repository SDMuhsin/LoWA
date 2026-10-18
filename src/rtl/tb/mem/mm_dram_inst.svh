// tb_dram instance + image load + statistics, shared by the P6a integration includes.
    `include "mm_ovl_chk.svh"
    tb_dram #(.AXL(3), .MEMW(`MM_DMEMW)) dram (
        .clk(clk), .rst_n(rst_n),
        .arvalid(m_arvalid), .arready(m_arready), .arid(m_arid), .araddr(m_araddr), .arlen(m_arlen), .arsize(m_arsize), .arburst(m_arburst),
        .rvalid(m_rvalid), .rready(m_rready), .rid(m_rid), .rdata(m_rdata), .rresp(m_rresp), .rlast(m_rlast),
        .awvalid(m_awvalid), .awready(m_awready), .awid(m_awid), .awaddr(m_awaddr), .awlen(m_awlen), .awsize(m_awsize), .awburst(m_awburst),
        .wvalid(m_wvalid), .wready(m_wready), .wdata(m_wdata), .wstrb(m_wstrb), .wlast(m_wlast),
        .bvalid(m_bvalid), .bready(m_bready), .bid(m_bid), .bresp(m_bresp));
    task mm_load;
        reg [1023:0] freg, fimg;
        integer t;
        begin
            if (!$value$plusargs("reg=%s", freg) || !$value$plusargs("img=%s", fimg)) begin
                $display("FAIL mm_dram need +reg +img"); $finish;
            end
            dram.lat = lat; dram.stall = stall;
            if ($value$plusargs("bw=%d", t)) dram.bw = t;
            if ($value$plusargs("wlat=%d", t)) dram.wlat = t;
            if ($value$plusargs("reorder=%d", t)) dram.reorder = t;
            if ($value$plusargs("ilv=%d", t)) dram.ilv = t;
            if ($value$plusargs("wfair=%d", t)) dram.wfair = t;
            dram.load(freg, fimg);
        end
    endtask
    // after the engine's done: wait until every posted write is B-acknowledged and every read has returned
    task mm_drain;
        integer w;
        begin
            w = 0;
            while (!(&mm_wr_idle && &mm_rd_idle) && w < 100000) begin @(posedge clk); w = w + 1; end
            $display("MMDRAIN %0d", w);
        end
    endtask
    final begin
        $display("DRAMSTATS rd_beats %0d wr_beats %0d ar %0d aw %0d max_rd_out %0d max_wr_out %0d first %0d last %0d err %0d axi_err %b",
                 dram.rd_beats, dram.wr_beats, dram.ar_n, dram.aw_n, dram.max_rd_out, dram.max_wr_out, dram.first_cyc,
                 dram.last_cyc, dram.n_err, u_mm.err);
        $display("DRAMIDLE none %0d lat %0d", dram.idle_none, dram.idle_lat);
        $display("DRAMID %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d", dram.id_beats[0], dram.id_beats[1],
                 dram.id_beats[2], dram.id_beats[3], dram.id_beats[4], dram.id_beats[5], dram.id_beats[6], dram.id_beats[7],
                 dram.id_beats[8], dram.id_beats[9], dram.id_beats[10], dram.id_beats[11], dram.id_beats[12], dram.id_beats[13],
                 dram.id_beats[14], dram.id_beats[15]);
        if (dram.n_err != 0 || u_mm.err != 0) $display("FAIL mm_dram protocol/response errors");
        $display("MMOVL_SUMMARY same-cycle rd/wr pairs %0d range-overlapping %0d word-overlapping %0d", mmo_pairs, mmo_rng, mmo_ovl);
        if (mmo_ovl != 0) $display("FAIL mm_dram same-cycle overlapping read/write");
    end

