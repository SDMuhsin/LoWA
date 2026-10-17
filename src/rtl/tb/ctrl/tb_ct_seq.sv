// tb_ct_seq: sequencer unit test (P6c).  ct_seq with a word memory (program + DMA data, random latency / gaps /
// write backpressure) and stub engines (random durations, random write-drain delay after done, merge stub).
// Writes a log of every unit start (with its payload words), done, write-idle and retire; the python checker
// (tb/ctrl/run_tests.py) compares it with the reference walk of the program (sw/isa.py) and checks issue
// order, unit serialisation, FENCE / WAIT / SIG semantics and the retire rule.  Plusargs: +prog=<hex>
// +log=<file> +maxcyc=N +seed=N +step_t=N +slot=N +slot_delta=N +nowrite +dlo=<word> +dn=<words>
`timescale 1ns/1ps
module tb_ct_seq;
    localparam GP = 36, VP = 25, LP = 10, MW = 1 << 20;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    reg         csr_we = 0, csr_re = 0;
    reg  [11:0] csr_addr = 0;
    reg  [31:0] csr_wdata = 0;
    wire [31:0] csr_rdata;
    wire        irq;
    wire        rd_req_valid, rd_dat_ready, wr_valid;
    reg         rd_req_ready, rd_dat_valid, wr_ready;
    wire [31:0] rd_req_addr, wr_addr;
    wire [6:0]  rd_req_len;
    reg  [255:0] rd_dat;
    wire [255:0] wr_data;
    wire [7:0]  wr_mask;
    reg  [3:0]  wr_idle;
    wire        gm_start, mg_start, vu_start, lw_start;
    wire [32*GP-1:0] gm_p;
    wire [32*VP-1:0] vu_p;
    wire [32*LP-1:0] lw_p;
    reg         gm_done, mg_done, vu_done, lw_done;
    wire [63:0] mk_seed;
    wire [23:0] mk_step;
    wire [3:0]  unit_busy;
    ct_seq #(.GP(GP), .VP(VP), .LP(LP)) dut (
        .clk(clk), .rst_n(rst_n), .csr_we(csr_we), .csr_re(csr_re), .csr_addr(csr_addr), .csr_wdata(csr_wdata),
        .csr_rdata(csr_rdata), .irq(irq),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready), .rd_req_addr(rd_req_addr), .rd_req_len(rd_req_len),
        .rd_dat_valid(rd_dat_valid), .rd_dat_ready(rd_dat_ready), .rd_dat(rd_dat),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_addr(wr_addr), .wr_data(wr_data), .wr_mask(wr_mask), .wr_idle(wr_idle),
        .gm_start(gm_start), .gm_p(gm_p), .gm_done(gm_done), .mg_done(mg_done), .mg_start(mg_start),
        .vu_start(vu_start), .vu_p(vu_p), .vu_done(vu_done), .lw_start(lw_start), .lw_p(lw_p), .lw_done(lw_done),
        .mk_seed(mk_seed), .mk_step(mk_step), .unit_busy(unit_busy));

    reg [31:0] mem [0:MW-1];
    integer fd, cyc, maxcyc, i, k, nowrite, dlo, dn, rs;
    reg [1023:0] fname;
    // ---------------------------------------------------------------- memory: read requests, beats
    reg [31:0] q_addr [0:15];
    reg [6:0]  q_len [0:15];
    integer    q_n, b_i, lat;
    always @(posedge clk) begin
        if (!rst_n) begin q_n <= 0; b_i <= 0; lat <= 0; rd_dat_valid <= 0; end
        else begin
            if (rd_req_valid && rd_req_ready) begin q_addr[q_n] = rd_req_addr; q_len[q_n] = rd_req_len; q_n = q_n + 1; end
            if (rd_dat_valid && rd_dat_ready) begin
                rd_dat_valid <= 0;
                b_i = b_i + 8;
                if (b_i >= q_len[0]) begin
                    for (k = 0; k < 15; k = k + 1) begin q_addr[k] = q_addr[k + 1]; q_len[k] = q_len[k + 1]; end
                    q_n = q_n - 1; b_i = 0; lat = 3 + ($urandom % 6);
                end
            end else if (!rd_dat_valid && q_n > 0) begin
                if (lat > 0) lat = lat - 1;
                else if ($urandom % 4 != 0) begin
                    for (k = 0; k < 8; k = k + 1) rd_dat[32*k +: 32] <= mem[((q_addr[0] >> 2) + b_i + k) % MW];
                    rd_dat_valid <= 1;
                end
            end
        end
    end
    always @(negedge clk) begin rd_req_ready = ($urandom % 3 != 0); wr_ready = ($urandom % 4 != 0); end
    // writes (DMA) + write channel idle of unit 0
    integer w_quiet;
    always @(posedge clk) begin
        if (!rst_n) w_quiet <= 0;
        else if (wr_valid && wr_ready) begin
            w_quiet <= 4 + ($urandom % 4);
            if (!nowrite) for (k = 0; k < 8; k = k + 1) if (wr_mask[k]) mem[((wr_addr >> 2) + k) % MW] <= wr_data[32*k +: 32];
        end else if (w_quiet > 0) w_quiet <= w_quiet - 1;
    end
    // ---------------------------------------------------------------- stub engines (1 GEMM, 2 VEC, 3 LOWA) + merge
    integer dur [1:3], drain [1:3], mdur;
    reg     run [1:3];
    reg     mrun;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (k = 1; k <= 3; k = k + 1) begin run[k] <= 0; drain[k] <= 0; end
            gm_done <= 0; vu_done <= 0; lw_done <= 0; mg_done <= 0; mrun <= 0; wr_idle <= 4'hF;
        end else begin
            gm_done <= 0; vu_done <= 0; lw_done <= 0; mg_done <= 0;
            wr_idle[0] <= (w_quiet == 0) && !wr_valid;
            for (k = 1; k <= 3; k = k + 1) begin
                if ((k == 1 && gm_start) || (k == 2 && vu_start) || (k == 3 && lw_start)) begin
                    if (run[k]) begin $display("FAIL ct_seq start while unit %0d busy", k); $finish; end
                    run[k] <= 1; dur[k] = 1 + ($urandom % 40); wr_idle[k] <= 0;
                    $fwrite(fd, "S %0d %0d", k, cyc);
                    if (k == 1) for (i = 0; i < GP; i = i + 1) $fwrite(fd, " %08x", gm_p[32*i +: 32]);
                    if (k == 2) for (i = 0; i < VP; i = i + 1) $fwrite(fd, " %08x", vu_p[32*i +: 32]);
                    if (k == 3) for (i = 0; i < LP; i = i + 1) $fwrite(fd, " %08x", lw_p[32*i +: 32]);
                    $fwrite(fd, "\n");
                end else if (run[k]) begin
                    if (dur[k] > 0) dur[k] = dur[k] - 1;
                    else begin
                        run[k] <= 0; drain[k] <= $urandom % 8;
                        if (k == 1) gm_done <= 1; else if (k == 2) vu_done <= 1; else lw_done <= 1;
                        $fwrite(fd, "D %0d %0d\n", k, cyc);
                    end
                end else if (!wr_idle[k] && !run[k]) begin
                    if (drain[k] > 0) drain[k] <= drain[k] - 1;
                    else begin wr_idle[k] <= 1; $fwrite(fd, "I %0d %0d\n", k, cyc + 1); end
                end
            end
            if (mg_start) begin mrun <= 1; mdur = 1 + ($urandom % 60); $fwrite(fd, "M %0d\n", cyc); end
            else if (mrun) begin
                if (mdur > 0) mdur = mdur - 1;
                else begin mrun <= 0; mg_done <= 1; $fwrite(fd, "N %0d\n", cyc); end
            end
        end
    end
    // DMA start / retire and all retires (busy falling)
    reg [3:0] busy_q;
    always @(posedge clk) begin
        busy_q <= unit_busy;
        if (rst_n) begin
            if (dut.dma_start) begin
                $fwrite(fd, "S 0 %0d", cyc);
                for (i = 0; i < 9; i = i + 1) $fwrite(fd, " %08x", dut.dma_p[32*i +: 32]);
                $fwrite(fd, "\n");
            end
            if (dut.dma_done) $fwrite(fd, "D 0 %0d\n", cyc);
            for (k = 0; k < 4; k = k + 1) if (busy_q[k] && !unit_busy[k]) $fwrite(fd, "R %0d %0d %0d\n", k, cyc, dut.flags);
        end
    end
    task csr_w(input [11:0] a, input [31:0] d);
        begin @(negedge clk); csr_we = 1; csr_addr = a; csr_wdata = d; @(negedge clk); csr_we = 0; end
    endtask
    task csr_r(input [11:0] a, output [31:0] d);
        begin @(negedge clk); csr_re = 1; csr_addr = a; @(negedge clk); csr_re = 0; d = csr_rdata; end
    endtask
    reg [31:0] v, st_t, sl, sd, sd2;
    initial begin
        for (i = 0; i < MW; i = i + 1) mem[i] = 0;
        if (!$value$plusargs("prog=%s", fname)) begin $display("FAIL ct_seq no +prog"); $finish; end
        $readmemh(fname, mem);
        if (!$value$plusargs("log=%s", fname)) fname = "ct.log";
        fd = $fopen(fname, "w");
        if (!$value$plusargs("maxcyc=%d", maxcyc)) maxcyc = 2000000;
        if (!$value$plusargs("seed=%d", rs)) rs = 1;
        v = $urandom(rs);
        nowrite = $test$plusargs("nowrite");
        if (!$value$plusargs("step_t=%d", st_t)) st_t = 0;
        if (!$value$plusargs("slot=%d", sl)) sl = 0;
        if (!$value$plusargs("slot_delta=%d", sd)) sd = 0;
        if (!$value$plusargs("dlo=%d", dlo)) dlo = 0;
        if (!$value$plusargs("dn=%d", dn)) dn = 0;
        cyc = 0;
        repeat (4) @(negedge clk);
        rst_n = 1;
        csr_r(12'h000, v); $fwrite(fd, "C ID %08x\n", v);
        csr_w(12'h020, 0); csr_w(12'h024, sd); csr_w(12'h018, st_t); csr_w(12'h060, 32'h309); csr_w(12'h064, 0);
        csr_w(12'h008, 32'h11);                                  // run-enable, IRQ_EN
        csr_w(12'h010, 32'h80000000 | sl);
        while (!irq && cyc < maxcyc && !(dut.st == 3'd6)) @(negedge clk);
        csr_r(12'h018, v); $fwrite(fd, "C STEP_T %0d\n", v);
        csr_r(12'h014, v); $fwrite(fd, "C STEP_DONE %0d\n", v);
        csr_r(12'h070, v); $fwrite(fd, "C ERR %0d\n", v);
        csr_r(12'h088, v); $fwrite(fd, "C DESC %0d\n", v);
        csr_r(12'h00C, v); $fwrite(fd, "C STATUS %08x\n", v);
        $fwrite(fd, "C IRQ %0d\n", irq);
        $fwrite(fd, "C MKSTEP %0d\n", mk_step);
        csr_w(12'h014, 0);
        @(negedge clk); $fwrite(fd, "C IRQ_AFTER_ACK %0d\n", irq);
        for (i = 0; i < dn; i = i + 1) $fwrite(fd, "W %0d %08x\n", dlo + i, mem[dlo + i]);
        $fwrite(fd, "C CYCLES %0d\n", cyc);
        $fclose(fd);
        $display("DONE ct_seq %0d cycles", cyc);
        $finish;
    end
    always @(posedge clk) cyc <= cyc + 1;
endmodule
