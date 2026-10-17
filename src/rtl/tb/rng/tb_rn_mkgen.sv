// tb_rn_mkgen: hw/rng/rn_mkgen on its mk channel vs model/rng.py keep_mask (streams from tb/rng/gen_rng.py mk).
//   +vec=<file> lines "idx keep"   +mkseed= +mkstep= +mksite= +mklayer= (hex, the stream configuration)
//   +gap=<pct> random request gaps   +maxcyc=<n> watchdog
// Requests are held (valid, idx) until ready, like the engines' mk ports; responses must come back in order,
// one per request, MKW bits each.  Reports cycles per request (throughput) and the request -> response latency.
`timescale 1ns/1ps
module tb_rn_mkgen #(parameter MKW = 16, parameter NCORE = 9, parameter STAGES = 4, parameter NV = 400000);
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    reg  [31:0]    vi [0:NV-1];
    reg  [MKW-1:0] vk [0:NV-1];
    reg  [63:0]    seed = 0;
    reg  [23:0]    step = 0;
    reg  [3:0]     site = 0, layer = 0;
    integer n = 0, ip = 0, op = 0, bad = 0, gap = 0, maxcyc = 50000000, cyc = 0, t0 = -1, lat = -1, infl = 0;
    reg            req_valid = 0;
    reg  [31:0]    req_idx = 0;
    wire           req_ready, rsp_valid, idle;
    wire [MKW-1:0] rsp_keep;
    rn_mkgen #(.MKW(MKW), .NCORE(NCORE), .STAGES(STAGES)) dut (.clk(clk), .rst_n(rst_n), .seed(seed), .step(step),
        .site(site), .layer(layer), .mk_req_valid(req_valid), .mk_req_ready(req_ready), .mk_req_idx(req_idx),
        .mk_rsp_valid(rsp_valid), .mk_rsp_keep(rsp_keep), .idle(idle));
    reg [1023:0] fv;
    integer fd, rc;
    initial begin
        if (!$value$plusargs("vec=%s", fv)) begin $display("FAIL rn_mkgen no +vec"); $finish; end
        if ($value$plusargs("mkseed=%h", seed)) ;
        if ($value$plusargs("mkstep=%h", step)) ;
        if ($value$plusargs("mksite=%h", site)) ;
        if ($value$plusargs("mklayer=%h", layer)) ;
        if ($value$plusargs("gap=%d", gap)) ;
        if ($value$plusargs("maxcyc=%d", maxcyc)) ;
        fd = $fopen(fv, "r");
        while (n < NV && !$feof(fd)) begin
            rc = $fscanf(fd, "%h %h\n", vi[n], vk[n]);
            if (rc == 2) n = n + 1;
        end
        $fclose(fd);
        repeat (3) @(posedge clk);
        rst_n <= 1;
    end
    always @(posedge clk) begin
        cyc <= cyc + 1;
        if (rst_n) begin
            if (req_valid && req_ready) begin
                if (ip == 1 && lat < 0) t0 = cyc;
                infl = infl + 1;
            end
            if (!req_valid || req_ready) begin
                if (ip < n && ($urandom % 100) >= gap) begin
                    req_valid <= 1; req_idx <= vi[ip]; ip <= ip + 1;
                end else req_valid <= 0;
            end
            if (rsp_valid) begin
                if (op == 0) lat = cyc - t0;
                if (infl == 0 || op >= n || rsp_keep !== vk[op]) begin
                    bad = bad + 1;
                    if (bad <= 5) $display("MISMATCH %0d: idx %h got %h want %h", op, vi[op], rsp_keep, vk[op]);
                end
                op = op + 1; infl = infl - 1;
            end
            if (op == n && n > 0) begin
                if (!idle) begin end
                else begin
                    $display("CYCLES %0d requests %0d latency %0d", cyc, n, lat);
                    if (bad == 0) $display("PASS rn_mkgen %0d vectors (MKW %0d NCORE %0d STAGES %0d)", n, MKW, NCORE, STAGES);
                    else          $display("FAIL rn_mkgen %0d/%0d mismatches", bad, n);
                    $finish;
                end
            end
            if (cyc > maxcyc) begin $display("FAIL rn_mkgen watchdog at %0d/%0d", op, n); $finish; end
        end
    end
endmodule
