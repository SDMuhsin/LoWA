// tb_rn_tf: hw/rng/rn_tf (+ rn_ks) vs model/rng.py threefry2x32_13 (vectors from tb/rng/gen_rng.py tf).
//   +vec=<file>  lines "c0 c1 k0 k1 x0 x1 keep"   +gap=<pct> random idle cycles   +maxcyc=<n> watchdog
// One block per cycle; the key words are held while blocks of the previous key are in flight (the tb
// drains the pipeline before a key change, as rn_mkgen holds a key per stream).  c1 is split as
// {step, site, layer} into rn_ks, so the counter word formation is part of the check.
`timescale 1ns/1ps
module tb_rn_tf #(parameter STAGES = 4, parameter NV = 1100000);
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    reg  [31:0] vc0 [0:NV-1], vc1 [0:NV-1], vk0 [0:NV-1], vk1 [0:NV-1], vx0 [0:NV-1], vx1 [0:NV-1];
    reg  [1:0]  vkp [0:NV-1];
    integer n = 0, ip = 0, op = 0, bad = 0, infl = 0, gap = 0, maxcyc = 50000000, cyc = 0;
    reg         in_valid = 0;
    reg  [31:0] a = 0, c1 = 0, k0 = 0, k1 = 0;
    wire [31:0] k0w, x1i, i1a, i1b, i2a, i2b, i3a, i3b, ox0, ox1;
    wire [1:0]  keep;
    wire        out_valid;
    rn_ks u_ks (.seed({k1, k0}), .step(c1[31:8]), .site(c1[7:4]), .layer(c1[3:0]), .k0(k0w), .x1i(x1i),
        .i1a(i1a), .i1b(i1b), .i2a(i2a), .i2b(i2b), .i3a(i3a), .i3b(i3b));
    rn_tf #(.STAGES(STAGES)) dut (.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .a(a), .b(k0w), .x1i(x1i),
        .i1a(i1a), .i1b(i1b), .i2a(i2a), .i2b(i2b), .i3a(i3a), .i3b(i3b),
        .out_valid(out_valid), .out_x0(ox0), .out_x1(ox1), .keep(keep));
    reg [1023:0] fv;
    integer fd, rc;
    initial begin
        if (!$value$plusargs("vec=%s", fv)) begin $display("FAIL rn_tf no +vec"); $finish; end
        if ($value$plusargs("gap=%d", gap)) ;
        if ($value$plusargs("maxcyc=%d", maxcyc)) ;
        fd = $fopen(fv, "r");
        while (n < NV && !$feof(fd)) begin
            rc = $fscanf(fd, "%h %h %h %h %h %h %h\n", vc0[n], vc1[n], vk0[n], vk1[n], vx0[n], vx1[n], vkp[n]);
            if (rc == 7) n = n + 1;
        end
        $fclose(fd);
        repeat (3) @(posedge clk);
        rst_n <= 1;
    end
    always @(posedge clk) begin
        cyc <= cyc + 1;
        if (rst_n) begin
            // drive
            if (ip < n && ($urandom % 100) >= gap &&
                ((infl == 0 && !in_valid) || (vk0[ip] == k0 && vk1[ip] == k1))) begin
                in_valid <= 1; a <= vc0[ip]; c1 <= vc1[ip]; k0 <= vk0[ip]; k1 <= vk1[ip]; ip <= ip + 1;
            end else in_valid <= 0;
            infl <= infl + (in_valid ? 1 : 0) - (out_valid ? 1 : 0);
            // check
            if (out_valid) begin
                if (op >= n || ox0 !== vx0[op] || ox1 !== vx1[op] || keep !== vkp[op]) begin
                    bad = bad + 1;
                    if (bad <= 5) $display("MISMATCH %0d: c %h %h k %h %h got %h %h %b want %h %h %b", op, vc0[op], vc1[op],
                                           vk0[op], vk1[op], ox0, ox1, keep, vx0[op], vx1[op], vkp[op]);
                end
                op = op + 1;
            end
            if (op == n && n > 0) begin
                if (bad == 0) $display("PASS rn_tf %0d vectors (STAGES %0d)", n, STAGES);
                else          $display("FAIL rn_tf %0d/%0d mismatches", bad, n);
                $finish;
            end
            if (cyc > maxcyc) begin $display("FAIL rn_tf watchdog at %0d/%0d", op, n); $finish; end
        end
    end
endmodule
