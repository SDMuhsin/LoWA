`timescale 1ns/1ps
// tb_fp_mul: streams vectors from +vec=<file> (tb/fp/gen_vectors.py, expected values from the L1
// entry points) into fp_mul one per cycle and compares every output bit for bit (!==), in order,
// exactly LATENCY = STAGES cycles later (checked).  Also checks that exactly one output appears per input.
// Final line: PASS mul <n> vectors | FAIL mul <bad>/<n> mismatches.
module tb_fp_mul;
    parameter STAGES = 1;
    reg clk = 1'b0, rst_n = 1'b0;
    always #20 clk = ~clk;
    reg         in_valid = 1'b0;
    reg  [31:0] a;
    reg  [31:0] b;
    wire        out_valid;
    wire [31:0] y;
`ifdef GLS
    fp_mul dut (                          // synthesised netlist (gls.v)
`else
    fp_mul #(.STAGES(STAGES)) dut (
`endif
       .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .a(a), .b(b),
        .out_valid(out_valid), .y(y));

    reg  [31:0] v_a;
    reg  [31:0] v_b;
    reg  [31:0] v_y;
    reg  [31:0] q_a [0:63];
    reg  [31:0] q_b [0:63];
    reg  [31:0] q_y [0:63];
    integer q_t [0:63];
    integer wp = 0, rp = 0, n_bad = 0, fd, rc, cyc = 0, last_in = 0;
    reg [1023:0] fname;
    reg done = 1'b0;

    always @(posedge clk) begin
        cyc <= cyc + 1;
        if (rst_n && out_valid) begin
            if (rp >= wp) begin
                n_bad = n_bad + 1;
                $display("EXTRA output at cycle %0d", cyc);
            end else begin
                if ((y !== q_y[rp[5:0]])) begin
                    if (n_bad < 10)
                        $display("MISMATCH #%0d a=%h b=%h got y=%h(want %h)", rp, q_a[rp[5:0]], q_b[rp[5:0]], y, q_y[rp[5:0]]);
                    n_bad = n_bad + 1;
                end
                if (cyc - q_t[rp[5:0]] != STAGES) begin
                    if (n_bad < 10) $display("LATENCY #%0d: %0d cycles, want %0d", rp, cyc - q_t[rp[5:0]], STAGES);
                    n_bad = n_bad + 1;
                end
                rp = rp + 1;
            end
        end
    end

    initial begin
        if (!$value$plusargs("vec=%s", fname)) begin
            $display("FAIL mul no +vec=<file>");
            $finish;
        end
        fd = $fopen(fname, "r");
        if (fd == 0) begin
            $display("FAIL mul cannot open vectors");
            $finish;
        end
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        while (!$feof(fd)) begin
            rc = $fscanf(fd, "%h %h %h\n", v_a, v_b, v_y);
            if (rc == 3) begin
                @(negedge clk);
                if (wp - rp > 60) begin
                    $display("FAIL mul output queue overflow (no outputs?)");
                    $finish;
                end
                in_valid = 1'b1; a = v_a; b = v_b;
                q_a[wp[5:0]] = v_a; q_b[wp[5:0]] = v_b; q_y[wp[5:0]] = v_y;
                q_t[wp[5:0]] = cyc;
                wp = wp + 1;
                last_in = cyc;
            end
        end
        @(negedge clk);
        in_valid = 1'b0;
        repeat (STAGES + 4) @(negedge clk);
        if (rp != wp) begin
            $display("MISSING outputs: %0d of %0d", wp - rp, wp);
            n_bad = n_bad + (wp - rp);
        end
        if (wp == 0) $display("FAIL mul 0/0 no vectors read");
        else if (n_bad == 0) $display("PASS mul %0d vectors", wp);
        else $display("FAIL mul %0d/%0d mismatches", n_bad, wp);
        $finish;
    end
endmodule
