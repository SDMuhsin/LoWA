`timescale 1ns/1ps
// tb_gm_pe: streams chunks from +vec=<file> (tb/gemm/gen_pe.py, expected partials from L1 fma
// chains) through one gm_pe exactly as the array does: a held for LAMBDA cycles per k, b_q per cycle,
// ctl = {valid, first, last, q}; chunks follow each other with a random gap (+gap=<max>, default 0:
// back to back, the hardest case).  After the last k of a chunk the result for q is captured
// FMA_STAGES + 1 cycles after the last op entered; the TB injects it into the drain chain the next
// cycle and compares dr_out one cycle later bit for bit.  err_overrun must stay 0.
// Final line: PASS gm_pe <n> vectors | FAIL gm_pe <m>/<n> mismatches.
module tb_gm_pe;
    parameter FMA_STAGES = 2;
    parameter LAMBDA = 4;
    parameter QW = 2;
    localparam CW = QW + 3;
    reg clk = 1'b0, rst_n = 1'b0;
    always #20 clk = ~clk;
    reg  [31:0]   a_in = 0, b_in = 0, dr_in = 32'd0;
    reg  [CW-1:0] ctl_in = 0;
    reg           inj = 1'b0;
    reg  [QW-1:0] inj_q = 0;
    wire [31:0]   a_out, b_out, dr_out;
    wire [CW-1:0] ctl_out;
    wire          err;
    gm_pe #(.FMA_STAGES(FMA_STAGES), .LAMBDA(LAMBDA), .QW(QW)) dut (
        .clk(clk), .rst_n(rst_n), .a_in(a_in), .b_in(b_in), .ctl_in(ctl_in),
        .a_out(a_out), .b_out(b_out), .ctl_out(ctl_out),
        .inj(inj), .inj_q(inj_q), .dr_in(dr_in), .dr_out(dr_out), .err_overrun(err));

    // expected results: queue of (q, value, capture cycle)
    reg [31:0] q_val [0:255];
    reg [7:0]  q_q   [0:255];
    integer    q_t   [0:255];
    integer wp = 0, rp = 0, n_bad = 0, n_vec = 0, cyc = 0, fd, rc, K, k, q, g, gapmax = 0;
    reg [31:0] va, vb [0:15], vp [0:15];
    reg [1023:0] fname;
    // check queue: (q, value, cycle at which dr_out must show it)
    reg [31:0] c_val [0:255];
    integer    c_q   [0:255];
    integer    c_t   [0:255];
    integer    cw = 0, cr = 0;

    always @(posedge clk) cyc <= cyc + 1;

    // drive at negedge; injection / check process
    always @(negedge clk) begin
        inj <= 1'b0;
        // check: the value injected at the posedge after cycle t is dr_out during cycle t + 1
        if (cr < cw && cyc == c_t[cr[7:0]]) begin
            if (dr_out !== c_val[cr[7:0]]) begin
                if (n_bad < 10) $display("MISMATCH vec %0d q=%0d got %h want %h", cr, c_q[cr[7:0]], dr_out, c_val[cr[7:0]]);
                n_bad = n_bad + 1;
            end
            cr = cr + 1;
        end
        // inject the oldest captured result (its capture cycle has passed)
        if (rp < wp && cyc > q_t[rp[7:0]]) begin
            inj <= 1'b1; inj_q <= q_q[rp[7:0]][QW-1:0];
            c_q[cw[7:0]] = q_q[rp[7:0]]; c_val[cw[7:0]] = q_val[rp[7:0]]; c_t[cw[7:0]] = cyc + 1;
            cw = cw + 1;
            rp = rp + 1;
        end
    end

    task stream_chunk;
        integer kk, qq, t0;
        begin
            for (kk = 0; kk < K; kk = kk + 1) begin
                rc = $fscanf(fd, "%h", va);
                for (qq = 0; qq < LAMBDA; qq = qq + 1) rc = $fscanf(fd, "%h", vb[qq]);
                for (qq = 0; qq < LAMBDA; qq = qq + 1) begin
                    @(negedge clk);
                    a_in = va; b_in = vb[qq];
                    ctl_in = {1'b1, (kk == 0), (kk == K - 1), qq[QW-1:0]};
                    if (kk == K - 1) begin
                        // op enters the hop register at the end of this cycle (cycle `cyc`),
                        // FMA input in cyc+1, y in cyc+1+FMA_STAGES, captured at the end of it
                        q_q[wp[7:0]] = qq; q_t[wp[7:0]] = cyc + 1 + FMA_STAGES;
                        wp = wp + 1;
                    end
                end
            end
            for (qq = 0; qq < LAMBDA; qq = qq + 1) begin
                rc = $fscanf(fd, "%h", vp[qq]);
                q_val[(wp - LAMBDA + qq) & 255] = vp[qq];
            end
            n_vec = n_vec + LAMBDA;
        end
    endtask

    initial begin
        if (!$value$plusargs("vec=%s", fname)) begin $display("FAIL gm_pe no +vec"); $finish; end
        if ($value$plusargs("gap=%d", gapmax)) ;
        fd = $fopen(fname, "r");
        if (fd == 0) begin $display("FAIL gm_pe cannot open vectors"); $finish; end
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        while (!$feof(fd)) begin
            rc = $fscanf(fd, "%h\n", K);
            if (rc == 1) begin
                stream_chunk;
                if (gapmax > 0) begin
                    g = $urandom % (gapmax + 1);
                    repeat (g) begin @(negedge clk); ctl_in = 0; end
                end
                while (wp - rp > 200) @(negedge clk);
            end
        end
        @(negedge clk); ctl_in = 0;
        repeat (FMA_STAGES + 2 * LAMBDA + 8) @(negedge clk);
        while (rp < wp || cr < cw) @(negedge clk);
        repeat (4) @(negedge clk);
        if (err) begin $display("OVERRUN flagged"); n_bad = n_bad + 1; end
        if (n_vec == 0) $display("FAIL gm_pe 0/0 no vectors read");
        else if (n_bad == 0) $display("PASS gm_pe %0d vectors", n_vec);
        else $display("FAIL gm_pe %0d/%0d mismatches", n_bad, n_vec);
        $finish;
    end
endmodule
