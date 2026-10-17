`timescale 1ns/1ps
// tb_fp_sfu: streams "fn x y" vectors from +vec=<file> (tb/fp/gen_vectors.py; y from the L1 entry
// points exp/recip/rsqrt/sqrt/tanh/erf/log) into fp_sfu, honouring in_ready, and checks every
// result bit for bit (!==) by tag, plus the fixed latency (N(fn) + 1) * (LF + 2) of each element.
// +gap=<0..15>: probability/16 of an idle input cycle before a vector (exercises free-slot accept).
// Final line: PASS sfu <n> vectors | FAIL sfu <bad>/<n> mismatches.
module tb_fp_sfu;
    parameter LF   = 2;
    parameter NTHR = LF + 2;
`ifdef GLS
    localparam TAGW = 4;                  // the gate-level netlist is synthesised with TAGW = 4
`else
    localparam TAGW = 8;
`endif
    localparam NTAG = 1 << TAGW;
    localparam T = LF + 2;
    reg clk = 1'b0, rst_n = 1'b0;
    always #20 clk = ~clk;

    reg             in_valid = 1'b0;
    reg  [2:0]      in_fn = 3'd0;
    reg  [31:0]     in_x = 32'd0;
    reg  [TAGW-1:0] in_tag = {TAGW{1'b0}};
    wire            in_ready, out_valid;
    wire [31:0]     out_y;
    wire [TAGW-1:0] out_tag;
`ifdef GLS
    fp_sfu dut (                          // synthesised netlist (flow/synth/synth.sh, gls.v)
`else
    fp_sfu #(.LF(LF), .NTHR(NTHR), .TAGW(TAGW)) dut (
`endif
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_ready(in_ready), .in_fn(in_fn),
        .in_x(in_x), .in_tag(in_tag), .out_valid(out_valid), .out_y(out_y), .out_tag(out_tag));

    reg  [31:0] e_y   [0:255];
    reg  [31:0] e_x   [0:255];
    reg  [2:0]  e_fn  [0:255];
    integer     e_t   [0:255];
    reg         pend  [0:255];
    integer     n_in = 0, n_out = 0, n_bad = 0, cyc = 0, fd, rc, gap = 0, i, last_out = 0, last_acc = 0;
    reg  [31:0] v_fn, v_x, v_y;
    reg  [15:0] lfsr = 16'hACE1;
    reg  [1023:0] fname;

    function integer nsteps(input [2:0] f);
        case (f)
            3'd0: nsteps = 13;  3'd1: nsteps = 9;   3'd2: nsteps = 13;  3'd3: nsteps = 17;
            3'd4: nsteps = 25;  3'd5: nsteps = 24;  3'd6: nsteps = 13;  default: nsteps = 1;
        endcase
    endfunction

    always @(posedge clk) begin
        cyc <= cyc + 1;
        if (n_in > n_out && cyc - last_out > 30 * T && cyc - last_acc > 30 * T) begin
            $display("FAIL sfu watchdog: no output and no input for %0d cycles (%0d of %0d done)", 30 * T, n_out, n_in);
            $finish;
        end
        if (rst_n && out_valid) begin
            if (!pend[out_tag]) begin
                n_bad = n_bad + 1;
                $display("UNEXPECTED output tag %0d at cycle %0d", out_tag, cyc);
            end else begin
                if (out_y !== e_y[out_tag]) begin
                    if (n_bad < 10)
                        $display("MISMATCH fn=%0d x=%h got %h want %h", e_fn[out_tag], e_x[out_tag],
                                 out_y, e_y[out_tag]);
                    n_bad = n_bad + 1;
                end
                if (cyc - e_t[out_tag] != (nsteps(e_fn[out_tag]) + 1) * T) begin
                    if (n_bad < 10)
                        $display("LATENCY fn=%0d: %0d cycles, want %0d", e_fn[out_tag],
                                 cyc - e_t[out_tag], (nsteps(e_fn[out_tag]) + 1) * T);
                    n_bad = n_bad + 1;
                end
                pend[out_tag] = 1'b0;
                n_out = n_out + 1;
                last_out = cyc;
            end
        end
    end

    initial begin
        for (i = 0; i < 256; i = i + 1) pend[i] = 1'b0;
        if (!$value$plusargs("vec=%s", fname)) begin
            $display("FAIL sfu no +vec=<file>");
            $finish;
        end
        if (!$value$plusargs("gap=%d", gap)) gap = 0;
        fd = $fopen(fname, "r");
        if (fd == 0) begin
            $display("FAIL sfu cannot open vectors");
            $finish;
        end
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        while (!$feof(fd)) begin
            rc = $fscanf(fd, "%h %h %h\n", v_fn, v_x, v_y);
            if (rc == 3) begin
                @(negedge clk);
                lfsr = {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
                if ({28'd0, lfsr[3:0]} < gap) begin
                    in_valid = 1'b0;
                    @(negedge clk);
                end
                while (pend[n_in % NTAG]) @(negedge clk);
                in_valid = 1'b1; in_fn = v_fn[2:0]; in_x = v_x; in_tag = n_in % NTAG;
                while (!in_ready) @(negedge clk);
                // accepted at the coming posedge
                e_y[n_in % NTAG] = v_y; e_x[n_in % NTAG] = v_x; e_fn[n_in % NTAG] = v_fn[2:0];
                e_t[n_in % NTAG] = cyc; pend[n_in % NTAG] = 1'b1; last_acc = cyc;
                n_in = n_in + 1;
            end
        end
        @(negedge clk);
        in_valid = 1'b0;
        repeat (30 * T) @(negedge clk);
        if (n_out != n_in) begin
            $display("MISSING outputs: %0d of %0d", n_in - n_out, n_in);
            n_bad = n_bad + (n_in - n_out);
        end
        if (n_in == 0) $display("FAIL sfu 0/0 no vectors read");
        else if (n_bad == 0) $display("PASS sfu %0d vectors", n_in);
        else $display("FAIL sfu %0d/%0d mismatches", n_bad, n_in);
        $finish;
    end
endmodule
