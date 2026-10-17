`timescale 1ns / 1ps
// tb_mm_sram_bank: mm_sram_bank port contract on the selected backend (+define+MM_TECH=<n>), against a reference
// array in this testbench (not RTL).  Random mix of reads / writes / idle cycles (write-heavy bursts, back-to-back
// write -> read of the same word, every word written then read), N ops, fixed seed (+seed).  Checks:
//   the cycle after every read q equals the reference word;
//   HOLD (default on, -DNO_HOLD_CHECK for gf180): q is unchanged by writes and idle cycles until the next read.
// Cycle limit: the op count is fixed (no waiting loops).  Prints PASS mm_sram_bank <n> vectors / FAIL ...
module tb_mm_sram_bank;
    parameter WORDS = 768, WIDTH = 32, AW = 10, N = 200000;
    reg clk = 0;
    always #5 clk = ~clk;
    reg en = 0, we = 0;
    reg [AW-1:0] addr = 0;
    reg [WIDTH-1:0] d = 0;
    wire [WIDTH-1:0] q;
    mm_sram_bank #(.WORDS(WORDS), .WIDTH(WIDTH), .AW(AW)) dut (.clk(clk), .en(en), .we(we), .addr(addr), .d(d), .q(q));

    reg [WIDTH-1:0] ref_m [0:WORDS-1];
    reg [WIDTH-1:0] exp_q;
    reg             have_q, chk_rd;
    integer i, n, bad, seed, r, mode, last_a;
    task op(input e, input w, input [AW-1:0] a, input [WIDTH-1:0] v);
        begin
            en <= e; we <= w; addr <= a; d <= v;
            @(posedge clk); #1;
            // the access was sampled at this edge: update the reference
            if (e && w) ref_m[a] = v;
            if (e && !w) begin exp_q = ref_m[a]; have_q = 1; end
            chk_rd = e && !w;
            n = n + 1;
            if (have_q) begin
`ifdef NO_HOLD_CHECK
                if (chk_rd && q !== exp_q) bad = bad + 1;
`else
                if (q !== exp_q) bad = bad + 1;
`endif
                if (bad == 1 && q !== exp_q) begin
                    $display("first mismatch op %0d addr %0d q %h exp %h", n, a, q, exp_q); bad = bad + 1;
                end
            end
        end
    endtask
    function [WIDTH-1:0] rnd(input integer dummy);        // WIDTH <= 64
        reg [63:0] t;
        begin
            t[31:0] = $random(seed); t[63:32] = $random(seed);
            rnd = t[WIDTH-1:0];
        end
    endfunction
    initial begin
        seed = 777; bad = 0; n = 0; have_q = 0; last_a = 0;
        if ($value$plusargs("seed=%d", r)) seed = r;
        @(posedge clk); #1;
        for (i = 0; i < WORDS; i = i + 1) op(1, 1, i, rnd(0));             // fill
        for (i = 0; i < WORDS; i = i + 1) op(1, 0, i, rnd(0));             // read all (d is don't-care)
        while (n < N) begin
            r = $random(seed); mode = (r >>> 8) & 15;
            r = ((r & 32'h7fffffff) % WORDS);
            case (mode)
                0, 1, 2, 3: op(1, 1, r, rnd(0));                           // write
                4, 5, 6, 7: op(1, 0, r, rnd(0));                           // read
                8:          begin op(1, 1, r, rnd(0)); op(1, 0, r, rnd(0)); end   // write then read same word
                9:          begin op(1, 0, r, rnd(0)); op(1, 1, r, rnd(0)); op(0, 0, r, rnd(0)); op(1, 0, r, rnd(0)); end
                10:         begin op(1, 1, last_a, rnd(0)); op(1, 1, last_a, rnd(0)); op(1, 0, last_a, rnd(0)); end
                11:         op(0, 1, r, rnd(0));                           // idle with we high: no write
                default:    op(0, 0, r, rnd(0));                           // idle
            endcase
            last_a = r;
        end
        if (bad == 0) $display("PASS mm_sram_bank %0d vectors", n);
        else $display("FAIL mm_sram_bank %0d/%0d mismatches", bad - 1, n);
        $finish;
    end
endmodule
