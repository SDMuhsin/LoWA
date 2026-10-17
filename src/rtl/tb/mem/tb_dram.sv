`timescale 1ns/1ps
// tb_dram: AXI4 slave DRAM model for simulation (test model, NOT synthesizable RTL; PLAN ruling R3:
// the DRAM controller / PHY and the host load path are off-die and only modelled here).
// Memory: a sparse image of up to 128 regions (memory_map.md layout) packed into one 2^MEMW-word array;
// region file lines "base_word nwords" (in packing order), data file = $readmemh of the packed words
// (both written by tb/mem/mm_image.py).  An access outside every region answers SLVERR and counts an
// error.  Timing knobs (set by the instantiating tb before reset release):
//   lat       cycles from AR acceptance to the first R beat           wlat  cycles from WLAST to B
//   bw        bandwidth cap in 1/100 byte per cycle, shared by R and W (3200 = 32 B/cycle = 800 MB/s
//             at 25 MHz); a beat costs 4 * AXW * 100 tokens, the bucket holds two beats
//   stall     percent of cycles with ARREADY / AWREADY / WREADY low and no R / B beat (backpressure)
//   reorder   1: R bursts and B responses of different IDs are served in random order (per ID in order)
//   ilv       1: R beats of different IDs interleave at random (AXI4 allows it)
//   wfair     0 (nominal): R has priority, a W beat needs a spare token while R streams; 1: W and R share tokens
// Protocol checks (each counted in n_err): INCR only, full-size beats, size-aligned address, <= 16
// beats, no 4 KiB crossing, WLAST on the last beat only, W only after its AW.
module tb_dram #(
    parameter AXL  = 3,
    parameter MEMW = 20,
    parameter AQ   = 32                      // AR / AW / B queue entries
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  arvalid,
    output reg                   arready,
    input  wire [3:0]            arid,
    input  wire [31:0]           araddr,
    input  wire [7:0]            arlen,
    input  wire [2:0]            arsize,
    input  wire [1:0]            arburst,
    output reg                   rvalid,
    input  wire                  rready,
    output reg  [3:0]            rid,
    output reg  [32*(1<<AXL)-1:0] rdata,
    output reg  [1:0]            rresp,
    output reg                   rlast,
    input  wire                  awvalid,
    output reg                   awready,
    input  wire [3:0]            awid,
    input  wire [31:0]           awaddr,
    input  wire [7:0]            awlen,
    input  wire [2:0]            awsize,
    input  wire [1:0]            awburst,
    input  wire                  wvalid,
    output reg                   wready,
    input  wire [32*(1<<AXL)-1:0] wdata,
    input  wire [4*(1<<AXL)-1:0] wstrb,
    input  wire                  wlast,
    output reg                   bvalid,
    input  wire                  bready,
    output reg  [3:0]            bid,
    output reg  [1:0]            bresp
);
    localparam AXW = 1 << AXL;
    localparam BEAT_TOK = 4 * AXW * 100;

    integer lat = 24, wlat = 8, bw = 3200, stall = 0, reorder = 0, ilv = 0;
    integer wfair = 0;                        // 0: R has priority (W only with a spare token); 1: W and R share tokens
    integer cyc = 0, n_err = 0;
    // statistics
    integer rd_beats = 0, wr_beats = 0, ar_n = 0, aw_n = 0, max_rd_out = 0, max_wr_out = 0, rd_out = 0, wr_out = 0;
    integer first_cyc = -1, last_cyc = 0;
    integer idle_none = 0, idle_lat = 0;      // R-slot use: cycles with tokens but no AR queued / none past latency
    integer id_beats [0:15];                  // R beats per ARID (separates the keep-bit stand-in traffic, R7)
    initial for (int q = 0; q < 16; q++) id_beats[q] = 0;

    // ---------------------------------------------------------------- memory + regions
    reg  [31:0] mem [0:(1<<MEMW)-1];
    reg  [31:0] rg_base [0:127];
    reg  [31:0] rg_n    [0:127];
    reg  [31:0] rg_off  [0:127];
    integer     n_rg = 0;
    task load(input [1023:0] freg, input [1023:0] fdat);
        integer fd, rc; reg [31:0] b, n, off;
        begin
            fd = $fopen(freg, "r");
            if (fd == 0) begin $display("DRAM cannot open region file"); $finish; end
            off = 0;
            rc = $fscanf(fd, "%h %h\n", b, n);
            while (rc == 2) begin
                rg_base[n_rg] = b; rg_n[n_rg] = n; rg_off[n_rg] = off; off = off + n; n_rg = n_rg + 1;
                rc = $fscanf(fd, "%h %h\n", b, n);
            end
            $fclose(fd);
            if (off > (1 << MEMW)) begin $display("DRAM image %0d words > 2^MEMW", off); $finish; end
            $readmemh(fdat, mem);
        end
    endtask
    // same region file, data as raw big-endian 32-bit words ($fread; P7a: the full-depth image is ~100 M words)
    task load_bin(input [1023:0] freg, input [1023:0] fdat);
        integer fd, rc; reg [31:0] b, n, off;
        begin
            fd = $fopen(freg, "r");
            if (fd == 0) begin $display("DRAM cannot open region file"); $finish; end
            off = 0;
            rc = $fscanf(fd, "%h %h\n", b, n);
            while (rc == 2) begin
                rg_base[n_rg] = b; rg_n[n_rg] = n; rg_off[n_rg] = off; off = off + n; n_rg = n_rg + 1;
                rc = $fscanf(fd, "%h %h\n", b, n);
            end
            $fclose(fd);
            if (off > (1 << MEMW)) begin $display("DRAM image %0d words > 2^MEMW", off); $finish; end
            fd = $fopen(fdat, "rb");
            if (fd == 0) begin $display("DRAM cannot open data file"); $finish; end
            rc = $fread(mem, fd, 0, off);
            $fclose(fd);
            if (rc != 4 * off) begin $display("DRAM data file: %0d bytes read, %0d expected", rc, 4 * off); $finish; end
        end
    endtask
    // word address -> backing index, -1 if unmapped
    function integer xlat(input [31:0] wa);
        integer r;
        begin
            xlat = -1;
            for (r = 0; r < n_rg; r = r + 1)
                if (wa >= rg_base[r] && wa - rg_base[r] < rg_n[r]) xlat = rg_off[r] + (wa - rg_base[r]);
        end
    endfunction
    function [31:0] peek(input [31:0] wa);
        integer x;
        begin
            x = xlat(wa);
            peek = (x < 0) ? 32'hDEADBEEF : mem[x];
        end
    endfunction

    function rnd_stall(input integer dummy);
        begin
            rnd_stall = (stall > 0) && (($urandom % 100) < stall);
        end
    endfunction
    function chk_burst(input [31:0] a, input [7:0] len, input [2:0] size, input [1:0] burst);
        begin
            chk_burst = (burst == 2'b01) && (size == AXL + 2) && (a[AXL+1:0] == 0) && (len <= 15)
                        && (((a & 32'hFFF) + (len + 1) * 4 * AXW) <= 4096);
        end
    endfunction

    // ---------------------------------------------------------------- queues
    reg         q_v   [0:AQ-1];  reg [3:0] q_id [0:AQ-1];  reg [31:0] q_a [0:AQ-1];
    reg [7:0]   q_len [0:AQ-1];  reg [7:0] q_b  [0:AQ-1];  integer q_t [0:AQ-1];  integer q_s [0:AQ-1];
    reg         q_bad [0:AQ-1];
    reg         w_v   [0:AQ-1];  reg [3:0] w_id [0:AQ-1];  reg [31:0] w_a [0:AQ-1];
    reg [7:0]   w_len [0:AQ-1];  reg [7:0] w_b  [0:AQ-1];  integer w_s [0:AQ-1];  reg w_bad [0:AQ-1];
    reg         b_v   [0:AQ-1];  reg [3:0] b_idq [0:AQ-1]; integer b_t [0:AQ-1];  integer b_s [0:AQ-1];
    reg         b_bad [0:AQ-1];
    integer     seq = 0, cur = -1, tok = 0, i, j, k, c, best, nc, x;
    reg         ok, rd_go;

    function integer count_v(input integer which);
        integer n2, m2;
        begin
            n2 = 0;
            for (m2 = 0; m2 < AQ; m2 = m2 + 1)
                if ((which == 0 && q_v[m2]) || (which == 1 && w_v[m2]) || (which == 2 && b_v[m2])) n2 = n2 + 1;
            count_v = n2;
        end
    endfunction
    function integer free_slot(input integer which);
        integer m2;
        begin
            free_slot = -1;
            for (m2 = AQ - 1; m2 >= 0; m2 = m2 - 1)
                if ((which == 0 && !q_v[m2]) || (which == 1 && !w_v[m2]) || (which == 2 && !b_v[m2])) free_slot = m2;
        end
    endfunction

    initial begin
        for (i = 0; i < AQ; i = i + 1) begin q_v[i] = 0; w_v[i] = 0; b_v[i] = 0; end
        arready = 0; awready = 0; wready = 0; rvalid = 0; bvalid = 0; rlast = 0; rresp = 0; bresp = 0;
        rid = 0; bid = 0; rdata = 0;
    end

    integer best_b = 0;
    reg rv_ = 0, rl_ = 0, bv_ = 0, arr_ = 0, awr_ = 0, wr_ = 0;
    reg [3:0] ri_ = 0, bi_ = 0;
    reg [1:0] rr_ = 0, br_ = 0;
    reg [32*(1<<AXL)-1:0] rd_ = 0;
    always @(posedge clk) begin
        cyc <= cyc + 1;
        if (rst_n) begin
            // ------------------------------------------------ AR
            if (arvalid && arr_) begin
                x = free_slot(0);
                ok = chk_burst(araddr, arlen, arsize, arburst);
                if (!ok) begin n_err = n_err + 1; if (n_err < 5) $display("DRAM AR protocol error a %h len %0d size %0d burst %0d", araddr, arlen, arsize, arburst); end
                q_v[x] = 1; q_id[x] = arid; q_a[x] = araddr; q_len[x] = arlen; q_b[x] = 0; q_t[x] = cyc + lat;
                q_s[x] = seq; q_bad[x] = !ok; seq = seq + 1; ar_n = ar_n + 1;
                rd_out = rd_out + 1; if (rd_out > max_rd_out) max_rd_out = rd_out;
                if (first_cyc < 0) first_cyc = cyc;
            end
            // ------------------------------------------------ AW
            if (awvalid && awr_) begin
                x = free_slot(1);
                ok = chk_burst(awaddr, awlen, awsize, awburst);
                if (!ok) begin n_err = n_err + 1; if (n_err < 5) $display("DRAM AW protocol error a %h len %0d", awaddr, awlen); end
                w_v[x] = 1; w_id[x] = awid; w_a[x] = awaddr; w_len[x] = awlen; w_b[x] = 0; w_s[x] = seq; w_bad[x] = !ok;
                seq = seq + 1; aw_n = aw_n + 1;
                wr_out = wr_out + 1; if (wr_out > max_wr_out) max_wr_out = wr_out;
                if (first_cyc < 0) first_cyc = cyc;
            end
            // ------------------------------------------------ W (applies to the oldest AW)
            if (wvalid && wr_) begin
                best = -1;
                for (k = 0; k < AQ; k = k + 1) if (w_v[k] && (best < 0 || w_s[k] < w_s[best])) best = k;
                if (best < 0) begin n_err = n_err + 1; $display("DRAM W without AW"); end
                else begin
                    for (j = 0; j < AXW; j = j + 1) begin
                        x = xlat(w_a[best] / 4 + w_b[best] * AXW + j);
                        if (x >= 0 && !w_bad[best])
                            for (c = 0; c < 4; c = c + 1)
                                if (wstrb[4*j + c]) mem[x][8*c +: 8] = wdata[32*j + 8*c +: 8];
                        if (x < 0 && wstrb[4*j +: 4] != 0) w_bad[best] = 1;
                    end
                    if (wlast != (w_b[best] == w_len[best])) begin n_err = n_err + 1; $display("DRAM WLAST mismatch"); end
                    wr_beats = wr_beats + 1; last_cyc = cyc;
                    if (w_b[best] == w_len[best]) begin
                        x = free_slot(2);
                        b_v[x] = 1; b_idq[x] = w_id[best]; b_t[x] = cyc + wlat; b_s[x] = w_s[best]; b_bad[x] = w_bad[best];
                        if (w_bad[best]) begin n_err = n_err + 1; $display("DRAM write outside the image a %h", w_a[best]); end
                        w_v[best] = 0;
                    end else w_b[best] = w_b[best] + 1;
                    tok = tok - BEAT_TOK;
                end
            end
            // ------------------------------------------------ R
            if (rv_ && rready) begin
                rd_beats = rd_beats + 1; last_cyc = cyc; id_beats[rid] = id_beats[rid] + 1;
                if (rl_) begin q_v[cur] = 0; cur = -1; rd_out = rd_out - 1; end
                else q_b[cur] = q_b[cur] + 1;
            end
            if (rv_ && !rready) ;                          // hold the beat
            else begin
                rv_ = 0;
                rd_go = !rnd_stall(0) && tok >= BEAT_TOK;
                if (rd_go && (cur < 0 || ilv || reorder)) begin
                    // eligible: latency elapsed, oldest of its ID; pick the oldest overall or a random one
                    nc = 0; best = -1;
                    for (k = 0; k < AQ; k = k + 1) if (q_v[k] && q_t[k] <= cyc) begin
                        ok = 1;
                        for (j = 0; j < AQ; j = j + 1) if (q_v[j] && q_id[j] == q_id[k] && q_s[j] < q_s[k]) ok = 0;
                        if (ok) begin
                            nc = nc + 1;
                            if (best < 0) best = k;
                            else if (reorder && ($urandom % nc) == 0) best = k;
                            else if (!reorder && q_s[k] < q_s[best]) best = k;
                        end
                    end
                    // without interleaving a started burst finishes first
                    if (cur >= 0 && !ilv) best = cur;
                    if (best >= 0) cur = best;
                end
                if (rd_go && cur < 0) begin             // a free R slot not used: nothing queued / still in latency
                    if (rd_out == 0) idle_none = idle_none + 1; else idle_lat = idle_lat + 1;
                end
                if (rd_go && cur >= 0) begin
                    rv_ = 1; ri_ = q_id[cur]; rl_ = (q_b[cur] == q_len[cur]); rr_ = 0;
                    for (j = 0; j < AXW; j = j + 1) begin
                        x = xlat(q_a[cur] / 4 + q_b[cur] * AXW + j);
                        rd_[32*j +: 32] = (x < 0) ? 32'hDEADBEEF : mem[x];
                        if (x < 0 || q_bad[cur]) rr_ = 2'b10;
                    end
                    if (rr_ != 0) begin n_err = n_err + 1; if (n_err < 5) $display("DRAM read outside the image a %h", q_a[cur]); end
                    tok = tok - BEAT_TOK;
                end
            end
            // ------------------------------------------------ B
            if (bv_ && bready) begin
                for (k = 0; k < AQ; k = k + 1) if (b_v[k] && b_s[k] == best_b) b_v[k] = 0;
                wr_out = wr_out - 1;
            end
            if (!(bv_ && !bready)) begin
                bv_ = 0;
                if (!rnd_stall(0)) begin
                    best = -1; nc = 0;
                    for (k = 0; k < AQ; k = k + 1) if (b_v[k] && b_t[k] <= cyc) begin
                        ok = 1;
                        for (j = 0; j < AQ; j = j + 1) if (b_v[j] && b_idq[j] == b_idq[k] && b_s[j] < b_s[k]) ok = 0;
                        if (ok) begin
                            nc = nc + 1;
                            if (best < 0 || (!reorder && b_s[k] < b_s[best]) || (reorder && ($urandom % nc) == 0)) best = k;
                        end
                    end
                    if (best >= 0) begin bv_ = 1; bi_ = b_idq[best]; br_ = b_bad[best] ? 2'b10 : 2'b00; best_b = b_s[best]; end
                end
            end
            // ------------------------------------------------ tokens, ready signals
            tok = tok + bw; if (tok > 2 * BEAT_TOK) tok = 2 * BEAT_TOK;
            arr_ = (count_v(0) < AQ) && !rnd_stall(0);
            awr_ = (count_v(1) < AQ) && (count_v(2) < AQ - 4) && !rnd_stall(0);
            wr_ = 0;
            for (k = 0; k < AQ; k = k + 1) if (w_v[k]) wr_ = 1;
            if (rnd_stall(0) || tok < BEAT_TOK || (rv_ && tok < 2 * BEAT_TOK && !wfair)) wr_ = 0;
        end
        rvalid <= rv_; rlast <= rl_; rid <= ri_; rdata <= rd_; rresp <= rr_;
        bvalid <= bv_; bid <= bi_; bresp <= br_; arready <= arr_; awready <= awr_; wready <= wr_;
    end
endmodule
