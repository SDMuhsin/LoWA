`default_nettype none
// ct_seq: the LoWA-TA sequencer / ISA decoder (P6c, docs/isa.md as built).
//  * host CSR bus (one access per cycle, registered read data; isa.md section 6): CTRL, STATUS, DOORBELL,
//    STEP_DONE, STEP_T, PROG_BASE, SLOT_DELTA, SEED, ERR, PERF; level interrupt `irq` at END (CTRL.IRQ_EN)
//  * fetch: one 48-word read request per descriptor at PC (read channel shared with DMA_COPY), 6 beats
//    into the 48-word decode buffer, then the REL term adds word by word (word += TERM[nibble], isa.md 1)
//  * issue in program order: header check (OP / UNIT / NSL, else ERR.decode and halt), WAIT flag (CLR
//    consumes), then a sequencer op (NOP, SETCSR, FENCE, LOOP, END) or a unit start: the unit's payload words
//    are copied to its descriptor register (held stable until the op retires) and `start` pulses
//  * retire: engine done (a GEMM with MERGE also needs the merge engine's done), then two cycles, then the
//    unit's memory write channel idle (all writes B-acknowledged, docs/p6a_report.md) -> unit free, SIG flag set.
//    Each unit runs one op at a time (queue depth 1 + the decoder's staging buffer).
//  * LOOP: body of `body` bytes after the LOOP descriptor, `count` iterations, TERM[13] = loop index from
//    `start` (+1 / -1), TERM[k] (k = 1..12) start at 0 and add +/- STR[k] (SETCSR k) per iteration.
// Units: 0 DMA (ct_dma inside), 1 GEMM (gm_top + lw_top merge queue), 2 VEC (vu_top), 3 LOWA (lw_top main).
module ct_seq #(
    parameter GP = 36,                 // GEMM payload words
    parameter VP = 25,                 // VEC payload words
    parameter LP = 10                  // LOWA payload words
) (
    input  wire          clk,
    input  wire          rst_n,
    // host CSR bus
    input  wire          csr_we,
    input  wire          csr_re,
    input  wire [11:0]   csr_addr,
    input  wire [31:0]   csr_wdata,
    output reg  [31:0]   csr_rdata,
    output wire          irq,
    // memory: one read channel (fetch + DMA), one write channel (DMA)
    output wire          rd_req_valid,
    input  wire          rd_req_ready,
    output wire [31:0]   rd_req_addr,
    output wire [6:0]    rd_req_len,
    input  wire          rd_dat_valid,
    output wire          rd_dat_ready,
    input  wire [255:0]  rd_dat,
    output wire          wr_valid,
    input  wire          wr_ready,
    output wire [31:0]   wr_addr,
    output wire [255:0]  wr_data,
    output wire [7:0]    wr_mask,
    input  wire [3:0]    wr_idle,      // memory write channel idle per unit: DMA, GEMM, VEC, LOWA
    // engines
    output reg           gm_start,
    output reg  [32*GP-1:0] gm_p,
    input  wire          gm_done,
    input  wire          mg_done,      // lw_top merge queue done (started with gm_start when MERGE)
    output reg           mg_start,
    output reg           vu_start,
    output reg  [32*VP-1:0] vu_p,
    input  wire          vu_done,
    output reg           lw_start,
    output reg  [32*LP-1:0] lw_p,
    input  wire          lw_done,
    // dropout generator configuration (mm_top mk_seed / mk_step; site / layer come from the payloads)
    output wire [63:0]   mk_seed,
    output wire [23:0]   mk_step,
    output wire [3:0]    unit_busy
);
    // reset tree: registered copies (kept hierarchy, docs/p3_report.md section 6), one for the sequencer, one for ct_dma
    wire rst_q, rst_dq;
    cm_rstbuf u_rb0 (.clk(clk), .rst_n(rst_n), .rst_n_q(rst_q));
    cm_rstbuf u_rb1 (.clk(clk), .rst_n(rst_n), .rst_n_q(rst_dq));
    // ------------------------------------------------------------------ CSRs
    reg  [31:0] prog_base, slot_delta, step_t, step_done, seed_lo, seed_hi, perf_cyc, perf_desc;
    reg  [8:0]  last_tag;
    reg  [5:0]  ctrl;
    reg  [2:0]  err;
    reg         irq_q, running, slot;
    reg  [31:0] pc;
    assign irq = irq_q & ctrl[4];
    assign mk_seed = {seed_hi, seed_lo};
    assign mk_step = step_t[23:0];
    // ------------------------------------------------------------------ sequencer state
    localparam S_IDLE = 3'd0, S_FETCH = 3'd1, S_RECV = 3'd2, S_DEC = 3'd3, S_EXEC = 3'd4, S_NEXT = 3'd5, S_HALT = 3'd6;
    reg  [2:0]  st;
    reg  [31:0] buf_w [0:47];
    reg  [5:0]  di;                   // decode word index
    reg  [2:0]  bi;                   // beat index
    reg  [31:0] str [1:12];
    reg  [31:0] lterm [0:15];          // entries 0, 13..15 stay 0 (TERM 0 / 13..15 come from tval)
    reg  [7:0]  lidx;
    reg         lp_act, lp_dir;
    reg  [31:0] lp_start, lp_end, lp_left;
    reg  [14:0] flags;
    // units: active, done seen, settle count, SIG
    reg  [3:0]  act, dseen;
    reg  [1:0]  settle [0:3];
    reg  [3:0]  sig_u [0:3];
    reg         gm_mg, mg_seen;
    wire        dma_done, dma_own;
    reg         dma_start;
    reg  [32*9-1:0] dma_p;
    assign unit_busy = act;
    // header of the buffered descriptor
    wire [31:0] h     = buf_w[0];
    wire [7:0]  h_op  = h[7:0];
    wire [3:0]  h_un  = h[11:8];
    wire [3:0]  h_wt  = h[15:12];
    wire [3:0]  h_sg  = h[19:16];
    wire        h_clr = h[20];
    wire [1:0]  h_ns  = h[22:21];
    wire [5:0]  nwords = (h_ns == 2'd0) ? 6'd16 : (h_ns == 2'd1) ? 6'd32 : 6'd48;
    wire [5:0]  pbase  = (h_ns == 2'd0) ? 6'd3 : (h_ns == 2'd1) ? 6'd5 : 6'd7;   // first payload word
    // opcode table: legal (OP, UNIT, NSL) combinations (isa.md section 3 as built)
    reg         h_ok;
    always @* begin
        case (h_op)
            8'h00, 8'h31, 8'h32, 8'h33, 8'h3F: h_ok = (h_un == 4'd0) && (h_ns == 2'd0);
            8'h01, 8'h02:                      h_ok = (h_un == 4'd2) && (h_ns == 2'd2);
            8'h10, 8'h16, 8'h22:               h_ok = (h_un == 4'd3) && (h_ns == 2'd1);
            8'h20, 8'h21, 8'h23:               h_ok = (h_un == 4'd4) && (h_ns == 2'd0);
            8'h30:                             h_ok = (h_un == 4'd1) && (h_ns == 2'd0);
            default:                           h_ok = 1'b0;
        endcase
    end
    // unit index of the op (0 DMA, 1 GEMM, 2 VEC, 3 LOWA)
    wire [1:0]  uidx = (h_un == 4'd1) ? 2'd0 : (h_un == 4'd2) ? 2'd1 : (h_un == 4'd3) ? 2'd2 : 2'd3;
    wire        is_unit = (h_un != 4'd0);
    wire        wait_ok = (h_wt == 4'hF) || flags[h_wt];
    /* verilator lint_off UNUSEDSIGNAL */   // LOOP start / dir use the low bits only; dma_busy = act[0] (kept for debug)
    wire [31:0] p0 = buf_w[3], p1 = buf_w[4], p2 = buf_w[5], p3 = buf_w[6];   // sequencer-op payload (NSL 1)
    wire        dma_busy;
    /* verilator lint_on UNUSEDSIGNAL */
    // TERM[sel] for the word being decoded
    wire [31:0] rel_w = buf_w[{3'd0, di[5:3]} + 6'd1];
    wire [3:0]  sel   = rel_w[{di[2:0], 2'b00} +: 4];
    reg  [31:0] tval;
    always @* begin
        case (sel)
            4'd0:  tval = 32'd0;
            4'd13: tval = {24'd0, lidx};
            4'd14: tval = step_t;
            4'd15: tval = slot ? slot_delta : 32'd0;
            default: tval = lterm[sel];
        endcase
    end
    wire        dma_rv, dma_rr;
    wire [31:0] dma_ra;
    wire [6:0]  dma_rl;
    assign rd_req_valid = dma_own ? dma_rv : (st == S_FETCH);
    assign rd_req_addr  = dma_own ? dma_ra : pc;
    assign rd_req_len   = dma_own ? dma_rl : 7'd48;
    assign rd_dat_ready = dma_own ? dma_rr : (st == S_RECV);
    ct_dma u_dma (.clk(clk), .rst_n(rst_dq), .start(dma_start), .p(dma_p), .busy(dma_busy), .done(dma_done),
        .rd_own(dma_own), .rd_req_valid(dma_rv), .rd_req_ready(rd_req_ready), .rd_req_addr(dma_ra), .rd_req_len(dma_rl),
        .rd_dat_valid(rd_dat_valid), .rd_dat_ready(dma_rr), .rd_dat(rd_dat),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_addr(wr_addr), .wr_data(wr_data), .wr_mask(wr_mask));
    wire [3:0] edone = {lw_done, vu_done, gm_done, dma_done};
    integer i;
    always @(posedge clk) begin
        if (!rst_q || ctrl[1]) begin
            st <= S_IDLE; running <= 1'b0; act <= 4'd0; dseen <= 4'd0; flags <= 15'd0; lp_act <= 1'b0;
            gm_start <= 1'b0; mg_start <= 1'b0; vu_start <= 1'b0; lw_start <= 1'b0; dma_start <= 1'b0;
            irq_q <= 1'b0; err <= 3'd0; step_done <= 32'd0; perf_cyc <= 32'd0; perf_desc <= 32'd0;
            if (!rst_q) begin
                ctrl <= 6'd0; step_t <= 32'd0; prog_base <= 32'd0; slot_delta <= 32'd0; seed_lo <= 32'd0;
                seed_hi <= 32'd0; slot <= 1'b0; last_tag <= 9'd0; pc <= 32'd0; lidx <= 8'd0;
            end else ctrl[1] <= 1'b0;                           // soft reset is one cycle
            for (i = 0; i < 4; i = i + 1) begin settle[i] <= 2'd0; sig_u[i] <= 4'hF; end
            gm_mg <= 1'b0; mg_seen <= 1'b0;
            for (i = 0; i < 16; i = i + 1) lterm[i] <= 32'd0;
        end else begin
            gm_start <= 1'b0; mg_start <= 1'b0; vu_start <= 1'b0; lw_start <= 1'b0; dma_start <= 1'b0;
            if (running) perf_cyc <= perf_cyc + 32'd1;
            // ---------------------------------------------------------- host CSR writes
            if (csr_we) begin
                case (csr_addr)
                    12'h008: begin ctrl <= {csr_wdata[5:3], 1'b0, csr_wdata[1:0]}; if (csr_wdata[2]) step_t <= 32'd0; end
                    12'h010: if (csr_wdata[31] && ctrl[0] && !running) begin
                                 slot <= csr_wdata[0]; pc <= prog_base; running <= 1'b1; st <= S_FETCH;
                                 perf_cyc <= 32'd0; perf_desc <= 32'd0; irq_q <= 1'b0;
                             end
                    12'h014: irq_q <= 1'b0;
                    12'h018: step_t <= csr_wdata;
                    12'h020: prog_base <= csr_wdata;
                    12'h024: slot_delta <= csr_wdata;
                    12'h060: seed_lo <= csr_wdata;
                    12'h064: seed_hi <= csr_wdata;
                    12'h070: err <= err & ~csr_wdata[2:0];
                    default: ;
                endcase
            end
            // ---------------------------------------------------------- retire
            if (mg_done) mg_seen <= 1'b1;
            for (i = 0; i < 4; i = i + 1) begin
                if (act[i]) begin
                    if (edone[i]) dseen[i] <= 1'b1;
                    if (dseen[i] && (i != 1 || !gm_mg || mg_seen)) begin
                        if (settle[i] != 2'd2) settle[i] <= settle[i] + 2'd1;
                        else if (wr_idle[i]) begin
                            act[i] <= 1'b0; dseen[i] <= 1'b0; settle[i] <= 2'd0;
                            if (sig_u[i] != 4'hF) flags[sig_u[i]] <= 1'b1;
                        end
                    end
                end
            end
            // ---------------------------------------------------------- fetch / decode / issue
            case (st)
                S_FETCH: if (!dma_own && rd_req_ready) begin bi <= 3'd0; st <= S_RECV; end
                S_RECV: if (rd_dat_valid) begin
                    for (i = 0; i < 8; i = i + 1) buf_w[{bi, 3'd0} + i[5:0]] <= rd_dat[32*i +: 32];
                    bi <= bi + 3'd1;
                    if (bi == 3'd5) begin st <= S_DEC; di <= 6'd0; end
                end
                S_DEC: begin
                    if (di == 6'd0) di <= pbase;
                    else begin
                        buf_w[di] <= buf_w[di] + tval;
                        if (di == nwords - 6'd1) st <= S_EXEC;
                        else di <= di + 6'd1;
                    end
                end
                S_EXEC: begin
                    if (!h_ok) begin err[0] <= 1'b1; running <= 1'b0; st <= S_HALT; end
                    else if (wait_ok && (!is_unit || !act[uidx]) && !(h_op == 8'h31 && (act & p0[3:0]) != 4'd0)
                             && !(h_op == 8'h3F && act != 4'd0)) begin
                        if (h_wt != 4'hF && h_clr) flags[h_wt] <= 1'b0;
                        last_tag <= h[31:23];
                        perf_desc <= perf_desc + 32'd1;
                        st <= S_NEXT;
                        if (is_unit) begin
                            act[uidx] <= 1'b1; dseen[uidx] <= 1'b0; settle[uidx] <= 2'd0; sig_u[uidx] <= h_sg;
                            case (uidx)
                                2'd0: begin
                                    dma_start <= 1'b1;
                                    for (i = 0; i < 9; i = i + 1) dma_p[32*i +: 32] <= buf_w[3 + i];
                                end
                                2'd1: begin
                                    gm_start <= 1'b1; gm_mg <= buf_w[10][24]; mg_start <= buf_w[10][24]; mg_seen <= 1'b0;
                                    for (i = 0; i < GP; i = i + 1) gm_p[32*i +: 32] <= buf_w[7 + i];
                                end
                                2'd2: begin
                                    vu_start <= 1'b1;
                                    for (i = 0; i < VP; i = i + 1) vu_p[32*i +: 32] <= buf_w[5 + i];
                                end
                                default: begin
                                    lw_start <= 1'b1;
                                    for (i = 0; i < LP; i = i + 1) lw_p[32*i +: 32] <= buf_w[3 + i];
                                end
                            endcase
                        end else begin
                            case (h_op)
                                8'h33: if (p0 >= 32'd1 && p0 <= 32'd12) str[p0[3:0]] <= p1;
                                8'h32: begin
                                    lp_act <= 1'b1; lp_start <= pc + 32'd64; lp_end <= pc + 32'd64 + p0; lp_left <= p1;
                                    lidx <= p2[7:0]; lp_dir <= p3[0];
                                    for (i = 0; i < 16; i = i + 1) lterm[i] <= 32'd0;
                                end
                                8'h3F: begin
                                    step_t <= step_t + 32'd1; step_done <= step_done + 32'd1; irq_q <= 1'b1;
                                    running <= 1'b0; st <= S_IDLE;
                                end
                                default: ;
                            endcase
                        end
                    end
                end
                S_NEXT: begin                                        // PC += 64 NSL, loop back-edge
                    st <= S_FETCH;
                    if (lp_act && (pc + {24'd0, nwords, 2'b00} == lp_end)) begin
                        if (lp_left > 32'd1) begin
                            lp_left <= lp_left - 32'd1; pc <= lp_start;
                            lidx <= lp_dir ? lidx - 8'd1 : lidx + 8'd1;
                            for (i = 1; i <= 12; i = i + 1) lterm[i] <= lp_dir ? lterm[i] - str[i] : lterm[i] + str[i];
                        end else begin lp_act <= 1'b0; pc <= lp_end; end
                    end else pc <= pc + {24'd0, nwords, 2'b00};
                end
                default: ;
            endcase
        end
    end
    // ------------------------------------------------------------------ CSR reads
    always @(posedge clk) begin
        if (csr_re) begin
            case (csr_addr)
                12'h000: csr_rdata <= 32'h4C575441;                  // 'LWTA'
                12'h008: csr_rdata <= {26'd0, ctrl};
                12'h00C: csr_rdata <= {pc[21:6], 9'd0, (err != 3'd0), running, 1'b0, act};
                12'h014: csr_rdata <= step_done;
                12'h018: csr_rdata <= step_t;
                12'h020: csr_rdata <= prog_base;
                12'h024: csr_rdata <= slot_delta;
                12'h060: csr_rdata <= seed_lo;
                12'h064: csr_rdata <= seed_hi;
                12'h070: csr_rdata <= {29'd0, err};
                12'h080: csr_rdata <= perf_cyc;
                12'h084: csr_rdata <= {23'd0, last_tag};
                12'h088: csr_rdata <= perf_desc;
                12'h08C: csr_rdata <= {17'd0, flags};
                12'h090: csr_rdata <= pc;
                default: csr_rdata <= 32'd0;
            endcase
        end
    end
endmodule
`default_nettype wire
