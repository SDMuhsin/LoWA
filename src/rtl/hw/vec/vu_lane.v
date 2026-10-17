`default_nettype none
// vu_lane: one vector lane (lane = one row for row ops, one element stream for element-wise ops).
// Holds the lane's two row-segment tile buffers (mm_sram_bank x 2, RB words each), the micro-op
// datapath (one fp_fma under the sequencer's micro-ops, docs/p4_report.md section 2), the
// accumulator unit (fp_fma with feedback, chunked sums of docs/arith.md section 2), the max/min
// unit (fp_cmp), the SFU (fp_sfu) behind in / out FIFOs, the in-place STORE FIFO, the output
// registers (NOB x 8 words) and the lane scalars s0..s3.
// All lanes receive the same control every cycle (SIMD); the sequencer (vu_seq) decides stalls
// from the AND of every lane's status.
// Element issue: in cycle t the sequencer reads A[addr] (buffer iss_abuf) and B[addr] (buffer
// ~iss_abuf) and the shared cvec banks; in cycle t+1 the lane captures them into slot iss_slot
// (r0 = A, r1 = B, r2..r4 = cvec0..2, keep / kmask / is_y, j = addr).
// Micro-op execution: in cycle t the sequencer presents ex_uop for slot ex_slot; the lane registers the
// control (stage L, one-hot slot select), reads the operands through AND-OR muxes into the operand
// register (stage O, cycle t+2) and issues the op from there; a result is written back in cycle t+4
// (FMA: fp_fma STAGES = 2; MIN / MOV / POP: delayed to the same cycle).  A slot's next op comes
// NS >= 8 cycles later, so there are no read-after-write hazards; FIFO pushes / pops happen at t+2.
module vu_lane #(
    parameter NS   = 8,                 // barrel slots (elements in flight), power of two >= 4
    parameter SW   = 3,                 // log2 NS
    parameter RB   = 256,               // words per tile buffer
    parameter RBW  = 8,                 // log2 RB
    parameter NOB  = 2,                 // 8-word output registers
    parameter LF   = 2,                 // SFU FMA stages
    parameter AFD  = 8,                 // accumulator FIFO depth
    parameter KC   = 64                 // chunk length of the contract
) (
    input  wire         clk,
    input  wire         rst_n,
    // tile buffer loads (one word per cycle, either buffer)
    input  wire         ld_we,
    input  wire         ld_buf,
    input  wire [RBW-1:0] ld_addr,
    input  wire [31:0]  ld_d,
    // per-row scalar loads: sld_sel 0 -> s0, 1 -> s1, 2 -> s2, 3 -> s3
    input  wire         sld_v,
    input  wire [1:0]   sld_sel,
    input  wire [31:0]  sld_d,
    // element issue (cycle t) and the cvec words (cycle t+1)
    input  wire         iss_v,
    input  wire         iss_abuf,
    input  wire [RBW-1:0] iss_addr,
    input  wire [SW-1:0] iss_slot,
    input  wire         iss_keep,
    input  wire         iss_isy,
    input  wire         km_d,               // key-mask bit of the element (cycle t+1)
    input  wire [31:0]  cv0_d, cv1_d, cv2_d,
    // micro-op execution
    input  wire         ex_v,
    input  wire [SW-1:0] ex_slot,
    /* verilator lint_off UNUSEDSIGNAL */   // ex_uop[35:32]: `last` and padding are the sequencer's
    input  wire [35:0]  ex_uop,
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire [31:0]  c0, c1, c2, c3, ds,
    // accumulator control
    input  wire         acc_start,          // reset both accumulators (p = tot = +0, cnt = 0)
    input  wire         acc_chunk,          // chunked mode for this pass
    input  wire         acc_flush,          // finish: tot = add(tot, p) if chunk and cnt != 0, else tot = p
    output wire         acc_idle,           // FIFO empty, nothing in flight, flush done
    output wire [3:0]   acc_cnt,            // FIFO occupancy
    // max unit
    input  wire         max_start,          // maxr = -inf
    // SFU status
    output wire [3:0]   sfu_out_cnt,        // results waiting in the SFU output FIFO
    // store FIFO
    output wire         st_busy,            // >= 2 pending stores: stall issue
    input  wire         cur_abuf,           // compute buffer for stores
    // output registers
    input  wire         ob_flush,           // close a partial register at the end of a step
    output wire         ob_room,            // an OUT may be pushed (fill register free, spill register free near its end)
    output wire         ob_rdy,             // oldest register complete
    output wire [3:0]   ob_nw,              // its valid word count
    output wire [255:0] ob_data,
    input  wire         ob_pop,
    // cross-lane value and the lane-0 scalars
    input  wire         xl_sel,             // 0 TOT0, 1 s2
    input  wire [31:0]  xtot,               // cross-lane result (broadcast by the top)
    output wire [31:0]  xl_val,
    output wire [31:0]  s0_o, s1_o, s3_o
);
    // ------------------------------------------------------------------ micro-op encoding
    localparam [3:0] O_FMA = 4'd1, O_SFU = 4'd2, O_POP = 4'd3, O_MAX = 4'd4, O_ACC = 4'd5, O_OUT = 4'd6,
                     O_STORE = 4'd7, O_MIN = 4'd8, O_MOV = 4'd9;

    // ------------------------------------------------------------------ tile buffers
    wire [31:0] bq0, bq1;
    // STORE FIFO (4 deep): drained into the compute buffer in cycles without an issue read
    reg  [31:0] stf_d [0:3];
    reg  [RBW-1:0] stf_a [0:3];
    reg  [2:0]  stf_wp, stf_rp;
    wire        stf_ne = (stf_wp != stf_rp);
    wire        st_fire = stf_ne & ~iss_v;
    assign st_busy = ((stf_wp - stf_rp) >= 3'd1);                       // up to 3 pushes in flight (depth 4)
    wire        b0_we = (ld_we & ~ld_buf) | (st_fire & ~cur_abuf);
    wire        b1_we = (ld_we &  ld_buf) | (st_fire &  cur_abuf);
    wire [RBW-1:0] b0_a = (ld_we & ~ld_buf) ? ld_addr : st_fire ? stf_a[stf_rp[1:0]] : iss_addr;
    wire [RBW-1:0] b1_a = (ld_we &  ld_buf) ? ld_addr : st_fire ? stf_a[stf_rp[1:0]] : iss_addr;
    wire [31:0] b0_d = (ld_we & ~ld_buf) ? ld_d : stf_d[stf_rp[1:0]];
    wire [31:0] b1_d = (ld_we &  ld_buf) ? ld_d : stf_d[stf_rp[1:0]];
    mm_sram_bank #(.WORDS(RB), .WIDTH(32), .AW(RBW)) u_b0 (.clk(clk), .en(b0_we | iss_v), .we(b0_we), .addr(b0_a), .d(b0_d), .q(bq0));
    mm_sram_bank #(.WORDS(RB), .WIDTH(32), .AW(RBW)) u_b1 (.clk(clk), .en(b1_we | iss_v), .we(b1_we), .addr(b1_a), .d(b1_d), .q(bq1));

    // ------------------------------------------------------------------ slot registers
    reg  [31:0] r0 [0:NS-1];
    reg  [31:0] r1 [0:NS-1];
    reg  [31:0] r2 [0:NS-1];
    reg  [31:0] r3 [0:NS-1];
    reg  [31:0] r4 [0:NS-1];
    reg  [31:0] r5 [0:NS-1];
    reg  [NS-1:0] f_keep, f_km, f_isy;
    reg  [RBW-1:0] f_j [0:NS-1];
    reg  [31:0] s0, s1, s2, s3;
    reg  [31:0] tot0, tot1, maxr;
    // issue capture (cycle t+1)
    reg         iss_v_q, iss_abuf_q, iss_keep_q, iss_isy_q;
    reg  [SW-1:0] iss_slot_q;
    reg  [RBW-1:0] iss_addr_q;
    always @(posedge clk) begin
        if (!rst_n) iss_v_q <= 1'b0;
        else        iss_v_q <= iss_v;
        iss_abuf_q <= iss_abuf; iss_keep_q <= iss_keep; iss_isy_q <= iss_isy;
        iss_slot_q <= iss_slot; iss_addr_q <= iss_addr;
    end

    // ------------------------------------------------------------------ stage L: lane-local control register
    // (fanout of the sequencer's control is one flop per lane); the slot select is a replicated one-hot
    localparam REP = 4;
    reg          ex_v_l;
    reg  [REP*NS-1:0] slot_oh_l;
    reg  [SW-1:0] ex_slot_l;
    /* verilator lint_off UNUSEDSIGNAL */   // uop_l: the source selects are decoded to one-hot at stage L
    reg  [35:0]  uop_l;
    /* verilator lint_on UNUSEDSIGNAL */
    reg  [25:0]  sa_oh_l, sb_oh_l, sc_oh_l;
    function [25:0] dec26;
        input [4:0] x;
        begin
            dec26 = 26'd0;
            if (x < 5'd26) dec26[x] = 1'b1;
        end
    endfunction
    function [NS-1:0] decns;
        input [SW-1:0] x;
        begin
            decns = {NS{1'b0}};
            decns[x] = 1'b1;
        end
    endfunction
    integer rp_i;
    always @(posedge clk) begin
        if (!rst_n) ex_v_l <= 1'b0;
        else        ex_v_l <= ex_v;
        ex_slot_l <= ex_slot; uop_l <= ex_uop;
        for (rp_i = 0; rp_i < REP; rp_i = rp_i + 1) slot_oh_l[rp_i*NS +: NS] <= decns(ex_slot);
        sa_oh_l <= dec26(ex_uop[8:4]); sb_oh_l <= dec26(ex_uop[13:9]); sc_oh_l <= dec26(ex_uop[18:14]);
    end
    // ------------------------------------------------------------------ operand read (stage L -> O)
    localparam SDW = 6 * 32 + RBW + 3;
    wire [NS*SDW-1:0] slotdat;
    genvar gs;
    generate
        for (gs = 0; gs < NS; gs = gs + 1) begin : g_sd
            assign slotdat[gs*SDW +: SDW] = {f_isy[gs], f_km[gs], f_keep[gs], f_j[gs], r5[gs], r4[gs], r3[gs], r2[gs], r1[gs], r0[gs]};
        end
    endgenerate
    wire [SDW-1:0] cur;
    cm_ohmux #(.W(SDW), .N(NS), .REP(REP)) u_slotmux (.sel(slot_oh_l), .d(slotdat), .y(cur));
    wire [31:0] c_r0 = cur[31:0], c_r1 = cur[63:32], c_r2 = cur[95:64], c_r3 = cur[127:96], c_r4 = cur[159:128], c_r5 = cur[191:160];
    wire [RBW-1:0] c_j = cur[192 +: RBW];
    wire        c_keep = cur[192 + RBW], c_km = cur[193 + RBW], c_isy = cur[194 + RBW];
    wire [26*32-1:0] srcs = {32'hFF800000, (c_isy ? 32'hBF800000 : 32'h80000000), ds, 32'h80000000, 32'h3F800000, 32'h00000000,
                             c3, c2, c1, c0, xtot, maxr, tot1, tot0, s3, s2, s1, s0, 32'd0, 32'd0, c_r5, c_r4, c_r3, c_r2, c_r1, c_r0};
    wire [31:0] opa_m, opb_m, opc_m;
    cm_ohmux #(.W(32), .N(26), .REP(1)) u_ma (.sel(sa_oh_l), .d(srcs), .y(opa_m));
    cm_ohmux #(.W(32), .N(26), .REP(1)) u_mb (.sel(sb_oh_l), .d(srcs), .y(opb_m));
    cm_ohmux #(.W(32), .N(26), .REP(1)) u_mc (.sel(sc_oh_l), .d(srcs), .y(opc_m));
    wire [3:0]  ul_opt = uop_l[3:0];
    wire        ul_zk = uop_l[25], ul_zm = uop_l[26];
    wire        zero_m = (ul_zk & ~c_keep) | (ul_zm & ~c_km);
    wire        skip_m = (ul_opt == O_MAX) & ul_zm & ~c_km;           // masked key: no MAX op
    // ------------------------------------------------------------------ stage O: operand register
    reg          ex_v_o, zero_o, capy_hit_o;
    reg  [SW-1:0] slot_o;
    reg  [3:0]   opt_o, dst_o;
    reg  [2:0]   fn_o;
    reg          accw_o;
    reg  [31:0]  opa, opb, opc_n, opa_n;
    reg  [RBW-1:0] j_o;
    always @(posedge clk) begin
        if (!rst_n) ex_v_o <= 1'b0;
        else        ex_v_o <= ex_v_l & ~skip_m;
        slot_o <= ex_slot_l; opt_o <= ul_opt; dst_o <= uop_l[22:19]; fn_o <= uop_l[30:28]; accw_o <= uop_l[31];
        zero_o <= zero_m; capy_hit_o <= uop_l[27] & c_isy; j_o <= c_j;
        opa <= opa_m; opb <= opb_m;
        opa_n <= {opa_m[31] ^ uop_l[23], opa_m[30:0]};
        opc_n <= {opc_m[31] ^ uop_l[24], opc_m[30:0]};
    end
    wire [3:0]  u_opt = opt_o;
    wire [3:0]  u_dst = dst_o;
    wire [2:0]  u_fn = fn_o;
    wire        u_accw = accw_o;
    wire        zero_res = zero_o;
    wire [31:0] opa_z = zero_res ? 32'h00000000 : opa;
    wire        ex_v_x = ex_v_o;
    wire [SW-1:0] ex_slot_x = slot_o;

    // ------------------------------------------------------------------ FMA + writeback (stage O -> O+2)
    wire        fma_iss = ex_v_x & (u_opt == O_FMA);
    wire        fma_ov;
    wire [31:0] fma_y;
    fp_fma #(.STAGES(2)) u_fma (.clk(clk), .rst_n(rst_n), .in_valid(fma_iss), .a(opa_n), .b(opb), .c(opc_n),
                                .out_valid(fma_ov), .y(fma_y));
    // MIN / MOV / POP: value captured at issue, delayed two cycles
    wire [31:0] sfo_y;
    wire        min_iss = ex_v_x & (u_opt == O_MIN);
    wire        cmp_ov;
    wire [31:0] cmp_mx, cmp_mn;
    wire        max_iss = ex_v_x & (u_opt == O_MAX);
    reg         wb_v1, wb_v2, wb_z1, wb_z2, wb_min1;
    wire [31:0] max_a = (cmp_ov & ~wb_min1) ? cmp_mx : maxr;   // forward the previous max (1 per cycle)
    /* verilator lint_off PINCONNECTEMPTY */   // lt / eq / gt / un and the SFU tag are not needed here
    fp_cmp #(.STAGES(1)) u_cmp (.clk(clk), .rst_n(rst_n), .in_valid(max_iss | min_iss),
                                .a(max_iss ? max_a : opa), .b(max_iss ? opa : opb),
                                .out_valid(cmp_ov), .lt(), .eq(), .gt(), .un(), .mx(cmp_mx), .mn(cmp_mn));
    reg  [SW-1:0] wb_s1, wb_s2;
    reg  [3:0]  wb_d1, wb_d2;
    reg  [31:0] wb_x1, wb_x2;
    wire        wb_iss = ex_v_x & ((u_opt == O_FMA) | (u_opt == O_MIN) | (u_opt == O_MOV) | (u_opt == O_POP));
    wire [31:0] wb_src = (u_opt == O_POP) ? sfo_y : opa;
    always @(posedge clk) begin
        if (!rst_n) begin wb_v1 <= 1'b0; wb_v2 <= 1'b0; end
        else begin wb_v1 <= wb_iss; wb_v2 <= wb_v1; end
        wb_s1 <= ex_slot_x; wb_d1 <= u_dst; wb_x1 <= wb_src; wb_z1 <= zero_res; wb_min1 <= min_iss;
        wb_s2 <= wb_s1; wb_d2 <= wb_d1; wb_x2 <= wb_min1 ? cmp_mn : wb_x1; wb_z2 <= wb_z1;
    end
    wire [31:0] wb_val = fma_ov ? fma_y : wb_x2;
    wire [31:0] wb_valz = wb_z2 ? 32'h00000000 : wb_val;
    always @(posedge clk) begin
        // slot capture
        if (iss_v_q) begin
            r0[iss_slot_q] <= iss_abuf_q ? bq1 : bq0;
            r1[iss_slot_q] <= iss_abuf_q ? bq0 : bq1;
            r2[iss_slot_q] <= cv0_d; r3[iss_slot_q] <= cv1_d; r4[iss_slot_q] <= cv2_d;
            f_keep[iss_slot_q] <= iss_keep_q; f_km[iss_slot_q] <= km_d; f_isy[iss_slot_q] <= iss_isy_q;
            f_j[iss_slot_q] <= iss_addr_q;
        end
        // writeback (never in the same cycle as a capture of the same slot: the slot is busy)
        if (wb_v2) begin
            case (wb_d2)
                4'd0: r0[wb_s2] <= wb_valz;  4'd1: r1[wb_s2] <= wb_valz;  4'd2: r2[wb_s2] <= wb_valz;
                4'd3: r3[wb_s2] <= wb_valz;  4'd4: r4[wb_s2] <= wb_valz;  4'd5: r5[wb_s2] <= wb_valz;
                default: ;
            endcase
        end
        if (sld_v) begin
            case (sld_sel)
                2'd0: s0 <= sld_d;  2'd1: s1 <= sld_d;  2'd2: s2 <= sld_d;  default: s3 <= sld_d;
            endcase
        end else if (wb_v2) begin
            case (wb_d2)
                4'd8: s0 <= wb_valz;  4'd9: s1 <= wb_valz;  4'd10: s2 <= wb_valz;  4'd11: s3 <= wb_valz;
                default: ;
            endcase
        end
        if (ex_v_x & (u_opt == O_MAX) & capy_hit_o) s3 <= opa;
        // max register
        if (max_start) maxr <= 32'hFF800000;
        else if (cmp_ov & ~wb_min1) maxr <= cmp_mx;
    end
    assign s0_o = s0;
    assign s1_o = s1;
    assign s3_o = s3;
    assign xl_val = xl_sel ? s2 : tot0;

    // ------------------------------------------------------------------ SFU with FIFOs
    reg  [34:0] sfi [0:7];
    reg  [3:0]  sfi_wp, sfi_rp;
    wire        sfi_ne = (sfi_wp != sfi_rp);
    wire        sfu_rdy, sfu_ov;
    wire [31:0] sfu_y;
    wire        sfu_push = ex_v_x & (u_opt == O_SFU);
    fp_sfu #(.LF(LF), .NTHR(LF + 2), .TAGW(1)) u_sfu (.clk(clk), .rst_n(rst_n), .in_valid(sfi_ne), .in_ready(sfu_rdy),
                                                     .in_fn(sfi[sfi_rp[2:0]][34:32]), .in_x(sfi[sfi_rp[2:0]][31:0]),
                                                     .in_tag(1'b0), .out_valid(sfu_ov), .out_y(sfu_y), .out_tag());
    /* verilator lint_on PINCONNECTEMPTY */
    reg  [31:0] sfo [0:7];
    reg  [3:0]  sfo_wp, sfo_rp;
    wire        sfu_pop = ex_v_x & (u_opt == O_POP);
    assign sfo_y = sfo[sfo_rp[2:0]];
    assign sfu_out_cnt = sfo_wp - sfo_rp;
    always @(posedge clk) begin
        if (!rst_n) begin sfi_wp <= 4'd0; sfi_rp <= 4'd0; sfo_wp <= 4'd0; sfo_rp <= 4'd0; end
        else begin
            if (sfu_push) sfi_wp <= sfi_wp + 4'd1;
            if (sfi_ne & sfu_rdy) sfi_rp <= sfi_rp + 4'd1;
            if (sfu_ov) sfo_wp <= sfo_wp + 4'd1;
            if (sfu_pop) sfo_rp <= sfo_rp + 4'd1;
        end
        if (sfu_push) sfi[sfi_wp[2:0]] <= {u_fn, opa};
        if (sfu_ov) sfo[sfo_wp[2:0]] <= sfu_y;
    end

    // ------------------------------------------------------------------ store FIFO
    wire st_push = ex_v_x & (u_opt == O_STORE);
    always @(posedge clk) begin
        if (!rst_n) begin stf_wp <= 3'd0; stf_rp <= 3'd0; end
        else begin
            if (st_push) stf_wp <= stf_wp + 3'd1;
            if (st_fire) stf_rp <= stf_rp + 3'd1;
        end
        if (st_push) begin stf_d[stf_wp[1:0]] <= opa_z; stf_a[stf_wp[1:0]] <= j_o; end
    end

    // ------------------------------------------------------------------ accumulator unit
    // FIFO of (x, y, w); p[w] = fma(x, y, p[w]); every KC pushes (chunk mode) tot[w] = add(tot[w], p[w]),
    // p[w] = +0; flush: tot = add(tot, p) if cnt != 0 (chunk) else tot = p.
    reg  [64:0] af [0:AFD-1];
    reg  [3:0]  af_wp, af_rp;
    wire        af_ne = (af_wp != af_rp);
    assign acc_cnt = af_wp - af_rp;
    wire        acc_push = ex_v_x & (u_opt == O_ACC);
    reg  [31:0] p0, p1;
    reg  [6:0]  cn0, cn1;
    reg  [1:0]  busy;                   // op in flight per accumulator
    reg  [1:0]  pend_chunk;             // chunk-end add pending per accumulator
    reg  [1:0]  fl_pend;                // flush add pending
    reg         chunk_m, flushing;
    wire        af_w = af[af_rp[2:0]][64];
    wire        afma_ov;
    wire [31:0] afma_y;
    reg         ai_v, ai_w, ai_kind;    // kind 0: p op, 1: tot op
    reg  [31:0] ai_a, ai_b, ai_c;
    // issue selection: chunk-end / flush adds first (they free the accumulator), then the FIFO head
    always @* begin
        ai_v = 1'b0; ai_w = 1'b0; ai_kind = 1'b0; ai_a = 32'd0; ai_b = 32'd0; ai_c = 32'd0;
        if ((pend_chunk[0] | fl_pend[0]) & ~busy[0]) begin
            ai_v = 1'b1; ai_w = 1'b0; ai_kind = 1'b1; ai_a = p0; ai_b = 32'h3F800000; ai_c = tot0;
        end else if ((pend_chunk[1] | fl_pend[1]) & ~busy[1]) begin
            ai_v = 1'b1; ai_w = 1'b1; ai_kind = 1'b1; ai_a = p1; ai_b = 32'h3F800000; ai_c = tot1;
        end else if (af_ne & ~busy[af_w] & ~pend_chunk[af_w] & ~fl_pend[af_w] & ~flushing) begin
            ai_v = 1'b1; ai_w = af_w; ai_kind = 1'b0; ai_a = af[af_rp[2:0]][63:32]; ai_b = af[af_rp[2:0]][31:0];
            ai_c = af_w ? p1 : p0;
        end
    end
    wire af_pop = ai_v & ~ai_kind;
    fp_fma #(.STAGES(2)) u_afma (.clk(clk), .rst_n(rst_n), .in_valid(ai_v), .a(ai_a), .b(ai_b), .c(ai_c),
                                 .out_valid(afma_ov), .y(afma_y));
    reg  [1:0] q_ww, q_kk;
    wire       ret_w = q_ww[1], ret_kind = q_kk[1];
    wire       cn_hit0 = (cn0 == KC[6:0] - 7'd1), cn_hit1 = (cn1 == KC[6:0] - 7'd1);
    assign acc_idle = ~af_ne & ~busy[0] & ~busy[1] & ~pend_chunk[0] & ~pend_chunk[1] & ~fl_pend[0] & ~fl_pend[1] & ~flushing;
    always @(posedge clk) begin
        if (!rst_n) begin
            af_wp <= 4'd0; af_rp <= 4'd0; busy <= 2'b00; pend_chunk <= 2'b00; fl_pend <= 2'b00; flushing <= 1'b0;
            cn0 <= 7'd0; cn1 <= 7'd0; chunk_m <= 1'b0;
        end else begin
            if (acc_push) af_wp <= af_wp + 4'd1;
            if (af_pop) af_rp <= af_rp + 4'd1;
            if (ai_v) busy[ai_w] <= 1'b1;
            if (afma_ov) begin
                busy[ret_w] <= 1'b0;
                if (ret_kind) begin                       // tot op done: clear the pending flag
                    if (ret_w) begin pend_chunk[1] <= 1'b0; fl_pend[1] <= 1'b0; end
                    else       begin pend_chunk[0] <= 1'b0; fl_pend[0] <= 1'b0; end
                end else if (chunk_m) begin
                    if (ret_w) begin cn1 <= cn_hit1 ? 7'd0 : cn1 + 7'd1; if (cn_hit1) pend_chunk[1] <= 1'b1; end
                    else       begin cn0 <= cn_hit0 ? 7'd0 : cn0 + 7'd1; if (cn_hit0) pend_chunk[0] <= 1'b1; end
                end
            end
            if (acc_start) begin
                cn0 <= 7'd0; cn1 <= 7'd0; chunk_m <= acc_chunk; flushing <= 1'b0;
            end
            if (acc_flush) begin
                flushing <= 1'b1;
                if (chunk_m) begin
                    fl_pend[0] <= (cn0 != 7'd0);
                    fl_pend[1] <= (cn1 != 7'd0);
                end
            end
            if (flushing & ~fl_pend[0] & ~fl_pend[1] & ~busy[0] & ~busy[1]) flushing <= 1'b0;
        end
        q_ww <= {q_ww[0], ai_w}; q_kk <= {q_kk[0], ai_kind};
        if (acc_push) af[af_wp[2:0]] <= {u_accw, opa_z, opb};
        if (acc_start) begin p0 <= 32'd0; p1 <= 32'd0; tot0 <= 32'd0; tot1 <= 32'd0; end
        else begin
            if (afma_ov) begin
                if (ret_kind) begin
                    if (ret_w) begin tot1 <= afma_y; p1 <= 32'd0; end
                    else       begin tot0 <= afma_y; p0 <= 32'd0; end
                end else begin
                    if (ret_w) p1 <= afma_y; else p0 <= afma_y;
                end
            end
            if (acc_flush & ~chunk_m) begin tot0 <= p0; tot1 <= p1; end   // sequential: the result is p
        end
    end

    // ------------------------------------------------------------------ output registers
    reg  [255:0] ob [0:NOB-1];
    reg  [3:0]   ob_n [0:NOB-1];
    reg  [NOB-1:0] ob_full;
    reg  [2:0]   ob_wpos;
    localparam OBW = (NOB <= 2) ? 1 : (NOB <= 4) ? 2 : 3;   // NOB = 2 or 4: the pointers wrap naturally
    reg  [OBW-1:0] ob_wsel, ob_rsel;
    wire         out_push = ex_v_x & (u_opt == O_OUT);
    wire         ob_close = (out_push & (ob_wpos == 3'd7)) | (ob_flush & (ob_wpos != 3'd0));
    wire [OBW-1:0] ob_wnext = ob_wsel + 1'b1;
    assign ob_room = ~ob_full[ob_wsel] & ((ob_wpos < 3'd5) | ~ob_full[ob_wnext]);   // up to 3 pushes in flight
    assign ob_rdy = ob_full[ob_rsel];
    assign ob_nw = ob_n[ob_rsel];
    assign ob_data = ob[ob_rsel];
    always @(posedge clk) begin
        if (!rst_n) begin
            ob_full <= {NOB{1'b0}}; ob_wpos <= 3'd0; ob_wsel <= {OBW{1'b0}}; ob_rsel <= {OBW{1'b0}};
        end else begin
            if (out_push) ob_wpos <= ob_wpos + 3'd1;
            if (ob_close) begin
                ob_full[ob_wsel] <= 1'b1;
                ob_n[ob_wsel] <= out_push ? 4'd8 : {1'b0, ob_wpos};
                ob_wpos <= 3'd0;
                ob_wsel <= ob_wsel + 1'b1;
            end
            if (ob_pop) begin
                ob_full[ob_rsel] <= 1'b0;
                ob_rsel <= ob_rsel + 1'b1;
            end
        end
        if (out_push) ob[ob_wsel][ob_wpos * 32 +: 32] <= opa_z;
    end
endmodule
`default_nettype wire
