`default_nettype none
// gm_gelu: GELU forward / backward for the GEMM epilogue (docs/arith.md section 4), op for op:
//   fwd : g  = mul(mul(u, 0.5), add(1, erf(mul(u, A))))                 A = 0x3F3504F3
//   bwd : du = mul(dg, fma(u, pdf, cdf)),  cdf = mul(0.5, add(1, erf(mul(u, A)))),
//         pdf = mul(exp(mul(mul(u, -0.5), u)), B)                        B = 0x3ECC422A
// erf and exp run on a pool of NSFU fp_sfu (one request per cycle to any SFU that is ready,
// results re-ordered by tag in a NROB-entry reorder buffer); mul(x, 0.5) / mul(u, -0.5) are exact
// power-of-two scalings (gm_pow2 + sign flip: mul(x, -0.5) = -(mul(x, 0.5)) bit for bit, the NaN
// result stays a NaN and is canonicalised by the next unit); the remaining ops are fp_mul,
// fp_add and fp_fma.  Elements leave in input order, 1 per cycle at most, with their meta word.
// Input: valid/ready (a bwd element needs two SFU requests and occupies the input for 2 cycles).
// `bypass` elements (masked columns) skip the SFUs and leave with an unspecified value.
// Output is valid-only; the consumer must have room (the epilogue's credits guarantee it).
module gm_gelu #(
    parameter NSFU = 16,
    parameter NROB = 64,
    parameter ROBW = 6,                     // log2 NROB
    parameter MW   = 16,                    // meta width
    parameter LF   = 2                      // SFU FMA stages
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          in_valid,
    output wire          in_ready,
    input  wire          in_bwd,            // 0 fwd (u = in_u), 1 bwd (u = in_u, dg = in_dg)
    input  wire          in_bypass,
    input  wire [31:0]   in_u,
    input  wire [31:0]   in_dg,
    input  wire [MW-1:0] in_meta,
    output reg           out_valid,
    output reg  [31:0]   out_y,
    output reg  [31:0]   out_u,
    output reg  [MW-1:0] out_meta,
    output reg           err                // SFU output FIFO overflow (sticky)
);
    localparam [31:0] C_A = 32'h3F3504F3, C_B = 32'h3ECC422A, ONE = 32'h3F800000;
    localparam [2:0]  FN_EXP = 3'd0, FN_ERF = 3'd5;
    localparam TAGW = ROBW + 1;

    // ------------------------------------------------------------------ reorder buffer
    // Flat register file; the head is read through a one-hot pointer (RREP keep-copies, each
    // driving one slice of the HW-bit entry: cm_ohmux) so that no flop fans out to the whole
    // NROB:1 mux (docs/p3_report.md section 6).  rp (binary) only serves the full / empty test.
    localparam HW   = 4*32 + MW;            // {u, dg, er, ex, meta}
    localparam RREP = 4;
    reg  [NROB*32-1:0] rb_u, rb_dg, rb_er, rb_ex;
    reg  [NROB*MW-1:0] rb_m;
    reg  [NROB-1:0] rb_fer, rb_fex;         // results present
    reg  [ROBW:0] wp, rp;
    (* keep *) reg [RREP*NROB-1:0] rp_ohr;  // one-hot copies of rp[ROBW-1:0]
    reg  [NROB-1:0] wp_oh;                  // one-hot wp[ROBW-1:0]: per-entry write enables
    wire rb_full  = (wp - rp) == (1 << ROBW);
    wire rb_empty = (wp == rp);
    // declared early (Icarus binds names in file order): collector result and head pop
    reg             col_v;
    reg  [31:0]     col_y;
    reg  [TAGW-1:0] col_tag;
    wire            pop;

    // ------------------------------------------------------------------ input / request side
    // cycle 0: accept, write ROB entry, start t = mul(u, A) and w = mul(-(u*0.5), u)
    // cycle 1: push erf(t) request (and exp(w) in cycle 2 for bwd)
    reg           ph2;                      // second request cycle of a bwd element
    reg           ph2_v;
    wire          rq_room;                  // request FIFO can take this element's requests
    assign in_ready = ~rb_full & ~ph2 & ~ph2_v & rq_room;
    wire          acc = in_valid & in_ready;
    wire [31:0]   half_u;
    gm_pow2 u_half (.x(in_u), .e(8'hFF), .y(half_u));     // mul(u, 0.5)
    wire [31:0]   t1, w1;
    wire          tv1, wv1;
    fp_mul #(.STAGES(1)) u_mt (.clk(clk), .rst_n(rst_n), .in_valid(acc), .a(in_u), .b(C_A), .out_valid(tv1), .y(t1));
    fp_mul #(.STAGES(1)) u_mw (.clk(clk), .rst_n(rst_n), .in_valid(acc), .a(half_u ^ 32'h80000000), .b(in_u), .out_valid(wv1), .y(w1));
    reg  [ROBW-1:0] slot1;
    reg             req1;
    reg  [31:0]     w2;
    reg  [ROBW-1:0] slot2;
    integer ri;
    always @(posedge clk) begin
        if (!rst_n) begin
            wp <= {(ROBW+1){1'b0}}; req1 <= 1'b0; ph2 <= 1'b0; ph2_v <= 1'b0; rb_fer <= {NROB{1'b0}}; rb_fex <= {NROB{1'b0}};
            rp <= {(ROBW+1){1'b0}};
            for (ri = 0; ri < RREP; ri = ri + 1) rp_ohr[ri*NROB +: NROB] <= {{(NROB-1){1'b0}}, 1'b1};
            wp_oh <= {{(NROB-1){1'b0}}, 1'b1};
        end else begin
            req1 <= acc & ~in_bypass;
            ph2  <= acc & in_bwd & ~in_bypass;
            ph2_v <= ph2;
            if (acc) begin
                rb_fer[wp[ROBW-1:0]] <= in_bypass;
                rb_fex[wp[ROBW-1:0]] <= in_bypass | ~in_bwd;
                wp <= wp + 1'b1;
                wp_oh <= {wp_oh[NROB-2:0], wp_oh[NROB-1]};
                slot1 <= wp[ROBW-1:0];
            end
            if (ph2) begin w2 <= w1; slot2 <= slot1; end
            // SFU result flags (the data is written below, per entry)
            if (col_v) begin
                if (col_tag[0]) rb_fex[col_tag[TAGW-1:1]] <= 1'b1;
                else            rb_fer[col_tag[TAGW-1:1]] <= 1'b1;
            end
            if (pop) begin
                rp <= rp + 1'b1;
                for (ri = 0; ri < RREP; ri = ri + 1)
                    rp_ohr[ri*NROB +: NROB] <= {rp_ohr[ri*NROB +: NROB-1], rp_ohr[ri*NROB + NROB-1]};
            end
        end
    end
    // entry data writes: one enable per entry (no shared decode net fans out to the whole file)
    genvar wi;
    generate
        for (wi = 0; wi < NROB; wi = wi + 1) begin : g_wr
            wire en_in = acc & wp_oh[wi];
            /* verilator lint_off WIDTHEXPAND */
            wire en_er = col_v & ~col_tag[0] & (col_tag[TAGW-1:1] == wi);
            wire en_ex = col_v &  col_tag[0] & (col_tag[TAGW-1:1] == wi);
            /* verilator lint_on WIDTHEXPAND */
            always @(posedge clk) begin
                if (en_in) begin
                    rb_u[wi*32 +: 32]  <= in_u;
                    rb_dg[wi*32 +: 32] <= in_dg;
                    rb_m[wi*MW +: MW]  <= in_meta;
                end
                if (en_er) rb_er[wi*32 +: 32] <= col_y;
                if (en_ex) rb_ex[wi*32 +: 32] <= col_y;
            end
        end
    endgenerate
    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_v = tv1 | wv1;
    /* verilator lint_on UNUSEDSIGNAL */

    // request FIFO (depth 8): {fn, x, tag}
    localparam RQW = 3 + 32 + TAGW;
    reg  [RQW-1:0] rq [0:7];
    reg  [3:0]     rq_wp, rq_rp;
    wire           rq_full  = (rq_wp - rq_rp) == 4'd8;
    wire           rq_empty = (rq_wp == rq_rp);
    assign rq_room = (rq_wp - rq_rp) <= 4'd4;            // <= 2 pending pushes + 2 new requests
    wire           rq_push  = req1 | ph2_v;
    wire [RQW-1:0] rq_in    = req1 ? {FN_ERF, t1, slot1, 1'b0} : {FN_EXP, w2, slot2, 1'b1};
    // in_ready throttles the input so that the FIFO never overflows; err flags it anyway
    wire           disp_fire;
    always @(posedge clk) begin
        if (!rst_n) begin
            rq_wp <= 4'd0; rq_rp <= 4'd0;
        end else begin
            if (rq_push & ~rq_full) begin
                rq[rq_wp[2:0]] <= rq_in;
                rq_wp <= rq_wp + 1'b1;
            end
            if (disp_fire) rq_rp <= rq_rp + 1'b1;
        end
    end
    wire [RQW-1:0] rq_head = rq[rq_rp[2:0]];

    // ------------------------------------------------------------------ SFU pool + dispatcher
    wire [NSFU-1:0] s_ready, s_ov;
    wire [31:0]     s_y   [0:NSFU-1];
    wire [TAGW-1:0] s_tag [0:NSFU-1];
    // first ready SFU
    reg  [NSFU-1:0] pick;
    integer pi;
    always @* begin
        pick = {NSFU{1'b0}};
        for (pi = NSFU - 1; pi >= 0; pi = pi - 1)
            if (s_ready[pi]) pick = {{(NSFU-1){1'b0}}, 1'b1} << pi;
    end
    assign disp_fire = ~rq_empty & (|s_ready);
    // per-SFU output FIFOs (depth 4) + round-robin collector into the ROB (one per cycle)
    reg  [NSFU-1:0] of_nonempty;
    wire [31:0]     of_y   [0:NSFU-1];
    wire [TAGW-1:0] of_tag [0:NSFU-1];
    wire [NSFU-1:0] of_pop;
    reg  [NSFU-1:0] grant;
    integer gi;
    always @* begin
        grant = {NSFU{1'b0}};
        for (gi = NSFU - 1; gi >= 0; gi = gi - 1)
            if (of_nonempty[gi]) grant = {{(NSFU-1){1'b0}}, 1'b1} << gi;
    end
    assign of_pop = grant;
    integer ci;
    always @* begin
        col_v = |of_nonempty; col_y = 32'd0; col_tag = {TAGW{1'b0}};
        for (ci = 0; ci < NSFU; ci = ci + 1)
            if (grant[ci]) begin col_y = col_y | of_y[ci]; col_tag = col_tag | of_tag[ci]; end
    end
    genvar i;
    generate
        for (i = 0; i < NSFU; i = i + 1) begin : g_sfu
            wire ov;
            fp_sfu #(.LF(LF), .NTHR(LF + 2), .TAGW(TAGW)) u_sfu (
                .clk(clk), .rst_n(rst_n), .in_valid(disp_fire & pick[i]), .in_ready(s_ready[i]),
                .in_fn(rq_head[RQW-1 -: 3]), .in_x(rq_head[TAGW +: 32]), .in_tag(rq_head[TAGW-1:0]),
                .out_valid(ov), .out_y(s_y[i]), .out_tag(s_tag[i]));
            reg [31+TAGW:0] f [0:3];
            reg [2:0] fwp, frp;
            wire fempty = (fwp == frp);
            wire ffull  = (fwp - frp) == 3'd4;
            always @(posedge clk) begin
                if (!rst_n) begin fwp <= 3'd0; frp <= 3'd0; end
                else begin
                    if (ov) begin f[fwp[1:0]] <= {s_y[i], s_tag[i]}; fwp <= fwp + 1'b1; end
                    if (of_pop[i] & ~fempty) frp <= frp + 1'b1;
                end
            end
            assign s_ov[i] = ov & ffull;
            always @* of_nonempty[i] = ~fempty;
            assign of_y[i]   = f[frp[1:0]][31+TAGW:TAGW];
            assign of_tag[i] = f[frp[1:0]][TAGW-1:0];
        end
    endgenerate
    always @(posedge clk)
        if (!rst_n) err <= 1'b0;
        else if ((|s_ov) | (rq_push & rq_full)) err <= 1'b1;

    // ------------------------------------------------------------------ ROB head -> tail pipeline
    wire head_ok = ~rb_empty & (|(rp_ohr[NROB-1:0] & rb_fer & rb_fex));
    assign pop = head_ok;
    wire [NROB*HW-1:0] rb_all;              // entry-major {u, dg, er, ex, meta}
    genvar ei;
    generate
        for (ei = 0; ei < NROB; ei = ei + 1) begin : g_ent
            assign rb_all[ei*HW +: HW] = {rb_u[ei*32 +: 32], rb_dg[ei*32 +: 32], rb_er[ei*32 +: 32],
                                          rb_ex[ei*32 +: 32], rb_m[ei*MW +: MW]};
        end
    endgenerate
    wire [31:0]   h_u, h_dg, h_er, h_ex;
    wire [MW-1:0] h_m;
    cm_ohmux #(.W(HW), .N(NROB), .REP(RREP)) u_head (.sel(rp_ohr), .d(rb_all), .y({h_u, h_dg, h_er, h_ex, h_m}));
    reg           bwd_mode;                 // constant per GEMM: taken from the last accepted element
    always @(posedge clk) if (acc) bwd_mode <= in_bwd;
    // t0: pop (registered operands)
    reg           p0_v;
    reg  [31:0]   p0_u, p0_dg, p0_er, p0_ex;
    reg  [MW-1:0] p0_m;
    always @(posedge clk) begin
        if (!rst_n) p0_v <= 1'b0; else p0_v <= pop;
        p0_u <= h_u; p0_dg <= h_dg; p0_er <= h_er; p0_ex <= h_ex; p0_m <= h_m;
    end
    // t1..t2: s = add(1, er); pdf = mul(ex, B) (t1, delayed to t2)
    wire [31:0] s2, pdf1, pdf2;
    wire        sv2, pv1;
    fp_add #(.STAGES(2)) u_add1 (.clk(clk), .rst_n(rst_n), .in_valid(p0_v), .a(ONE), .b(p0_er), .sub(1'b0), .out_valid(sv2), .y(s2));
    fp_mul #(.STAGES(1)) u_pdf (.clk(clk), .rst_n(rst_n), .in_valid(p0_v), .a(p0_ex), .b(C_B), .out_valid(pv1), .y(pdf1));
    cm_delay #(.W(32), .N(1), .RST(0)) u_pd (.clk(clk), .rst_n(rst_n), .d(pdf1), .q(pdf2));
    wire [31:0]   u2, dg2;
    wire [MW-1:0] m2;
    cm_delay #(.W(64+MW), .N(2), .RST(0)) u_d2 (.clk(clk), .rst_n(rst_n), .d({p0_u, p0_dg, p0_m}), .q({u2, dg2, m2}));
    // t2: fwd: h = mul(u, 0.5), g = mul(h, s) (t3);  bwd: cdf = mul(s, 0.5), f = fma(u, pdf, cdf) (t4)
    wire [31:0] hu2, cdf2, f4;
    wire        fv4;
    gm_pow2 u_hu  (.x(u2), .e(8'hFF), .y(hu2));
    gm_pow2 u_cdf (.x(s2), .e(8'hFF), .y(cdf2));
    fp_fma #(.STAGES(2)) u_fma (.clk(clk), .rst_n(rst_n), .in_valid(sv2), .a(u2), .b(pdf2), .c(cdf2), .out_valid(fv4), .y(f4));
    wire [31:0]   dg4;
    cm_delay #(.W(32), .N(2), .RST(0)) u_d4 (.clk(clk), .rst_n(rst_n), .d(dg2), .q(dg4));
    // shared final multiplier: fwd (h, s) at t2 -> t3;  bwd (dg, f) at t4 -> t5
    wire [31:0] mo;
    wire        mov;
    fp_mul #(.STAGES(1)) u_mo (.clk(clk), .rst_n(rst_n), .in_valid(bwd_mode ? fv4 : sv2),
        .a(bwd_mode ? dg4 : hu2), .b(bwd_mode ? f4 : s2), .out_valid(mov), .y(mo));
    // align: fwd result (t3) delayed 2 -> t5
    wire [31:0]   g5;
    wire          gv5;
    wire [31:0]   u3, u5;
    wire [MW-1:0] m3, m5;
    cm_delay #(.W(32), .N(2), .RST(0)) u_g5 (.clk(clk), .rst_n(rst_n), .d(mo), .q(g5));
    cm_delay #(.W(1), .N(2), .RST(1)) u_gv5 (.clk(clk), .rst_n(rst_n), .d(mov), .q(gv5));
    cm_delay #(.W(32+MW), .N(1), .RST(0)) u_d3 (.clk(clk), .rst_n(rst_n), .d({u2, m2}), .q({u3, m3}));
    cm_delay #(.W(32+MW), .N(2), .RST(0)) u_d5 (.clk(clk), .rst_n(rst_n), .d({u3, m3}), .q({u5, m5}));
    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_t = pv1;
    /* verilator lint_on UNUSEDSIGNAL */
    always @(posedge clk) begin
        if (!rst_n) out_valid <= 1'b0;
        else        out_valid <= bwd_mode ? mov : gv5;
        out_y    <= bwd_mode ? mo : g5;
        out_u    <= u5;
        out_meta <= m5;
    end
endmodule
`default_nettype wire
