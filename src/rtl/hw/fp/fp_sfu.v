`default_nettype none
// fp_sfu: special-function unit, exp / recip / rsqrt / sqrt / tanh / erf / ln exactly as
// docs/arith.md section 3 (same seeds, constants, Newton/FMA sequences), bit-identical to the L1
// entry points.  Every rounded op runs on ONE fp_fma (mul = fma(x, y, -0), add = fma(x, 1, y),
// sub = fma(x, 1, -y), all bit-identical to the separate ops); the integer steps (special-value
// tests, exponent split, rsqrt seed, n = bits(t) - bits(M), 2^k construction, float(E), sign OR)
// are small integer logic around it.
//
// Micro-sequenced barrel: T = LF + 2 thread slots take turns, one per cycle (slot = cycle mod T).
// In its turn a busy slot issues the next micro-op of its program (fp_sfu_ucode) to the FMA; the
// result is written back to the slot's 4 registers before the slot's next turn, so there are no
// hazards and the FMA takes one op per cycle when all slots are busy.
//   cycle k      : read slot context, micro-op ROM, operand muxes      -> issue registers
//   k+1 .. k+LF  : fp_fma (STAGES = LF)
//   k+LF+1       : write-back (+ EXPN / RSPLIT integer side effects) or output post-processing
// Steps per element N (programs incl. branch padding, so the latency is fixed per function):
//   exp 13, recip 9, rsqrt 13, sqrt 17, tanh 25, erf 24, ln 13 (fn 0..6; fn 7 -> NaN, N = 1).
// Interface: in_valid/in_ready on the input (a new element enters slot `cnt` in its turn, either
// when the slot is free or in the turn in which its previous element issues its last op);
// out_valid/out_y/out_tag are valid-only.  Latency = (N + 1) * T cycles, fixed per function;
// throughput = NTHR / (N * T) elements per cycle (= 1/N with NTHR = T).  Results of the same
// function leave in order; mixed functions may reorder, hence out_tag (= in_tag).
module fp_sfu #(
    parameter LF   = 2,               // fp_fma STAGES
    parameter NTHR = LF + 2,          // active thread slots, 1 .. LF + 2
    parameter TAGW = 4
) (
    input  wire            clk,
    input  wire            rst_n,
    input  wire            in_valid,
    output wire            in_ready,
    input  wire [2:0]      in_fn,
    input  wire [31:0]     in_x,
    input  wire [TAGW-1:0] in_tag,
    output reg             out_valid,
    output reg  [31:0]     out_y,
    output reg  [TAGW-1:0] out_tag
);
    localparam T  = LF + 2;
    localparam SW = (T <= 2) ? 1 : (T <= 4) ? 2 : (T <= 8) ? 3 : 4;   // slot index width (T <= 16)
    /* verilator lint_off WIDTHTRUNC */   // sized copies of integer parameters (values fit)
    localparam [SW-1:0] TM1 = T - 1;
    localparam [SW:0]   NT  = NTHR;       // SW+1 bits: NTHR = T must not wrap to 0
    /* verilator lint_on WIDTHTRUNC */

    // ------------------------------------------------------------------ slot counter
    reg  [SW-1:0] cnt;
    always @(posedge clk)
        if (!rst_n || cnt == TM1) cnt <= {SW{1'b0}};
        else                      cnt <= cnt + 1'b1;
    wire act = ({1'b0, cnt} < NT);

    // ------------------------------------------------------------------ slot contexts
    // control: busy fn br step s spc sps e1 e2 tag; data: R0..R3
    reg            busy [0:T-1];
    reg  [2:0]     cfn  [0:T-1];
    reg            cbr  [0:T-1];
    reg  [4:0]     cstp [0:T-1];
    reg            cs   [0:T-1];
    reg  [2:0]     cspc [0:T-1];
    reg            csps [0:T-1];
    reg  [7:0]     ce1  [0:T-1];
    reg  [7:0]     ce2  [0:T-1];
    reg  [TAGW-1:0] ctag [0:T-1];
    reg  [31:0]    r0   [0:T-1];
    reg  [31:0]    r1   [0:T-1];
    reg  [31:0]    r2   [0:T-1];
    reg  [31:0]    r3   [0:T-1];

    wire        cur_busy = busy[cnt];
    wire [2:0]  cur_fn   = cfn[cnt];
    wire        cur_br   = cbr[cnt];
    wire [4:0]  cur_stp  = cstp[cnt];
    wire [7:0]  cur_e1   = ce1[cnt];
    wire [7:0]  cur_e2   = ce2[cnt];
    wire [31:0] cur_r0 = r0[cnt], cur_r1 = r1[cnt], cur_r2 = r2[cnt], cur_r3 = r3[cnt];

    // ------------------------------------------------------------------ micro-op ROM + operands
    wire [18:0] uc;
    wire [31:0] kab, kc;
    fp_sfu_ucode u_rom (.fn(cur_fn), .br(cur_br), .step(cur_stp), .c(uc), .kab(kab), .kc(kc));
    // uc = {iss, asel[2:0], nega, bsel[2:0], csel[2:0], negc, wen, dst[1:0], wb[1:0], last, ors}
    wire [2:0] u_asel = uc[17:15];
    wire       u_nega = uc[14];
    wire [2:0] u_bsel = uc[13:11];
    wire [2:0] u_csel = uc[10:8];
    wire       u_negc = uc[7];
    wire       u_wen  = uc[6];
    wire [1:0] u_dst  = uc[5:4];
    wire [1:0] u_wb   = uc[3:2];
    wire       u_last = uc[1];
    wire       u_ors  = uc[0];
    /* verilator lint_off UNUSEDSIGNAL */   // uc[18] (iss) only documents NOPs: they issue harmlessly
    wire       u_iss  = uc[18];
    /* verilator lint_on UNUSEDSIGNAL */

    reg [31:0] opa, opb, opc;
    always @* begin
        case (u_asel)
            3'd0: opa = cur_r0;  3'd1: opa = cur_r1;  3'd2: opa = cur_r2;  3'd3: opa = cur_r3;
            default: opa = kab;
        endcase
        case (u_bsel)
            3'd0: opb = cur_r0;  3'd1: opb = cur_r1;  3'd2: opb = cur_r2;  3'd3: opb = cur_r3;
            3'd5: opb = {1'b0, cur_e1, 23'd0};
            3'd6: opb = {1'b0, cur_e2, 23'd0};
            default: opb = kab;
        endcase
        case (u_csel)
            3'd0: opc = cur_r0;  3'd1: opc = cur_r1;  3'd2: opc = cur_r2;  3'd3: opc = cur_r3;
            default: opc = kc;
        endcase
    end

    wire iss_v  = act & cur_busy;
    assign in_ready = act & (~cur_busy | u_last);
    wire accept = in_valid & in_ready;

    // ------------------------------------------------------------------ accept-time pre-processing
    wire        xs  = in_x[31];
    wire [7:0]  xe  = in_x[30:23];
    wire [22:0] xf  = in_x[22:0];
    wire        xz  = (xe == 8'd0);                      // zero or subnormal (DAZ)
    wire        xn  = (xe == 8'hFF) & (xf != 23'd0);
    wire        xi  = (xe == 8'hFF) & (xf == 23'd0);
    wire [31:0] xd  = xz ? {xs, 31'd0} : in_x;            // daz(x)
    wire [30:0] xa  = xd[30:0];                           // abs(daz(x))
    wire [31:0] m1  = {9'h07F, xf};                       // significand, exponent field 127
    wire [8:0]  ee  = {1'b0, xe} - 9'd127;                // E (two's complement, 9 bits)
    // recip: k = -E
    wire [8:0]  rk  = 9'd127 - {1'b0, xe};
    /* verilator lint_off UNUSEDSIGNAL */   // bit 8 of the halved exponents: only [7:0] is used
    wire [8:0]  rk1 = {rk[8], rk[8:1]};
    wire [7:0]  r_e1 = rk1[7:0] + 8'd127;
    wire [7:0]  r_e2 = rk[7:0] - rk1[7:0] + 8'd127;
    // rsqrt / sqrt: k = E >> 1 (floor), m = x 2^-2k in [1, 4), seed 0x5F3759DF - (bits(m) >> 1)
    wire [8:0]  qk  = {ee[8], ee[8:1]};
    /* verilator lint_on UNUSEDSIGNAL */
    wire [31:0] qm  = {1'b0, 8'd127 + {7'd0, ee[0]}, xf};
    wire [31:0] qy  = 32'h5F3759DF - {1'b0, qm[31:1]};
    wire [7:0]  q_e1r = 8'd127 - qk[7:0];                 // 2^-k  (rsqrt)
    wire [7:0]  q_e1s = 8'd127 + qk[7:0];                 // 2^k   (sqrt)
    // ln: mb in (sqrt(.5), sqrt(2)], E adjusted, float(E) exact (|E| <= 128)
    wire        lbig = (xf > 23'h3504F3);
    wire [31:0] lmb  = {1'b0, lbig ? 8'd126 : 8'd127, xf};
    wire [8:0]  le   = ee + {8'd0, lbig};
    wire [7:0]  lmag = le[8] ? (~le[7:0] + 8'd1) : le[7:0];
    /* verilator lint_off UNUSEDSIGNAL */   // top bit: hidden 1 / sign copy, only the low bits are used
    wire [7:0]  lnm;
    /* verilator lint_on UNUSEDSIGNAL */
    wire [2:0]  llz;
    fp_norm #(.W(8), .CW(3)) u_lnorm (.x(lmag), .y(lnm), .lz(llz));
    wire [31:0] lef  = (lmag == 8'd0) ? 32'd0 : {le[8], 8'd134 - {5'd0, llz}, lnm[6:0], 16'd0};
    // tanh / erf thresholds on a = abs(daz(x))
    wire        t_sat = (xa >= 31'h41100000);             // a >= 9
    wire        t_br  = (xa >= 31'h3F800000);             // a >= 1
    wire        e_sat = (xa >= 31'h407AE148);             // a >= 3.92
    wire        e_br  = (xa >= 31'h3F6C0000);             // a >= 0.921875

    localparam [2:0] SP_NONE = 3'd0, SP_NAN = 3'd1, SP_INF = 3'd2, SP_ZERO = 3'd3, SP_ONE = 3'd4;
    reg [2:0]  p_spc;
    reg        p_sps, p_br;
    reg [7:0]  p_e1, p_e2;
    reg [31:0] p_r0, p_r1;
    always @* begin
        p_spc = SP_NONE; p_sps = 1'b0; p_br = 1'b0;
        p_e1 = 8'd0; p_e2 = 8'd0; p_r0 = xd; p_r1 = {1'b0, xa};
        case (in_fn)
            3'd0: begin                                   // exp
                if (xn)                                   p_spc = SP_NAN;
                else if (~xs & (xa > 31'h42B20000))       p_spc = SP_INF;    // x > 89
                else if (xs & (xa > 31'h42D00000))        p_spc = SP_ZERO;   // x < -104
            end
            3'd1: begin                                   // recip
                if (xn)      p_spc = SP_NAN;
                else if (xi) begin p_spc = SP_ZERO; p_sps = xs; end
                else if (xz) begin p_spc = SP_INF;  p_sps = xs; end
                p_r0 = m1; p_e1 = r_e1; p_e2 = r_e2;
            end
            3'd2, 3'd3: begin                             // rsqrt, sqrt
                if (xn)      p_spc = SP_NAN;
                else if (xz) begin p_spc = (in_fn == 3'd2) ? SP_INF : SP_ZERO; p_sps = xs; end
                else if (xs) p_spc = SP_NAN;
                else if (xi) p_spc = (in_fn == 3'd2) ? SP_ZERO : SP_INF;
                p_r0 = qm; p_r1 = qy; p_e1 = (in_fn == 3'd2) ? q_e1r : q_e1s;
            end
            3'd4: begin                                   // tanh
                if (xn)         p_spc = SP_NAN;
                else if (t_sat) begin p_spc = SP_ONE; p_sps = xs; end
                p_br = t_br;
            end
            3'd5: begin                                   // erf
                if (xn)         p_spc = SP_NAN;
                else if (e_sat) begin p_spc = SP_ONE; p_sps = xs; end
                p_br = e_br;
            end
            3'd6: begin                                   // ln
                if (xn)      p_spc = SP_NAN;
                else if (xz) begin p_spc = SP_INF; p_sps = 1'b1; end
                else if (xs) p_spc = SP_NAN;
                else if (xi) p_spc = SP_INF;
                p_r0 = lmb; p_r1 = lef;
            end
            default: p_spc = SP_NAN;                      // fn 7: reserved
        endcase
    end

    // ------------------------------------------------------------------ issue registers + FMA
    // sideband: {last, ors, s, spc, sps, tag, slot, wen, dst, wb}
    localparam SBW = 1 + 1 + 1 + 3 + 1 + TAGW + SW + 1 + 2 + 2;
    reg            iv;
    reg  [31:0]    ia, ib, ic;
    reg  [SBW-1:0] isb;
    always @(posedge clk) begin
        if (!rst_n) iv <= 1'b0;
        else        iv <= iss_v;
        ia  <= opa ^ {u_nega, 31'd0};
        ib  <= opb;
        ic  <= opc ^ {u_negc, 31'd0};
        isb <= {u_last, u_ors, cs[cnt], cspc[cnt], csps[cnt], ctag[cnt], cnt, u_wen, u_dst, u_wb};
    end

    wire        fv;
    wire [31:0] fy;
    fp_fma #(.STAGES(LF)) u_fma (.clk(clk), .rst_n(rst_n), .in_valid(iv), .a(ia), .b(ib), .c(ic),
                                 .out_valid(fv), .y(fy));
    wire [SBW-1:0] wsb;
    cm_delay #(.W(SBW), .N(LF), .RST(0)) u_sb (.clk(clk), .rst_n(rst_n), .d(isb), .q(wsb));
    wire            w_last = wsb[SBW-1];
    wire            w_ors  = wsb[SBW-2];
    wire            w_s    = wsb[SBW-3];
    wire [2:0]      w_spc  = wsb[SBW-4 -: 3];
    wire            w_sps  = wsb[SBW-7];
    wire [TAGW-1:0] w_tag  = wsb[SBW-8 -: TAGW];
    wire [SW-1:0]   w_slot = wsb[SW+4 : 5];
    wire            w_wen  = wsb[4];
    wire [1:0]      w_dst  = wsb[3:2];
    wire [1:0]      w_wb   = wsb[1:0];

    // ------------------------------------------------------------------ write-back side effects
    // EXPN: n = bits(t) - 0x4B400000 in [-150, 128]; e1 = (n >> 1) + 127, e2 = (n - (n >> 1)) + 127
    wire [31:0] xn32 = fy - 32'h4B400000;
    wire [8:0]  nn   = xn32[8:0];
    /* verilator lint_off UNUSEDSIGNAL */   // top bit: hidden 1 / sign copy, only the low bits are used
    wire [8:0]  nn1  = {nn[8], nn[8:1]};
    /* verilator lint_on UNUSEDSIGNAL */
    wire [7:0]  x_e1 = nn1[7:0] + 8'd127;
    wire [7:0]  x_e2 = nn[7:0] - nn1[7:0] + 8'd127;
    // RSPLIT: d -> m = significand with exponent field 127, k = -E
    wire [8:0]  dk   = 9'd127 - {1'b0, fy[30:23]};
    /* verilator lint_off UNUSEDSIGNAL */   // top bit: hidden 1 / sign copy, only the low bits are used
    wire [8:0]  dk1  = {dk[8], dk[8:1]};
    /* verilator lint_on UNUSEDSIGNAL */
    wire [7:0]  d_e1 = dk1[7:0] + 8'd127;
    wire [7:0]  d_e2 = dk[7:0] - dk1[7:0] + 8'd127;
    wire [31:0] wdat = (w_wb == 2'd2) ? {9'h07F, fy[22:0]} : fy;
    wire        wr   = fv & w_wen;
    wire        we12 = fv & (w_wb != 2'd0);
    wire [7:0]  w_e1 = (w_wb == 2'd2) ? d_e1 : x_e1;
    wire [7:0]  w_e2 = (w_wb == 2'd2) ? d_e2 : x_e2;
    /* verilator lint_off UNUSEDSIGNAL */   // xn32[31:9]: n fits in 9 bits for every in-range t
    wire        unused_x = |xn32[31:9];
    /* verilator lint_on UNUSEDSIGNAL */

    // ------------------------------------------------------------------ context update
    integer j;
    always @(posedge clk) begin
        for (j = 0; j < T; j = j + 1) begin
            // turn of slot j: accept a new element or advance
            if (act && cnt == j[SW-1:0]) begin
                if (accept) begin
                    busy[j] <= 1'b1;   cfn[j] <= in_fn;  cbr[j] <= p_br;  cstp[j] <= 5'd0;
                    cs[j]   <= xs;     cspc[j] <= p_spc; csps[j] <= p_sps;
                    ce1[j]  <= p_e1;   ce2[j] <= p_e2;   ctag[j] <= in_tag;
                    r0[j]   <= p_r0;   r1[j]  <= p_r1;
                end else if (cur_busy) begin
                    cstp[j] <= cur_stp + 5'd1;
                    if (u_last) busy[j] <= 1'b0;
                end
            end
            // write-back of slot j (never in slot j's own turn: it lands at turn - 1)
            if (fv && w_slot == j[SW-1:0]) begin
                if (wr) begin
                    case (w_dst)
                        2'd0: r0[j] <= wdat;
                        2'd1: r1[j] <= wdat;
                        2'd2: r2[j] <= wdat;
                        default: r3[j] <= wdat;
                    endcase
                end
                if (we12) begin
                    ce1[j] <= w_e1;
                    ce2[j] <= w_e2;
                end
            end
            if (!rst_n) busy[j] <= 1'b0;
        end
    end

    // ------------------------------------------------------------------ output
    wire [31:0] yo = fy | {w_ors & w_s, 31'd0};
    reg  [31:0] yf;
    always @* begin
        case (w_spc)
            SP_NAN:  yf = 32'h7FC00000;
            SP_INF:  yf = {w_sps, 8'hFF, 23'd0};
            SP_ZERO: yf = {w_sps, 31'd0};
            SP_ONE:  yf = {w_sps, 8'h7F, 23'd0};
            default: yf = yo;
        endcase
    end
    always @(posedge clk) begin
        if (!rst_n) out_valid <= 1'b0;
        else        out_valid <= fv & w_last;
        out_y   <= yf;
        out_tag <= w_tag;
    end
endmodule
`default_nettype wire
