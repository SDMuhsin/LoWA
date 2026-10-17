`default_nettype none
// lw_scal: LOWA_SCAL, the per-step optimizer scalars of docs/dataflow.md section 6.2 in fp32 on chip,
// op for op as model/hwmodel/step.py (L1 entry point H.step_scalars):
//   sf = i2f(s); f = s < w ? mul(sf, INV_W) : fmax(mul(i2f(T - s), INV_TW), 0)
//   b1t = mul(b1t, b1); b2t = mul(b2t, b2); bc1 = sub(1, b1t); rbc2 = recip(sqrt(sub(1, b2t)))
//   per group g: lr = mul(lr0_g, f); step = mul(lr, recip(bc1)); decay = sub(1, mul(lr, wd))
// s = completed steps, so the first update runs at lr = 0 exactly.  Sequential micro-sequence on one
// fp_mul, fp_add, fp_i2f, fp_cmp and one fp_sfu (one op in flight at a time; ~300 cycles).
// CONST block (16 words at d_const_base): inv_w, inv_tw, lr0[0], lr0[1], wd, b1, b2, b1c, b2c, eps.
// OPT_SCAL block (24 words at d_scal_base): in  [0] b1t_prev, [1] b2t_prev (1.0 before the first step);
//   out [0..7] b1t, b2t, bc1, rbc2, f, lr[0], lr[1], 0;  [8..15] group 0 and [16..23] group 1:
//   decay, step, rbc2, b1c, b2, b2c, eps, (word 7 = the clip coefficient, written by the vector
//   unit's GN_FIN, left untouched: write mask 0x7F).  Groups are what lw_adamw reads (d_sc_base).
module lw_scal (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    output wire         busy,
    output reg          done,
    input  wire [31:0]  d_s, d_w, d_t, d_const_base, d_scal_base,
    output wire         rd_req_valid,
    input  wire         rd_req_ready,
    output wire [31:0]  rd_req_addr,
    output wire [6:0]   rd_req_len,
    input  wire         rd_dat_valid,
    output wire         rd_dat_ready,
    input  wire [255:0] rd_dat,
    output reg          wr_valid,
    input  wire         wr_ready,
    output reg  [31:0]  wr_addr,
    output reg  [255:0] wr_data,
    output reg  [7:0]   wr_mask
);
    localparam [2:0] S_IDLE = 3'd0, S_CONST = 3'd1, S_PREV = 3'd2, S_RUN = 3'd3, S_W0 = 3'd4, S_W1 = 3'd5, S_W2 = 3'd6;
    reg  [2:0]  st;
    assign busy = (st != S_IDLE);
    reg         rq_go;
    reg  [31:0] rq_base;
    reg  [15:0] rq_n;
    wire        rq_busy, rq_done, rw_valid;
    wire [31:0] rw_d;
    wire [15:0] rw_idx;
    lw_rdseq u_rd (.clk(clk), .rst_n(rst_n), .go(rq_go), .base(rq_base), .nwords(rq_n), .busy(rq_busy), .done(rq_done),
                   .req_valid(rd_req_valid), .req_ready(rd_req_ready), .req_addr(rd_req_addr), .req_len(rd_req_len),
                   .dat_valid(rd_dat_valid), .dat_ready(rd_dat_ready), .dat(rd_dat),
                   .w_valid(rw_valid), .w_ready(1'b1), .w_d(rw_d), .w_idx(rw_idx));
    // constants and state
    reg  [31:0] c_inv_w, c_inv_tw, c_lr0_0, c_lr0_1, c_wd, c_b1, c_b2, c_b1c, c_b2c, c_eps, b1t_p, b2t_p;
    always @(posedge clk) if (rw_valid) begin
        if (st == S_CONST) case (rw_idx[3:0])
            4'd0: c_inv_w <= rw_d;  4'd1: c_inv_tw <= rw_d;  4'd2: c_lr0_0 <= rw_d;  4'd3: c_lr0_1 <= rw_d;
            4'd4: c_wd <= rw_d;     4'd5: c_b1 <= rw_d;      4'd6: c_b2 <= rw_d;     4'd7: c_b1c <= rw_d;
            4'd8: c_b2c <= rw_d;    4'd9: c_eps <= rw_d;     default: ;
        endcase
        if (st == S_PREV) begin if (rw_idx[0]) b2t_p <= rw_d; else b1t_p <= rw_d; end
    end
    // results
    reg  [31:0] sf, tms, f1, f2, f2m, fr, b1t, b2t, bc1, xb2, sq, rbc2, rb1, lr0, st0, lw0, dc0, lr1, st1, lw1, dc1;
    // units
    reg         i_v, m_v, a_v, c_v, s_v;
    reg  [31:0] i_x, m_a, m_b, a_a, a_b, c_a, c_b, s_x;
    reg  [2:0]  s_fn;
    wire        i_ov, m_ov, a_ov, c_ov, s_ov, s_rdy;
    wire [31:0] i_y, m_y, a_y, c_mx, s_y;
    wire        c_lt, c_eq, c_gt, c_un;
    wire [31:0] c_mn;
    wire [3:0]  s_tag;
    fp_i2f #(.STAGES(1)) u_i (.clk(clk), .rst_n(rst_n), .in_valid(i_v), .x(i_x), .out_valid(i_ov), .y(i_y));
    fp_mul #(.STAGES(2)) u_m (.clk(clk), .rst_n(rst_n), .in_valid(m_v), .a(m_a), .b(m_b), .out_valid(m_ov), .y(m_y));
    fp_add #(.STAGES(2)) u_a (.clk(clk), .rst_n(rst_n), .in_valid(a_v), .a(a_a), .b(a_b), .sub(1'b1), .out_valid(a_ov), .y(a_y));
    fp_cmp #(.STAGES(1)) u_c (.clk(clk), .rst_n(rst_n), .in_valid(c_v), .a(c_a), .b(c_b), .out_valid(c_ov),
                               .lt(c_lt), .eq(c_eq), .gt(c_gt), .un(c_un), .mx(c_mx), .mn(c_mn));
    fp_sfu #(.LF(2), .NTHR(1), .TAGW(4)) u_s (.clk(clk), .rst_n(rst_n), .in_valid(s_v), .in_ready(s_rdy), .in_fn(s_fn), .in_x(s_x),
                                               .in_tag(4'd0), .out_valid(s_ov), .out_y(s_y), .out_tag(s_tag));
    wire unused_u = c_lt | c_eq | c_gt | c_un | (|c_mn) | (|s_tag) | rq_busy | (|rw_idx[15:4]);
    localparam [2:0] FN_RECIP = 3'd1, FN_SQRT = 3'd3;
    reg  [4:0]  step;
    reg         wait_r;                   // an op is in flight
    wire [31:0] tsub = d_t - d_s;
    wire        res_v = i_ov | m_ov | a_ov | c_ov | s_ov;
    wire [31:0] res = i_ov ? i_y : m_ov ? m_y : a_ov ? a_y : c_ov ? c_mx : s_y;
    always @(posedge clk)
        if (!rst_n) begin
            st <= S_IDLE; rq_go <= 1'b0; done <= 1'b0; wr_valid <= 1'b0; step <= 5'd0; wait_r <= 1'b0;
            i_v <= 1'b0; m_v <= 1'b0; a_v <= 1'b0; c_v <= 1'b0; s_v <= 1'b0;
        end else begin
            rq_go <= 1'b0; done <= 1'b0;
            i_v <= 1'b0; m_v <= 1'b0; a_v <= 1'b0; c_v <= 1'b0;
            if (s_v & s_rdy) s_v <= 1'b0;
            case (st)
                S_IDLE: if (start) begin st <= S_CONST; rq_go <= 1'b1; rq_base <= d_const_base; rq_n <= 16'd16; end
                S_CONST: if (rq_done) begin st <= S_PREV; rq_go <= 1'b1; rq_base <= d_scal_base; rq_n <= 16'd2; end
                S_PREV: if (rq_done) begin st <= S_RUN; step <= 5'd0; wait_r <= 1'b0; end
                S_RUN: begin
                    if (!wait_r) begin
                        wait_r <= 1'b1;
                        case (step)
                            5'd0:  begin i_v <= 1'b1; i_x <= d_s; end
                            5'd1:  begin i_v <= 1'b1; i_x <= tsub; end
                            5'd2:  begin m_v <= 1'b1; m_a <= sf; m_b <= c_inv_w; end
                            5'd3:  begin m_v <= 1'b1; m_a <= tms; m_b <= c_inv_tw; end
                            5'd4:  begin c_v <= 1'b1; c_a <= f2; c_b <= 32'd0; end
                            5'd5:  begin wait_r <= 1'b0; fr <= (d_s < d_w) ? f1 : f2m; end
                            5'd6:  begin m_v <= 1'b1; m_a <= b1t_p; m_b <= c_b1; end
                            5'd7:  begin m_v <= 1'b1; m_a <= b2t_p; m_b <= c_b2; end
                            5'd8:  begin a_v <= 1'b1; a_a <= 32'h3F800000; a_b <= b1t; end
                            5'd9:  begin a_v <= 1'b1; a_a <= 32'h3F800000; a_b <= b2t; end
                            5'd10: begin s_v <= 1'b1; s_fn <= FN_SQRT; s_x <= xb2; end
                            5'd11: begin s_v <= 1'b1; s_fn <= FN_RECIP; s_x <= sq; end
                            5'd12: begin s_v <= 1'b1; s_fn <= FN_RECIP; s_x <= bc1; end
                            5'd13: begin m_v <= 1'b1; m_a <= c_lr0_0; m_b <= fr; end
                            5'd14: begin m_v <= 1'b1; m_a <= lr0; m_b <= rb1; end
                            5'd15: begin m_v <= 1'b1; m_a <= lr0; m_b <= c_wd; end
                            5'd16: begin a_v <= 1'b1; a_a <= 32'h3F800000; a_b <= lw0; end
                            5'd17: begin m_v <= 1'b1; m_a <= c_lr0_1; m_b <= fr; end
                            5'd18: begin m_v <= 1'b1; m_a <= lr1; m_b <= rb1; end
                            5'd19: begin m_v <= 1'b1; m_a <= lr1; m_b <= c_wd; end
                            5'd20: begin a_v <= 1'b1; a_a <= 32'h3F800000; a_b <= lw1; end
                            default: begin
                                wait_r <= 1'b0; st <= S_W0; wr_valid <= 1'b1; wr_addr <= d_scal_base; wr_mask <= 8'hFF;
                                wr_data <= {32'd0, lr1, lr0, fr, rbc2, bc1, b2t, b1t};
                            end
                        endcase
                        if (step != 5'd5) step <= step + 5'd1; else step <= 5'd6;
                    end else if (res_v) begin
                        wait_r <= 1'b0;
                        case (step)
                            5'd1:  sf <= res;    5'd2:  tms <= res;   5'd3:  f1 <= res;    5'd4:  f2 <= res;
                            5'd5:  f2m <= res;   5'd7:  b1t <= res;   5'd8:  b2t <= res;   5'd9:  bc1 <= res;
                            5'd10: xb2 <= res;   5'd11: sq <= res;    5'd12: rbc2 <= res;  5'd13: rb1 <= res;
                            5'd14: lr0 <= res;   5'd15: st0 <= res;   5'd16: lw0 <= res;   5'd17: dc0 <= res;
                            5'd18: lr1 <= res;   5'd19: st1 <= res;   5'd20: lw1 <= res;   5'd21: dc1 <= res;
                            default: ;
                        endcase
                    end
                end
                S_W0: if (wr_ready) begin
                    wr_addr <= d_scal_base + 32'd32; wr_mask <= 8'h7F; st <= S_W1;
                    wr_data <= {32'd0, c_eps, c_b2c, c_b2, c_b1c, rbc2, st0, dc0};
                end
                S_W1: if (wr_ready) begin
                    wr_addr <= d_scal_base + 32'd64; st <= S_W2;
                    wr_data <= {32'd0, c_eps, c_b2c, c_b2, c_b1c, rbc2, st1, dc1};
                end
                S_W2: if (wr_ready) begin wr_valid <= 1'b0; st <= S_IDLE; done <= 1'b1; end
                default: st <= S_IDLE;
            endcase
        end
endmodule
`default_nettype wire
