`default_nettype none
// lw_sg: LoWA sampled coefficient gradient (docs/dataflow.md section 4, PI ruling R1: summed per
// atom).  For t = 0 .. tokens-1 the dY row and the X row (D words each) are loaded into two
// single-port row banks; for every atom j (ascending) the 7-word windows dY[t, refl(ci - 3 .. ci + 3)]
// and X[t, refl(cj - 3 .. cj + 3)] are read (one word per cycle per bank, 7 cycles per atom) and
// handed to one of NL x 2 chain slots; the slot's lane does, for tap = 0 .. 24 in nonzero(H) order,
//     p = mul(yw[di + 3], xw[dj + 3]);  acc_j = fma(h_tap, p, acc_j)
// one mul and one fma every other cycle (the lane's two slots alternate, so a 2-stage FMA feeds its
// own result back without a bubble; L1 lw_lowa_sg order: t outer, tap inner, per atom).  acc_j lives
// in a 256-word bank (+0 at the start); at the end grad_j = mul(gamma, acc_j) is written out.
// Reflection fold: p < 0 -> -1 - p, p >= D -> 2D - 1 - p (sgwa_adapter._reflect).
// Rates: 7 cycles per atom (window bound), D cycles per row pair (single-buffered, 1 word/cycle).
module lw_sg #(
    parameter D  = 768,                  // row length (m = n = D)
    parameter NL = 4                     // lanes (2 slots each); 8 slots x 7 cycles >= the 50-cycle chain
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    output wire         busy,
    output reg          done,
    input  wire [31:0]  d_dy_base, d_dy_rs, d_x_base, d_x_rs, d_ctr_base, d_h_base, d_gamma, d_grad_base,
    input  wire [15:0]  d_tok,
    // rd0: h, centres, dY rows;  rd1: X rows
    output wire         rd0_req_valid,
    input  wire         rd0_req_ready,
    output wire [31:0]  rd0_req_addr,
    output wire [6:0]   rd0_req_len,
    input  wire         rd0_dat_valid,
    output wire         rd0_dat_ready,
    input  wire [255:0] rd0_dat,
    output wire         rd1_req_valid,
    input  wire         rd1_req_ready,
    output wire [31:0]  rd1_req_addr,
    output wire [6:0]   rd1_req_len,
    input  wire         rd1_dat_valid,
    output wire         rd1_dat_ready,
    input  wire [255:0] rd1_dat,
    // write channel (gradient, 256 words)
    output reg          wr_valid,
    input  wire         wr_ready,
    output reg  [31:0]  wr_addr,
    output reg  [255:0] wr_data,
    output wire [7:0]   wr_mask
);
    localparam NS = 2 * NL;
    localparam [10:0] DM = 2 * D - 1;
    localparam [10:0] DD = D;
    localparam [15:0] DW = D;
    localparam SW = (NL <= 2) ? 2 : (NL <= 4) ? 3 : 4;      // slot index width
    localparam [SW-1:0] NSM1 = NS - 1;
    localparam [3:0] S_IDLE = 4'd0, S_H = 4'd1, S_C = 4'd2, S_INIT = 4'd3, S_ROW = 4'd4, S_ATOM = 4'd5,
                     S_FIN = 4'd6, S_OUT = 4'd7, S_OUTW = 4'd8;
    reg  [3:0]  st;
    assign busy = (st != S_IDLE);
    assign wr_mask = 8'hFF;
    // ------------------------------------------------------------------ readers
    reg         r0_go, r1_go;
    reg  [31:0] r0_base, r1_base;
    reg  [15:0] r0_n;
    wire        r0_busy, r0_done, r1_busy, r1_done, w0_valid, w1_valid;
    wire [31:0] w0_d, w1_d;
    wire [15:0] w0_idx, w1_idx;
    lw_rdseq u_r0 (.clk(clk), .rst_n(rst_n), .go(r0_go), .base(r0_base), .nwords(r0_n), .busy(r0_busy), .done(r0_done),
                   .req_valid(rd0_req_valid), .req_ready(rd0_req_ready), .req_addr(rd0_req_addr), .req_len(rd0_req_len),
                   .dat_valid(rd0_dat_valid), .dat_ready(rd0_dat_ready), .dat(rd0_dat),
                   .w_valid(w0_valid), .w_ready(1'b1), .w_d(w0_d), .w_idx(w0_idx));
    lw_rdseq u_r1 (.clk(clk), .rst_n(rst_n), .go(r1_go), .base(r1_base), .nwords(DW), .busy(r1_busy), .done(r1_done),
                   .req_valid(rd1_req_valid), .req_ready(rd1_req_ready), .req_addr(rd1_req_addr), .req_len(rd1_req_len),
                   .dat_valid(rd1_dat_valid), .dat_ready(rd1_dat_ready), .dat(rd1_dat),
                   .w_valid(w1_valid), .w_ready(1'b1), .w_d(w1_d), .w_idx(w1_idx));
    // ------------------------------------------------------------------ taps, banks
    reg  [25*32-1:0] hreg;
    genvar gi;
    generate
        for (gi = 0; gi < 25; gi = gi + 1) begin : g_h
            localparam [4:0] GI = gi;
            always @(posedge clk) if ((st == S_H) & w0_valid & (w0_idx[4:0] == GI)) hreg[gi*32 +: 32] <= w0_d;
        end
    endgenerate
    reg         ct_en, ct_we;
    reg  [7:0]  ct_addr;
    reg  [31:0] ld_d0, ld_d1;                           // words registered with the bank controls
    wire [31:0] ct_q;
    always @(posedge clk) begin ld_d0 <= w0_d; ld_d1 <= w1_d; end
    mm_sram_bank #(.WORDS(256), .WIDTH(32), .AW(8)) u_ct (.clk(clk), .en(ct_en), .we(ct_we), .addr(ct_addr), .d(ld_d0), .q(ct_q));
    reg         dy_en, x_en, rows_we;
    reg  [9:0]  dy_addr, x_addr;
    wire [31:0] dy_q, x_q;
    mm_sram_bank #(.WORDS(D), .WIDTH(32), .AW(10)) u_dy (.clk(clk), .en(dy_en), .we(rows_we), .addr(dy_addr), .d(ld_d0), .q(dy_q));
    mm_sram_bank #(.WORDS(D), .WIDTH(32), .AW(10)) u_x  (.clk(clk), .en(x_en), .we(rows_we), .addr(x_addr), .d(ld_d1), .q(x_q));
    reg         ac_en, ac_we;
    reg  [7:0]  ac_addr;
    reg  [31:0] ac_d;
    wire [31:0] ac_q;
    mm_sram_bank #(.WORDS(256), .WIDTH(32), .AW(8)) u_ac (.clk(clk), .en(ac_en), .we(ac_we), .addr(ac_addr), .d(ac_d), .q(ac_q));
    // ------------------------------------------------------------------ dispatcher
    reg  [15:0] tok;
    reg  [31:0] dy_row, x_row;
    reg  [8:0]  j;                        // next atom to dispatch, 0 .. 256
    reg  [2:0]  w;                        // window phase 0 .. 6
    reg  [9:0]  ci_n, cj_n;
    reg  [SW-1:0] tgt;                    // slot being filled
    reg  [SW-1:0] rr;                     // round-robin slot pointer
    reg         atom_act;                 // an atom window is in progress
    reg         ct_rdy;                   // the first centre of the token is in ci_n / cj_n
    reg         init_s1, init_s2;         // acc_init read: en registered, access, q
    reg  [2:0]  w_s1, w_s2;               // window index of the word arriving (read en registered, access, q)
    reg         w_s1_v, w_s2_v;
    reg  [SW-1:0] tgt_s1, tgt_s2;
    reg         ct_s1, ct_s2;             // centre read in flight
    reg         ct0_s1, ct0_s2, ct0_s3;   // first-centre read of a token
    // unfolded window positions ci - 3 + w, cj - 3 + w as running counters (loaded at dispatch, +1 per
    // window cycle); only the fold (subtract / compare / mux) is in front of the address registers
    reg  [10:0] py, px;
    wire [10:0] pyf = DM - py, pxf = DM - px;
    wire [9:0]  ry = py[10] ? ~py[9:0] : (py >= DD) ? pyf[9:0] : py[9:0];
    wire [9:0]  rx = px[10] ? ~px[9:0] : (px >= DD) ? pxf[9:0] : px[9:0];
    // slots
    reg  [NS-1:0]   s_busy;               // chain running (window complete)
    reg  [NS-1:0]   s_fill;               // window being filled
    reg  [4:0]      s_tm [0:NS-1];        // muls issued
    reg  [4:0]      s_tf [0:NS-1];        // fmas issued
    reg  [7:0]      s_j  [0:NS-1];
    reg  [31:0]     s_acc [0:NS-1];
    reg  [7*32-1:0] s_yw [0:NS-1];
    reg  [7*32-1:0] s_xw [0:NS-1];
    wire [NS-1:0]   s_free = ~s_busy & ~s_fill;
    // lane results -> acc bank
    reg  [NL-1:0]   res_v;
    reg  [7:0]      res_j [0:NL-1];
    reg  [31:0]     res_d [0:NL-1];
    reg  [NL-1:0]   res_take;
    reg             res_any;
    wire [NL-1:0]   res_pend;             // a chain's last FMA is in flight (s_busy already dropped)
    integer k, kq;                        // kq: the clocked loop below (Yosys: one loop variable per process)
    always @* begin
        res_take = {NL{1'b0}}; res_any = 1'b0;
        for (k = 0; k < NL; k = k + 1)
            if (res_v[k] & ~res_any) begin res_take[k] = 1'b1; res_any = 1'b1; end
    end
    wire        atoms_left = (j != 9'd256);
    wire        disp_start = (st == S_ATOM) & ct_rdy & (~atom_act | (w == 3'd6)) & atoms_left & s_free[rr] & ~res_any;
    reg         ph;                       // slot parity turn
    // out phase
    reg  [8:0]  o_j;
    reg  [3:0]  o_rd;                     // reads issued in the current 8-word group
    reg  [2:0]  o_wi;
    reg         o_s1, o_s2;
    wire        o_issue = (st == S_OUT) & (o_rd != 4'd8);
    wire        om_ov;
    wire [31:0] om_y;
    fp_mul #(.STAGES(2)) u_om (.clk(clk), .rst_n(rst_n), .in_valid(o_s2), .a(d_gamma), .b(ac_q), .out_valid(om_ov), .y(om_y));
    // bank controls (registered: access the cycle after, q two cycles after)
    always @(posedge clk) begin
        ct_en <= 1'b0; ct_we <= 1'b0; dy_en <= 1'b0; x_en <= 1'b0; rows_we <= 1'b0; ac_en <= 1'b0; ac_we <= 1'b0;
        ct_addr <= w0_idx[7:0]; dy_addr <= w0_idx[9:0]; x_addr <= w1_idx[9:0];
        if ((st == S_C) & w0_valid) begin ct_en <= 1'b1; ct_we <= 1'b1; end
        if (st == S_ROW) begin
            if (w0_valid) begin dy_en <= 1'b1; rows_we <= 1'b1; end
            if (w1_valid) begin x_en <= 1'b1; rows_we <= 1'b1; end
        end
        if (st == S_INIT) begin ac_en <= 1'b1; ac_we <= 1'b1; ac_addr <= j[7:0]; ac_d <= 32'd0; end
        if (res_any) begin
            ac_en <= 1'b1; ac_we <= 1'b1;
            for (kq = 0; kq < NL; kq = kq + 1) if (res_take[kq]) begin ac_addr <= res_j[kq]; ac_d <= res_d[kq]; end
        end else if (disp_start) begin
            ac_en <= 1'b1; ac_addr <= j[7:0];
        end else if (o_issue) begin
            ac_en <= 1'b1; ac_addr <= o_j[7:0];
        end
        if (atom_act) begin dy_en <= 1'b1; dy_addr <= ry; x_en <= 1'b1; x_addr <= rx; end
        if (atom_act & (w == 3'd3)) begin ct_en <= 1'b1; ct_addr <= j[7:0]; end     // next atom's centre
        if (ct0_s1) begin ct_en <= 1'b1; ct_addr <= 8'd0; end
    end
    always @(posedge clk)
        if (!rst_n) begin
            atom_act <= 1'b0; w <= 3'd0; w_s1_v <= 1'b0; w_s2_v <= 1'b0; init_s1 <= 1'b0; init_s2 <= 1'b0;
            ct_s1 <= 1'b0; ct_s2 <= 1'b0; ct0_s2 <= 1'b0; ct0_s3 <= 1'b0; ct_rdy <= 1'b0; rr <= {SW{1'b0}}; ph <= 1'b0;
        end else begin
            ph <= ~ph;
            w_s1_v <= atom_act; w_s1 <= w; tgt_s1 <= tgt;
            w_s2_v <= w_s1_v; w_s2 <= w_s1; tgt_s2 <= tgt_s1;
            ct_s1 <= atom_act & (w == 3'd3); ct_s2 <= ct_s1;
            ct0_s2 <= ct0_s1; ct0_s3 <= ct0_s2;
            if (ct_s2 | ct0_s3) begin ci_n <= ct_q[25:16]; cj_n <= ct_q[9:0]; end
            if (ct0_s3) ct_rdy <= 1'b1;
            if (st == S_ROW) ct_rdy <= 1'b0;
            init_s1 <= disp_start; init_s2 <= init_s1;
            if (disp_start) begin
                atom_act <= 1'b1; w <= 3'd0; tgt <= rr; rr <= (rr == NSM1) ? {SW{1'b0}} : rr + 1'b1;
                py <= {1'b0, ci_n} + 11'h7FD; px <= {1'b0, cj_n} + 11'h7FD;
            end else if (atom_act) begin
                py <= py + 11'd1; px <= px + 11'd1;
                if (w == 3'd6) begin atom_act <= 1'b0; w <= 3'd0; end
                else w <= w + 3'd1;
            end
        end
    // ------------------------------------------------------------------ slots and lanes
    // tap offsets in nonzero(H) order (row-major over the 7 x 7 box, |di| + |dj| <= 3) as window indices di + 3, dj + 3
    function [2:0] tap_di; input [4:0] t; begin
        case (t)
            5'd0: tap_di = 3'd0;
            5'd1, 5'd2, 5'd3: tap_di = 3'd1;
            5'd4, 5'd5, 5'd6, 5'd7, 5'd8: tap_di = 3'd2;
            5'd9, 5'd10, 5'd11, 5'd12, 5'd13, 5'd14, 5'd15: tap_di = 3'd3;
            5'd16, 5'd17, 5'd18, 5'd19, 5'd20: tap_di = 3'd4;
            5'd21, 5'd22, 5'd23: tap_di = 3'd5;
            default: tap_di = 3'd6;
        endcase end
    endfunction
    function [2:0] tap_dj; input [4:0] t; begin
        case (t)
            5'd0, 5'd24: tap_dj = 3'd3;
            5'd1, 5'd21: tap_dj = 3'd2;  5'd2, 5'd22: tap_dj = 3'd3;  5'd3, 5'd23: tap_dj = 3'd4;
            5'd4, 5'd16: tap_dj = 3'd1;  5'd5, 5'd17: tap_dj = 3'd2;  5'd6, 5'd18: tap_dj = 3'd3;
            5'd7, 5'd19: tap_dj = 3'd4;  5'd8, 5'd20: tap_dj = 3'd5;
            5'd9: tap_dj = 3'd0;  5'd10: tap_dj = 3'd1;  5'd11: tap_dj = 3'd2;  5'd12: tap_dj = 3'd3;
            5'd13: tap_dj = 3'd4;  5'd14: tap_dj = 3'd5;  default: tap_dj = 3'd6;
        endcase end
    endfunction
    genvar gl, gp;
    generate
        for (gl = 0; gl < NL; gl = gl + 1) begin : g_lane
            localparam [SW-2:0] GL = gl;
            // ph copies per lane (the global ph would fan out to every operand mux select): toggled in lockstep
            reg        ph_a, ph_b;
            always @(posedge clk) if (!rst_n) begin ph_a <= 1'b0; ph_b <= 1'b0; end else begin ph_a <= ~ph_a; ph_b <= ~ph_b; end
            wire [SW-1:0] sid = {GL, ph_b};              // slot whose turn it is
            wire [SW-1:0] nsid = {GL, ~ph_a};            // slot whose turn is next: its operands are selected
            wire [SW-1:0] nsid_b = {GL, ~ph_b};          // now and registered (the fp inputs see one mux level)
            wire [4:0] tf = s_tf[sid];
            wire [4:0] ntm = s_tm[nsid], ntf = s_tf[nsid_b];
            wire [2:0] ndi = tap_di(ntm), ndj = tap_dj(ntm);
            wire [6:0] ndi_oh = 7'd1 << ndi, ndj_oh = 7'd1 << ndj;
            wire [24:0] ntf_oh = 25'd1 << ntf;
            wire [31:0] ya_n, xb_n, h_n, m_y, f_y;
            wire        m_ov, f_ov;
            cm_ohmux #(.W(32), .N(7), .REP(1)) u_ys (.sel(ndi_oh), .d(s_yw[nsid]), .y(ya_n));
            cm_ohmux #(.W(32), .N(7), .REP(1)) u_xs (.sel(ndj_oh), .d(s_xw[nsid]), .y(xb_n));
            cm_ohmux #(.W(32), .N(25), .REP(1)) u_hs (.sel(ntf_oh), .d(hreg), .y(h_n));
            reg         m_issue;                         // = s_busy & tm != 25 of the slot now in turn
            reg  [31:0] ya_q, xb_q, h_q, acc_q;
            reg         first_q;
            always @(posedge clk) begin
                if (!rst_n) m_issue <= 1'b0;
                else m_issue <= s_busy[nsid] & (ntm != 5'd25);
                ya_q <= ya_n; xb_q <= xb_n; h_q <= h_n; acc_q <= s_acc[nsid_b]; first_q <= (ntf == 5'd0);
            end
            fp_mul #(.STAGES(2)) u_m (.clk(clk), .rst_n(rst_n), .in_valid(m_issue), .a(ya_q), .b(xb_q), .out_valid(m_ov), .y(m_y));
            // the product of tap tf arrives exactly in this slot's next turn (2 cycles): m_ov marks it
            wire        f_issue = m_ov;
            wire [31:0] c_in = first_q ? acc_q : f_y;
            fp_fma #(.STAGES(2)) u_f (.clk(clk), .rst_n(rst_n), .in_valid(f_issue), .a(h_q), .b(m_y), .c(c_in), .out_valid(f_ov), .y(f_y));
            reg        last_s1, last_s2;                // the chain's last FMA was issued 1 / 2 cycles ago
            reg  [7:0] lj_s1, lj_s2;
            always @(posedge clk) begin
                if (!rst_n) begin last_s1 <= 1'b0; last_s2 <= 1'b0; end
                else begin last_s1 <= f_issue & (tf == 5'd24); last_s2 <= last_s1; end
                lj_s1 <= s_j[sid]; lj_s2 <= lj_s1;
            end
            assign res_pend[gl] = last_s1 | last_s2;
            wire unused_l = f_ov;
            always @(posedge clk)
                if (!rst_n) res_v[gl] <= 1'b0;
                else begin
                    if (last_s2) begin res_v[gl] <= 1'b1; res_d[gl] <= f_y; res_j[gl] <= lj_s2; end
                    else if (res_take[gl]) res_v[gl] <= 1'b0;
                end
            for (gp = 0; gp < 2; gp = gp + 1) begin : g_slot
                localparam [SW-1:0] SID = gl * 2 + gp;
                localparam [0:0] GP = gp;
                wire my_turn = (ph == GP);
                always @(posedge clk)
                    if (!rst_n) begin
                        s_busy[SID] <= 1'b0; s_fill[SID] <= 1'b0; s_tm[SID] <= 5'd0; s_tf[SID] <= 5'd0;
                    end else begin
                        if (disp_start & (rr == SID)) begin s_fill[SID] <= 1'b1; s_j[SID] <= j[7:0]; end
                        if (w_s2_v & (tgt_s2 == SID) & (w_s2 == 3'd6)) begin
                            s_fill[SID] <= 1'b0; s_busy[SID] <= 1'b1; s_tm[SID] <= 5'd0; s_tf[SID] <= 5'd0;
                        end
                        if (my_turn & m_issue) s_tm[SID] <= s_tm[SID] + 5'd1;
                        if (my_turn & f_issue) begin
                            s_tf[SID] <= tf + 5'd1;
                            if (tf == 5'd24) s_busy[SID] <= 1'b0;   // window registers were free since tm == 25
                        end
                    end
                always @(posedge clk) begin
                    if (w_s2_v & (tgt_s2 == SID)) begin
                        s_yw[SID][w_s2*32 +: 32] <= dy_q;
                        s_xw[SID][w_s2*32 +: 32] <= x_q;
                    end
                    if (init_s2 & (tgt == SID)) s_acc[SID] <= ac_q;
                end
            end
        end
    endgenerate
    // ------------------------------------------------------------------ FSM
    // s_busy drops at the last FMA issue; the result reaches res_v three cycles later (res_pend covers the gap,
    // otherwise S_OUT starts early and its ac read of atom 1 is pre-empted by the final acc write)
    wire all_idle = ~(|s_busy) & ~(|s_fill) & ~(|res_v) & ~(|res_pend) & ~atom_act;
    always @(posedge clk)
        if (!rst_n) begin
            st <= S_IDLE; r0_go <= 1'b0; r1_go <= 1'b0; done <= 1'b0; wr_valid <= 1'b0; j <= 9'd0;
            o_s1 <= 1'b0; o_s2 <= 1'b0; ct0_s1 <= 1'b0; o_rd <= 4'd8;
        end else begin
            r0_go <= 1'b0; r1_go <= 1'b0; done <= 1'b0; ct0_s1 <= 1'b0;
            o_s1 <= o_issue; o_s2 <= o_s1;
            if (disp_start) j <= j + 9'd1;
            case (st)
                S_IDLE: if (start) begin
                    st <= S_H; r0_go <= 1'b1; r0_base <= d_h_base; r0_n <= 16'd25;
                    tok <= 16'd0; dy_row <= d_dy_base; x_row <= d_x_base;
                end
                S_H: if (r0_done) begin st <= S_C; r0_go <= 1'b1; r0_base <= d_ctr_base; r0_n <= 16'd256; j <= 9'd0; end
                S_C: if (r0_done) st <= S_INIT;
                S_INIT: begin                                   // acc[j] = +0, j ascending
                    j <= j + 9'd1;
                    if (j == 9'd255) begin
                        st <= S_ROW; j <= 9'd0;
                        r0_go <= 1'b1; r0_base <= dy_row; r0_n <= DW;
                        r1_go <= 1'b1; r1_base <= x_row;
                    end
                end
                S_ROW: if (~r0_busy & ~r1_busy & ~r0_go & ~r1_go) begin   // both rows resident
                    st <= S_ATOM; j <= 9'd0; ct0_s1 <= 1'b1;
                end
                S_ATOM: if (~atoms_left & ~atom_act) begin         // all windows of this token read
                    tok <= tok + 16'd1;
                    if (tok + 16'd1 == d_tok) st <= S_FIN;
                    else begin
                        st <= S_ROW; dy_row <= dy_row + d_dy_rs; x_row <= x_row + d_x_rs;
                        r0_go <= 1'b1; r0_base <= dy_row + d_dy_rs; r0_n <= DW;
                        r1_go <= 1'b1; r1_base <= x_row + d_x_rs;
                    end
                end
                S_FIN: if (all_idle) begin st <= S_OUT; o_j <= 9'd0; o_rd <= 4'd0; o_wi <= 3'd0; wr_addr <= d_grad_base; end
                S_OUT: begin
                    if (o_issue) begin o_j <= o_j + 9'd1; o_rd <= o_rd + 4'd1; end
                    if (om_ov) begin
                        wr_data[o_wi*32 +: 32] <= om_y;
                        o_wi <= o_wi + 3'd1;
                        if (o_wi == 3'd7) begin wr_valid <= 1'b1; st <= S_OUTW; end
                    end
                end
                S_OUTW: if (wr_ready) begin
                    wr_valid <= 1'b0; wr_addr <= wr_addr + 32'd32; o_rd <= 4'd0;
                    if (o_j == 9'd256) begin st <= S_IDLE; done <= 1'b1; end
                    else st <= S_OUT;
                end
                default: st <= S_IDLE;
            endcase
        end
    wire unused_s = (|w0_idx[15:10]) | (|w1_idx[15:10]) | r0_done | r1_done | (|ct_q[31:26]) | (|ct_q[15:10]) | pyf[10] | pxf[10];
endmodule
`default_nettype wire
