`default_nettype none
// vu_seq: the vector-unit sequencer (docs/p4_report.md section 2).  Executes one descriptor:
//   planner : walks (tile of V rows, phase of the op, row segment of RB words) in order and fills a
//             two-entry step queue with what each step loads (tile operands A / B into the lane
//             buffers, column vectors, per-row scalars) and the buffer parity it uses; a later phase
//             re-reading a row segment that is still resident (rows <= RB) loads nothing.
//   loader  : vu_loader executes the loads of the queue entry after the one being computed
//             (double buffering); dual-operand steps (A and B) use both buffers and are not overlapped.
//   compute : ELEM phases issue the segment's elements in order into the lanes' barrel slots and
//             drive the kernel micro-ops (vu_ucode) in slot turns; FIFO ops (SFU push / pop, MAX, ACC,
//             OUT, STORE) are gated to element order by per-type sequence counters.  SCAL phases run
//             a one-element kernel; XL phases do the cross-lane sequential sum (chunked or not);
//             LDS / WST / W0 / W03 phases load / write per-row scalars through the loader / writer.
// Reduction orders are the contract's (docs/dataflow.md 5, arith.md 2): every row is one lane,
// sequential in j; chunks of KC = 64 in the lane accumulator; cross-lane sums sequential in row order.
module vu_seq #(
    parameter V    = 16,
    parameter VW   = 4,
    parameter NS   = 8,
    parameter SW   = 3,
    parameter RB   = 256,
    parameter RBW  = 8,
    parameter KC   = 64
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    output reg          busy,
    output reg          done,
    // descriptor
    input  wire [4:0]   d_vop,
    input  wire [15:0]  d_rows, d_cols, d_rows_per_b,
    input  wire         d_st_en, d_x_reset, d_g_reset,
    input  wire [31:0]  d_mk_base, d_mk_rs,
    // lane status (reduced over lanes by the top)
    input  wire         acc_idle_all, acc_room_all, st_busy_any, st_ok_all, ob_room_all,
    input  wire [3:0]   sfu_cnt_min,          // SFU results waiting (min over lanes)
    input  wire [31:0]  xl_lane_val,          // lane xl_row's xl_val
    // lane control
    output reg          iss_v,
    output reg          iss_abuf,
    output reg  [RBW-1:0] iss_addr,
    output reg  [SW-1:0] iss_slot,
    output reg          cv_par,
    output reg          km_mode,              // cvec0 holds key-mask bits (word j >> 5, bit j & 31)
    output reg          ex_v,
    output reg  [SW-1:0] ex_slot,
    output reg  [35:0]  ex_uop,
    output reg          acc_start, acc_chunk, acc_flush, max_start, ob_flush, cur_abuf,
    output reg          xl_sel,
    output reg  [VW-1:0] xl_row,
    output reg  [31:0]  xtot,
    output reg          sreg_clr,             // s2 <= +0 in every lane (d_g_reset)
    // loader
    output reg          ld_start,
    input  wire         ld_done,
    output reg          ld_a, ld_b, ld_abuf, ld_gather, ld_km, ld_cvpar,
    output reg  [2:0]   ld_cv,
    output reg  [1:0]   ld_sreg,
    output reg  [15:0]  ld_row0, ld_seg_off, ld_b_idx,
    output reg  [VW:0]  ld_nrows,
    output reg  [RBW:0] ld_seg_len,
    // mask fetcher
    output reg          mk_start,
    output reg  [31:0]  mk_idx0,
    output reg  [VW:0]  mk_nrows,
    output reg  [RBW:0] mk_len,
    input  wire [2:0]   mk_cnt_min,           // keep-bit entries available (min over lanes)
    output reg          mk_pop,               // consume one bit per lane (with iss_v)
    // writer
    output reg          wr_req,               // push a geometry entry
    output reg  [1:0]   wr_kind,              // 0 OUT stream, 1 WST (s0 -> st0, s1 -> st1), 2 W0, 3 W03
    output reg  [15:0]  wr_row0,
    output reg  [VW:0]  wr_nrows,
    output reg  [15:0]  wr_seg_off,
    output reg  [RBW:0] wr_seg_len,
    input  wire         wr_room,              // the writer queue can take an entry
    input  wire         wr_idle               // queue empty and no beat in flight
);
    // ------------------------------------------------------------------ ROM
    wire [5:0]  rom_kid;
    wire [3:0]  rom_pc;
    wire [35:0] rom_uop;
    wire [19:0] rom_ph;
    reg  [3:0]  p_ph;
    vu_ucode u_rom (.kid(rom_kid), .pc(rom_pc), .uop(rom_uop), .vop(d_vop), .ph(p_ph), .phase(rom_ph));
    localparam [2:0] K_END = 3'd0, K_ELEM = 3'd1, K_SCAL = 3'd2, K_XL = 3'd3, K_LDS = 3'd4, K_WST = 3'd5, K_W0 = 3'd6;
    localparam [3:0] O_SFU = 4'd2, O_POP = 4'd3, O_MAX = 4'd4, O_ACC = 4'd5, O_OUT = 4'd6, O_STORE = 4'd7;
    localparam [4:0] VOP_EMB = 5'd5;

    // ------------------------------------------------------------------ geometry
    /* verilator lint_off WIDTHTRUNC */   // V (an integer parameter) in 16-bit row arithmetic
    /* verilator lint_off WIDTHEXPAND */
    localparam [15:0] V16 = V;
    wire [15:0] ntiles = (d_rows + V16 - 16'd1) >> VW;
    wire [15:0] nseg   = (d_cols + RB - 1) >> RBW;

    // ------------------------------------------------------------------ step queue (2 entries)
    reg  [1:0]  q_valid, q_loaded, q_dual, q_needld, q_abuf, q_cvpar, q_lastseg, q_firstseg, q_lasttile;
    reg  [19:0] q_phase [0:1];
    reg  [15:0] q_row0 [0:1];
    reg  [VW:0] q_nrows [0:1];
    reg  [15:0] q_segoff [0:1];
    reg  [RBW:0] q_seglen [0:1];
    reg  [15:0] q_bidx [0:1];
    reg         pp, lp, cp;                   // planner / loader / compute pointers
    // planner iteration state
    reg  [15:0] p_tile, p_seg, p_bidx, p_rows_in_b;
    reg         p_lastbuf, p_cvpar, p_tile_loaded, p_active, p_final, p_done;
    wire [2:0]  ph_kind  = rom_ph[2:0];
    wire        ph_useb  = rom_ph[9], ph_cv0 = rom_ph[10], ph_cv1 = rom_ph[11], ph_cv2 = rom_ph[12],
                ph_final = rom_ph[19];
    wire        ph_is_elem = (ph_kind == K_ELEM);
    wire        ph_dual  = ph_is_elem & ph_useb;
    wire        ph_anycv = ph_cv0 | ph_cv1 | ph_cv2 | rom_ph[14];
    wire [15:0] p_row0   = p_tile << VW;
    wire [15:0] p_remrows = d_rows - p_row0;
    wire [VW:0] p_nrows  = (p_remrows > V) ? V[VW:0] : p_remrows[VW:0];
    /* verilator lint_on WIDTHTRUNC */
    /* verilator lint_on WIDTHEXPAND */
    wire [15:0] p_segoff = p_seg << RBW;
    wire [15:0] p_remcols = d_cols - p_segoff;
    wire [RBW:0] p_seglen = (p_remcols > RB) ? RB[RBW:0] : p_remcols[RBW:0];
    wire        p_lastseg = (p_seg == nseg - 16'd1);
    wire        p_needld = ph_is_elem & ((nseg != 16'd1) | ~p_tile_loaded);
    wire        p_tile_end = (ph_kind == K_END) | (~p_final & ph_final);
    wire        p_skip = (p_final & ~ph_final & (ph_kind != K_END)) | ((ph_kind == K_WST) & ~d_st_en);
    wire        p_step = p_active & ~q_valid[pp];
    // compute engine
    reg  [3:0]  cs;
    localparam [3:0] C_IDLE = 4'd0, C_WAITLD = 4'd1, C_ELEM = 4'd2, C_DRAIN = 4'd3, C_FLUSH = 4'd4, C_SCAL = 4'd5,
                     C_XL = 4'd6, C_XLFL = 4'd7, C_WR = 4'd8, C_WRWAIT = 4'd9, C_NEXT = 4'd10, C_FIN = 4'd11;
    wire        lane_free_cp = (cs == C_FLUSH) | (cs == C_XL) | (cs == C_XLFL) | (cs == C_WR) | (cs == C_WRWAIT) | (cs == C_NEXT);
    /* verilator lint_off UNUSEDSIGNAL */   // c_ph: operand-use bits are the planner's / loader's
    wire [19:0] c_ph = q_phase[cp];
    /* verilator lint_on UNUSEDSIGNAL */
    wire [2:0]  c_kind = c_ph[2:0];
    wire [5:0]  c_kid = c_ph[8:3];
    wire        c_usekeep = c_ph[13], c_usekm = c_ph[14], c_out = c_ph[15], c_chunk = c_ph[16];
    wire        c_xls2 = c_ph[17], c_xlchunk = c_ph[18];
    reg         e_run, e_scal, e_done, dr_done, fl_done, xl_done, xlf_done;
    reg  [2:0]  dr_cnt;
    // loader hand-off
    reg         l_busy;
    wire        l_can = q_valid[lp] & ~q_loaded[lp] & ~l_busy & ((lp == cp) | ~(q_dual[cp] | q_dual[lp]) | lane_free_cp);
    wire        c_start_ok = q_valid[cp] & q_loaded[cp] & (~((c_kind == K_ELEM) & c_out) | wr_room);
    always @(posedge clk) begin
        ld_start <= 1'b0; done <= 1'b0; sreg_clr <= 1'b0;
        if (!rst_n) begin
            q_valid <= 2'b00; q_loaded <= 2'b00; pp <= 1'b0; lp <= 1'b0; cp <= 1'b0; p_active <= 1'b0; p_done <= 1'b0;
            l_busy <= 1'b0; busy <= 1'b0; cs <= C_IDLE; p_tile <= 16'd0; p_seg <= 16'd0; p_ph <= 4'd0; p_bidx <= 16'd0;
            p_rows_in_b <= 16'd0; p_lastbuf <= 1'b0; p_cvpar <= 1'b0; p_tile_loaded <= 1'b0; p_final <= 1'b0;
        end else begin
            if (start & ~busy) begin
                busy <= 1'b1; p_active <= 1'b1; p_done <= 1'b0; p_tile <= 16'd0; p_seg <= 16'd0; p_ph <= 4'd0;
                p_bidx <= 16'd0; p_rows_in_b <= 16'd0; p_tile_loaded <= 1'b0; p_final <= (ntiles == 16'd0);
                cs <= C_WAITLD; sreg_clr <= d_g_reset;
            end
            // ---------------------------------------------------------- planner
            if (p_step) begin
                if (p_tile_end) begin
                    if (p_final) begin
                        p_done <= 1'b1; p_active <= 1'b0;
                    end else if (p_tile == ntiles - 16'd1) begin
                        p_final <= 1'b1; p_ph <= 4'd0;
                    end else begin
                        p_tile <= p_tile + 16'd1; p_ph <= 4'd0; p_tile_loaded <= 1'b0;
                        if (p_rows_in_b + V16 >= d_rows_per_b) begin p_rows_in_b <= 16'd0; p_bidx <= p_bidx + 16'd1; end
                        else p_rows_in_b <= p_rows_in_b + V16;
                    end
                end else if (p_skip) begin
                    p_ph <= p_ph + 4'd1;
                end else begin
                    q_valid[pp] <= 1'b1; q_loaded[pp] <= ~(p_needld | (ph_is_elem & ph_anycv) | (ph_kind == K_LDS));
                    q_phase[pp] <= rom_ph; q_row0[pp] <= p_row0; q_nrows[pp] <= p_nrows; q_segoff[pp] <= p_segoff;
                    q_seglen[pp] <= p_seglen; q_bidx[pp] <= p_bidx;
                    q_dual[pp] <= ph_dual | (ph_kind == K_LDS); q_needld[pp] <= p_needld;
                    q_lastseg[pp] <= p_lastseg | ~ph_is_elem; q_firstseg[pp] <= (p_seg == 16'd0) | ~ph_is_elem;
                    q_lasttile[pp] <= (p_tile == ntiles - 16'd1);
                    q_abuf[pp] <= ph_dual ? 1'b0 : (p_needld ? ~p_lastbuf : p_lastbuf);
                    q_cvpar[pp] <= (ph_is_elem & ph_anycv) ? ~p_cvpar : p_cvpar;
                    if (ph_is_elem & ph_anycv) p_cvpar <= ~p_cvpar;
                    if (ph_is_elem) begin
                        if (p_needld) p_lastbuf <= ph_dual ? 1'b1 : ~p_lastbuf;
                        if (nseg == 16'd1) p_tile_loaded <= 1'b1;
                        if (p_lastseg) begin p_seg <= 16'd0; p_ph <= p_ph + 4'd1; end
                        else p_seg <= p_seg + 16'd1;
                    end else p_ph <= p_ph + 4'd1;
                    pp <= ~pp;
                end
            end
            // ---------------------------------------------------------- loader hand-off
            if (l_can) begin
                l_busy <= 1'b1; ld_start <= 1'b1;
                ld_a <= q_needld[lp]; ld_b <= q_needld[lp] & q_phase[lp][9]; ld_abuf <= q_abuf[lp];
                ld_gather <= q_needld[lp] & (d_vop == VOP_EMB);
                ld_km <= q_phase[lp][14]; ld_cvpar <= q_cvpar[lp];
                ld_cv <= (q_phase[lp][2:0] == K_ELEM) ? {q_phase[lp][12], q_phase[lp][11], q_phase[lp][10] | q_phase[lp][14]} : 3'b000;
                ld_sreg <= (q_phase[lp][2:0] == K_LDS) ? 2'b11 : 2'b00;
                ld_row0 <= q_row0[lp]; ld_nrows <= q_nrows[lp]; ld_seg_off <= q_segoff[lp]; ld_seg_len <= q_seglen[lp];
                ld_b_idx <= q_bidx[lp];
            end
            if (l_busy & ld_done) begin
                l_busy <= 1'b0; q_loaded[lp] <= 1'b1; lp <= ~lp;
            end else if (q_valid[lp] & q_loaded[lp] & ~l_busy & (lp != cp | cs == C_WAITLD)) begin
                lp <= ~lp;                                        // nothing to load for this step
            end
            // ---------------------------------------------------------- compute engine
            case (cs)
                C_WAITLD: begin
                    if (p_done & ~q_valid[cp]) cs <= C_FIN;
                    else if (c_start_ok) begin
                        case (c_kind)
                            K_ELEM: cs <= C_ELEM;
                            K_SCAL: cs <= C_SCAL;
                            K_XL:   cs <= C_XL;
                            K_LDS:  cs <= C_NEXT;
                            default: cs <= C_WR;                 // WST / W0 / W03
                        endcase
                    end
                end
                C_ELEM: if (e_done) cs <= C_DRAIN;
                C_SCAL: if (e_done) cs <= C_DRAIN;
                C_DRAIN: if (dr_done) cs <= ((c_kind == K_ELEM) & q_lastseg[cp]) ? C_FLUSH : C_NEXT;
                C_FLUSH: if (fl_done) cs <= C_NEXT;
                C_XL:   if (xl_done) cs <= q_lasttile[cp] ? C_XLFL : C_NEXT;
                C_XLFL: if (xlf_done) cs <= C_NEXT;
                C_WR:   if (wr_idle & wr_room) cs <= C_WRWAIT;
                C_WRWAIT: if (dr_done) cs <= C_NEXT;
                C_NEXT: begin
                    q_valid[cp] <= 1'b0; q_loaded[cp] <= 1'b0; cp <= ~cp; cs <= C_WAITLD;
                end
                C_FIN: if (wr_idle) begin busy <= 1'b0; done <= 1'b1; cs <= C_IDLE; end
                default: ;
            endcase
        end
    end

    // ------------------------------------------------------------------ element engine
    reg  [RBW:0] e_j;                     // next element to issue
    reg  [NS-1:0] sl_busy, sl_pend1, sl_pend2;
    reg  [2:0]  sl_cool [0:NS-1];
    reg  [3:0]  sl_pc [0:NS-1];
    reg  [RBW:0] sl_seq [0:NS-1];
    reg  [RBW:0] nseq [0:6];             // next element in order per FIFO-op type
    reg  [SW-1:0] turn;
    wire [RBW:0] c_len = e_scal ? {{RBW{1'b0}}, 1'b1} : q_seglen[cp];
    wire [SW-1:0] nxt_slot = e_j[SW-1:0];
    wire        slot_free = ~sl_busy[nxt_slot] & ~sl_pend1[nxt_slot] & ~sl_pend2[nxt_slot] & (sl_cool[nxt_slot] == 3'd0);
    // in-flight accounting: a POP decision at t pops the lane FIFO at t+3 (registered ex_v, lane control and
    // operand stages), an issue pops the keep FIFO at t+1: the decisions still in flight are subtracted
    reg  [2:0]  pop_fire_d;
    reg  [1:0]  mk_end_d;
    wire [3:0]  pop_inflight = {3'd0, pop_fire_d[0]} + {3'd0, pop_fire_d[1]} + {3'd0, pop_fire_d[2]};
    wire [2:0]  mk_pend = {2'd0, mk_end_d[0]} + {2'd0, mk_end_d[1]};
    wire        pop_ok = (sfu_cnt_min > pop_inflight);
    wire        mk_ok = (mk_cnt_min > mk_pend);
    wire        e_grp_end = (e_j[3:0] == 4'd15) | (e_j == c_len - 1'b1);
    wire        can_issue = e_run & (e_j < c_len) & slot_free & acc_room_all & ob_room_all & ~st_busy_any &
                            (~c_usekeep | mk_ok | e_scal);
    // turn execution
    wire [3:0]  t_pc = sl_pc[turn];
    assign rom_kid = c_kid;
    assign rom_pc  = t_pc;
    wire [3:0]  t_opt = rom_uop[3:0];
    wire        t_last = rom_uop[32];
    wire        t_accw = rom_uop[31];
    // FIFO-op type index: 0 SFU, 1 POP, 2 MAX, 3 ACC0, 4 ACC1, 5 OUT, 6 STORE
    wire [2:0]  t_type = (t_opt == O_SFU) ? 3'd0 : (t_opt == O_POP) ? 3'd1 : (t_opt == O_MAX) ? 3'd2 :
                         (t_opt == O_ACC) ? (t_accw ? 3'd4 : 3'd3) : (t_opt == O_OUT) ? 3'd5 : 3'd6;
    wire        t_fifo = (t_opt == O_SFU) | (t_opt == O_POP) | (t_opt == O_MAX) | (t_opt == O_ACC) | (t_opt == O_OUT) | (t_opt == O_STORE);
    wire        t_ord = e_scal | ~t_fifo | (sl_seq[turn] == nseq[t_type]);
    wire        t_ok = (t_opt == O_POP) ? pop_ok :
                       (t_opt == O_ACC) ? acc_room_all :
                       (t_opt == O_OUT) ? ob_room_all :
                       (t_opt == O_STORE) ? st_ok_all : 1'b1;
    wire        t_fire = e_run & sl_busy[turn] & t_ok & t_ord;
    integer i;
    always @(posedge clk) begin
        iss_v <= 1'b0; ex_v <= 1'b0; acc_start <= 1'b0; acc_flush <= 1'b0; max_start <= 1'b0; ob_flush <= 1'b0;
        mk_pop <= 1'b0; mk_start <= 1'b0; wr_req <= 1'b0; e_done <= 1'b0; dr_done <= 1'b0; fl_done <= 1'b0;
        if (!rst_n) begin
            e_run <= 1'b0; e_scal <= 1'b0; sl_busy <= {NS{1'b0}}; turn <= {SW{1'b0}}; e_j <= {(RBW+1){1'b0}};
            sl_pend1 <= {NS{1'b0}}; sl_pend2 <= {NS{1'b0}}; dr_cnt <= 3'd0; pop_fire_d <= 3'b000; mk_end_d <= 2'b00;
            for (i = 0; i < NS; i = i + 1) begin sl_cool[i] <= 3'd0; sl_pc[i] <= 4'd0; sl_seq[i] <= {(RBW+1){1'b0}}; end
            for (i = 0; i < 7; i = i + 1) nseq[i] <= {(RBW+1){1'b0}};
        end else begin
            turn <= turn + 1'b1;
            pop_fire_d <= {pop_fire_d[1:0], t_fire & (t_opt == O_POP)};
            mk_end_d <= {mk_end_d[0], can_issue & c_usekeep & ~e_scal & e_grp_end};
            for (i = 0; i < NS; i = i + 1) if (sl_cool[i] != 3'd0) sl_cool[i] <= sl_cool[i] - 3'd1;
            sl_pend1 <= {NS{1'b0}};
            sl_pend2 <= sl_pend1;
            sl_busy <= sl_busy | sl_pend2;
            // step start
            if ((cs == C_WAITLD) & c_start_ok & ((c_kind == K_ELEM) | (c_kind == K_SCAL))) begin
                e_run <= 1'b1; e_scal <= (c_kind == K_SCAL); e_j <= {(RBW+1){1'b0}};
                for (i = 0; i < 7; i = i + 1) nseq[i] <= {(RBW+1){1'b0}};
                cur_abuf <= q_abuf[cp]; cv_par <= q_cvpar[cp]; km_mode <= c_usekm;
                if ((c_kind == K_ELEM) & q_firstseg[cp]) begin
                    acc_start <= 1'b1; acc_chunk <= c_chunk; max_start <= 1'b1;
                end
                if ((c_kind == K_ELEM) & c_usekeep) begin
                    mk_start <= 1'b1; mk_idx0 <= d_mk_base + q_row0[cp] * d_mk_rs + {16'd0, q_segoff[cp]};
                    mk_nrows <= q_nrows[cp]; mk_len <= q_seglen[cp];
                end
                if ((c_kind == K_ELEM) & c_out) begin
                    wr_req <= 1'b1; wr_kind <= 2'd0; wr_row0 <= q_row0[cp]; wr_nrows <= q_nrows[cp];
                    wr_seg_off <= q_segoff[cp]; wr_seg_len <= q_seglen[cp];
                end
            end
            // issue
            if (can_issue) begin
                iss_v <= 1'b1; iss_abuf <= cur_abuf; iss_addr <= e_j[RBW-1:0]; iss_slot <= nxt_slot;
                mk_pop <= c_usekeep & ~e_scal;
                sl_pend1[nxt_slot] <= 1'b1;
                sl_pc[nxt_slot] <= 4'd0;
                sl_seq[nxt_slot] <= e_j;
                e_j <= e_j + 1'b1;
            end
            // turn
            if (t_fire) begin
                ex_v <= 1'b1; ex_slot <= turn; ex_uop <= rom_uop;
                sl_pc[turn] <= t_pc + 4'd1;
                if (t_fifo) nseq[t_type] <= nseq[t_type] + 1'b1;
                if (t_last) begin sl_busy[turn] <= 1'b0; sl_cool[turn] <= 3'd5; end   // last write-back lands at t+4
            end
            // step end: all issued and every slot free
            if (e_run & (e_j >= c_len) & (sl_busy == {NS{1'b0}}) & (sl_pend1 == {NS{1'b0}}) & (sl_pend2 == {NS{1'b0}}) & ~iss_v) begin
                e_run <= 1'b0; e_done <= 1'b1;
            end
            // drain: six cycles for the write-backs and FIFO pushes, then close the output registers
            if (cs == C_DRAIN) begin
                if (dr_cnt == 3'd6) begin
                    dr_cnt <= 3'd0; dr_done <= 1'b1;
                    if ((c_kind == K_ELEM) & c_out) ob_flush <= 1'b1;
                end else dr_cnt <= dr_cnt + 3'd1;
            end
            // accumulator flush at the end of the last segment: wait idle, flush, wait idle
            if (cs == C_FLUSH) begin
                case (dr_cnt)
                    3'd0: if (acc_idle_all) begin acc_flush <= 1'b1; dr_cnt <= 3'd1; end
                    3'd1, 3'd2: dr_cnt <= dr_cnt + 3'd1;
                    default: if (acc_idle_all) begin dr_cnt <= 3'd0; fl_done <= 1'b1; end
                endcase
            end
            // scalar writes: push the entry, then wait for the writer to drain it
            if ((cs == C_WR) & wr_idle & wr_room) begin
                wr_req <= 1'b1; wr_kind <= (c_kind == K_WST) ? 2'd1 : (c_kind == K_W0) ? 2'd2 : 2'd3;
                wr_row0 <= q_row0[cp]; wr_nrows <= q_nrows[cp]; wr_seg_off <= 16'd0; wr_seg_len <= {(RBW+1){1'b0}};
                dr_cnt <= 3'd0;
            end
            if (cs == C_WRWAIT) begin
                if (dr_cnt != 3'd3) dr_cnt <= dr_cnt + 3'd1;
                else if (wr_idle) begin dr_cnt <= 3'd0; dr_done <= 1'b1; end
            end
        end
    end

    // ------------------------------------------------------------------ cross-lane unit
    // xp = add(xp, lane[r]) for r ascending (fma(v, 1, xp)); chunk mode: every KC rows
    // xtot_c = add(xtot_c, xp), xp = +0; flush (last tile): xtot = chunk ? (cnt ? add(xtot_c, xp) : xtot_c) : xp.
    reg  [31:0] xp, xtot_c;
    reg  [6:0]  xcnt;
    reg  [VW:0] x_r;
    reg  [2:0]  x_st;
    wire        x_ov;
    wire [31:0] x_y;
    localparam [2:0] X_IDLE = 3'd0, X_ROW = 3'd1, X_WROW = 3'd2, X_ICH = 3'd3, X_WCH = 3'd4, X_FL = 3'd5, X_WFL = 3'd6,
                     X_WFL2 = 3'd7;
    wire        x_iss = (x_st == X_ROW) | (x_st == X_ICH) | (x_st == X_FL);
    wire        x_tot_op = (x_st == X_ICH) | (x_st == X_FL);
    fp_fma #(.STAGES(2)) u_xfma (.clk(clk), .rst_n(rst_n), .in_valid(x_iss), .a(x_tot_op ? xp : xl_lane_val),
                                 .b(32'h3F800000), .c(x_tot_op ? xtot_c : xp), .out_valid(x_ov), .y(x_y));
    always @(posedge clk) begin
        xl_done <= 1'b0; xlf_done <= 1'b0;
        if (!rst_n) begin
            x_st <= X_IDLE; x_r <= {(VW+1){1'b0}}; xcnt <= 7'd0; xl_sel <= 1'b0; xl_row <= {VW{1'b0}};
        end else begin
            if (start & ~busy & d_x_reset) begin xp <= 32'd0; xtot_c <= 32'd0; xcnt <= 7'd0; end
            case (x_st)
                X_IDLE: begin
                    if (cs == C_XL & ~xl_done) begin
                        x_r <= {(VW+1){1'b0}}; xl_sel <= c_xls2; xl_row <= {VW{1'b0}};
                        x_st <= X_WROW;                           // settle the lane mux, then issue
                    end else if (cs == C_XLFL & ~xlf_done) begin
                        if (c_xlchunk & (xcnt != 7'd0)) x_st <= X_FL;
                        else begin
                            xtot <= c_xlchunk ? xtot_c : xp;
                            xp <= 32'd0; xtot_c <= 32'd0; xcnt <= 7'd0; xlf_done <= 1'b1;
                        end
                    end
                end
                X_WROW: begin
                    if (x_r >= q_nrows[cp]) begin xl_done <= 1'b1; x_st <= X_IDLE; end
                    else x_st <= X_ROW;
                end
                X_ROW: x_st <= X_WCH;                              // issued: wait for the result (2 cycles)
                X_WCH: if (x_ov) begin
                    xp <= x_y;
                    if (c_xlchunk & (xcnt == KC[6:0] - 7'd1)) begin xcnt <= 7'd0; x_st <= X_ICH; end
                    else begin
                        if (c_xlchunk) xcnt <= xcnt + 7'd1;
                        x_r <= x_r + 1'b1; xl_row <= xl_row + 1'b1; x_st <= X_WROW;
                    end
                end
                X_ICH: x_st <= X_WFL;
                X_WFL: if (x_ov) begin
                    xtot_c <= x_y; xp <= 32'd0;
                    x_r <= x_r + 1'b1; xl_row <= xl_row + 1'b1; x_st <= X_WROW;
                end
                X_FL: x_st <= X_WFL2;
                default: if (x_ov) begin                          // X_WFL2
                    xtot <= x_y; xp <= 32'd0; xtot_c <= 32'd0; xcnt <= 7'd0; xlf_done <= 1'b1; x_st <= X_IDLE;
                end
            endcase
        end
    end
endmodule
`default_nettype wire
