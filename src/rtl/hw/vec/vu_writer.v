`default_nettype none
// vu_writer: the vector unit's write path.  A queue of geometry entries (pushed by the sequencer
// in program order) is drained in order:
//   kind 0 OUT : the lanes' 8-word output registers of one row-segment step, lane r = row row0 + r,
//                words [taken, taken + nw) of the segment at c_base + row * c_rs + (seg_off + taken) * 4;
//                per visit of a row, up to WK consecutive groups of that row are written back to back
//                (contiguous addresses, so mm_wrport coalesces them into one burst; P6b, ruling R8),
//                then the next row; the lanes produce in lockstep, so with WK <= NOB no lane waits on
//                another (the WK-th group of row r exists once every lane has produced it).  Default WK = 1
//                (one group per row visit, the P4 order): on real_ln_fwd at 800 MB/s WK = 2 / NOB 2 measured
//                3,348,576 cycles and WK = 4 / NOB 4 3,356,732 vs 2,963,526 (docs/p6b_report.md section 3).
//   kind 1 WST : lane scalars s0 -> st0_base + row * 4, s1 -> st1_base + row * 4 (per-row stats);
//   kind 2 W0  : lane 0's s0 -> o2_base;    kind 3 W03 : lane 0's s0 -> o2_base, s3 -> o2_base + 4.
// Every write is one or two 256-bit beats with a word mask (a group of up to 8 words at any word
// address is split at the 32-byte boundary), on the same wr channel as the GEMM engine.
module vu_writer #(
    parameter V    = 16,
    parameter VW   = 4,
    parameter RBW  = 8,
    parameter QD   = 4,
    parameter WK   = 1                   // OUT: 8-word groups written per row visit (<= the lanes' NOB; P6b, R8)
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire [31:0]  d_c_base, d_c_rs, d_st0_base, d_st1_base, d_o2_base,
    // entries
    input  wire         wr_req,
    input  wire [1:0]   wr_kind,
    input  wire [15:0]  wr_row0,
    input  wire [VW:0]  wr_nrows,
    input  wire [15:0]  wr_seg_off,
    input  wire [RBW:0] wr_seg_len,
    output wire         wr_room,
    output wire         wr_idle,
    // lanes
    input  wire [V-1:0] ob_rdy,
    input  wire [V*4-1:0] ob_nw,
    input  wire [V*256-1:0] ob_data,
    output reg  [V-1:0] ob_pop,
    /* verilator lint_off UNUSEDSIGNAL */   // lane_s3: only lane 0's (clip coefficient); grp_addr[1:0]: word aligned
    input  wire [V*32-1:0] lane_s0, lane_s1, lane_s3,
    /* verilator lint_on UNUSEDSIGNAL */
    // write channel
    output reg          wr_valid,
    input  wire         wr_ready,
    output reg  [31:0]  wr_addr,
    output reg  [255:0] wr_data,
    output reg  [7:0]   wr_mask
);
    // ------------------------------------------------------------------ entry queue
    reg  [1:0]   e_kind [0:QD-1];
    reg  [15:0]  e_row0 [0:QD-1];
    reg  [VW:0]  e_nrows [0:QD-1];
    reg  [15:0]  e_segoff [0:QD-1];
    reg  [RBW:0] e_seglen [0:QD-1];
    reg  [2:0]   qw, qr;
    wire         q_ne = (qw != qr);
    assign wr_room = ((qw - qr) < QD);
    wire [1:0]   c_kind = e_kind[qr[1:0]];
    wire [15:0]  c_row0 = e_row0[qr[1:0]];
    wire [VW:0]  c_nrows = e_nrows[qr[1:0]];
    wire [15:0]  c_segoff = e_segoff[qr[1:0]];
    wire [RBW:0] c_seglen = e_seglen[qr[1:0]];
    // ------------------------------------------------------------------ engine
    reg  [3:0]   st;
    localparam [3:0] S_IDLE = 4'd0, S_MUL1 = 4'd1, S_MUL2 = 4'd2, S_LANE = 4'd3, S_B1 = 4'd4, S_B2 = 4'd5,
                     S_VEC = 4'd6, S_VB1 = 4'd7, S_VB2 = 4'd8, S_NEXT = 4'd9;
    reg  [VW+1:0] lane;                       // current lane (OUT) / first lane of the vector group (groups of 8)
    /* verilator lint_off WIDTHTRUNC */   // V < 2**(VW+2) by construction
    localparam [VW+1:0] VV2 = V;
    /* verilator lint_on WIDTHTRUNC */
    localparam [VW+1:0] EIGHT = 8;
    reg  [RBW:0] taken;
    reg  [RBW:0] koff;                        // words of the current row visit already taken
    reg  [3:0]   kc;                          // groups of the current row visit
    localparam integer WKI = WK - 1;
    localparam [3:0] WKM = WKI[3:0];
    wire [RBW:0] kofs = (kc == 4'd0) ? {(RBW+1){1'b0}} : koff;
    /* verilator lint_off UNUSEDSIGNAL */
    reg  [31:0]  rowaddr, grp_addr, row0addr; // current row base, current group address, row 0 base
    /* verilator lint_on UNUSEDSIGNAL */
    reg  [255:0] g_data;
    reg  [3:0]   g_nw;
    reg          vstep;                       // WST: 0 = s0, 1 = s1; W03: 0 = s0, 1 = s3
    reg  [31:0]  mulp;
    wire [2:0]   pos = grp_addr[4:2];
    wire [3:0]   first_n = (4'd8 - {1'b0, pos} < g_nw) ? (4'd8 - {1'b0, pos}) : g_nw;
    wire [4:0]   pend = {2'b00, pos} + {1'b0, g_nw};
    wire [3:0]   rest_n = g_nw - first_n;
    // word-wise barrel placement
    reg  [511:0] shifted;
    always @* shifted = {256'd0, g_data} << (pos * 32);
    reg  [7:0]   m1, m2;
    integer k;
    always @* begin
        m1 = 8'd0; m2 = 8'd0;
        for (k = 0; k < 8; k = k + 1) begin
            if ((k >= {29'd0, pos}) & (k < {27'd0, pend})) m1[k] = 1'b1;
            if (k < {28'd0, rest_n}) m2[k] = 1'b1;
        end
    end
    wire [255:0] lane_ob = ob_data[lane[VW-1:0] * 256 +: 256];
    wire [3:0]   lane_nw = ob_nw[lane[VW-1:0] * 4 +: 4];
    wire         lane_rdy = ob_rdy[lane[VW-1:0]];
    // vector group data: 8 lane scalars from group lane[VW:3] (lanes 8 g .. 8 g + 7), masked by nrows
    reg  [255:0] vdat;
    reg  [3:0]   vn;
    reg  [VW+1:0] vbase, vrem;
    reg  [VW-1:0] vi;
    always @* begin
        vbase = {lane[VW+1:3], 3'b000};
        vrem = c_nrows - vbase;
        for (k = 0; k < 8; k = k + 1) begin
            vi = vbase[VW-1:0] + k[VW-1:0];
            vdat[k*32 +: 32] = (c_kind == 2'd1) ? (vstep ? lane_s1[vi * 32 +: 32] : lane_s0[vi * 32 +: 32])
                             : (vstep ? lane_s3[31:0] : lane_s0[31:0]);
        end
        vn = (c_kind != 2'd1) ? 4'd1 : (vrem > 8) ? 4'd8 : vrem[3:0];   // V < 8: one group of V lanes
    end
    wire [31:0] vaddr = (c_kind == 2'd1) ? ((vstep ? d_st1_base : d_st0_base) + {14'd0, c_row0 + {{(16-VW-2){1'b0}}, vbase}, 2'b00})
                      : (d_o2_base + (vstep ? 32'd4 : 32'd0));
    assign wr_idle = ~q_ne & (st == S_IDLE) & ~wr_valid;
    always @(posedge clk) begin
        ob_pop <= {V{1'b0}};
        if (!rst_n) begin
            qw <= 3'd0; qr <= 3'd0; st <= S_IDLE; wr_valid <= 1'b0; lane <= {(VW+2){1'b0}}; taken <= {(RBW+1){1'b0}};
            vstep <= 1'b0;
        end else begin
            if (wr_req) begin
                e_kind[qw[1:0]] <= wr_kind; e_row0[qw[1:0]] <= wr_row0; e_nrows[qw[1:0]] <= wr_nrows;
                e_segoff[qw[1:0]] <= wr_seg_off; e_seglen[qw[1:0]] <= wr_seg_len; qw <= qw + 3'd1;
            end
            if (wr_valid & wr_ready) wr_valid <= 1'b0;
            case (st)
                S_IDLE: if (q_ne) begin
                    lane <= {(VW+2){1'b0}}; taken <= {(RBW+1){1'b0}}; vstep <= 1'b0; koff <= {(RBW+1){1'b0}}; kc <= 4'd0;
                    mulp <= c_row0 * d_c_rs;
                    st <= (c_kind == 2'd0) ? S_MUL1 : S_VEC;
                end
                S_MUL1: st <= S_MUL2;
                S_MUL2: begin
                    rowaddr <= d_c_base + mulp + {14'd0, c_segoff, 2'b00}; row0addr <= d_c_base + mulp + {14'd0, c_segoff, 2'b00};
                    st <= S_LANE;
                end
                // up to WK 8-word groups of a row back to back, then the next row (lanes fill in lockstep)
                S_LANE: begin
                    if (taken >= c_seglen) st <= S_NEXT;
                    else if (lane >= VV2) begin
                        lane <= {(VW+2){1'b0}}; rowaddr <= row0addr; taken <= taken + koff; koff <= {(RBW+1){1'b0}}; kc <= 4'd0;
                    end else if (lane_rdy) begin
                        // rows >= nrows (lanes beyond the tile) are popped and dropped
                        g_data <= lane_ob; g_nw <= lane_nw; grp_addr <= rowaddr + {{(29-RBW){1'b0}}, taken + kofs, 2'b00};
                        ob_pop[lane[VW-1:0]] <= 1'b1;
                        if ((kc == WKM) | (taken + kofs + {{(RBW-3){1'b0}}, lane_nw} >= c_seglen)) begin
                            lane <= lane + 1'b1; rowaddr <= rowaddr + d_c_rs; kc <= 4'd0; koff <= kofs + {{(RBW-3){1'b0}}, lane_nw};
                        end else begin
                            kc <= kc + 4'd1; koff <= kofs + {{(RBW-3){1'b0}}, lane_nw};
                        end
                        if (lane < {1'b0, c_nrows}) st <= S_B1;
                    end
                end
                S_B1: if (~wr_valid | wr_ready) begin
                    wr_valid <= 1'b1; wr_addr <= {grp_addr[31:5], 5'd0}; wr_data <= shifted[255:0]; wr_mask <= m1;
                    st <= (rest_n != 4'd0) ? S_B2 : S_LANE;
                end
                S_B2: if (~wr_valid | wr_ready) begin
                    wr_valid <= 1'b1; wr_addr <= {grp_addr[31:5], 5'd0} + 32'd32; wr_data <= shifted[511:256]; wr_mask <= m2;
                    st <= S_LANE;
                end
                S_VEC: begin
                    // vector / scalar write: one group of <= 8 words
                    if ((c_kind == 2'd1) & (lane >= {1'b0, c_nrows})) begin
                        if (vstep) st <= S_NEXT;
                        else begin vstep <= 1'b1; lane <= {(VW+2){1'b0}}; end
                    end else begin
                        g_data <= vdat; g_nw <= vn; grp_addr <= vaddr; st <= S_VB1;
                    end
                end
                S_VB1: if (~wr_valid | wr_ready) begin
                    wr_valid <= 1'b1; wr_addr <= {grp_addr[31:5], 5'd0}; wr_data <= shifted[255:0]; wr_mask <= m1;
                    st <= S_VB2;
                    if (rest_n == 4'd0) begin
                        if (c_kind == 2'd1) lane <= lane + EIGHT;
                        else vstep <= 1'b1;
                        st <= ((c_kind == 2'd1) | ((c_kind == 2'd3) & ~vstep)) ? S_VEC : S_NEXT;
                    end
                end
                S_VB2: if (~wr_valid | wr_ready) begin
                    wr_valid <= 1'b1; wr_addr <= {grp_addr[31:5], 5'd0} + 32'd32; wr_data <= shifted[511:256]; wr_mask <= m2;
                    if (c_kind == 2'd1) lane <= lane + EIGHT;
                    else vstep <= 1'b1;
                    st <= ((c_kind == 2'd1) | ((c_kind == 2'd3) & ~vstep)) ? S_VEC : S_NEXT;
                end
                S_NEXT: begin qr <= qr + 3'd1; st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
`default_nettype wire
