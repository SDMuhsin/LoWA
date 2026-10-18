`default_nettype none
// gm_array: ROWS x COLS output-stationary systolic array of gm_pe.  a operands flow along a row
// (left to right, one register per PE), b operands and the control word flow down a column, the
// drain chains run down the columns (dr_out[c] = the bottom PE's drain register).  The operand
// skew (row r / column c delayed by r + c hops) is produced by the feeders (docs/p3_report.md
// section 2): a_row[r] must present A[r, k] during the LAMBDA cycles in which PE(r, 0) consumes
// (k, 0..LAMBDA-1), and b_col[c] / ctl_col[c] present W row c*LAMBDA+q at k in the cycle PE(0, c)
// consumes it, i.e. c cycles after column 0.
// Drain: inj_row[r] / inj_q[r] from gm_drain inject res[q] of every PE of row r whose column has
// col_en[c] = 1.  P8c2 (PLAN R14a): with DRS > 0 the chain has one extra register after every DRS rows (between
// PE(r, c) and PE(r + 1, c) when (r + 1) mod DRS = 0), so a PE-to-PE hop is two register stages; gm_drain injects
// a row's words as many cycles earlier as there are extra registers below it, so the words still reach the chain
// bottom in the same cycles (no latency change; see gm_drain).  err_overrun = OR of the PE flags (sticky, reset only by rst_n).
// Reset distribution: rst_n is re-registered once per row (rst_row[r]) so that no single net
// drives the ROWS*COLS PEs; the PEs leave reset one cycle after the rest of the engine, which
// is idle for many cycles after reset anyway (P8's reset tree starts from these flops).
module gm_array #(
    parameter ROWS       = 16,
    parameter COLS       = 16,
    parameter LAMBDA     = 4,
    parameter QW         = 2,
    parameter FMA_STAGES = 2,
    parameter DRS        = 1                // drain chain: extra register after every DRS rows (0 = none)
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire [ROWS*32-1:0]     a_row,
    input  wire [COLS*32-1:0]     b_col,
    input  wire [COLS*(QW+3)-1:0] ctl_col,
    input  wire [ROWS-1:0]        inj_row,
    input  wire [ROWS*QW-1:0]     inj_q,
    input  wire [COLS-1:0]        col_en,
    output wire [COLS*32-1:0]     dr_out,
    output wire                   err_overrun
);
    localparam CW = QW + 3;
    wire [31:0]   a_w   [0:ROWS-1][0:COLS];
    wire [31:0]   b_w   [0:ROWS][0:COLS-1];
    wire [CW-1:0] ctl_w [0:ROWS][0:COLS-1];
    wire [31:0]   dr_w  [0:ROWS][0:COLS-1];
    wire [ROWS*COLS-1:0] err;
    reg  [ROWS-1:0] rst_row;
    always @(posedge clk) rst_row <= {ROWS{rst_n}};
    genvar r, c;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : g_r
            assign a_w[r][0] = a_row[r*32 +: 32];
            for (c = 0; c < COLS; c = c + 1) begin : g_c
                if (r == 0) begin : g_top
                    assign b_w[0][c]   = b_col[c*32 +: 32];
                    assign ctl_w[0][c] = ctl_col[c*CW +: CW];
                    assign dr_w[0][c]  = 32'd0;
                end
                wire [31:0] dr_o;
                gm_pe #(.FMA_STAGES(FMA_STAGES), .LAMBDA(LAMBDA), .QW(QW)) u_pe (
                    .clk(clk), .rst_n(rst_row[r]),
                    .a_in(a_w[r][c]), .b_in(b_w[r][c]), .ctl_in(ctl_w[r][c]),
                    .a_out(a_w[r][c+1]), .b_out(b_w[r+1][c]), .ctl_out(ctl_w[r+1][c]),
                    .inj(inj_row[r] & col_en[c]), .inj_q(inj_q[r*QW +: QW]),
                    .dr_in(dr_w[r][c]), .dr_out(dr_o),
                    .err_overrun(err[r*COLS+c]));
                if (DRS > 0 && r < ROWS - 1 && ((r + 1) % (DRS > 0 ? DRS : 1)) == 0) begin : g_dreg
                    reg [31:0] dr_p;                // chain pipeline register (datapath: not reset)
                    always @(posedge clk) dr_p <= dr_o;
                    assign dr_w[r+1][c] = dr_p;
                end else begin : g_dwire
                    assign dr_w[r+1][c] = dr_o;
                end
                if (r == ROWS - 1) begin : g_bot
                    assign dr_out[c*32 +: 32] = dr_w[ROWS][c];
                end
            end
            /* verilator lint_off UNUSEDSIGNAL */   // the last column's a_out leaves the array
            wire [31:0] unused_a = a_w[r][COLS];
            /* verilator lint_on UNUSEDSIGNAL */
        end
        for (c = 0; c < COLS; c = c + 1) begin : g_unused
            /* verilator lint_off UNUSEDSIGNAL */   // the bottom row's b / ctl outputs leave the array
            wire [31:0]   unused_b = b_w[ROWS][c];
            wire [CW-1:0] unused_c = ctl_w[ROWS][c];
            /* verilator lint_on UNUSEDSIGNAL */
        end
    endgenerate
    assign err_overrun = |err;
endmodule
`default_nettype wire
