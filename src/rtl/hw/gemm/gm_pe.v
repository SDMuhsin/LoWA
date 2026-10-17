`default_nettype none
// gm_pe: one output-stationary processing element of the GEMM array (docs/dataflow.md section 2).
//
// Operands enter from the left (a, held by the feeder for LAMBDA cycles per k) and from the top
// (b, one W row c*LAMBDA+q per cycle, q = 0..LAMBDA-1, with the control word) and are passed on
// one hop (one register) to the right / downward.  The PE keeps LAMBDA chunk partials in flight
// inside its FMA pipeline: the op for (k, q) is issued LAMBDA cycles after the op for (k-1, q), so
// the result of the previous op is its c operand (delayed LAMBDA - FMA_STAGES cycles).
//   p = +0 at k = 0 of every K chunk (ctl.first selects +0 instead of the feedback: the
//   restart rule), p = fma(a_k, b_k, p) for k ascending (ctl.last marks the chunk's last k).
// At ctl.last the FMA result for q is captured in res[q] (the LAMBDA accumulator registers of
// spec.md) and held until the drain engine injects it into the column drain chain:
//   dr_s <= inj ? res[inj_q] : dr_in     (dr_in = dr_s of the PE above, 0 at row 0).
// err_overrun flags a capture into a res[q] that was not yet drained (a scheduling bug).
// Control word ctl = {valid, first, last, q[QW-1:0]}; valid is the only reset state.
//
// Timing: FMA_STAGES = 2 closes 40 ns at ss_125C_3v00 (docs/p3_report.md): the only logic added
// to fp_fma's critical path is the 2:1 feedback mux on c; the operand / control / drain registers
// are plain flops.  FMA_STAGES must be <= LAMBDA.
module gm_pe #(
    parameter FMA_STAGES = 2,
    parameter LAMBDA     = 4,
    parameter QW         = 2               // log2(LAMBDA)
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire [31:0]   a_in,
    input  wire [31:0]   b_in,
    input  wire [QW+2:0] ctl_in,
    output wire [31:0]   a_out,
    output wire [31:0]   b_out,
    output wire [QW+2:0] ctl_out,
    input  wire          inj,
    input  wire [QW-1:0] inj_q,
    input  wire [31:0]   dr_in,
    output wire [31:0]   dr_out,
    output reg           err_overrun
);
    localparam CW = QW + 3;
    // ------------------------------------------------------------------ hop registers
    reg [31:0]   a_s, b_s;
    reg [CW-1:0] ctl_s;
    always @(posedge clk) begin
        a_s <= a_in;
        b_s <= b_in;
        if (!rst_n) ctl_s <= {CW{1'b0}};
        else        ctl_s <= ctl_in;
    end
    assign a_out = a_s;
    assign b_out = b_s;
    assign ctl_out = ctl_s;
    wire          v_s     = ctl_s[CW-1];
    wire          first_s = ctl_s[CW-2];
    wire          last_s  = ctl_s[CW-3];
    wire [QW-1:0] q_s     = ctl_s[QW-1:0];

    // ------------------------------------------------------------------ FMA with feedback
    wire [31:0] y, fb;
    wire        yv;
    cm_delay #(.W(32), .N(LAMBDA - FMA_STAGES), .RST(0)) u_fb (.clk(clk), .rst_n(rst_n), .d(y), .q(fb));
    wire [31:0] cc = first_s ? 32'd0 : fb;
    fp_fma #(.STAGES(FMA_STAGES)) u_fma (.clk(clk), .rst_n(rst_n), .in_valid(v_s),
        .a(a_s), .b(b_s), .c(cc), .out_valid(yv), .y(y));

    // ------------------------------------------------------------------ chunk-result capture
    wire          cap;
    wire [QW-1:0] cap_q;
    cm_delay #(.W(1), .N(FMA_STAGES), .RST(1)) u_cv (.clk(clk), .rst_n(rst_n), .d(v_s & last_s), .q(cap));
    cm_delay #(.W(QW), .N(FMA_STAGES), .RST(0)) u_cq (.clk(clk), .rst_n(rst_n), .d(q_s), .q(cap_q));
    /* verilator lint_off UNUSEDSIGNAL */   // out_valid of the FMA is implied by cap
    wire unused_yv = yv;
    /* verilator lint_on UNUSEDSIGNAL */

    reg [31:0]     res  [0:LAMBDA-1];
    reg [LAMBDA-1:0] pend;
    integer i;
    always @(posedge clk) begin
        for (i = 0; i < LAMBDA; i = i + 1) begin
            /* verilator lint_off WIDTHEXPAND */
            if (cap && cap_q == i) res[i] <= y;
            if (!rst_n)                       pend[i] <= 1'b0;
            else if (cap && cap_q == i)       pend[i] <= 1'b1;
            else if (inj && inj_q == i)       pend[i] <= 1'b0;
            /* verilator lint_on WIDTHEXPAND */
        end
        if (!rst_n) err_overrun <= 1'b0;
        else if (cap && pend[cap_q] && !(inj && inj_q == cap_q)) err_overrun <= 1'b1;
    end

    // ------------------------------------------------------------------ drain chain
    reg [31:0] dr_s;
    always @(posedge clk) dr_s <= inj ? res[inj_q] : dr_in;
    assign dr_out = dr_s;
endmodule
`default_nettype wire
