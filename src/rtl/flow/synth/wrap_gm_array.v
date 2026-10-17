`default_nettype none
// Synthesis/STA wrapper (not part of the design): registers every input of gm_array so all timing
// paths are register to register (the feeders' outputs are registers in the design).
module wrap_gm_array #(parameter ROWS = 4, parameter COLS = 4, parameter LAMBDA = 4, parameter QW = 2, parameter FMA_STAGES = 2) (
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
    reg [ROWS*32-1:0] a_r; reg [COLS*32-1:0] b_r; reg [COLS*(QW+3)-1:0] c_r;
    reg [ROWS-1:0] i_r; reg [ROWS*QW-1:0] q_r; reg [COLS-1:0] e_r;
    reg rst_r;
    always @(posedge clk) begin a_r <= a_row; b_r <= b_col; c_r <= ctl_col; i_r <= inj_row; q_r <= inj_q; e_r <= col_en; rst_r <= rst_n; end
    (* keep_hierarchy *) gm_array #(.ROWS(ROWS), .COLS(COLS), .LAMBDA(LAMBDA), .QW(QW), .FMA_STAGES(FMA_STAGES)) u_dut (
        .clk(clk), .rst_n(rst_r), .a_row(a_r), .b_col(b_r), .ctl_col(c_r), .inj_row(i_r), .inj_q(q_r), .col_en(e_r),
        .dr_out(dr_out), .err_overrun(err_overrun));
endmodule
`default_nettype wire
