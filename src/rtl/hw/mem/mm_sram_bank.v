`default_nettype none
// mm_sram_bank: a WORDS x WIDTH single-port SRAM composed from the GF180 sram256x8 macro
// (gf180mcu_ocd_ip_sram__sram256x8m8wm1): ceil(WIDTH/8) macros side by side form one word,
// ceil(WORDS/256) such rows are stacked; the upper address bits select the row (CEN per row),
// the low 8 bits go to every macro.  One access per cycle (read or write, write wins), read data
// `q` is valid the cycle after the access (the macro's registered Q, plus one combinational row
// mux); q is held until the next read.  No reset (datapath).
// P6 binds the real macros through the `SYNTHESIS` blackbox path of the macro file.
module mm_sram_bank #(
    parameter WORDS = 256,
    parameter WIDTH = 32,
    parameter AW    = 8                     // address width, 2**AW >= WORDS
) (
    input  wire             clk,
    input  wire             en,             // access this cycle
    input  wire             we,             // 1 = write, 0 = read
    input  wire [AW-1:0]    addr,
    input  wire [WIDTH-1:0] d,
    output wire [WIDTH-1:0] q
);
    localparam NB   = (WIDTH + 7) / 8;      // macros per word
    localparam NR   = (WORDS + 255) / 256;  // macro rows
    localparam RW   = (AW > 8) ? AW - 8 : 1;
    localparam WP   = NB * 8;               // padded width

    wire [WP-1:0] dp = {{(WP-WIDTH){1'b0}}, d};
    wire [RW-1:0] row_sel;
    generate
        if (AW > 8) begin : g_rs
            assign row_sel = addr[AW-1:8];
        end else begin : g_rs0
            assign row_sel = {RW{1'b0}};
        end
    endgenerate
    wire [7:0] a8 = addr[7:0];

    reg [RW-1:0] row_q;                     // row of the last read (selects the output mux)
    always @(posedge clk) if (en & ~we) row_q <= row_sel;

    wire [WP-1:0] qrow [0:NR-1];
    genvar r, b;
    generate
        for (r = 0; r < NR; r = r + 1) begin : g_row
            /* verilator lint_off WIDTHEXPAND */
            wire cen_n = ~(en & (row_sel == r));
            /* verilator lint_on WIDTHEXPAND */
            for (b = 0; b < NB; b = b + 1) begin : g_mac
                gf180mcu_ocd_ip_sram__sram256x8m8wm1 u_sram (
                    .CLK(clk), .CEN(cen_n), .A(a8), .GWEN(~we), .WEN({8{~we}}),
                    .D(dp[b*8 +: 8]), .Q(qrow[r][b*8 +: 8]));
            end
        end
    endgenerate
    wire [WP-1:0] qp = qrow[row_q];
    assign q = qp[WIDTH-1:0];
endmodule
`default_nettype wire
