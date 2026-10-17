`default_nettype none
// Synthesis/STA wrapper (not part of the design): registers every input of vu_lane so that all
// timing paths are register to register (the sequencer's control and the loader's writes are flops).
module wrap_vu_lane #(parameter NS = 8, parameter SW = 3, parameter NOB = 2) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         ld_we, ld_buf, sld_v, iss_v, iss_abuf, iss_keep, iss_isy, km_d, ex_v,
    input  wire [7:0]   ld_addr, iss_addr,
    input  wire [31:0]  ld_d, sld_d, cv0_d, cv1_d, cv2_d, c0, c1, c2, c3, ds, xtot,
    input  wire [1:0]   sld_sel,
    input  wire [SW-1:0] iss_slot, ex_slot,
    input  wire [35:0]  ex_uop,
    input  wire         acc_start, acc_chunk, acc_flush, max_start, cur_abuf, ob_flush, ob_pop, xl_sel,
    output wire         acc_idle, st_busy, ob_room, ob_rdy,
    output wire [3:0]   acc_cnt, sfu_out_cnt, ob_nw,
    output wire [255:0] ob_data,
    output wire [31:0]  xl_val, s0_o, s1_o, s3_o
);
    reg         q_ld_we, q_ld_buf, q_sld_v, q_iss_v, q_iss_abuf, q_iss_keep, q_iss_isy, q_km_d, q_ex_v;
    reg  [7:0]  q_ld_addr, q_iss_addr;
    reg  [31:0] q_ld_d, q_sld_d, q_cv0, q_cv1, q_cv2, q_c0, q_c1, q_c2, q_c3, q_ds, q_xtot;
    reg  [1:0]  q_sld_sel;
    reg  [SW-1:0] q_iss_slot, q_ex_slot;
    reg  [35:0] q_uop;
    reg         q_as, q_ac, q_af, q_ms, q_ca, q_of, q_op, q_xs;
    always @(posedge clk) begin
        q_ld_we <= ld_we; q_ld_buf <= ld_buf; q_sld_v <= sld_v; q_iss_v <= iss_v; q_iss_abuf <= iss_abuf;
        q_iss_keep <= iss_keep; q_iss_isy <= iss_isy; q_km_d <= km_d; q_ex_v <= ex_v; q_ld_addr <= ld_addr;
        q_iss_addr <= iss_addr; q_ld_d <= ld_d; q_sld_d <= sld_d; q_cv0 <= cv0_d; q_cv1 <= cv1_d; q_cv2 <= cv2_d;
        q_c0 <= c0; q_c1 <= c1; q_c2 <= c2; q_c3 <= c3; q_ds <= ds; q_xtot <= xtot; q_sld_sel <= sld_sel;
        q_iss_slot <= iss_slot; q_ex_slot <= ex_slot; q_uop <= ex_uop; q_as <= acc_start; q_ac <= acc_chunk;
        q_af <= acc_flush; q_ms <= max_start; q_ca <= cur_abuf; q_of <= ob_flush; q_op <= ob_pop; q_xs <= xl_sel;
    end
    vu_lane #(.NS(NS), .SW(SW), .NOB(NOB)) u (
        .clk(clk), .rst_n(rst_n), .ld_we(q_ld_we), .ld_buf(q_ld_buf), .ld_addr(q_ld_addr), .ld_d(q_ld_d),
        .sld_v(q_sld_v), .sld_sel(q_sld_sel), .sld_d(q_sld_d), .iss_v(q_iss_v), .iss_abuf(q_iss_abuf), .iss_addr(q_iss_addr),
        .iss_slot(q_iss_slot), .iss_keep(q_iss_keep), .iss_isy(q_iss_isy), .km_d(q_km_d), .cv0_d(q_cv0), .cv1_d(q_cv1), .cv2_d(q_cv2),
        .ex_v(q_ex_v), .ex_slot(q_ex_slot), .ex_uop(q_uop), .c0(q_c0), .c1(q_c1), .c2(q_c2), .c3(q_c3), .ds(q_ds),
        .acc_start(q_as), .acc_chunk(q_ac), .acc_flush(q_af), .acc_idle(acc_idle), .acc_cnt(acc_cnt), .max_start(q_ms),
        .sfu_out_cnt(sfu_out_cnt), .st_busy(st_busy), .cur_abuf(q_ca), .ob_flush(q_of), .ob_room(ob_room), .ob_rdy(ob_rdy),
        .ob_nw(ob_nw), .ob_data(ob_data), .ob_pop(q_op), .xl_sel(q_xs), .xtot(q_xtot), .xl_val(xl_val), .s0_o(s0_o),
        .s1_o(s1_o), .s3_o(s3_o));
endmodule
`default_nettype wire
