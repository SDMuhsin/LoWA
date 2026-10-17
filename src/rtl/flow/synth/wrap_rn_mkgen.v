`default_nettype none
// Synthesis/STA wrapper (not part of the design): one rn_mkgen with every input registered so all paths are
// register to register.
module wrap_rn_mkgen #(parameter MKW = 16, parameter NCORE = 9, parameter STAGES = 3) (
    input  wire           clk, rst_n,
    input  wire [63:0]    seed,
    input  wire [23:0]    step,
    input  wire [3:0]     site, layer,
    input  wire           mk_req_valid,
    input  wire [31:0]    mk_req_idx,
    output wire           mk_req_ready, mk_rsp_valid, idle,
    output wire [MKW-1:0] mk_rsp_keep
);
    reg [63:0] q_seed; reg [23:0] q_step; reg [3:0] q_site, q_layer; reg q_v; reg [31:0] q_i;
    always @(posedge clk) begin
        q_seed <= seed; q_step <= step; q_site <= site; q_layer <= layer; q_v <= mk_req_valid; q_i <= mk_req_idx;
    end
    rn_mkgen #(.MKW(MKW), .NCORE(NCORE), .STAGES(STAGES)) u (.clk(clk), .rst_n(rst_n), .seed(q_seed), .step(q_step),
        .site(q_site), .layer(q_layer), .mk_req_valid(q_v), .mk_req_ready(mk_req_ready), .mk_req_idx(q_i),
        .mk_rsp_valid(mk_rsp_valid), .mk_rsp_keep(mk_rsp_keep), .idle(idle));
endmodule
`default_nettype wire
