// +define+MK_GEN (included by tb/vec/tb_vu_top.sv and tb/gemm/tb_gm_top.sv, ideal-memory path): the engine's mk
// channel is served by hw/rng/rn_mkgen (the on-chip generator, P6b) instead of the tb's mask array.  Configuration
// +mkseed= +mkstep= +mksite= +mklayer= (hex); `MKGEN_NC = Threefry pipelines (default 9).  Every response is also
// checked against the L1 keep bits of the case (+mask=, model/rng.py via the case generator): MKCHECK line.
`ifndef MKGEN_NC
`define MKGEN_NC 9
`endif
    reg  [63:0] mg_seed = 0;
    reg  [23:0] mg_step = 0;
    reg  [3:0]  mg_site = 0, mg_layer = 0;
    integer     mg_bad = 0, mg_chk = 0, mg_j;
    initial begin
        if ($value$plusargs("mkseed=%h", mg_seed)) ;
        if ($value$plusargs("mkstep=%h", mg_step)) ;
        if ($value$plusargs("mksite=%h", mg_site)) ;
        if ($value$plusargs("mklayer=%h", mg_layer)) ;
    end
    wire             mg_ready, mg_valid;
    wire [MKG_W-1:0] mg_keep;
    rn_mkgen #(.MKW(MKG_W), .NCORE(`MKGEN_NC)) u_mkgen (.clk(clk), .rst_n(rst_n), .seed(mg_seed), .step(mg_step),
        .site(mg_site), .layer(mg_layer), .mk_req_valid(mk_req_valid), .mk_req_ready(mg_ready), .mk_req_idx(mk_req_idx),
        .mk_rsp_valid(mg_valid), .mk_rsp_keep(mg_keep), .idle());
    always @(posedge clk) begin
        if (mg_valid) begin
            mg_chk = mg_chk + 1;
            for (mg_j = 0; mg_j < MKG_W; mg_j = mg_j + 1)
                if ((qm_i[qm_r % 64] + mg_j) < mask_n && mg_keep[mg_j] !== mask[qm_i[qm_r % 64] + mg_j]) begin
                    if (mg_bad < 5) $display("MKGEN MISMATCH idx %0d bit %0d got %b want %b", qm_i[qm_r % 64], mg_j,
                                             mg_keep[mg_j], mask[qm_i[qm_r % 64] + mg_j]);
                    mg_bad = mg_bad + 1;
                end
        end
    end
