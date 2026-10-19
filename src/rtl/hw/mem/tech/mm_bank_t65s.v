`default_nettype none
// mm_bank_t65s (mm_sram_bank backend TECH = 3, tsmc65_sram; PLAN R16): WORDS x WIDTH single-port bank built on
// one compiled SRAM / register-file macro (mm_sram_macro; the site wrapper maps it to the generated macro of this
// shape).  Same port contract as every mm_sram_bank backend:
//   read : en & ~we at posedge N; q is the addressed word after N and is held until the next read;
//   write: en & we at posedge N stores d; a read at posedge N+1 sees it; writes never change q.
// Adapter: the macros are write-through (q follows d on a write), so the contract's "writes never change q"
// needs a hold register.  rd_last = 1 while the macro's q still shows the last read (no write since);
// on a write, q_h captures the current q and q switches to it until the next read.  The macro's q is held
// on cycles without an access, so idle cycles need nothing.  Cost: WIDTH + 1 flops and a WIDTH-bit 2:1 mux
// after the macro output.  Read latency, address / data timing and the one-access-per-cycle rule are unchanged.
module mm_bank_t65s #(
    parameter WORDS = 256,
    parameter WIDTH = 32,
    parameter AW    = 8
) (
    input  wire             clk,
    input  wire             en,
    input  wire             we,
    input  wire [AW-1:0]    addr,
    input  wire [WIDTH-1:0] d,
    output wire [WIDTH-1:0] q
);
    wire [WIDTH-1:0] mq;
    reg              rd_last;               // the macro output holds the data of the last read
    reg [WIDTH-1:0]  q_h;                   // read data held across writes

    mm_sram_macro #(.WORDS(WORDS), .WIDTH(WIDTH), .AW(AW)) u_mac (
        .clk(clk), .cen_n(~en), .wen_n(~we), .a(addr), .d(d), .q(mq));

    always @(posedge clk) begin
        if (en) rd_last <= ~we;
        if (en & we & rd_last) q_h <= mq;
    end
    assign q = rd_last ? mq : q_h;
endmodule
`default_nettype wire
