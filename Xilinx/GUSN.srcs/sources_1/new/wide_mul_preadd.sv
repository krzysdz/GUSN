module wide_mul_preadd #(
    parameter int AW = 27,
    parameter int BW = 9,
    parameter int MW = 38,
    localparam int OW = MW + (AW > BW ? AW : BW) + 2 
) (
    input signed [AW-1:0] a,
    input signed [BW-1:0] b,
    input [MW-1:0] m,
    output signed [OW-1:0] o
);
    assign o = (a + b) * $signed({1'b0, m});
endmodule
