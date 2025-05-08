module fp_muladd #(
    parameter int EXP_W = 8,
    parameter int FRAC_W = 23
) (
    input logic [WIDTH-1:0] a,
    input logic [WIDTH-1:0] b,
    input logic [WIDTH-1:0] c,
    output logic [WIDTH-1:0] acc
);
    localparam int WIDTH = 1 + EXP_W + FRAC_W;

    logic [WIDTH-1:0] tmp;
    fp_mul #(.EXP_W(EXP_W), .FRAC_W(FRAC_W)) i_mul(.a(a), .b(b), .product(tmp));
    fp_add #(.EXP_W(EXP_W), .FRAC_W(FRAC_W)) i_add(.a(tmp), .b(c), .sum(acc));
endmodule
