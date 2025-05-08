module fp_mul #(
    parameter int EXP_W = 8,
    parameter int FRAC_W = 23
) (
    input logic [WIDTH-1:0] a,
    input logic [WIDTH-1:0] b,
    output logic [WIDTH-1:0] product
);
    localparam int WIDTH = 1 + EXP_W + FRAC_W;
    localparam int MUL_W = 2*(FRAC_W+1);
    // Decomposed numbers
    logic a_sign;
    logic [EXP_W-1:0] a_exp;
    logic [FRAC_W-1:0] a_frac;
    logic b_sign;
    logic [EXP_W-1:0] b_exp;
    logic [FRAC_W-1:0] b_frac;
    logic prod_sign;
    logic [EXP_W-1:0] prod_exp;
    logic [FRAC_W-1:0] prod_frac;
    // Temporary results
    logic [MUL_W-1:0] frac_mul; // Multiplied fractions
    logic a_zero;
    logic b_zero;
    logic result_zero;

    assign {a_sign, a_exp, a_frac} = a;
    assign {b_sign, b_exp, b_frac} = b;

    // Look for 0
    assign a_zero = a_exp == 0 && a_frac == 0;
    assign b_zero = b_exp == 0 && b_frac == 0;
    assign result_zero = a_zero | b_zero;

    // Calculate sign
    assign prod_sign = a_sign ^ b_sign;

    // Intermediate multiplication result (should use dedicated DSP unit), remember about leading 1/0
    assign frac_mul = {|a_exp, a_frac} * {|b_exp, b_frac};

    // If top bit of frac_mul is 1 the mantissa is 1x,abcd and has to be shifted to 1,xabc and 1 added to exponent
    // Remember that exponents are biased: (e1+b)+(e2+b)=e1+e2+2*b, so b has to be subtracted
    assign prod_exp = a_exp + b_exp - {1'b0, {(FRAC_W-1){1'b1}}} + frac_mul[MUL_W-1];
    assign prod_frac = frac_mul[MUL_W-1] ? frac_mul[2*FRAC_W:FRAC_W+1] : frac_mul[2*FRAC_W-1:FRAC_W];

    // Bind wires into a single output
    assign product = {prod_sign,
                      result_zero ? {EXP_W{1'b0}} : prod_exp,
                      prod_frac};
endmodule
