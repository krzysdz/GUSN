module fp_add #(
    parameter int EXP_W = 8,
    parameter int FRAC_W = 23
) (
    input logic [WIDTH-1:0] a,
    input logic [WIDTH-1:0] b,
    output logic [WIDTH-1:0] sum
);
    localparam int WIDTH = 1 + EXP_W + FRAC_W;

    logic [WIDTH-1:0] bigger;
    logic [WIDTH-1:0] smaller;
    logic must_swap;
    logic subtract;
    // Decomposed numbers (after swap), including leading bit in mantissa
    logic a_sign;
    logic [EXP_W-1:0] a_exp;
    logic [FRAC_W:0] a_frac;
    logic b_sign;
    logic [EXP_W-1:0] b_exp;
    logic [FRAC_W:0] b_frac;
    logic [FRAC_W:0] b_frac_sh; // Shifted to match a's exponent for fixed-point addition
    logic [EXP_W-1:0] exp_diff;
    logic [FRAC_W+1:0] fixed_result; // add/sub fixed-point result (must be normalised)
    logic requires_shl; // Variable-length shift left is required for subtractions when e_a-e_b < 2 - e.g.:
                        // 1.000 - 0.111 = 0.001 (diff 1)
                        // 1.001 - 1.000 = 0.001 (diff 0)
                        // 1.000 - 0.011 = 0.101 (diff 2 - at most 1 bit has to be shifted left)
    // no_lz = counting leading zeros and variable shl are not necessary
    logic [FRAC_W-1:0] no_lz_frac;
    logic [EXP_W-1:0] no_lz_exp;
    logic [$clog2(FRAC_W)-1:0] leading_zeros;
    logic result_is_zero;
    logic [FRAC_W-1:0] lz_frac;
    logic [EXP_W-1:0] lz_exp;

    assign subtract = a[WIDTH-1] ^ b[WIDTH-1];

    // Compare exponents, then fractional part - in case of subtraction there must be no underflow (FP is sign magnitude)
    assign must_swap = a[WIDTH-2:0] < b[WIDTH-2:0];
    assign bigger = must_swap ? b : a;
    assign smaller = must_swap ? a : b;

    assign {a_sign, a_exp, a_frac[FRAC_W-1:0]} = bigger;
    assign {b_sign, b_exp, b_frac[FRAC_W-1:0]} = smaller;
    // Set leading bit based on whether the number is normalised or subnormal
    assign a_frac[FRAC_W] = a_exp != 0;
    assign b_frac[FRAC_W] = b_exp != 0;

    assign exp_diff = a_exp - b_exp;
    assign b_frac_sh = b_frac >> exp_diff;

    assign fixed_result = subtract ? a_frac - b_frac_sh : a_frac + b_frac_sh;

    assign requires_shl = subtract && exp_diff <= 1;

    assign no_lz_frac = subtract ?
                            (fixed_result[FRAC_W] ? fixed_result[FRAC_W-1:0] : {fixed_result[FRAC_W-2:0], 1'b0}):
                            (fixed_result[FRAC_W+1] ? fixed_result[FRAC_W:1] : fixed_result[FRAC_W-1:0]);
    assign no_lz_exp = subtract ?
                            (fixed_result[FRAC_W] ? a_exp : a_exp - 1) :
                            (fixed_result[FRAC_W+1] ? a_exp + 1 : a_exp);

    always_comb begin
        for (leading_zeros = 1; leading_zeros < FRAC_W; ++leading_zeros)
            if (fixed_result[FRAC_W-leading_zeros]) break;
    end
    assign lz_frac = fixed_result[FRAC_W-1:0] << leading_zeros;
    assign lz_exp = a_exp - lz_frac;
    assign result_is_zero = fixed_result == 0;

    assign sum = requires_shl ?
                     (result_is_zero ? {WIDTH{1'b0}} : {a_sign, lz_exp, lz_frac}) :
                     {a_sign, no_lz_exp, no_lz_frac};
endmodule
