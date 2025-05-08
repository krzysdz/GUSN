module q_tensor #(
    parameter int N_INPUTS = 8
)(
    input logic signed [7:0] weights[N_INPUTS],
    input logic signed [7:0] inputs[N_INPUTS],
    input logic signed [31:0] bias,
    input logic signed [31:0] input_offset,
    input logic signed [31:0] weight_offset,
    input logic signed [31:0] output_multiplier,
    input logic signed [31:0] output_offset,
    // -31 + output_shift from data - result is [-62, -24] - right shift by 24 to 62 bits
    // 24 is constant shifted, so adjusted_shift specifies right shift in range [0, 38]
    input logic [5:0] adjusted_shift,
    output logic signed [7:0] result
);
    logic signed [31:0] acc;
    logic signed [63:0] acc_mul;
    logic signed [31:0] acc_mul_sh;
    logic signed [31:0] acc_scaled;

    always_comb begin
        acc = 0;
        // I hope it wont create a chain of additions, please be a tree
        for (int i = 0; i < N_INPUTS; ++i)
            acc += (weights[i] + weight_offset) * (inputs[i] + input_offset);
        acc += bias;
    end
    assign acc_mul = acc * output_multiplier;
    assign acc_mul_sh = acc_mul[63:24] >> adjusted_shift;
    assign acc_scaled = acc_mul_sh + output_offset;
    always_comb assert (acc_scaled >= -128 && acc_scaled <= 127);
    assign result = acc_scaled;

    // https://github.com/tensorflow/tflite-micro/blob/bc68d362d6f3ac93ce11d8712974d05b1d6a8305/tensorflow/lite/kernels/internal/common.cc#L80-L87
    // Inputs:
    // - quantized_multiplier has fixed point at bit 31
    // - shift is -31 to +7 (negative for right shift)
    //
    // Assumptions: The following input ranges are assumed
    // - quantize_scale>=0  (the usual range is (1<<30) to (1>>31)-1)
    // - scaling is chosen so final scaled result fits in int32_t
    // - input x is in the range -(1<<47) <= x < (1<<47)
endmodule
