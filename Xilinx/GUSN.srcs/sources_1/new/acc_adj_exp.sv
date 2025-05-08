/*module acc_adj_exp #(
    parameter int ACC_W = 27,
    parameter int BIAS_W = 9,
    parameter int MUL_W = 16,
    parameter int SH_W = 6, // shift
    parameter int OOF_W = 8, // output offset
    localparam int OW = MUL_W + (ACC_W > BIAS_W ? ACC_W : BIAS_W) + 2 
) (
    input logic signed [ACC_W-1:0] acc,
    input logic signed [BIAS_W-1:0] bias,
    input logic signed [MUL_W-1:0] mul,
    input logic [SH_W-1:0] shr,
    input logic signed [OOF_W-1:0] out_off,
    output logic signed [OW-1:0] o
);
    logic signed [(ACC_W > BIAS_W ? ACC_W : BIAS_W):0] biased_acc;
    logic signed [OW-1:0] shifted_offset_rd;
    logic signed [OW-1:0] unshifted_result;
    
    assign biased_acc = (acc + bias);
    assign shifted_offset_rd = (out_off << shr) | (1 << (shr - 1));
    assign unshifted_result = biased_acc * mul + shifted_offset_rd;
    assign o = unshifted_result >> shr;
endmodule*/

`define REG_PPL

`ifndef REG_PPL
module acc_adj_exp #(
    parameter int ACC_W = 29,
    parameter int BIAS_W = 8,
    parameter int MUL_W = 16,
    parameter int SH_W = 6, // shift
    parameter int OOF_W = 8, // output offset
    localparam int OW = 48 
) (
    input logic signed [ACC_W-1:0] acc,
    input logic signed [BIAS_W-1:0] bias,
    input logic signed [MUL_W-1:0] mul,
    input logic [SH_W-1:0] shr,
    input logic signed [OOF_W-1:0] out_off,
    output logic signed [OW-1:0] o
);
    logic signed [ACC_W:0] biased_acc;
    logic signed [42:0] mul_res;
    logic signed [OW-1:0] shifted_offset_rd;
    logic signed [OW-1:0] unshifted_result;
    
    assign shifted_offset_rd = (out_off << shr) | (1 << (shr - 1));
    assign biased_acc = acc + bias;
    assign mul_res = biased_acc * mul;
    assign unshifted_result = mul_res + shifted_offset_rd;
    assign o = unshifted_result >> shr;
endmodule
`else
module acc_adj_exp #(
    parameter int ACC_W = 29,
    parameter int BIAS_W = 8,
    parameter int MUL_W = 16,
    parameter int SH_W = 6, // shift
    parameter int OOF_W = 8, // output offset
    localparam int OW = 48 
) (
    input logic clk,
    input logic signed [ACC_W-1:0] acc,
    input logic signed [BIAS_W-1:0] bias,
    input logic signed [MUL_W-1:0] mul,
    input logic [SH_W-1:0] shr,
    input logic signed [OOF_W-1:0] out_off,
    output logic signed [OW-1:0] o
);
    // registered inputs
    logic signed [ACC_W-1:0] acc_r;
    logic signed [BIAS_W-1:0] bias_r;
    logic signed [MUL_W-1:0] mul_r;
    logic [SH_W-1:0] shr_r;
    logic signed [OOF_W-1:0] out_off_r;
    // 1st stage out
    logic signed [ACC_W-1:0] biased_acc;
    logic signed [MUL_W-1:0] mul_r2;
    logic [SH_W-1:0] shr_r2;
    logic signed [OW-1:0] shifted_offset;
    logic signed [OW-1:0] shifted_round;
    // 2nd stage out
    logic signed [OW-1:0] mul_res;
    logic [SH_W-1:0] shr_r3;
    logic signed [OW-1:0] shifted_offset_rd;
    // 3rd stage out
    logic signed [OW-1:0] unshifted_result;
    logic [SH_W-1:0] shr_r4;
    
    always_ff @(posedge clk) begin
        // register inputs
        acc_r <= acc;
        bias_r <= bias;
        mul_r <= mul;
        shr_r <= shr;
        out_off_r <= out_off;
        // Preadder and shl
        biased_acc <= acc_r + bias_r;
        shifted_offset <= out_off_r << shr_r;
        shifted_round <= 1 << (shr_r - 1);
        mul_r2 <= mul_r;
        shr_r2 <= shr_r;
        // Multiplication and finalizing offset+round
        shifted_offset_rd <= shifted_offset | shifted_round;
        mul_res <= biased_acc * mul_r2;
        shr_r3 <= shr_r2;
        // Final DSP output
        unshifted_result <= mul_res + shifted_offset_rd;
        shr_r4 <= shr_r3;
        // Registered utput
        o <= unshifted_result >>> shr_r4;
    end
endmodule
`endif
