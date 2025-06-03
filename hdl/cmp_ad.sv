// Compare (signed) with associated data
module cmp_ad #(
    parameter int NUM_W = 8,
    parameter int AD_W = 8
) (
    input logic signed [NUM_W-1:0] a_num,
    input logic [AD_W-1:0] a_data,
    input logic signed [NUM_W-1:0] b_num,
    input logic [AD_W-1:0] b_data,
    output logic signed [NUM_W-1:0] o_num,
    output logic [AD_W-1:0] o_data
);
    logic a_bigger_or_eq;

    assign a_bigger_or_eq = a_num >= b_num;
    assign o_num = a_bigger_or_eq ? a_num : b_num;
    assign o_data = a_bigger_or_eq ? a_data : b_data;
endmodule
