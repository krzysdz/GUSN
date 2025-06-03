module tree_test #(
    parameter int WIDTH_IN = 8,
    parameter int WIDTH_OUT = 8,
    parameter int N = 27
) (
    input logic clk,
    input logic signed [WIDTH_IN-1:0] inputs[N],
    output logic [WIDTH_OUT-1:0] sum
);
    adder_tree3 #(
        .WIDTH_IN(WIDTH_IN),
        .WIDTH_OUT(WIDTH_OUT),
        .N(N),
        .PIPELINE(1)
    ) adder_i (.clk(clk), .inputs(inputs), .sum(sum));
endmodule
