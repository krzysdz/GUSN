module dsp_mul_test(
    input logic clk,
    input logic signed [7:0] a,
    input logic signed [7:0] b,
    input logic signed [7:0] c,
    input logic no_acc,
    output logic signed [19:0] result
);
    logic signed [7:0] s1_a;
    logic signed [7:0] s1_b;
    logic signed [7:0] s1_c;
    logic s1_no_acc;
    logic signed [8:0] s2_apc;
    logic signed [7:0] s2_b;
    logic s2_no_acc;
    logic signed [16:0] s3_prod;
    logic s3_no_acc;
    
    always_ff @(posedge clk) begin
        s1_a <= a;
        s1_b <= b;
        s1_c <= c;
        s1_no_acc <= no_acc;
        s2_apc <= s1_a + s1_c;
        s2_b <= s1_b;
        s2_no_acc <= s1_no_acc;
        s3_prod <= s2_apc * s2_b;
        s3_no_acc <= s2_no_acc;
        result <= (s3_no_acc ? 20'sd0 : result) + s3_prod;
    end
endmodule
