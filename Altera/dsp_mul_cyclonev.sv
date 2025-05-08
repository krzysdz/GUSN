(* use_dsp = "yes" *) module dsp_mul_cyclonev(
    input logic clk,
    input logic signed [7:0] a,
    input logic signed [7:0] b,
    input logic signed [7:0] c,
    input logic acc,
    output logic signed [19:0] result
);
    logic signed [7:0] a_r;
    logic signed [7:0] b_r;
    logic signed [7:0] c_r;
    logic              acc_r;
    logic signed [19:0] acc_src_c;
    assign acc_src_c = acc_r ? result : 0;

    always_ff @(posedge clk) begin
        a_r <= a;
        b_r <= b;
        c_r <= c;
        acc_r <= acc;
        result <= acc_src_c + ((a_r - c_r) * b_r);
    end
endmodule
