module adder_tree3 #(
    parameter int WIDTH_IN = 8,
    parameter int WIDTH_OUT = 8,
    parameter int N = 27,
    parameter bit PIPELINE = 1
) (
    input logic clk,
    input logic signed [WIDTH_IN-1:0] inputs[N],
    output logic [WIDTH_OUT-1:0] sum
);
    function automatic int smaller_pow_3(input int x);
        // int p = 1;
        // int next_p;
        // while (1) begin // This never ends in Quartus
        //     next_p = p * 3;
        //     if (next_p >= x) return p;
        //     p = next_p;
        // end
        // Quartus-compatible version
        // if (x <= 3) return 1;
        // if (x <= 9) return 3;
        // if (x <= 27) return 9;
        // if (x <= 81) return 27;
        // if (x <= 243) return 81;
        // if (x <= 729) return 243;
        // if (x <= 2187) return 729;
        // if (x <= 6561) return 2187;
        // if (x <= 19683) return 6561;
        // if (x <= 59049) return 19683;
        // return 'x;
        // https://community.intel.com/t5/Intel-Quartus-Prime-Software/SystemVerilog-loops-in-functions-are-completely-broken/m-p/1682861
        // This works in Quartus
        smaller_pow_3 = 1;
        while (smaller_pow_3 * 3 < x) smaller_pow_3 *= 3;
    endfunction
    function automatic int min(input int a, input int b);
        return a <= b ? a : b;
    endfunction

    localparam int MaxSubN = smaller_pow_3(N);
    localparam int Sub1N = MaxSubN;
    localparam int Sub2N = min(MaxSubN, N - Sub1N);
    localparam int Sub3N = min(MaxSubN, N - Sub1N - Sub2N);

    initial $display("adder N=%d, S1N=%d, S2N=%d, S3N=%d", N, Sub1N, Sub2N, Sub3N);

    logic [WIDTH_OUT-1:0] sum_c;

    // Quartus for some reason requires generate keyword (conditional generate constructs are not supported outside generate regions)
    generate
        if (N == 3) assign sum_c = inputs[0] + inputs[1] + inputs[2];
        if (N == 2) assign sum_c = inputs[0] + inputs[1];
        if (N == 1) assign sum_c = inputs[0];
        if (N > 3) begin : g_adder_subadders
            logic [WIDTH_OUT-1:0] s1;
            logic [WIDTH_OUT-1:0] s2;

            adder_tree3 #(
                .WIDTH_IN(WIDTH_IN),
                .WIDTH_OUT(WIDTH_OUT),
                .N(Sub1N),
                .PIPELINE(PIPELINE)
            ) i_at3_1(.clk(clk), .inputs(inputs[0:Sub1N-1]), .sum(s1));
            adder_tree3 #(
                .WIDTH_IN(WIDTH_IN),
                .WIDTH_OUT(WIDTH_OUT),
                .N(Sub2N),
                .PIPELINE(PIPELINE)
            ) i_at3_2(.clk(clk), .inputs(inputs[Sub1N:Sub1N+Sub2N-1]), .sum(s2));

            if (Sub3N > 0) begin : g_adder_sub_full3
                logic [WIDTH_OUT-1:0] s3;
                adder_tree3 #(
                    .WIDTH_IN(WIDTH_IN),
                    .WIDTH_OUT(WIDTH_OUT),
                    .N(Sub3N),
                    .PIPELINE(PIPELINE)
                ) i_at3_3(.clk(clk), .inputs(inputs[Sub1N+Sub2N:N-1]), .sum(s3));

                assign sum_c = s1 + s2 + s3;
            end else begin : g_adder_sub_only2
                assign sum_c = s1 + s2;
            end
        end
    endgenerate
    generate
        if (PIPELINE)
            always_ff @(posedge clk) sum <= sum_c;
        else
            assign sum = sum_c;
    endgenerate
endmodule
