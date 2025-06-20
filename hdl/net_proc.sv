`include "datatypes.svh"

module net_proc #(
    parameter int MUL_WIDTH = 27,
    parameter int NUM_CHUNKS = 35
) (
    input logic clk,
    input logic start,
    output logic done,
    output logic [3:0] max_idx_10, // max index of first 10 memory items

    input logic ext_mem_rst,
    input logic ext_mem_we,
    input logic [7:0] ext_mem_wdata
);
    localparam int InstAddrW = 13;
    localparam int SingleMacW = 24;
    localparam int AccAllW = 24;
    localparam int MulW = 15; // this is width of unsigned
    localparam int PreShrW = AccAllW + MulW + 2; // 1, because mul will be signed; 1 for additions ovf

    logic running;
    logic wait_wr;
    full_inst_t instruction;
    full_inst_t instruction_tmp;
    logic [InstAddrW-1:0] inst_ptr;
    logic [$bits(full_inst_t)-1:0] prog_mem[2**InstAddrW];
    initial $readmemb("prog.dat", prog_mem);

`ifdef RAM_ACLR
    // This still does not have registered output in M10K, but at least uses memory blocks, unlike version with sync reset
    always_ff @(posedge clk, posedge start)
`else
    always_ff @(posedge clk)
`endif
    begin
        if (start) begin
            inst_ptr <= 0;
            instruction <= '{1'b0, 1'b0, {27*8{1'b0}}, 1'b0, 1'b0, NOP, 15'd0};
            instruction_tmp <= '{1'b0, 1'b0, {27*8{1'b0}}, 1'b0, 1'b0, NOP, 15'd0};
        end else begin
            if (running && !wait_wr && instruction.proc_inst != FIN) begin
                inst_ptr <= inst_ptr + 1;
                instruction_tmp <= full_inst_t'(prog_mem[inst_ptr]);
                instruction <= instruction_tmp;
            end
        end
    end
    always_ff @(posedge clk) begin
        if (start || ext_mem_rst || instruction.proc_inst == FIN)
            running <= start;
    end

    oof_d_t oof_d;
    mul_d_t mul_d;
    shr_d_t shr_d;
    act_d_t act_d;
    st_d_t st_d;

`ifndef QUARTUS
    assign oof_d = instruction.proc_data.oof_d;
    assign mul_d = instruction.proc_data.mul_d;
    assign shr_d = instruction.proc_data.shr_d;
    assign act_d = instruction.proc_data.act_d;
    assign st_d  = instruction.proc_data.st_d;
`else
    assign oof_d = oof_d_t'(instruction.proc_data);
    assign mul_d = mul_d_t'(instruction.proc_data);
    assign shr_d = shr_d_t'(instruction.proc_data);
    assign act_d = act_d_t'(instruction.proc_data);
    assign st_d  = st_d_t'(instruction.proc_data);
`endif

    logic signed [7:0] in_offset_neg;
    // Parameters set by instructions
    logic signed [7:0] out_offset;
    logic [MulW-1:0] multiplier;
    logic [5:0] shift;
    logic signed [7:0] act_min;
    // Memory read pointer saving
    logic [$clog2(NUM_CHUNKS)-1:0] saved_rptr;
    logic [$clog2(NUM_CHUNKS)-1:0] mem_rptr;
    // Internal wires
    param_arr_t mem_data;
    logic signed [SingleMacW-1:0] mac_outs[27];
    // Final neuron processing pipeline
    logic signed [AccAllW-1:0] mac_sum; // this is already after 2 steps
    logic signed [AccAllW:0] sum_preadded_bias;
    logic signed [45:0] offset_rd;
    logic signed [PreShrW-1:0] mul_no_shr;
    logic signed [46:0] mul_w_off_rd;
    logic signed [46:0] mul_shifted;
    logic signed [7:0] activated;
    logic signed [7:0] bias_ppl [4];
    logic [7:0] is_write_ppl;
    logic [7:0] last_in_layer_ppl;

    assign wait_wr = |last_in_layer_ppl;

    // Setting (most) parameters
    always_ff @(posedge clk) begin
        if (start) begin
            out_offset <= 0;
            multiplier <= 0;
            shift <= 0;
            act_min <= 0;
        end else if (running && !wait_wr) begin
            unique case (instruction.proc_inst)
                SET_OUT_OFFSET: out_offset <= oof_d.out_offset;
                SET_MUL: multiplier <= mul_d.mul;
                SET_SHR: begin
                    assert (shr_d.shift <= 38)
                        else $error("shift outside [0, 38] range");
                    shift <= shr_d.shift;
                end
                SET_MIN: act_min <= act_d.act_min;
                default: ;
            endcase
        end
    end
    // Memory read pointer control
    always_ff @(posedge clk) begin
        if (start)
            saved_rptr <= 0;
        else if (instruction.save_rptr)
            saved_rptr <= mem_rptr;
    end
    // Shifted offset and rounding
    // this is calculated at most 1 cycle after setting shift or offset,
    // but won't be needed until 3? more and is set only once per layer (eol blocks until write)
    always_ff @(posedge clk) begin
        offset_rd <= {{38{out_offset[7]}}, out_offset, 1'b1} << shift;
    end
    // Neuron finalizing pipeline
    assign sum_preadded_bias = (mac_sum + bias_ppl[3]); // Quartus requires this as a separate net, or the preadder won't be inferred, because inputs are resized to multiplier output
    always_ff @(posedge clk) begin
        if (start) begin
            is_write_ppl <= 0;
            last_in_layer_ppl <= 0;
            in_offset_neg <= net_config::input_quant_offset;
        end else begin
            is_write_ppl <= {is_write_ppl[6:0], instruction.proc_inst == STORE && !wait_wr};
            last_in_layer_ppl <= {last_in_layer_ppl[6:0],
                                (instruction.proc_inst == STORE && !wait_wr && st_d.last_in_layer)};
            if (last_in_layer_ppl[7]) begin
                in_offset_neg <= out_offset;
            end
        end
        for (int n = 3; n > 0; n--) bias_ppl[n] <= bias_ppl[n-1];
        bias_ppl[0] <= st_d.bias;
        mul_no_shr <= sum_preadded_bias * $signed({1'b0, multiplier});
        mul_w_off_rd <= mul_no_shr + offset_rd;
        mul_shifted <= $signed(mul_w_off_rd[46:1]) >>> shift;
`ifdef SIM_ONLY
        if (is_write_ppl[6]) $display("Pre-act %0d", mul_shifted);
`endif
        activated <= mul_shifted > 8'sd127 ?
                        8'sd127 : (
                            mul_shifted < act_min ?
                                act_min :
                                mul_shifted
                        );
    end

    sdp_mem #(
        .WRITE_WIDTH(8),
        .READ_WIDTH_MUL(27),
        .NUM_CHUNKS(NUM_CHUNKS)
    ) i_data_mem (
        .clk(clk),
        .rst(ext_mem_rst),
        .wdata(ext_mem_we ? (ext_mem_wdata + net_config::input_quant_offset) : $unsigned(activated)),
        .we(ext_mem_we | is_write_ppl[7]),
        .w_next_chunk((~ext_mem_we & last_in_layer_ppl[7]) | start),
        .rp_inc(instruction.mul_en && !wait_wr),
        .rp_load_val(saved_rptr),
        .rp_load(instruction.load_rptr),
        .rp(mem_rptr),
        .rdata(mem_data)
    );

    generate
        genvar i;
        for (i = 0; i < 27; ++i) begin : g_mac
            poor_mac #(
                .REGISTER_INPUTS(1),
                .O_WIDTH(SingleMacW),
                .PREADDER_SUB(1)
            ) i_mac (
                .clk(clk),
                .en(instruction.mul_en),
                .a(mem_data[i]),
                .b(instruction.weights[i]),
                .c(in_offset_neg),
                .acc(instruction.mul_acc),
                .result(mac_outs[i])
            );
        end
    endgenerate

    adder_tree3 #(
        .WIDTH_IN(SingleMacW),
        .WIDTH_OUT(AccAllW),
        .N(27),
        .PIPELINE(1)
    ) i_sum_tree (
        .clk(clk),
        .inputs(mac_outs),
        .sum(mac_sum)
    );

    typedef logic signed [7:0] t_10_num [10];
    t_10_num first_10_ok_order;
    always_comb for (int i = 0; i < 10; ++i) first_10_ok_order[i] = mem_data[i];
    max_idx10 #(
        .WIDTH(8)
    ) i_max_idx (
        .clk(clk),
        .reset(start),
        .inputs(first_10_ok_order),
        .start(!wait_wr && instruction.proc_inst == FIN && !done),
        .idx(max_idx_10),
        .done(done)
    );
endmodule
