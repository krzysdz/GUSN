package net_config;
    // Quantization offset of NN inputs. Applied to incoming data and used in first layer.
    localparam logic signed [7:0] input_quant_offset = 8'b10000000;
endpackage
