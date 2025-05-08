module net_test();
	bit clk;

	initial begin
		clk = 0;
		forever #5 clk = ~clk;
	end

	bit [7:0] image [28*28];
	initial $readmemh("image.dat", image);

	// Inputs
	bit start;
	bit prepare_write;
	bit we;
	bit [7:0] in_byte;
	// Outputs
	logic done;
	logic [3:0] result;

	net_proc i_net(
        .clk(clk),
        .start(start),
        .done(done),
        .max_idx_10(result),
        .ext_mem_rst(prepare_write),
        .ext_mem_we(we),
        .ext_mem_wdata(in_byte)
    );

	initial begin
		start = 0;
		prepare_write = 1;
		we = 0;
		in_byte = 0;
		@(posedge clk) #1;
		prepare_write = 0;
		for (int i = 0; i < 28*28; ++i) begin
			in_byte = image[i];
			we = 1;
			@(posedge clk) #1;
			$display("Wrote byte %d: 0x%02h", i, in_byte);
			// In reality there should be (probably with repeat):
			// we = 0;
			// @(posedge clk) #1;
		end
		we = 0;
		start = 1;
		@(posedge clk) #1;
		start = 0;
		while (!done) @(posedge clk);
		$display("Result is %0d", result);
		repeat (5) @(posedge clk);
		$finish;
	end
endmodule
