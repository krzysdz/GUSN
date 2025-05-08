create_clock -period 10.000 -name clk -waveform {0.000 5.000} [get_ports clk]

create_clock -period 5.000 -name clk -waveform {0.000 2.500} -add [get_ports clk]
create_clock -period 4.000 -name clk -waveform {0.000 2.000} -add [get_ports clk]

