# Clock and PLL clock
create_clock -name clk -period 20 [get_ports clk]
derive_pll_clocks
derive_clock_uncertainty

# Ignore async paths in timing analysis
set_false_path -from [get_ports rx]
set_false_path -from [get_ports rst_btn]
set_false_path -to [get_ports tx]
set_false_path -to [get_ports {segments*}]
