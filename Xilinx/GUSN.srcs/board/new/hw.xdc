# Clock definition and pin
create_clock -name clk -period 10.000 [get_ports clk]
set_property -dict {PACKAGE_PIN N15 IOSTANDARD LVCMOS33} [get_ports clk]

# UART and reset button
set_property -dict {PACKAGE_PIN B16 IOSTANDARD LVCMOS33} [get_ports rx]
set_property -dict {PACKAGE_PIN A16 IOSTANDARD LVCMOS33} [get_ports tx]
set_property -dict {PACKAGE_PIN A8 IOSTANDARD LVCMOS25} [get_ports rst_btn]

# 7 segment display output
set_property -dict {PACKAGE_PIN B3 IOSTANDARD LVCMOS25} [get_ports {d0_anodes[3]}]
set_property -dict {PACKAGE_PIN C3 IOSTANDARD LVCMOS25} [get_ports {d0_anodes[2]}]
set_property -dict {PACKAGE_PIN H6 IOSTANDARD LVCMOS25} [get_ports {d0_anodes[1]}]
set_property -dict {PACKAGE_PIN G6 IOSTANDARD LVCMOS25} [get_ports {d0_anodes[0]}]
set_property -dict {PACKAGE_PIN B5 IOSTANDARD LVCMOS25} [get_ports {d0_segments[7]}]
set_property -dict {PACKAGE_PIN C4 IOSTANDARD LVCMOS25} [get_ports {d0_segments[6]}]
set_property -dict {PACKAGE_PIN D6 IOSTANDARD LVCMOS25} [get_ports {d0_segments[5]}]
set_property -dict {PACKAGE_PIN D7 IOSTANDARD LVCMOS25} [get_ports {d0_segments[4]}]
set_property -dict {PACKAGE_PIN C5 IOSTANDARD LVCMOS25} [get_ports {d0_segments[3]}]
set_property -dict {PACKAGE_PIN D5 IOSTANDARD LVCMOS25} [get_ports {d0_segments[2]}]
set_property -dict {PACKAGE_PIN B4 IOSTANDARD LVCMOS25} [get_ports {d0_segments[1]}]
set_property -dict {PACKAGE_PIN E6 IOSTANDARD LVCMOS25} [get_ports {d0_segments[0]}]

# Ignore async paths in timing analysis
set_false_path -to [get_ports rx]
set_false_path -to [get_ports tx]
set_false_path -to [get_ports rst_btn]
set_false_path -to [get_ports {d0_anodes*}]
set_false_path -to [get_ports {d0_segments*}]

# Other config
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
