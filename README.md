# FPGA-based digit recognition

This repository, among many unnecessary files (some notes, old models, attempts at understanding TFLite models and processing), contains code and data used for model training, analysis, compilation and processing on FPGA.

More important files and directories:

- [`m.py`](./m.py) - script responsible for creation and training of the model,
- [`model_notnorm_nosoft`](./model_notnorm_nosoft) - quantized model created by the above script,
- [`model_notnorm_nosoft.tflite`](./model_notnorm_nosoft.tflite) - copy of `model_notnorm_nosoft` that is used by many other tools,
- [`tflite-micro/`](./tflite-micro/) - a copy of [tensorflow/tflite-micro](https://github.com/tensorflow/tflite-micro) with added [mnist\_gusn](./tflite-micro/tensorflow/lite/micro/examples/mnist_gusn/) project that was very useful for understanding TFLM operations and debugging own implementation. This project can be built using `make -f tensorflow/lite/micro/tools/make/Makefile BUILD_TYPE=debug mnist_gusn_bin` (I'd suggest adding `-j` to parallelise the build) from [`tflite-micro/`](./tflite-micro/).
- [`net-preprocessor/`](./net-preprocessor/) - _"compiler"_ that processes a TFLM model (with many assumptions) from a hardcoded path and outputs a configuration file [`hdl/net_config.sv`](./hdl/net_config.sv), program memory [initialisation file](./hdl/prog.dat) and a [text form](./hdl/prog.txt) of the program for debugging purposes,
- [`hdl/`](./hdl/) - HDL sources of the project, including simulation and top-level modules for laboratory DE1-SoC board and another one that I use personally. Some files (`fp_*` and `q_tensor.sv`) are completely unused. The important files (and modules) are:
  - [`sim/net_test.sv`](./hdl/sim/net_test.sv) - testbench used in simulation
  - [`boards/`](./hdl/boards/) - top level modules for specific boards
  - [`top_qi8.sv`](./hdl/top_qi8.sv) - module that glues UART with the main processing unit
  - [`uart/`](./hdl/uart/) - UART rx/tx modules copied from MKwSC laboratories that have terrible code quality and inconsistent behaviour
  - [**`net_proc.sv`**](./hdl/net_proc.sv) - the main processing unit. Contains program memory, data memory (instance of `sdp_mem`), 27 DSP-based MAC units (instances of `poor_mac`), ternary adder tree that accumulates results from MACs (`adder_tree3`), some control logic, final neuron output processing (bias, scaling and activation) pipeline and hard-wired module that produces final classification result of 10 possible classes (`max_idx10`).
  - [`sdp_mem.sv`](./hdl/sdp_mem.sv) - a simple dual port memory that writes byte by byte and reads in larger (27 bytes by default) chunks
  - [`poor_mac.sv`](./hdl/poor_mac.sv) - multiply-accumulate (MAC) module with preadder (configurable +/-) and optionally registered inputs (used only in this configuration, because of timing problems) that can map well to DSP blocks on FPGAs from different manufacturers
  - [`adder_tree3.sv`](./hdl/adder_tree3.sv) - ternary adder tree (addition of 3 numbers on 6 input LUT architectures uses the same amount of resources as adding just 2) with optional pipeline registers after each stage (necessary unless negative slack of multiple clock cycles is desired or the clock is very slow). This module uses a very funny loop to calculate parameters, because [Quartus has some problems](https://community.intel.com/t5/Intel-Quartus-Prime-Software/SystemVerilog-loops-in-functions-are-completely-broken/m-p/1682861)
  - [`max_idx10.sv`](./hdl/max_idx10.sv) - find index of maximum value in an input array of 10 numbers, using 3 time-multiplexed comparators (`cmp_ad`)
  - [`cmp_ad.sv`](./hdl/cmp_ad.sv) - outputs greater of 2 inputs with its associated data
  - [`datatypes.svh`](./hdl/datatypes.svh) - some type definitions used mostly for instruction encoding. Unfortunately Quartus supports packed unions since version 21 and Standard and Lite editions [do not receive feature updates since 18](community.intel.com/t5/Intel-Quartus-Prime-Software/SystemVerilog-loops-in-functions-are-completely-broken/m-p/1684523#M86149), so some workarounds were necessary.
- [`Altera/`](./Altera/) - contains a Quartus [project](./Altera/GUSN.qpf) targetting DE1-SoC board including constraints and pin assignments
- [`Xilinx/`](./Xilinx/) - contains a Vivado [project](./Xilinx/GUSN.xpr) targetting XC7S50-based board, some unnecessary files **and simulation sources ready to run in built-in xsim simulator**
- [`drawing_gui.py`](./drawing_gui.py) - module with simple GUI for drawing numbers. Should not be used directly.
- [`fpga_predict.py`](./fpga_predict.py) - script that can communicate with FPGA running the project to recognize handwritten digits. Can be used through CLI or show an optional GUI.

## Running the project (hardware)

The project is almost ready to run and its usage should not be complicated.

### (Optional) 0.a) Train own model

> [!NOTE]
> This step requires TensorFlow and TensorFlow Model Optimization to be installed.\
> `pip install tensorflow tensorflow_model_optimization`

Run (or modify and run) [`m.py`](./m.py) to train the model. Rename the output file to `model_notnorm_nosoft.tflite`, because that's the name expected in the next steps.

> [!IMPORTANT]
> Using a model with different number of neurons in layers may require increasing the data memory size. Failure to do so will result in memory corruption and wrong results. The parameter which controls that is `NUM_CHUNKS`.

### (Optional) 0.b) Convert the model to instruction data

> [!IMPORTANT]
> This step requires Rust to be installed to first compile the converter. If you don't have Rust installed, skip to the next step and use the default pre-generated program.

From the `net-preprocessor/` directory execute `cargo run` to compile and run the program that will generate `nat_config.sv`, `prog.dat` and `prog.txt` from the model:

```console
user@PC:~/GUSN$ cd net-preprocessor
user@PC:~/GUSN/net-preprocessor$ cargo run
   Compiling net-preprocessor v0.1.0 (/home/user/GUSN/net-preprocessor)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.57s
     Running `target/debug/net-preprocessor`
Remember that it only works if layers are purely sequential and output from each of them is immediately consumed by the next one.
Processing subgraph main
Operation QUANTIZE, scale 1, zero point -128
Operation RESHAPE - skipping
Operation FULLY_CONNECTED, activation function RELU
TensorFlow's quantized multiplier: 1714578666 with shift -11, adjusted to [-38, 0] and u15 26162 (15 bits) >> (25 + 1)
Q: 3.898500531249738e-4, A: 3.898441791534424e-4, diff 5.873971531400457e-9
Operation FULLY_CONNECTED, activation function NONE
TensorFlow's quantized multiplier: 1391761579 with shift -9, adjusted to [-38, 0] and u15 21236 (15 bits) >> (23 + 1)
Q: 1.2657997822316247e-3, A: 1.2657642364501953e-3, diff 3.5545781429391354e-8
Operation DEQUANTIZE - skipping
Accumulators in layer will be in range [-2786386, 1961251] and require 23 bits as signed integers
Accumulators in layer will be in range [-376379, 1171986] and require 22 bits as signed integers
user@PC:~/GUSN/net-preprocessor$ # DONE
```

### 1. Synthesize the project for FPGA

There are projects for Quartus (in the [`Altera/`](./Altera/) directory) and Vivado (in [`Xilinx/`](./Xilinx/)). Modify those to match your hardware, run synthesis and program the device.

> [!TIP]
> If elaboration in Vivado fails, because the memory is too big, execute the [`fix_mem_size.tcl`](./Xilinx/fix_mem_size.tcl) script.

### 2. Perform digit recognition

> [!NOTE]
> This step requires `pillow` and `pyserial`. Both can be installed through `requirements.io.txt` - `pip install -r requirements.io.txt`.

When the board is working and connected over UART to the computer, run `python fpga_predict.py PORT --gui` to connect over port `PORT` (replace it with appropriate `COMx` or `/dev/ttyUSBx`) and launch a GUI. When _Send_ is clicked, the image is downscaled to 28x28 px and sent over UART. Once the board responds, the answer is shown at the bottom in GUI (and on 7-segment display of the board if available).

## Running simulation

For now the simulation has only been tested in xsim (Vivado's built-in simulator). The [Vivado project](./Xilinx/GUSN.xpr) has everything already configured, so just launching "behavioral simulation" should be enough. The optional "0." steps from [Running the project](#running-the-project-hardware) section can be performed to use a different net than the one provided in this repository.
