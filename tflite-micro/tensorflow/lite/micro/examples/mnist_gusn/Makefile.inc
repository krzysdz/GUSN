MNIST_GUSN_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_gusn/mnist_gusn_test.cc

MNIST_GUSN_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_gusn/mnist_gusn_test.cc

MNIST_GUSN_HDRS :=

MNIST_GUSN_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_gusn/models/mnist_gusn_int8.tflite

MNIST_GUSN_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_gusn/models/mnist_gusn_int8_model_data.cc

MNIST_GUSN_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_gusn/models/mnist_gusn_int8_model_data.h

# Tests loading and running the sine model.
$(eval $(call microlite_test,mnist_gusn_test,\
$(MNIST_GUSN_TEST_SRCS),,$(MNIST_GUSN_GENERATOR_INPUTS)))

# Builds a standalone binary.
$(eval $(call microlite_test,mnist_gusn,\
$(MNIST_GUSN_SRCS),,$(MNIST_GUSN_GENERATOR_INPUTS)))

# Add sources and headers generated from $(MNIST_GUSN_GENERATOR_INPUTS).
MNIST_GUSN_SRCS += $(MNIST_GUSN_GENERATED_SRCS)
MNIST_GUSN_HDRS += $(MNIST_GUSN_GENERATED_HDRS)

# Try disabling optimisation
CORE_OPTIMIZATION_LEVEL := -O0
KERNEL_OPTIMIZATION_LEVEL := -O0

list_mnist_gusn_example_sources:
	@echo $(MNIST_GUSN_SRCS)

list_mnist_gusn_example_headers:
	@echo $(MNIST_GUSN_HDRS)
