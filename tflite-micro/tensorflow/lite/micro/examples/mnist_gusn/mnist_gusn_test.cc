/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <math.h>
#include <inttypes.h>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/examples/mnist_gusn/models/mnist_gusn_int8_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
// #include "tensorflow/lite/micro/testing/micro_test.h" // TFLITE_CHECK_EQ macro

const float TEST_IMG[784] = {
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,  84., 185., 159., 151., 60.,  36.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0., 222., 254., 254., 254.,254., 241., 198., 198., 198., 198., 198., 198., 198., 198.,170.,  52.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,  67., 114.,  72., 114.,163., 227., 254., 225., 254., 254., 254., 250., 229., 254.,254., 140.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,  17.,  66.,  14.,  67.,  67.,  67.,  59.,  21., 236.,254., 106.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  83., 253.,209.,  18.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,  22., 233., 255., 83.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0., 129., 254., 238., 44.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,  59., 249., 254.,  62.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0., 133., 254., 187.,   5.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   9., 205., 248.,  58.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0., 126., 254., 182.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,  75., 251., 240.,  57.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,  19., 221., 254., 166.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   3., 203., 254., 219.,  35.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,  38., 254., 254.,  77.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,  31., 224., 254., 115.,   1.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0., 133., 254., 254.,  52.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 61., 242., 254., 254.,  52.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,121., 254., 254., 219.,  40.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,121., 254., 207.,  18.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.};

size_t argmax(float *arr, size_t n) {
  float maxval = -INFINITY;
  size_t idx = 0;
  for (size_t i = 0; i < n; ++i) {
    if (arr[i] > maxval) {
      maxval = arr[i];
      idx = i;
    }
  }
  return idx;
}

namespace {
using MnistgusnOpResolver = tflite::MicroMutableOpResolver<4>;

TfLiteStatus RegisterOps(MnistgusnOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());
  return kTfLiteOk;
}
}  // namespace

TfLiteStatus ProfileMemoryAndLatency() {
  tflite::MicroProfiler profiler;
  MnistgusnOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 30000;
  uint8_t tensor_arena[kTensorArenaSize];
  constexpr int kNumResourceVariables = 24;

  tflite::RecordingMicroAllocator* allocator(
      tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));
  tflite::RecordingMicroInterpreter interpreter(
      tflite::GetModel(g_mnist_gusn_int8_model_data), op_resolver, allocator,
      tflite::MicroResourceVariables::Create(allocator, kNumResourceVariables),
      &profiler);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
  TFLITE_CHECK_EQ(interpreter.inputs_size(), 1);
  for (size_t i = 0; i < 784; ++i)
    interpreter.input(0)->data.f[1] = 1.f;
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  MicroPrintf("");  // Print an empty new line
  profiler.LogTicksPerTagCsv();

  MicroPrintf("");  // Print an empty new line
  interpreter.GetMicroAllocator().PrintAllocations();
  return kTfLiteOk;
}

TfLiteStatus LoadQuantModelAndPerformInference() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      ::tflite::GetModel(g_mnist_gusn_int8_model_data);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  MnistgusnOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 30000;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  TfLiteTensor* input = interpreter.input(0);
  TFLITE_CHECK_NE(input, nullptr);

  TfLiteTensor* output = interpreter.output(0);
  TFLITE_CHECK_NE(output, nullptr);

  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;

  TFLITE_CHECK_EQ(3, input->dims->size);
  TFLITE_CHECK_EQ(1, input->dims->data[0]);
  TFLITE_CHECK_EQ(28, input->dims->data[1]);
  TFLITE_CHECK_EQ(28, input->dims->data[2]);
  TFLITE_CHECK_EQ(kTfLiteFloat32, input->type);
  TFLITE_CHECK_EQ(784 * sizeof(float), input->bytes);
  static_assert (sizeof(float) == 4);
  TFLITE_CHECK_EQ(2, output->dims->size);
  TFLITE_CHECK_EQ(1, output->dims->data[0]);
  TFLITE_CHECK_EQ(10, output->dims->data[1]);
  TFLITE_CHECK_EQ(kTfLiteFloat32, output->type);

  for (size_t i = 0; i < 784; ++i) {
    input->data.f[i] = TEST_IMG[i];
  }

  TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  float results[10];
  for (size_t i = 0; i < 10; ++i) {
    results[i] = (output->data.f[i] - output_zero_point) * output_scale;
    MicroPrintf("%f ", static_cast<double>(results[i]));
  }
  size_t predicted = argmax(results, 10);
  MicroPrintf("\nThe result is %" PRIu32 "\n", static_cast<uint32_t>(predicted));
  TFLITE_CHECK_EQ(7U, predicted);

  return kTfLiteOk;
}

int main(int argc, char* argv[]) {
  tflite::InitializeTarget();
  TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency());
  TF_LITE_ENSURE_STATUS(LoadQuantModelAndPerformInference());
  MicroPrintf("~~~ALL TESTS PASSED~~~\n");
  return kTfLiteOk;
}
