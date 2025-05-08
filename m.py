import tensorflow as tf
import tensorflow_model_optimization as tfmot
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  # tf.keras.layers.Dense(10, activation='softmax')
  tf.keras.layers.Dense(10)
])

q_aware_model = tfmot.quantization.keras.quantize_model(model)

# q_aware_model.compile(optimizer='adam',
#   loss='sparse_categorical_crossentropy',
#   metrics=['accuracy'])
q_aware_model.compile(
  optimizer=tf.keras.optimizers.Adam(0.001),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

q_aware_model.fit(x_train, y_train, epochs=6)
q_aware_model.evaluate(x_test, y_test)
q_aware_model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8  # or tf.int8
# converter.inference_output_type = tf.uint8  # or tf.int8

quantized_tflite_model: bytes = converter.convert()
with open("model_notnorm_nosoft", "wb") as out:
	out.write(quantized_tflite_model)
