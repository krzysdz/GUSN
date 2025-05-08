import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
input_data = tf.constant(x_test[0], shape=[1, 28, 28], dtype=tf.float32)

interpreter = tf.lite.Interpreter(model_path="model_notnorm_nosoft", experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()
interpreter.set_tensor(interpreter.get_input_details()[0], input_data)
interpreter.invoke()
