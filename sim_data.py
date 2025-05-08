import tensorflow as tf
import numpy as np
import numpy.typing as npt
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

image: np.ndarray[tuple[int, int], np.dtypes.UInt8DType] = x_train[0]
assert len(image.shape) == 2
assert image.shape == (28, 28)

flat_img = image.flatten()

with open("./hdl/sim/image.dat", "wt") as hex_file:
	for byte in flat_img:
		hex_file.write(f"{byte:0>2X}\n")
