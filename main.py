import numpy as np
from models.character_model import CharacterModel
from keras.datasets import mnist
from models.mnist_model import MnistModel
from models.simple_mnist_model import SimpleMnistModel


def one_hot_encode(data):
	encoded_data = np.zeros((data.size, data.max() + 1))
	encoded_data[np.arange(data.size), data] = 1
	return encoded_data


(input_data, label_data), (input_test, label_test) = mnist.load_data()

input_data = input_data.reshape(input_data.shape[0], 28, 28, 1)
input_test = input_test.reshape(input_test.shape[0], 28, 28, 1)

input_data = input_data / 255
input_test = input_test / 255

label_data = one_hot_encode(label_data)
label_test = one_hot_encode(label_test)

model1 = SimpleMnistModel(input_shape=(28, 28, 1), output_shape=10)
model1.compile()
model1.fit(input_data, label_data, epochs=5, val_proportion=0.3)
accuracy = model1.evaluate(input_test, label_test)
model1.plot_loss()
model1.save_model({'accuracy': accuracy})
