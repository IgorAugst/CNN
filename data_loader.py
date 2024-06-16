import numpy as np
from keras.datasets import mnist


def one_hot_encode(data):
    encoded_data = np.zeros((data.size, data.max() + 1))
    encoded_data[np.arange(data.size), data] = 1
    return encoded_data


def load_data():
    (input_data, label_data), (input_test, label_test) = mnist.load_data()

    input_data = input_data.reshape(input_data.shape[0], 28, 28, 1)
    input_test = input_test.reshape(input_test.shape[0], 28, 28, 1)

    input_data = input_data / 255
    input_test = input_test / 255

    return (input_data, label_data), (input_test, label_test)


def load_data_binary_classificator(desired_class: int):
    (input_data, label_data), (input_test, label_test) = load_data()

    label_data = np.where(label_data == desired_class, 0, 1)
    label_test = np.where(label_test == desired_class, 0, 1)

    return (input_data, label_data), (input_test, label_test)


def load_and_encode_data(binary_classificator):
    (input_data, label_data), (input_test, label_test) = load_data()
    (binary_input_data, binary_label_data), (binary_input_test, binary_label_test) = load_data_binary_classificator(binary_classificator)

    label_data = one_hot_encode(label_data)
    label_test = one_hot_encode(label_test)
    binary_label_data = one_hot_encode(binary_label_data)
    binary_label_test = one_hot_encode(binary_label_test)

    return (input_data, label_data, input_test, label_test), (binary_input_data, binary_label_data, binary_input_test, binary_label_test)
