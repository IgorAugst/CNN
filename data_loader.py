import numpy as np
from keras.datasets import mnist


# Função para realizar o one hot encode, converte a categoria para um vetor binário
def one_hot_encode(data):
    encoded_data = np.zeros((data.size, data.max() + 1))
    encoded_data[np.arange(data.size), data] = 1
    return encoded_data


# carrega os dados do MNIST
def load_data():
    (input_data, label_data), (input_test, label_test) = mnist.load_data()

    input_data = input_data.reshape(input_data.shape[0], 28, 28, 1)  # redimensiona o array para o formato usado no tensorflow
    input_test = input_test.reshape(input_test.shape[0], 28, 28, 1)

    input_data = input_data / 255  # normaliza os valores para o intervalo [0, 1], melhor treinamento
    input_test = input_test / 255

    return (input_data, label_data), (input_test, label_test)


def load_data_binary_classificator(desired_class: int):
    (input_data, label_data), (input_test, label_test) = load_data()

    label_data = np.where(label_data == desired_class, 0, 1)  # converte a categoria desejada para 0 e as demais para 1
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
