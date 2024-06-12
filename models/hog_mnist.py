import tensorflow as tf
import numpy as np
import cv2
from models.base_model import BaseModel


class HogMnist(BaseModel):
	def __init__(self, input_shape=(324,), output_shape=10):
		super().__init__(input_shape, output_shape)

	@staticmethod
	def extract_hog_features(data):
		print('Extraindo características HOG')
		# defininco tamanho do bloco, imagem e célula
		win_size = (32, 32)
		block_size = (16, 16)
		block_stride = (8, 8)
		cell_size = (8, 8)
		nbins = 9

		hog_features = []  # array para armazenar as características HOG

		for img in data:
			img = cv2.resize(img, (32, 32))  # redimensiona a imagem para 32x32
			img = (img * 255).astype('uint8')  # converte a imagem para uint8, de 0 a 255
			hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)  # cria o descritor HOG
			hog_features.append(hog.compute(img).flatten())  # extrai as características HOG e adiciona ao array

		return np.array(hog_features)

	def early_stopping(self, patience=10):
		return tf.keras.callbacks.EarlyStopping(patience=patience)

	def build_model(self):
		model = tf.keras.models.Sequential([
			tf.keras.layers.Input(shape=self.input_shape),
			tf.keras.layers.Dense(64, activation='leaky_relu'),  # camada MLP
			tf.keras.layers.Dense(self.output_shape, activation='softmax')  # camada de saida com softmax
		])

		return model

	def fit(self, input_data, target_data, epochs, val_proportion=0.3):
		hog_features = self.extract_hog_features(input_data)
		super().fit(hog_features, target_data, epochs, val_proportion)

	def evaluate(self, test_data, test_target, threshold=None):
		hog_features = self.extract_hog_features(test_data)
		return super().evaluate(hog_features, test_target, threshold)
