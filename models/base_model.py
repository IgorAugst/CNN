import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class BaseModel:

	def compile(self, optimizer='adam', loss='mean_squared_error'):
		self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

	def early_stopping(self, patience=5):
		return tf.keras.callbacks.EarlyStopping(patience=patience)

	def build_model(self):
		raise NotImplementedError('Método build_model não implementado')

	def __init__(self, input_shape=(10, 12, 1), output_shape=26):
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.model = self.build_model()
		self.training_history = None

	def fit(self, input_data, target_data, epochs, val_proportion=0.3):
		self.training_history = self.model.fit(input_data, target_data, epochs=epochs, validation_split=val_proportion,
											   callbacks=[self.early_stopping()])

	def evaluate(self, test_data, test_target, threshold=None):
		predictions = self.model.predict(test_data, verbose=False)
		accuracy = 0

		for i, pred in enumerate(predictions):
			if threshold is None:
				accuracy += (np.argmax(pred) == np.argmax(test_target[i]))
			else:
				arr = np.where(pred > threshold, 1, 0)
				if sum(arr) == 1:
					accuracy += np.argmax(arr) == np.argmax(test_target[i])

		return accuracy / len(predictions)

	def plot_loss(self):
		training_loss = self.training_history.history['loss']
		validation_loss = self.training_history.history['val_loss']

		plt.plot(training_loss, label='Treinamento')
		plt.plot(validation_loss, label='Validação')
		plt.xlabel('Época')
		plt.ylabel('Erro')
		plt.legend()
		plt.show()
