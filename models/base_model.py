import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class BaseModel:
	def save_model(self, custom_data: dict):
		os.makedirs('./modelos', exist_ok=True)

		if not os.path.exists(f'./modelos/models.json'):
			with open(f'./modelos/models.json', 'w') as f:
				json.dump([], f)

		with open(f'./modelos/models.json', 'r') as f:
			models = json.load(f)

		model_name = self.__class__.__name__ + '_' + str(len(models))

		os.makedirs(f'./modelos/{model_name}', exist_ok=True)

		self.model.save(f'./modelos/{model_name}/modelo.keras')
		self.plot_loss(save=True, path=f'./modelos/{model_name}/loss.png')

		model_info = {
			'model_name': model_name,
			'history': self.training_history.history,
			'custom_data': custom_data,
		}

		models.append(model_info)

		with open(f'./modelos/models.json', 'w') as f:
			json.dump(models, f)

	def load_model(self, model_name):
		self.model = tf.keras.models.load_model(f'./modelos/{model_name}/modelo.keras')

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

		confusion_matrix = np.zeros((self.output_shape, self.output_shape))

		for i, pred in enumerate(predictions):
			if threshold is None:
				accuracy += (np.argmax(pred) == np.argmax(test_target[i]))
				confusion_matrix[np.argmax(test_target[i])][np.argmax(pred)] += 1
			else:
				arr = np.where(pred > threshold, 1, 0)
				if sum(arr) == 1:
					accuracy += np.argmax(arr) == np.argmax(test_target[i])
					confusion_matrix[np.argmax(test_target[i])][np.argmax(arr)] += 1

		return {'accuracy': accuracy / len(test_data), 'confusion_matrix': confusion_matrix}

	def plot_loss(self, save=False, path=None):
		training_loss = self.training_history.history['loss']
		validation_loss = self.training_history.history['val_loss']

		plt.plot(training_loss, label='Treinamento')
		plt.plot(validation_loss, label='Validação')
		plt.xlabel('Época')
		plt.ylabel('Erro')
		plt.legend()

		if save:
			plt.savefig(path)
			plt.close()
		else:
			plt.show()

	@staticmethod
	def plot_matrix(matrix):
		plt.matshow(matrix, cmap='viridis')

		for i in range(matrix.shape[0]):
			for j in range(matrix.shape[1]):
				plt.text(j, i, str(matrix[i, j]), ha='center', va='center', color='red')

		plt.xlabel('Predicted')
		plt.ylabel('Actual')

		plt.show()