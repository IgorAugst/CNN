import tensorflow as tf
from models.base_model import BaseModel


class SimpleMnistModel(BaseModel):
	def early_stopping(self, patience=10):
		return tf.keras.callbacks.EarlyStopping(patience=patience)

	def build_model(self):
		model = tf.keras.models.Sequential([
			tf.keras.layers.Input(shape=self.input_shape),
			tf.keras.layers.Conv2D(32, (3, 3), activation='leaky_relu'),
			tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(self.output_shape, activation='softmax')
		])

		return model
