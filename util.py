import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from data_loader import load_and_encode_data
from keras.models import Model

(input_data, label_data, input_test, label_test), (
	binary_input_data, binary_label_data, binary_input_test, binary_label_test) = load_and_encode_data(
	9)

model = tf.keras.models.load_model('./modelos_apresentacao/mnist_model_multiclasse/modelo.keras')
feature_map_model = Model(inputs=model.inputs, outputs=model.layers[2].output)  # cria o modelo com a camada de extração de características
feature_maps = feature_map_model.predict(input_data[0:1])

# plota os mapas de características
fig, axs = plt.subplots(4, 8, figsize=(15, 8))

for i, ax in enumerate(axs.flat):
	if i < feature_maps.shape[-1]:
		# exibe o mapa de características
		ax.imshow(feature_maps[0, :, :, i], cmap='gray')
		ax.axis('off')

plt.show()