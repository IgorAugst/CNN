import numpy as np
from models.character_model import CharacterModel
from models.mnist_model import MnistModel
from models.simple_mnist_model import SimpleMnistModel
from data_loader import load_data, one_hot_encode, load_data_binary_classificator
from models.hog_mnist import HogMnistModel

(input_data, label_data), (input_test, label_test) = load_data_binary_classificator(5)

label_data = one_hot_encode(label_data)
label_test = one_hot_encode(label_test)

model1 = HogMnistModel(output_shape=2)
model1.compile()
model1.fit(input_data, label_data, epochs=5, val_proportion=0.3, batch_size=100)
accuracy = model1.evaluate(input_test, label_test)
model1.save_model({'accuracy': accuracy['accuracy']}, confusion_matrix=accuracy['confusion_matrix'])
model1.plot_loss()
model1.plot_matrix(accuracy['confusion_matrix'])
model1.summary()
