import numpy as np
from models.character_model import CharacterModel
from models.mnist_model import MnistModel
from models.simple_mnist_model import SimpleMnistModel
from data_loader import load_data, one_hot_encode, load_data_binary_classificator
from models.hog_mnist import HogMnist

(input_data, label_data), (input_test, label_test) = load_data_binary_classificator(0)

label_data = one_hot_encode(label_data)
label_test = one_hot_encode(label_test)

model1 = HogMnist(output_shape=2)
model1.compile()
model1.fit(input_data, label_data, epochs=2, val_proportion=0.3)
accuracy = model1.evaluate(input_test, label_test)
model1.plot_loss()
model1.save_model({'accuracy': accuracy['accuracy']})
model1.plot_matrix(accuracy['confusion_matrix'])
model1.summary()
