from models import models
from data_loader import load_data, one_hot_encode, load_data_binary_classificator, load_and_encode_data
import inquirer


# função para treinar, exibir resultados e salvar o modelo
def train_and_evaluate_model(model, input_data, label_data, input_test, label_test):
	model.compile()  # compila o modelo
	model.fit(input_data, label_data, epochs=10, val_proportion=0.3, batch_size=100)  # realiza o treinamento
	accuracy = model.evaluate(input_test, label_test)
	model.save_model({'accuracy': accuracy['accuracy']}, confusion_matrix=accuracy['confusion_matrix'])
	model.plot_loss()
	model.plot_matrix(accuracy['confusion_matrix'])
	model.summary()  # exibe o resumo do modelo


question = [
	inquirer.Checkbox(
		'modelo',
		message='Selecione o Modelo desejado. Use as setas para cima e para baixo para navegar, e a barra de espaço '
				'para selecionar',
		choices=[
			('Modelo usando HOG', 'hog_mnist'),
			('Modelo usando CNN', 'mnist_model'),
		],
		default=['hog_mnist']
	)
]

answers = inquirer.prompt(question)

for model_name in answers['modelo']:
	(input_data, label_data, input_test, label_test), (
		binary_input_data, binary_label_data, binary_input_test, binary_label_test) = load_and_encode_data(5)  # carrega os dados multiclasse e binário

	model = models[model_name]()
	model_binary = models[model_name](output_shape=2)  # o modelo binário possui apenas dois neurônios de saída

	train_and_evaluate_model(model, input_data, label_data, input_test, label_test)
	train_and_evaluate_model(model_binary, binary_input_data, binary_label_data, binary_input_test, binary_label_test)
