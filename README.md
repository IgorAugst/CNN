# CNN - Classificação de digitos

## Integrantes
* BRUNO LEITE DE ANDRADE - 11369642
* FRANCISCO OLIVEIRA GOMES JUNIOR - 12683190
* GUILHERME DIAS JIMENES - 11911021
* IGOR AUGUSTO DOS SANTOS - 11796851
* LAURA PAIVA DE SIQUEIRA – 11207515

## Observações
* O arquivo `main.py` é responsável por treinar os modelos e armazenar os resultados na pasta `./modelos`.
* O arquivo `base_model.py` contém a implementação abstrata da rede neural
* O arquivo `hog_mnist.py` contém a implementação da rede neural com HOG
* O arquivo `mnist_model.py` contém a implementação da rede neural com CNN
* O arquivo `data_loader.py` é responsável por carregar os dados de treino e teste.
* O código foi desenvolvido em Python 3.12

## Como executar
1. Clone o repositório
2. Crie o ambiente virtual com o comando:
```bash
python3 -m venv .venv
```
3. Ative o ambiente virtual com o comando:
```bash
source .venv/bin/activate
```
4. Instale as dependências com o comando:
```bash
pip install -r requirements.txt
```
5. Execute o arquivo `main.py` com o comando:
```bash
python main.py
```

## Utilização
```
[?] Selecione o Modelo desejado. Use as setas para cima e para baixo para navegar, e a barra de espaço para selecionar:
 > [X] Modelo usando HOG
   [ ] Modelo usando CNN
```

1. Os modelos serão treinados conforme a seleção do usuário e os resultados serão armazenados na pasta `./modelos`.