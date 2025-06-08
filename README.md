# Reconhecimento do Alfabeto em Libras (ASL) com MediaPipe e Scikit-learn 🤘

Este projeto utiliza visão computacional e machine learning para reconhecer as letras do alfabeto da Língua de Sinais Americana (ASL) em tempo real, através de uma webcam.

> **Status do Projeto:** 🚧 Em Desenvolvimento (Estágios Iniciais) 🚧
>
> Este projeto é uma prova de conceito inicial e demonstra um fluxo de trabalho robusto. Há muitas oportunidades para melhorias e expansões futuras!

---

### Demonstração

*(Sugestão: Grave um GIF do programa funcionando e coloque aqui para um README incrível!)*

![Demonstração do Projeto](https://github.com/SamuelMauli/Dijkstra_Largura_Profundidade/blob/main/asl.gif) 
---

## Índice

* [Sobre o Projeto](#sobre-o-projeto-)
* [Como Funciona? O Fluxo de Trabalho](#como-funciona-o-fluxo-de-trabalho-)
* [Tecnologias Utilizadas](#tecnologias-utilizadas-)
* [Como Executar o Projeto](#como-executar-o-projeto-)
* [Estrutura de Pastas](#estrutura-de-pastas-)
* [Próximos Passos](#próximos-passos--melhorias-futuras-)

---

## Sobre o Projeto 🦾

O objetivo principal é criar um modelo de Machine Learning capaz de identificar as 26 letras do alfabeto em ASL a partir de uma imagem ou de um vídeo ao vivo. O projeto não se baseia na imagem bruta, mas sim em uma representação estrutural da mão, tornando o modelo mais resiliente a diferentes fundos, iluminações e tons de pele.

## Como Funciona? O Fluxo de Trabalho ⚙️

O script segue um pipeline de Machine Learning profissional e bem definido:

#### 1. Extração de Características com MediaPipe

Em vez de usar pixels, o projeto utiliza a solução **MediaPipe Hands** do Google para extrair 21 pontos-chave (landmarks) da mão em cada imagem.

* **Normalização:** Para que o modelo seja robusto à posição e ao tamanho da mão na câmera, todos os 21 pontos são normalizados em relação ao pulso (landmark 0). Isso significa que o modelo aprende a forma do sinal, e não onde a mão está na tela.

#### 2. Modelo de Machine Learning

* Foi utilizado um `RandomForestClassifier` do Scikit-learn, um modelo de ensemble poderoso que combina múltiplas árvores de decisão para obter uma predição mais precisa e estável.

#### 3. Validação Robusta

* Para garantir que o modelo tem um bom desempenho e não está "viciado" nos dados de treino, utilizamos **Validação Cruzada Estratificada K-Fold**. Isso divide os dados em 5 partes (folds), treinando e testando o modelo 5 vezes para garantir que a acurácia reportada seja confiável.

#### 4. Análise de Performance

* Ao final da validação, o script gera um **Relatório de Classificação** (com precisão, recall, F1-score) e uma **Matriz de Confusão** visual. A matriz ajuda a entender quais letras o modelo confunde com mais frequência (por exemplo, 'A' com 'S' ou 'M' com 'N').

#### 5. Reconhecimento em Tempo Real

* Após ser treinado com todos os dados, o modelo final é carregado e a webcam é iniciada. Para cada frame do vídeo, o mesmo processo de extração e normalização de landmarks é aplicado, e o modelo prevê a letra em tempo real, mostrando a predição e a confiança na tela.

---

## Tecnologias Utilizadas 🛠️

* **Python 3.x**
* **OpenCV:** Para captura e manipulação de imagem e vídeo.
* **MediaPipe:** Para a detecção e extração dos landmarks da mão.
* **Scikit-learn:** Para o modelo RandomForest, validação cruzada e métricas de avaliação.
* **NumPy:** Para manipulação de arrays e cálculos numéricos.
* **Matplotlib & Seaborn:** Para a visualização da matriz de confusão.
* **tqdm:** Para criar barras de progresso elegantes durante o processamento.

---

## Como Executar o Projeto 🚀

Siga os passos abaixo para colocar o projeto em funcionamento na sua máquina.

#### 1. Pré-requisitos

* Python 3.8 ou superior instalado.
* pip (gerenciador de pacotes do Python).
* Uma webcam conectada.

#### 2. Clone o Repositório

```bash
git clone [https://github.com/SamuelMauli/Dijkstra_Largura_Profundidade.git](https://github.com/SamuelMauli/Dijkstra_Largura_Profundidade.git)
cd Dijkstra_Largura_Profundidade
```

#### 3. Baixe o Dataset

O modelo é treinado com o dataset "ASL Alphabet" do Kaggle.

* **Baixe aqui:** [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
* Após o download, descompacte o arquivo `archive.zip` dentro da pasta do projeto.

#### 4. Crie um Ambiente Virtual (Recomendado)

```bash
# Cria o ambiente virtual
python -m venv venv

# Ativa o ambiente
# No Windows:
venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate
```

#### 5. Instale as Dependências

Crie um arquivo chamado `requirements.txt` na pasta do projeto com o seguinte conteúdo:

```txt
opencv-python
mediapipe
scikit-learn
numpy
matplotlib
seaborn
tqdm
```

Em seguida, instale todas as bibliotecas de uma vez com o comando:

```bash
pip install -r requirements.txt
```

#### 6. Execute o Script

Com o ambiente virtual ativado e as dependências instaladas, basta rodar o script principal:

```bash
python seu_script_asl.py 
```
*(Substitua `seu_script_asl.py` pelo nome real do seu arquivo Python)*

O script irá primeiro processar o dataset, treinar o modelo e, ao final, abrirá a janela da sua webcam para o reconhecimento em tempo real.

---

## Estrutura de Pastas 📂

Para que o script funcione corretamente, a estrutura de pastas após o download do dataset deve ser a seguinte:

```
Dijkstra_Largura_Profundidade/
├── archive/
│   └── asl_alphabet_train/
│       └── asl_alphabet_train/
│           ├── A/
│           │   ├── A1.jpg
│           │   └── ...
│           ├── B/
│           ├── C/
│           └── ... (outras letras)
├── seu_script_asl.py
├── requirements.txt
└── README.md
```

---

## Próximos Passos & Melhorias Futuras 🔮

Como este é um projeto em estágio inicial, há muitas avenidas para exploração:

* **Melhorar a Acurácia:** Testar outros modelos de classificação, como `XGBoost`, `SVM` ou até mesmo uma pequena rede neural (`MLPClassifier`).
* **Reconhecer Sinais Dinâmicos:** Expandir o modelo para reconhecer palavras ou frases, que envolvem movimento (requer modelos como LSTMs ou Transformers).
* **Incluir Números e Símbolos:** Adicionar mais classes ao dataset para um vocabulário mais completo.
* **Interface Gráfica:** Criar uma interface mais robusta usando `Tkinter` ou `PyQt` para exibir informações adicionais.
* **Otimização:** Explorar técnicas para tornar o reconhecimento em tempo real ainda mais rápido e fluido.