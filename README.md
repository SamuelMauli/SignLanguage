# 🤖 Reconhecimento do Alfabeto em ASL com MediaPipe e Scikit-learn

Este projeto utiliza visão computacional e machine learning para reconhecer, em tempo real, as letras do alfabeto da Língua de Sinais Americana (ASL) utilizando webcam e análise de landmarks da mão.

> **Status:** 🚧 Em desenvolvimento – prova de conceito inicial.

---

## 📌 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Fluxo de Funcionamento](#fluxo-de-funcionamento)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Como Executar](#como-executar)
- [Estrutura de Pastas](#estrutura-de-pastas)
- [Vídeo Demonstrativo](#vídeo-demonstrativo)
- [Melhorias Futuras](#melhorias-futuras)

---

## 🦾 Sobre o Projeto

O objetivo é desenvolver um sistema capaz de reconhecer as 26 letras do alfabeto em ASL a partir de imagens ou vídeo em tempo real. Em vez de processar imagens brutas, a aplicação utiliza landmarks (pontos de referência) da mão para tornar o modelo mais robusto em relação à iluminação, fundo e tom de pele.

---

## ⚙️ Fluxo de Funcionamento

1. **Extração de Características (MediaPipe Hands):**
   - São utilizados os 21 pontos-chave (landmarks) da mão.
   - Os dados são normalizados em relação ao ponto do pulso (landmark 0) para eliminar variações de posição e escala.

2. **Treinamento com Random Forest:**
   - O modelo `RandomForestClassifier` da Scikit-learn é utilizado pela sua precisão e robustez.

3. **Validação com K-Fold Estratificado:**
   - Utiliza validação cruzada com 5 divisões para garantir a generalização do modelo.

4. **Análise de Desempenho:**
   - Geração de relatório de classificação (precision, recall, F1-score).
   - Visualização da matriz de confusão.

5. **Reconhecimento em Tempo Real:**
   - O modelo final é salvo e usado para prever letras capturadas via webcam, com exibição ao vivo da predição.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **OpenCV** – Captura e manipulação de vídeo.
- **MediaPipe** – Extração dos landmarks da mão.
- **Scikit-learn** – Treinamento, validação e métricas.
- **NumPy** – Operações matriciais e normalização.
- **Matplotlib / Seaborn** – Visualização gráfica.
- **tqdm** – Feedback visual em loops demorados.

---

## 🚀 Como Executar

### 1. Pré-requisitos

- Python 3.8+
- pip
- Webcam

### 2. Clonar o Repositório

```bash
git clone https://github.com/SamuelMauli/SignLanguage.git
cd SignLanguage
```

### 3. Baixar o Dataset

- Faça o download do dataset ASL Alphabet:  
  👉 [ASL Alphabet - Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- Extraia o conteúdo dentro da pasta `archive/`.

### 4. Criar Ambiente Virtual

```bash
python -m venv .venv
# Ativar no Linux/macOS
source .venv/bin/activate
# Ativar no Windows
.venv\Scripts\activate
```

### 5. Instalar Dependências

```bash
pip install -r requirements.txt
```

> Exemplo do arquivo `requirements.txt`:

```txt
opencv-python
mediapipe
scikit-learn
numpy
matplotlib
seaborn
tqdm
```

### 6. Executar o Reconhecimento

```bash
python treinar_modelo.py
```

---

## 📁 Estrutura de Pastas

```
SignLanguage/
├── archive/
│   ├── asl_alphabet_train/
│   │   ├── A/, B/, C/, ..., Z/
│   └── asl_alphabet_test/
│       ├── A_test.jpg, ..., Z_test.jpg
├── asl_class_labels.npy
├── asl_random_forest_model.joblib
├── treinar_modelo.py
├── reconhecimento.py
├── requirements.txt
└── README.md
```

---

## 🎥 Vídeo Demonstrativo

Assista à demonstração do sistema em funcionamento:  
👉 [Ver no Google Drive](https://drive.google.com/file/d/1D4EhIK6ydQQXVrySmS_nyaniHppaC8t4/view?usp=sharing)

---

## 🔮 Melhorias Futuras

- 📈 Testar outros modelos: `XGBoost`, `SVM`, `MLPClassifier`.
- 🧠 Reconhecimento de sinais dinâmicos (ex.: frases) com LSTM/Transformer.
- ➕ Adicionar números e símbolos ao vocabulário.
- 🖥️ Criar uma interface gráfica com `Tkinter` ou `PyQt`.
- ⚡ Otimizar tempo de execução para real-time mais fluido.