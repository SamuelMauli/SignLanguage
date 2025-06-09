
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

2. **Treinamento com Múltiplos Modelos:**
   - Modelos utilizados: `RandomForest`, `SVM`, `XGBoost`, `MLPClassifier`.
   - Acurácias e relatórios são exibidos para cada modelo.

3. **Validação com Holdout e Relatórios:**
   - Divisão treino/teste com stratificação.
   - Geração de matriz de confusão e classificação detalhada.

4. **Reconhecimento em Tempo Real:**
   - Aplicação terminal ou interface web via Flask permite uso com webcam em tempo real.
   - Permite troca de modelo ao vivo via interface web.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **OpenCV** – Captura e manipulação de vídeo.
- **MediaPipe** – Extração dos landmarks da mão.
- **Scikit-learn / XGBoost** – Modelos de aprendizado de máquina.
- **NumPy / Matplotlib / Seaborn / tqdm**
- **Flask** – Interface Web (opcional)

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
xgboost
numpy
matplotlib
seaborn
tqdm
flask
```

### 6. Treinar os Modelos

```bash
python treinar_modelo.py
```

### 7. Executar o Reconhecimento em Tempo Real (Terminal)

```bash
python reconhecimento.py
```

### 8. Iniciar Interface Web com Flask

```bash
python app.py
```

---

## 📁 Estrutura de Pastas

```
SignLanguage/
├── archive/
│   └── asl_alphabet_train/
│       └── A/, B/, ..., Z/
├── asl_class_labels.npy
├── model_randomforest.joblib
├── model_svm.joblib
├── model_xgboost.joblib
├── model_mlpclassifier.joblib
├── treinar_modelo.py
├── reconhecimento.py
├── app.py
├── templates/
│   ├── layout.html
│   └── index.html
├── static/
├── requirements.txt
└── README.md
```

---

## 🎥 Vídeo Demonstrativo

Assista à demonstração do sistema em funcionamento:  
👉 [Ver no Google Drive](https://drive.google.com/file/d/1D4EhIK6ydQQXVrySmS_nyaniHppaC8t4/view?usp=sharing)

---

## 🔮 Melhorias Futuras

- 📈 Ajustar hiperparâmetros dos modelos.
- 🧠 Suporte a sinais dinâmicos (ex: frases completas).
- 🖥️ Interface mais robusta (ex: PyQt5, Electron).
- 📱 Adaptar para dispositivos móveis (Flutter, React Native).
- 📦 Deploy via Docker.