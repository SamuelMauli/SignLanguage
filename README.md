# 🤖 Reconhecimento do Alfabeto em ASL com MediaPipe, Scikit-learn e LightGBM

Este projeto aplica visão computacional e aprendizado de máquina para reconhecer, em tempo real, as letras do alfabeto da Língua de Sinais Americana (ASL) utilizando uma webcam. A nova versão inclui **data augmentation**, **extração de features geométricas**, e **otimização de hiperparâmetros** com validação cruzada.

> **Status:** ✅ Estável – MVP funcional com reconhecimento em tempo real e troca dinâmica de modelo.

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

O sistema reconhece letras do alfabeto em ASL com base nos landmarks da mão, extraídos pelo MediaPipe. Evita o uso de imagens cruas, tornando-se robusto a iluminação, fundo e características físicas.

Esta versão aplica **Data Augmentation com albumentations** e **extração de features geométricas**, como ângulos e distâncias entre pontos da mão. O modelo pode ser executado tanto via terminal quanto por uma interface web em Flask.

---

## ⚙️ Fluxo de Funcionamento

1. **Extração de Landmarks (MediaPipe):**

   - Captura dos 21 pontos-chave da mão.
   - Normalização dos pontos em relação ao pulso.

2. **Engenharia de Features:**

   - Distâncias e ângulos entre landmarks.
   - Aplicação de `StandardScaler` para padronização.

3. **Data Augmentation:**

   - Aumento do dataset com brilho, ruído, rotação, flip horizontal etc.

4. **Treinamento e Otimização de Modelos:**

   - Modelos: `RandomForest`, `LightGBM`.
   - Otimização com `GridSearchCV` e validação cruzada estratificada.

5. **Inferência em Tempo Real:**
   - Streaming via OpenCV e Flask.
   - Troca dinâmica de modelos pela interface.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **MediaPipe** – Extração dos landmarks da mão.
- **OpenCV** – Captura de vídeo.
- **Scikit-learn / LightGBM** – Treinamento de modelos.
- **Albumentations** – Data augmentation.
- **Flask** – Interface web.
- **tqdm, numpy, joblib, matplotlib** – Suporte geral.

---

## 🚀 Como Executar

### 1. Pré-requisitos

- Python 3.8+
- pip
- Webcam

### 2. Clonar o Projeto

```
git clone https://github.com/SamuelMauli/SignLanguage.git
cd SignLanguage
```

### 3. Baixar Dataset

- Dataset ASL Alphabet:  
  👉 [ASL Alphabet - Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)  
  ➤ Extraia para `data/asl_alphabet_train/asl_alphabet_train/`

### 4. Ambiente Virtual

```
python -m venv .venv
source .venv/bin/activate # Linux/macOS
.venv\\Scripts\\activate # Windows
```

### 5. Instalar Dependências

```
pip install -r requirements.txt
```

> Exemplo do `requirements.txt`:

```
opencv-python
mediapipe
scikit-learn
lightgbm
numpy
matplotlib
seaborn
tqdm
flask
joblib
albumentations
```

### 6. Pré-processar Dataset (com Augmentation)

```
python data_processor.py
```

### 7. Treinar e Otimizar Modelos

```
python model_trainer.py
```

### 8. Rodar Interface Web

```
python app.py
```

---

## 📁 Estrutura de Pastas

```
SignLanguage/
├── app.py # Interface Flask
├── config.py # Parâmetros do projeto
├── data_processor.py # Extração de features e augmentation
├── model_trainer.py # Treinamento com GridSearch
├── models/ # Modelos otimizados e scaler
│ ├── model_randomforest.joblib
│ ├── model_lightgbm.joblib
│ ├── class_labels.npy
│ └── processed_data.joblib
├── templates/
│ └── index.html # Interface Web
├── static/ # Estáticos para o Flask
├── data/ # Dataset ASL (extraído)
│ └── asl_alphabet_train/
├── requirements.txt
└── README.md
```

---

## 🎥 Vídeo Demonstrativo

Assista a uma demonstração real do sistema em uso:  
👉 [Ver no Google Drive](https://drive.google.com/file/d/1D4EhIK6ydQQXVrySmS_nyaniHppaC8t4/view?usp=sharing)

---
