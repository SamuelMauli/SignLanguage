# -*- coding: utf-8 -*-
"""
TREINAMENTO DE MÚLTIPLOS MODELOS - Reconhecimento de Alfabeto em Libras 

1.  Carrega o dataset "ASL Alphabet".
2.  Usa o MediaPipe para extrair landmarks das mãos.
3.  Treina um modelo especificado (RandomForest, SVM, XGBoost, MLP).
4.  Gera relatórios de performance.
5.  Salva o modelo treinado e os rótulos das classes.
"""

# Importação das bibliotecas
import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import joblib

# Configuração do MediaPipe Hands
mp_hands = mp.solutions.hands

# --- FUNÇÕES DE PROCESSAMENTO DE DADOS ---

def extract_hand_landmarks(image_path):
    """Lê uma imagem, extrai os landmarks da mão e os normaliza."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Normalização pela posição do pulso (landmark 0)
        wrist_landmark = hand_landmarks.landmark[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x - wrist_landmark.x, lm.y - wrist_landmark.y])
            
        return landmarks

def load_data_from_local_folder(dataset_path):
    """Percorre as pastas do dataset, extrai landmarks e retorna os dados."""
    if not os.path.exists(dataset_path):
        print(f"Erro: O diretório do dataset não foi encontrado em '{dataset_path}'")
        return None, None, None

    X, y = [], []
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    print("Processando imagens e extraindo landmarks...")
    for label in tqdm(classes, desc="Processando classes"):
        class_path = os.path.join(dataset_path, label)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            landmarks = extract_hand_landmarks(img_path)
            if landmarks:
                X.append(landmarks)
                y.append(label)
    
    return np.array(X), np.array(y), classes

def plot_confusion_matrix_func(y_true, y_pred, class_names, model_name):
    """Gera e exibe uma matriz de confusão visual."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matriz de Confusão - {model_name}', fontsize=20)
    plt.ylabel('Classe Verdadeira', fontsize=16)
    plt.xlabel('Classe Predita', fontsize=16)
    plt.show()

# --- BLOCO DE EXECUÇÃO PRINCIPAL ---

def train_and_evaluate_model(classifier, model_name, X_train, y_train, X_test, y_test, class_labels):
    """Função para treinar, avaliar e salvar um modelo."""
    print(f"\n--- Treinando o modelo: {model_name} ---")
    
    # Treinamento
    classifier.fit(X_train, y_train)
    
    # Avaliação
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do {model_name}: {accuracy:.2%}")
    
    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(y_test, y_pred, target_names=class_labels, zero_division=0))
    
    # Matriz de Confusão
    plot_confusion_matrix_func(y_test, y_pred, class_labels, model_name)
    
    # Treinar modelo final com todos os dados
    print(f"\nTreinando o modelo final {model_name} com 100% dos dados...")
    classifier.fit(np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)))
    
    # Salvar o modelo
    model_filename = f"model_{model_name.lower().replace(' ', '_')}.joblib"
    print(f"Salvando o modelo em '{model_filename}'...")
    joblib.dump(classifier, model_filename)
    
    print(f"--- Treinamento do {model_name} concluído! ---")


if __name__ == "__main__":
    DATASET_PATH = os.path.join('archive', 'asl_alphabet_train', 'asl_alphabet_train')
    
    X_data, y_data_str, class_labels = load_data_from_local_folder(DATASET_PATH)

    if X_data is None:
        exit()

    # Mapear rótulos de string para numérico
    label_map = {label: i for i, label in enumerate(class_labels)}
    y_data_numeric = np.array([label_map[label] for label in y_data_str])
    
    # Salvar os rótulos para uso na aplicação Flask
    labels_filename = "asl_class_labels.npy"
    print(f"\nSalvando os rótulos das classes em '{labels_filename}'...")
    np.save(labels_filename, class_labels)
    
    print(f"\nTotal de amostras processadas: {len(X_data)}")
    print(f"Número de classes: {len(class_labels)}")

    # Dividir os dados em treino e teste (uma única vez)
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data_numeric, test_size=0.2, random_state=42, stratify=y_data_numeric
    )
    
    # --- Definir os modelos a serem treinados ---
    models_to_train = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42), # probability=True é útil, mas deixa o treino mais lento
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1),
        'MLPClassifier': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    # Treinar e avaliar cada modelo
    for name, model in models_to_train.items():
        train_and_evaluate_model(model, name, X_train, y_train, X_test, y_test, class_labels)

    print("\n--- Processo de Treinamento de Todos os Modelos Concluído! ---")