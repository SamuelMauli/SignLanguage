# -*- coding: utf-8 -*-
"""
TREINAMENTO DO MODELO - Reconhecimento de Alfabeto em Língua de Sinais (ASL)

Este script executa o fluxo de treinamento completo:
1.  Carrega o dataset "ASL Alphabet".
2.  Usa o MediaPipe para extrair landmarks das mãos.
3.  Avalia o modelo RandomForest com Validação Cruzada.
4.  Gera relatórios de performance.
5.  Treina o modelo final com todos os dados.
6.  SALVA o modelo treinado e os rótulos das classes em arquivos.
"""

# Passo 1: Importação das bibliotecas
import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import joblib  # <--- Biblioteca para salvar o modelo

# Configuração do MediaPipe Hands
mp_hands = mp.solutions.hands

# --- FUNÇÕES DE PROCESSAMENTO DE DADOS E MODELO (sem alterações) ---

def extract_hand_landmarks(image_path):
    """
    Lê uma imagem, extrai os landmarks da mão e os normaliza.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        
        wrist_landmark = hand_landmarks.landmark[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x - wrist_landmark.x, lm.y - wrist_landmark.y])
            
        return landmarks

def load_data_from_local_folder(dataset_path):
    """
    Percorre as pastas do dataset local, extrai landmarks e retorna os dados.
    """
    if not os.path.exists(dataset_path):
        print(f"Erro: O diretório do dataset não foi encontrado em '{dataset_path}'")
        exit()

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

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Gera e exibe uma matriz de confusão visual.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão', fontsize=20)
    plt.ylabel('Classe Verdadeira', fontsize=16)
    plt.xlabel('Classe Predita', fontsize=16)
    plt.show()

# --- BLOCO DE EXECUÇÃO PRINCIPAL ---

if __name__ == "__main__":
    # Passo 2: Definir o caminho para o dataset
    DATASET_PATH = os.path.join('archive', 'asl_alphabet_train', 'asl_alphabet_train')
    
    # Passo 3: Carregar dados e extrair features
    X_data, y_data_str, class_labels = load_data_from_local_folder(DATASET_PATH)

    label_map = {label: i for i, label in enumerate(class_labels)}
    y_data_numeric = np.array([label_map[label] for label in y_data_str])
    
    print(f"\nTotal de amostras processadas: {len(X_data)}")
    print(f"Número de classes: {len(class_labels)} -> {class_labels}")

    # Passo 4: Avaliar o modelo com Validação Cruzada K-Fold (opcional, bom para análise)
    print("\nIniciando avaliação do modelo com Validação Cruzada (K=5)...")
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    accuracies = []
    classifier_eval = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_data_numeric)):
        print(f"--- Processando Fold {fold + 1}/{k_folds} ---")
        X_train_fold, y_train_fold = X_data[train_idx], y_data_numeric[train_idx]
        X_val_fold, y_val_fold = X_data[val_idx], y_data_numeric[val_idx]
        
        classifier_eval.fit(X_train_fold, y_train_fold)
        y_pred_fold = classifier_eval.predict(X_val_fold)
        acc = accuracy_score(y_val_fold, y_pred_fold)
        accuracies.append(acc)
        print(f"Acurácia do Fold {fold + 1}: {acc:.2%}")

    print("\n--- Resultados da Validação Cruzada ---")
    print(f"Acurácia Média: {np.mean(accuracies):.2%}")

    # Passo 5: Gerar Relatório de Classificação e Matriz de Confusão
    print("\nGerando relatório de classificação e matriz de confusão...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data_numeric, test_size=0.2, random_state=42, stratify=y_data_numeric
    )
    classifier_eval.fit(X_train, y_train)
    y_pred_test = classifier_eval.predict(X_test)
    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(y_test, y_pred_test, target_names=class_labels, zero_division=0))
    plot_confusion_matrix(y_test, y_pred_test, class_labels)

    # Passo 6: Treinar o modelo final com todos os dados
    print("\nTreinando o modelo final com 100% dos dados...")
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(X_data, y_data_numeric)
    print("Modelo final treinado com sucesso!")

    # Passo 7: SALVAR o modelo e os rótulos das classes
    MODEL_PATH = "asl_random_forest_model.joblib"
    LABELS_PATH = "asl_class_labels.npy"

    print(f"\nSalvando o modelo em '{MODEL_PATH}'...")
    joblib.dump(final_model, MODEL_PATH)
    
    print(f"Salvando os rótulos das classes em '{LABELS_PATH}'...")
    np.save(LABELS_PATH, class_labels)
    
    print("\n--- Processo de Treinamento Concluído! ---")
    print(f"Agora você pode executar o arquivo 'reconhecimento.py' para usar a câmera.")