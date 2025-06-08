# -*- coding: utf-8 -*-
"""
Reconhecimento de Alfabeto em Língua de Sinais (ASL) - Versão Profissional (Corrigida)

Este script aprimorado implementa um fluxo de trabalho de Machine Learning mais robusto:
1.  Carrega o dataset "ASL Alphabet" de uma pasta local (caminho corrigido).
2.  Usa o MediaPipe para extrair landmarks das mãos de cada imagem.
3.  Avalia o modelo RandomForest usando Validação Cruzada Estratificada K-Fold.
4.  Gera um relatório de classificação e uma matriz de confusão para análise de performance.
5.  Treina o modelo final com todos os dados.
6.  Inicia o reconhecimento em tempo real usando a webcam.
"""

# Passo 1: Importação das bibliotecas necessárias
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

# Configuração do MediaPipe Hands para detecção
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- FUNÇÕES DE PROCESSAMENTO DE DADOS E MODELO ---

def extract_hand_landmarks(image_path):
    """
    Lê uma imagem, extrai os landmarks da mão e os normaliza.
    A normalização em relação ao pulso torna o modelo robusto a variações de 
    posição e escala da mão na imagem.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Processa a imagem com o MediaPipe
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            return None

        # Coleta os landmarks da primeira mão detectada
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Normaliza os pontos subtraindo as coordenadas do pulso (landmark 0)
        wrist_landmark = hand_landmarks.landmark[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x - wrist_landmark.x, lm.y - wrist_landmark.y])
            
        return landmarks

def load_data_from_local_folder(dataset_path):
    """
    Percorre as pastas do dataset local, extrai landmarks de cada imagem e retorna os dados.
    Esta função lê todas as imagens dentro de cada subpasta de letra (A, B, C...).
    """
    if not os.path.exists(dataset_path):
        print(f"Erro: O diretório do dataset não foi encontrado em '{dataset_path}'")
        print("Verifique se o caminho está correto e se a estrutura de pastas corresponde a 'archive/asl_alphabet_train/asl_alphabet_train'.")
        exit()

    X, y = [], []
    # Garante que as classes sejam lidas em ordem alfabética (A, B, C...)
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    print("Processando imagens e extraindo landmarks do dataset local...")
    for label in tqdm(classes, desc="Processando classes"):
        class_path = os.path.join(dataset_path, label)
        # Itera sobre todas as imagens dentro da pasta da letra
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            landmarks = extract_hand_landmarks(img_path)
            if landmarks:
                X.append(landmarks)
                y.append(label)
    
    return np.array(X), np.array(y), classes

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Gera e exibe uma matriz de confusão visual (heatmap).
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão', fontsize=20)
    plt.ylabel('Classe Verdadeira', fontsize=16)
    plt.xlabel('Classe Predita', fontsize=16)
    plt.show()

# --- FUNÇÃO DE RECONHECIMENTO EM TEMPO REAL ---

def real_time_recognition(model, class_names):
    """
    Inicia a webcam para reconhecimento de sinais em tempo real.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar frame da câmera.")
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                wrist_landmark = hand_landmarks.landmark[0]
                landmarks_for_model = []
                for lm in hand_landmarks.landmark:
                    landmarks_for_model.extend([lm.x - wrist_landmark.x, lm.y - wrist_landmark.y])
                
                prediction_proba = model.predict_proba([landmarks_for_model])[0]
                confidence = np.max(prediction_proba)
                predicted_class_index = np.argmax(prediction_proba)
                predicted_class_name = class_names[predicted_class_index]
                
                text = f"Letra: {predicted_class_name} ({confidence:.2%})"
                cv2.putText(frame, text, (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

            cv2.imshow("Reconhecimento de Sinais ASL (Pressione 'q' para sair)", frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Câmera desligada.")

# --- BLOCO DE EXECUÇÃO PRINCIPAL ---

if __name__ == "__main__":
    # Passo 2: Definir o caminho CORRETO para o dataset local
    # Com base na sua estrutura de pastas, o caminho inclui a pasta 'archive'.
    DATASET_PATH = os.path.join('archive', 'asl_alphabet_train', 'asl_alphabet_train')
    
    # Passo 3: Carregar dados e extrair features (landmarks)
    X_data, y_data_str, class_labels = load_data_from_local_folder(DATASET_PATH)

    # Mapeia os rótulos de string (A, B, C...) para números (0, 1, 2...) para o Scikit-Learn
    label_map = {label: i for i, label in enumerate(class_labels)}
    y_data_numeric = np.array([label_map[label] for label in y_data_str])
    
    print(f"\nTotal de amostras processadas: {len(X_data)}")
    print(f"Número de classes: {len(class_labels)} -> {class_labels}")

    # Passo 4: Avaliar o modelo com Validação Cruzada K-Fold (k=5)
    print("\nIniciando avaliação do modelo com Validação Cruzada (K=5)...")
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    accuracies = []
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_data_numeric)):
        print(f"--- Processando Fold {fold + 1}/{k_folds} ---")
        X_train_fold, y_train_fold = X_data[train_idx], y_data_numeric[train_idx]
        X_val_fold, y_val_fold = X_data[val_idx], y_data_numeric[val_idx]
        
        classifier.fit(X_train_fold, y_train_fold)
        y_pred_fold = classifier.predict(X_val_fold)
        acc = accuracy_score(y_val_fold, y_pred_fold)
        accuracies.append(acc)
        print(f"Acurácia do Fold {fold + 1}: {acc:.2%}")

    print("\n--- Resultados da Validação Cruzada ---")
    print(f"Acurácias de cada Fold: {[f'{acc:.2%}' for acc in accuracies]}")
    print(f"Acurácia Média: {np.mean(accuracies):.2%}")
    print(f"Desvio Padrão da Acurácia: {np.std(accuracies):.4f}")

    # Passo 5: Gerar Relatório de Classificação e Matriz de Confusão
    print("\nGerando relatório de classificação detalhado e matriz de confusão...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data_numeric, test_size=0.2, random_state=42, stratify=y_data_numeric
    )
    classifier.fit(X_train, y_train)
    y_pred_test = classifier.predict(X_test)
    
    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(y_test, y_pred_test, target_names=class_labels, zero_division=0))
    
    plot_confusion_matrix(y_test, y_pred_test, class_labels)

    # Passo 6: Treinar o modelo final com todos os dados
    print("\nTreinando o modelo final com 100% dos dados para o reconhecimento em tempo real...")
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(X_data, y_data_numeric)
    print("Modelo final treinado com sucesso!")

    # Passo 7: Iniciar o reconhecimento em tempo real
    print("\nIniciando reconhecimento em tempo real. Pressione 'q' na janela da câmera para sair.")
    real_time_recognition(final_model, class_labels)