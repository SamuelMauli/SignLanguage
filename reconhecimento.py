# -*- coding: utf-8 -*-
"""
RECONHECIMENTO EM TEMPO REAL - Alfabeto ASL

Este script carrega um modelo de Machine Learning já treinado e o utiliza
para reconhecer sinais da Língua de Sinais Americana (ASL) em tempo real
através da webcam.

Pré-requisitos:
1.  Os arquivos 'asl_random_forest_model.joblib' e 'asl_class_labels.npy'
    devem estar na mesma pasta que este script.
2.  Execute o script 'treinar_modelo.py' primeiro para gerar esses arquivos.
"""

# Passo 1: Importação das bibliotecas necessárias
import cv2
import numpy as np
import mediapipe as mp
import joblib  # <--- Biblioteca para carregar o modelo
import os

# --- Verificação de Arquivos ---
MODEL_PATH = "asl_random_forest_model.joblib"
LABELS_PATH = "asl_class_labels.npy"

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    print("Erro: Arquivos de modelo não encontrados!")
    print(f"Certifique-se de que '{MODEL_PATH}' e '{LABELS_PATH}' estão na pasta correta.")
    print("Execute o script 'treinar_modelo.py' primeiro para criar esses arquivos.")
    exit()

# Passo 2: Carregar o modelo e os rótulos das classes
print("Carregando modelo treinado...")
try:
    model = joblib.load(MODEL_PATH)
    class_names = np.load(LABELS_PATH)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Ocorreu um erro ao carregar os arquivos: {e}")
    exit()

# Passo 3: Configuração do MediaPipe e da Câmera
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def real_time_recognition(model, class_names):
    """
    Inicia a webcam para reconhecimento de sinais em tempo real.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return

    print("\nIniciando reconhecimento... Pressione 'q' na janela da câmera para sair.")
    
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar frame da câmera.")
                break
            
            # Espelhar a imagem para uma visualização tipo 'selfie'
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Lógica de detecção e predição
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Desenha os landmarks na imagem
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Normaliza os landmarks da mesma forma que no treinamento
                wrist_landmark = hand_landmarks.landmark[0]
                landmarks_for_model = []
                for lm in hand_landmarks.landmark:
                    landmarks_for_model.extend([lm.x - wrist_landmark.x, lm.y - wrist_landmark.y])
                
                # Faz a predição e obtém a confiança
                prediction_proba = model.predict_proba([landmarks_for_model])[0]
                confidence = np.max(prediction_proba)
                predicted_class_index = np.argmax(prediction_proba)
                predicted_class_name = class_names[predicted_class_index]
                
                # Exibe o resultado na tela
                text = f"Letra: {predicted_class_name} ({confidence:.2%})"
                cv2.putText(frame, text, (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

            cv2.imshow("Reconhecimento de Sinais ASL (Pressione 'q' para sair)", frame)

            # Condição de parada
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Câmera desligada.")

# --- BLOCO DE EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    real_time_recognition(model, class_names)