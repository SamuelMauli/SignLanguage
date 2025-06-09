# -*- coding: utf-8 -*-
"""
APLICAÇÃO WEB FLASK - Reconhecimento de Alfabeto em Libras

Esta aplicação utiliza Flask para criar uma interface web que permite
ao usuário selecionar um modelo treinado e usar a webcam para 
reconhecimento de sinais em tempo real.
"""

import cv2
import numpy as np
import mediapipe as mp
import joblib
from flask import Flask, render_template, Response, jsonify, request

# --- CONFIGURAÇÃO INICIAL ---
app = Flask(__name__)

# Carregar os rótulos das classes
try:
    class_labels = np.load('asl_class_labels.npy')
except FileNotFoundError:
    print("Erro: Arquivo 'asl_class_labels.npy' não encontrado. Execute o script de treinamento primeiro.")
    exit()

# Dicionário para carregar os modelos sob demanda
models = {
    'randomforest': 'model_randomforest.joblib',
    'svm': 'model_svm.joblib',
    'xgboost': 'model_xgboost.joblib',
    'mlpclassifier': 'model_mlpclassifier.joblib'
}
current_model = None
current_model_name = ""

# Configuração do MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# --- FUNÇÕES AUXILIARES ---

def extract_landmarks(image):
    """Extrai landmarks da mão na imagem e normaliza."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    landmarks_data = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Desenhar os landmarks na imagem para visualização
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )

        # Normalização dos landmarks
        wrist_landmark = hand_landmarks.landmark[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x - wrist_landmark.x, lm.y - wrist_landmark.y])
        landmarks_data = np.array(landmarks).reshape(1, -1)
        
    return image, landmarks_data

# --- ROTAS DO FLASK ---

@app.route('/')
def index():
    """Página inicial que renderiza o layout com a barra lateral."""
    return render_template('index.html', models=models.keys())

def generate_frames():
    """Gera frames da webcam com predições."""
    global current_model
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Inverter o frame horizontalmente para efeito de espelho
        frame = cv2.flip(frame, 1)
        
        # Extrair landmarks
        frame, landmarks = extract_landmarks(frame)
        
        prediction_text = "Nenhuma mao detectada"
        
        if landmarks is not None and current_model is not None:
            try:
                # Fazer a predição
                prediction_idx = current_model.predict(landmarks)[0]
                predicted_char = class_labels[prediction_idx]
                
                # Obter probabilidades (se o modelo suportar)
                if hasattr(current_model, "predict_proba"):
                    probabilities = current_model.predict_proba(landmarks)[0]
                    confidence = probabilities[prediction_idx]
                    prediction_text = f'LETRA: {predicted_char} ({confidence:.2%})'
                else:
                    prediction_text = f'LETRA: {predicted_char}'
            except Exception as e:
                prediction_text = f"Erro na predicao: {e}"

        # Escrever a predição no frame
        cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Modelo: {current_model_name.upper()}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


        # Codificar o frame como JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Enviar o frame para o browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Rota para o streaming de vídeo."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/select_model', methods=['POST'])
def select_model():
    """Carrega o modelo selecionado pelo usuário."""
    global current_model, current_model_name
    model_name = request.json['model']
    
    if model_name in models:
        try:
            current_model = joblib.load(models[model_name])
            current_model_name = model_name
            print(f"Modelo '{model_name}' carregado com sucesso.")
            return jsonify(success=True, model_name=model_name)
        except FileNotFoundError:
            print(f"Erro: Arquivo do modelo '{models[model_name]}' não encontrado.")
            return jsonify(success=False, error="Arquivo do modelo não encontrado.")
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
            return jsonify(success=False, error=str(e))
            
    return jsonify(success=False, error="Modelo inválido.")

if __name__ == '__main__':
    app.run(debug=True)