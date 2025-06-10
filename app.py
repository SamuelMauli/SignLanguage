# app.py
import cv2
import numpy as np
import mediapipe as mp
import joblib
from flask import Flask, render_template, Response, jsonify, request
import os

# Importa as funções e variáveis de configuração dos nossos módulos
from config import MODELS_DIR, PROCESSED_DATA_FILE
from data_processor import extract_features 

# --- CONFIGURAÇÃO INICIAL DA APLICAÇÃO FLASK ---
app = Flask(__name__)

# Tenta carregar os recursos essenciais (scaler, rótulos) no início.
# Se falhar, a aplicação não inicia, o que é um comportamento seguro.
try:
    class_labels = np.load(os.path.join(MODELS_DIR, 'class_labels.npy'), allow_pickle=True)
    # Carrega o scaler que foi salvo junto com os dados processados
    scaler = joblib.load(PROCESSED_DATA_FILE)['scaler']
    print("Recursos essenciais (scaler, rótulos) carregados com sucesso.")
except FileNotFoundError:
    print("ERRO CRÍTICO: Arquivos essenciais não encontrados no diretório 'models/'.")
    print(f"Verifique se '{PROCESSED_DATA_FILE}' e 'class_labels.npy' existem.")
    print("Execute 'data_processor.py' e 'model_trainer.py' primeiro.")
    exit()

# Dicionário para carregar os modelos otimizados sob demanda
available_models = {
    'randomforest': os.path.join(MODELS_DIR, 'model_randomforest.joblib'),
    'lightgbm': os.path.join(MODELS_DIR, 'model_lightgbm.joblib'),
}
current_model = None
current_model_name = ""

# --- LÓGICA DE RECONHECIMENTO EM TEMPO REAL ---

def generate_frames():
    """
    Gera frames da webcam com predições em tempo real.
    Esta função representa o "pipeline de inferência".
    """
    global current_model
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Não foi possível capturar o frame. Fim do stream.")
            break
        
        frame = cv2.flip(frame, 1)
        
        # 1. Extração de Features: Usa a mesma função do treino para consistência
        features = extract_features(frame)
        prediction_text = "Nenhuma mao detectada"
        
        if features is not None and current_model is not None:
            try:
                # 2. Padronização: Aplica a mesma escala (Z-score) do treino. Etapa CRÍTICA!
                features_scaled = scaler.transform([features])
                
                # 3. Predição: Usa o modelo otimizado para prever a probabilidade de cada classe
                prediction_proba = current_model.predict_proba(features_scaled)[0]
                confidence = np.max(prediction_proba)
                prediction_idx = np.argmax(prediction_proba)
                predicted_char = class_labels[prediction_idx]
                
                prediction_text = f'LETRA: {predicted_char} ({confidence:.2%})'
            except Exception as e:
                prediction_text = f"Erro na predicao: {e}"

        # Visualização no frame
        cv2.rectangle(frame, (0, 0), (640, 50), (20, 20, 20), -1)
        cv2.putText(frame, prediction_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 120), 3, cv2.LINE_AA)
        
        # Codifica o frame para streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- ROTAS FLASK ---
@app.route('/')
def index():
    """Renderiza a página inicial."""
    return render_template('index.html', models=available_models.keys())

@app.route('/video_feed')
def video_feed():
    """Rota para o streaming de vídeo."""
    if current_model is None:
        return "Nenhum modelo selecionado", 400
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/select_model', methods=['POST'])
def select_model():
    """Carrega o modelo selecionado pelo usuário de forma segura."""
    global current_model, current_model_name
    model_name = request.json.get('model')
    
    if not model_name or model_name not in available_models:
        return jsonify(success=False, error="Nome do modelo inválido."), 400
    
    try:
        model_path = available_models[model_name]
        if not os.path.exists(model_path):
            error_msg = f"Arquivo do modelo '{model_path}' nao foi encontrado. Treine o modelo primeiro."
            return jsonify(success=False, error=error_msg), 404
            
        current_model = joblib.load(model_path)
        current_model_name = model_name
        print(f"Modelo '{model_name}' carregado com sucesso.")
        return jsonify(success=True, model_name=model_name)
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return jsonify(success=False, error=f"Erro interno ao carregar o modelo: {e}"), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')