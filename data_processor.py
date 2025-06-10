# data_processor.py
import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import joblib
import albumentations as A
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

from config import *

# --- Data Augmentation Pipeline ---
augmentation_pipeline = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.6),
    A.HorizontalFlip(p=0.5),
])

# --- Funções de Feature Engineering ---
mp_hands = mp.solutions.hands

def calculate_angles(landmarks, hand_connections):
    """Calcula ângulos entre conexões de landmarks."""
    angles = []
    for connection in hand_connections:
        p1 = landmarks[connection[0]]
        p2 = landmarks[connection[1]]
        # Vetor entre os pontos
        vector = p2 - p1
        # Ângulo com o eixo x
        angle = np.arctan2(vector[1], vector[0])
        angles.append(angle)
    return angles

def extract_features(image):
    """Extrai um vetor de features completo de uma imagem."""
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

        # 1. Landmarks Normalizados (relativos ao pulso)
        wrist_coords = coords[0]
        normalized_landmarks = (coords - wrist_coords).flatten()

        # 2. Features Geométricas (se ativado)
        if USE_GEOMETRIC_FEATURES:
            # Distâncias euclidianas entre todos os pares de landmarks
            distances = pdist(coords[:, :2]) # Usando apenas x, y para distâncias 2D

            # Ângulos entre as conexões dos dedos
            angles = calculate_angles(coords[:, :2], mp_hands.HAND_CONNECTIONS)
            
            # Concatena todas as features
            feature_vector = np.concatenate((normalized_landmarks, distances, angles))
        else:
            feature_vector = normalized_landmarks
        
        return feature_vector

def create_processed_dataset():
    """
    Carrega imagens, aplica augmentation, extrai features e salva um arquivo de dados processado.
    """
    if not os.path.exists(DATASET_PATH):
        print(f"Erro: Diretório do dataset não encontrado em '{DATASET_PATH}'")
        return

    X, y = [], []
    classes = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    
    print("Processando imagens, aplicando augmentation e extraindo features...")
    for label_idx, label in enumerate(tqdm(classes, desc="Processando classes")):
        class_path = os.path.join(DATASET_PATH, label)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is None: continue

            # Extrai features da imagem original
            features = extract_features(image)
            if features is not None:
                X.append(features)
                y.append(label_idx)

            # Aplica Data Augmentation
            if USE_AUGMENTATION:
                for _ in range(AUGMENTATION_FACTOR):
                    augmented_image = augmentation_pipeline(image=image)['image']
                    aug_features = extract_features(augmented_image)
                    if aug_features is not None:
                        X.append(aug_features)
                        y.append(label_idx)

    X = np.array(X)
    y = np.array(y)
    
    # Padroniza os dados (Normalização Z-score)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Garante que o diretório de modelos exista
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Salva os dados processados para não precisar refazer
    joblib.dump({
        'X': X_scaled,
        'y': y,
        'classes': classes,
        'scaler': scaler
    }, PROCESSED_DATA_FILE)

    print("\nProcessamento concluído!")
    print(f"Total de amostras geradas: {len(X)}")
    print(f"Dados processados e salvos em '{PROCESSED_DATA_FILE}'")


if __name__ == "__main__":
    create_processed_dataset()