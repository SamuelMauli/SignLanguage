# config.py
import os

# --- Caminhos do Projeto ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'asl_alphabet_train', 'asl_alphabet_train')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Arquivo para salvar os dados processados e evitar reprocessamento
PROCESSED_DATA_FILE = os.path.join(MODELS_DIR, 'processed_data.joblib')

# --- Parâmetros de Processamento de Dados ---
# Ativar/desativar a criação de imagens aumentadas
USE_AUGMENTATION = True
AUGMENTATION_FACTOR = 2 # Cria N novas imagens para cada imagem original

# --- Parâmetros de Feature Engineering ---
# Ativar/desativar a extração de features geométricas
USE_GEOMETRIC_FEATURES = True

# --- Configurações de Treinamento ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5 # Número de folds para a Validação Cruzada