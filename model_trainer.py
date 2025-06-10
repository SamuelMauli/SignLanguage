# model_trainer.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score

from config import *

def train_optimized_models():
    """
    Carrega dados processados e usa GridSearchCV para treinar e otimizar modelos.
    """
    try:
        processed_data = joblib.load(PROCESSED_DATA_FILE)
        X = processed_data['X']
        y = processed_data['y']
        class_labels = processed_data['classes']
    except FileNotFoundError:
        print(f"Erro: Arquivo de dados processados '{PROCESSED_DATA_FILE}' não encontrado.")
        print("Execute o script 'data_processor.py' primeiro.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Dataset carregado: {len(X_train)} amostras de treino, {len(X_test)} de teste.")

    # --- Definição dos Modelos e Grids de Hiperparâmetros ---
    models_and_params = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5]
            }
        },
        'LightGBM': {
            'model': LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [31, 50, 60]
            }
        }
    }

    # Validação Cruzada Estratificada para o GridSearchCV
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Salva os nomes das classes para a aplicação Flask
    np.save(os.path.join(MODELS_DIR, 'class_labels.npy'), class_labels)

    for name, config in models_and_params.items():
        print(f"\n--- Otimizando o modelo: {name} com GridSearchCV ---")
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=cv,
            scoring='accuracy',
            verbose=2,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        print(f"\nMelhores parâmetros para {name}: {grid_search.best_params_}")
        print(f"Melhor acurácia (validação cruzada): {grid_search.best_score_:.2%}")

        # Avaliação final no conjunto de teste
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Acurácia final no conjunto de Teste: {test_accuracy:.2%}")
        print("Relatório de Classificação Detalhado:")
        print(classification_report(y_test, y_pred, target_names=class_labels, zero_division=0))

        # Salva o melhor modelo encontrado
        model_path = os.path.join(MODELS_DIR, f'model_{name.lower()}.joblib')
        joblib.dump(best_model, model_path)
        print(f"Modelo otimizado salvo em '{model_path}'")

    print("\n--- Processo de Treinamento e Otimização Concluído! ---")

if __name__ == "__main__":
    train_optimized_models()