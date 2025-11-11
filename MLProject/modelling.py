# /Workflow-CI/MLProject/modelling.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import os
import argparse

# Sesuaikan PATH agar model dapat membaca dataset dari sub-folder
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'diabetes_preprocessing', 'diabetes_preprocessing.csv')

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    return X, y

# Ganti nama fungsi menjadi main() untuk MLProject, dan tambahkan argumen parameter
def main(test_size=0.2, random_state=42):
    """Melatih model dengan Hyperparameter Tuning dan Manual Logging (Dijalankan oleh MLflow Project)."""
    
    X, y = load_data(DATASET_PATH)
    # Gunakan parameter yang diterima dari MLProject
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 1. Definisikan Model dan Hyperparameter Grid 
    model = RandomForestClassifier(random_state=random_state)
    param_grid = {
        'n_estimators': [100, 200],      
        'max_depth': [10, 20],           
        'min_samples_split': [2, 5]
    }
    
    # 2. Setup GridSearchCV
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=3, 
        scoring='accuracy',
        n_jobs=-1
    )

    # MLflow akan otomatis membuat run saat dipanggil oleh mlflow run .
    # Tidak perlu mlflow.start_run()
    
    print("Mulai proses Hyperparameter Tuning...")

    # Latih dengan Grid Search
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Hitung Metrik
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy (Manual Log): {accuracy:.4f}")
    
    # 3. MANUAL LOGGING 
    
    # Log Parameters (Manual)
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    
    # Log Metriks (Manual)
    mlflow.log_metric("test_accuracy", accuracy) 
    
    # Log Model (menyimpan artefak model di folder mlruns)
    mlflow.sklearn.log_model(best_model, "best_ci_model")
    
    print("Model Tuning dan Logging Selesai.")

if __name__ == "__main__":
    # Parsing argumen jika dipanggil langsung
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    
    main(args.test_size, args.random_state)