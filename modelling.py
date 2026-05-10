import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set tracking URI & nama eksperimen
# mlflow.set_experiment("Heart Disease CI")

# Load data hasil preprocessing
data_path = os.path.join(os.path.dirname(__file__), 'dataset_processed.joblib')
X_train, X_test, y_train, y_test = joblib.load(data_path)

# Pastikan target biner (0 = Sehat, 1 = Sakit)
# num: 0=normal, 1-4=heart disease
y_train = (y_train > 0).astype(int)
y_test = (y_test > 0).astype(int)

print(f"Ukuran X_train : {X_train.shape}")
print(f"Ukuran X_test  : {X_test.shape}")

# Target sudah biner (0 = Sehat, 1 = Penyakit Jantung) dari preprocessing
print(f"\nDistribusi y_train:\n{y_train.value_counts().to_string()}")
print(f"\nDistribusi y_test:\n{y_test.value_counts().to_string()}")

# Aktifkan autolog MLflow 
# Mencatat parameter default, metrik, dan model secara otomatis
# mlflow.sklearn.autolog()

with mlflow.start_run(nested=True):
    #  Training model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Hitung metrik evaluasi
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary')
    rec  = recall_score(y_test, y_pred, average='binary')
    f1   = f1_score(y_test, y_pred, average='binary')

    # Log parameter manual (Basic but good practice)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # Log metrik manual ke MLflow
    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall",    rec)
    mlflow.log_metric("f1_score",  f1)

    # Simpan Model Manual
    mlflow.sklearn.log_model(model, "model")

    # Confusion Matrix → log sebagai artefak
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Sehat (0)', 'Sakit (1)'],
        yticklabels=['Sehat (0)', 'Sakit (1)']
    )
    plt.title('Confusion Matrix — Heart Disease')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')

    # Feature Importance -> Artefak ke-2
    import pandas as pd
    import numpy as np
    importances = model.feature_importances_
    # Feature names from preprocessing (numeric + categorical)
    # numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    # categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    feature_names = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    feat_importances = pd.Series(importances, index=feature_names)
    plt.figure(figsize=(8, 6))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    mlflow.log_artifact('feature_importance.png')

    os.remove('confusion_matrix.png')
    os.remove('feature_importance.png')

    # Ringkasan hasil
    print("\n" + "="*45)
    print("   HASIL EVALUASI — Random Forest")
    print("="*45)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("="*45)

# print("\nCek MLflow UI di http://127.0.0.1:5000")