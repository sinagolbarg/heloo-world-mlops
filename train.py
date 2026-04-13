"""
Simple training script:
- loads iris dataset from sklearn
- trains a LogisticRegression
- saves model to model.pkl
- computes accuracy, AUC, KS
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import joblib
import os
import json
import numpy as np

os.makedirs("artifacts", exist_ok=True)

def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    # Convert to binary classification: setosa (0) vs non-setosa (1)
    y_binary = (y != 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.5, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Save model
    model_path = os.path.join("artifacts", "model.pkl")
    joblib.dump(model, model_path)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = model.score(X_test, y_test)

    # AUC
    auc = roc_auc_score(y_test, y_proba)

    # KS statistic
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    ks = max(tpr - fpr)

    # Save metrics
    metrics = {
        "accuracy": float(accuracy),
        "auc": float(auc),
        "ks": float(ks)
    }

    with open(os.path.join("artifacts", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved model to {model_path}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"KS: {ks:.4f}")

if __name__ == "__main__":
    main()
