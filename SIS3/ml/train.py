import os
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "iris_classification"
MODEL_REGISTRY_NAME = "iris_random_forest"

# Save model relative to the project root (one level up from ml/)
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "models", "model.joblib")


def train():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "random_state": 42,
    }

    with mlflow.start_run(run_name="random_forest_baseline"):
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"F1 Score : {f1:.4f}")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME,
        )
        print(f"Model registered in MLflow registry as '{MODEL_REGISTRY_NAME}'")

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Model saved locally → {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train()
