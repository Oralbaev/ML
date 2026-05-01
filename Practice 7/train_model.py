"""
train_model.py - Train a simple ML model and save it as model.pkl.

Uses a synthetic regression dataset so no external data file is needed.
"""

import joblib
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

MODEL_PATH = "model.pkl"


def train_and_save():
    # Generate a small synthetic dataset with 3 features
    X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate on the test split so we can report a quick sanity metric
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model trained. Test MSE: {mse:.2f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()
