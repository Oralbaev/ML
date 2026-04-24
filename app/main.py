import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

from .schemas import IrisFeatures, PredictionResponse

app = FastAPI(
    title="Iris Classification API",
    description="Predict Iris flower species from sepal and petal measurements.",
    version="1.0.0",
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

try:
    _model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    _model = None


@app.get("/")
def root():
    return {"message": "Iris Classification API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python ml/train.py' first, then restart the API.",
        )
    data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])
    pred = int(_model.predict(data)[0])
    confidence = float(_model.predict_proba(data)[0][pred])
    return PredictionResponse(
        prediction=pred,
        class_name=CLASS_NAMES[pred],
        confidence=round(confidence, 4),
    )
