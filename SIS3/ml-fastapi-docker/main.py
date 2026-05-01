from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.joblib")


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float  
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(features: IrisFeatures):
    data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])
    prediction = model.predict(data)[0]
    class_names = ["setosa", "versicolor", "virginica"]
    return {"prediction": int(prediction), "class": class_names[prediction]}
