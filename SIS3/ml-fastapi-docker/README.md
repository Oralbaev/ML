# ML FastAPI Docker

A simple machine learning REST API built with FastAPI, scikit-learn, and Docker.

The model is trained on the Iris dataset using a RandomForestClassifier and served via FastAPI.

---

## Train the model

```bash
python train.py
```

This creates `model.joblib` in the current directory.

---

## Run locally

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

---

## Test the API

### Health check

```bash
curl http://localhost:8000/
```

Response:
```json
{"message": "ML API is running"}
```

### Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

Response:
```json
{"prediction": 0, "class": "setosa"}
```

---

## Swagger docs

Open in browser: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Docker

### Build image

```bash
docker build -t ml-fastapi-app .
```

### Run container

```bash
docker run -p 8000:8000 ml-fastapi-app
```

The API will be available at `http://localhost:8000`
