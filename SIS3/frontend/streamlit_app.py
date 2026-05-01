import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Iris Classifier", layout="centered")

st.title("Iris Flower Classifier")
st.markdown(
    "Enter the flower measurements below, then click **Predict** to classify the species."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)

with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        st.success(
            f"**Predicted species: {result['class_name'].capitalize()}**  \n"
            f"Class index: {result['prediction']}  \n"
            f"Confidence: {result['confidence'] * 100:.1f}%"
        )
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to the API at `{API_URL}`. "
            "Make sure the FastAPI service is running."
        )
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.json().get("detail", str(exc))
        st.error(f"API returned an error: {detail}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")

st.divider()
st.caption("SIS-3 Project — Iris Classification System")
