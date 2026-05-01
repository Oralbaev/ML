"""
batch_predict.py - Load the trained model and generate predictions for all
rows in input_data, then write results to the predictions table.
"""

import os
import joblib
import numpy as np
from datetime import datetime, timezone

from db import get_connection, create_tables

MODEL_PATH = "model.pkl"


def load_model():
    """Load the trained model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run 'python train_model.py' first."
        )
    return joblib.load(MODEL_PATH)


def fetch_input_data(conn):
    """Return all rows from input_data as a list of dicts."""
    cursor = conn.cursor()
    cursor.execute("SELECT id, feature1, feature2, feature3 FROM input_data")
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def already_predicted_ids(conn):
    """Return a set of input_ids that already have a prediction."""
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT input_id FROM predictions")
    return {row[0] for row in cursor.fetchall()}


def run_batch_prediction():
    # Make sure tables exist before we start
    create_tables()

    conn = get_connection()

    rows = fetch_input_data(conn)
    if not rows:
        print("No input data found — nothing to predict.")
        conn.close()
        return

    model = load_model()

    # Skip rows that were already predicted in a previous run
    done_ids = already_predicted_ids(conn)
    new_rows = [r for r in rows if r["id"] not in done_ids]

    if not new_rows:
        print("All rows already have predictions — nothing new to do.")
        conn.close()
        return

    # Build a feature matrix and run inference in one vectorised call
    features = np.array(
        [[r["feature1"], r["feature2"], r["feature3"]] for r in new_rows]
    )
    predictions = model.predict(features)

    timestamp = datetime.now(timezone.utc).isoformat()

    cursor = conn.cursor()
    records = [
        (row["id"], float(pred), timestamp)
        for row, pred in zip(new_rows, predictions)
    ]
    cursor.executemany(
        "INSERT INTO predictions (input_id, prediction, prediction_timestamp) VALUES (?, ?, ?)",
        records,
    )
    conn.commit()
    conn.close()

    print(f"[{timestamp}] Inserted {len(records)} prediction(s) into the database.")


if __name__ == "__main__":
    run_batch_prediction()
