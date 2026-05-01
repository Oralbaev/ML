# Practical Task 7 — Batch Prediction Pipeline

## Goal

Build a simple ML batch prediction pipeline that:
- Reads input data from a SQLite database
- Loads a trained scikit-learn model
- Generates predictions
- Saves results back to the database
- Runs automatically on a schedule (every 5 minutes)

---

## Project Structure

```
15week/
├── db.py             # Database setup and seed data
├── train_model.py    # Train and save the ML model
├── batch_predict.py  # Core prediction logic
├── scheduler.py      # Runs batch_predict every 5 minutes
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## Database Structure

**`pipeline.db`** — SQLite database with two tables:

### `input_data`
| Column   | Type    | Description             |
|----------|---------|-------------------------|
| id       | INTEGER | Primary key             |
| feature1 | REAL    | Numeric input feature   |
| feature2 | REAL    | Numeric input feature   |
| feature3 | REAL    | Numeric input feature   |

### `predictions`
| Column               | Type    | Description                        |
|----------------------|---------|------------------------------------|
| id                   | INTEGER | Primary key                        |
| input_id             | INTEGER | Foreign key → input_data.id        |
| prediction           | REAL    | Model output value                 |
| prediction_timestamp | TEXT    | UTC timestamp of the prediction    |

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows:** `venv\Scripts\activate`
- **macOS/Linux:** `source venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Train the model

Generates a synthetic dataset, trains a `LinearRegression` model, and saves it as `model.pkl`.

```bash
python train_model.py
```

### Initialise the database

Creates `pipeline.db`, sets up tables, and inserts 8 sample rows into `input_data`.

```bash
python db.py
```

### Run a single batch prediction

Reads all unprocessed rows from `input_data`, generates predictions with `model.pkl`, and writes them to `predictions`.

```bash
python batch_predict.py
```

### Run the scheduler

Executes batch prediction immediately, then repeats every **5 minutes** automatically. Stop with `Ctrl+C`.

```bash
python scheduler.py
```

---

## Full Example (run in order)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

python train_model.py          # → model.pkl created
python db.py                   # → pipeline.db created with sample data
python batch_predict.py        # → predictions written to database
python scheduler.py            # → keeps running every 5 minutes
```

---

## Notes

- Re-running `batch_predict.py` is safe — rows that already have a prediction are skipped automatically.
- The scheduler catches errors per run and keeps going without crashing.
- SQLite stores everything in a single file (`pipeline.db`) — no server needed.
