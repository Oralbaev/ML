"""
db.py - Database setup for the batch prediction pipeline.
Creates SQLite database, tables, and seeds sample input data.
"""

import sqlite3

DB_PATH = "pipeline.db"


def get_connection():
    """Return a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)


def create_tables():
    """Create input_data and predictions tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS input_data (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            feature1 REAL NOT NULL,
            feature2 REAL NOT NULL,
            feature3 REAL NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            input_id             INTEGER NOT NULL,
            prediction           REAL    NOT NULL,
            prediction_timestamp TEXT    NOT NULL,
            FOREIGN KEY (input_id) REFERENCES input_data(id)
        )
    """)

    conn.commit()
    conn.close()
    print("Tables created (or already exist).")


def seed_input_data():
    """Insert sample rows into input_data if the table is empty."""
    sample_rows = [
        (1.5, 2.3, 0.8),
        (3.1, 0.5, 1.2),
        (2.0, 1.7, 3.4),
        (0.3, 4.1, 2.2),
        (5.0, 0.9, 0.1),
        (1.1, 1.1, 1.1),
        (4.4, 3.3, 2.2),
        (0.7, 2.9, 3.8),
    ]

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM input_data")
    count = cursor.fetchone()[0]

    if count == 0:
        cursor.executemany(
            "INSERT INTO input_data (feature1, feature2, feature3) VALUES (?, ?, ?)",
            sample_rows,
        )
        conn.commit()
        print(f"Inserted {len(sample_rows)} sample rows into input_data.")
    else:
        print(f"input_data already has {count} rows — skipping seed.")

    conn.close()


if __name__ == "__main__":
    create_tables()
    seed_input_data()
    print("Database initialised at:", DB_PATH)
