"""
scheduler.py - Run batch_predict.py automatically every 5 minutes.

Uses the 'schedule' library for simplicity.
Stop the process with Ctrl+C.
"""

import time
import schedule

from batch_predict import run_batch_prediction

INTERVAL_MINUTES = 5


def job():
    print(f"\n--- Scheduled batch prediction started ---")
    try:
        run_batch_prediction()
    except Exception as e:
        # Log the error but keep the scheduler alive
        print(f"Error during batch prediction: {e}")


# Run once immediately on startup so we don't wait 5 minutes for the first result
job()

# Then repeat every INTERVAL_MINUTES minutes
schedule.every(INTERVAL_MINUTES).minutes.do(job)

print(f"\nScheduler running. Next run in {INTERVAL_MINUTES} minutes. Press Ctrl+C to stop.\n")

while True:
    schedule.run_pending()
    time.sleep(30)  # check every 30 seconds whether a job is due
