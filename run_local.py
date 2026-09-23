import os
import sys
import threading
import time
import webbrowser
import uvicorn
from loguru import logger

from signal_engine.database import init_db
from signal_engine.scheduler import start_scheduler
from signal_engine.main import handle_trade_alert

def start_background_signal_engine():
    logger.info("Starting background market scanning thread...")
    try:
        start_scheduler(alert_callback=handle_trade_alert)
    except Exception as e:
        logger.error(f"Signal engine error: {e}")

def main():
    logger.info("Initializing Finance AI Local System...")
    init_db()

    # Launch signal engine in daemon background thread
    se_thread = threading.Thread(target=start_background_signal_engine, daemon=True)
    se_thread.start()

    logger.info("Opening dashboard in browser...")
    time.sleep(1.5)
    try:
        webbrowser.open("http://localhost:8080")
    except Exception:
        pass

    logger.info("Starting FastAPI & Dashboard on http://localhost:8080 ...")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8080, log_level="info")

if __name__ == "__main__":
    main()

