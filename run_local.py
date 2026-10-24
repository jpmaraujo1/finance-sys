import os
import threading
import time
import webbrowser

import uvicorn
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from signal_engine.database import init_db, DATA_DIR
from signal_engine.scheduler import start_scheduler
from signal_engine.main import handle_trade_alert

# 127.0.0.1 = only this computer can open the dashboard (safe default).
# Set HOST=0.0.0.0 in .env to reach it from your phone or other devices on your
# home network, and set API_KEY too when you do.
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8080"))
OPEN_BROWSER = os.getenv("OPEN_BROWSER", "true").lower() in ("1", "true", "yes")

# Keep a rotating log file so you can see what happened while you weren't watching
LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(str(LOG_DIR / "finance-ai.log"), rotation="10 MB", retention=10, level="INFO", enqueue=True)


def start_background_signal_engine():
    """Run the scheduler; if it ever crashes, log it and restart after a pause."""
    while True:
        try:
            logger.info("Starting background market scanning thread...")
            start_scheduler(alert_callback=handle_trade_alert)
            return  # clean shutdown
        except Exception as e:
            logger.exception(f"Signal engine crashed, restarting in 60s: {e}")
            time.sleep(60)


def main():
    logger.info("Initializing Finance AI Local System...")
    if HOST not in ("127.0.0.1", "localhost") and not os.getenv("API_KEY"):
        logger.warning(
            f"Dashboard is reachable from other devices (HOST={HOST}) without an API_KEY. "
            "Anyone on your network can use it. Set API_KEY in .env."
        )
    init_db()

    se_thread = threading.Thread(target=start_background_signal_engine, daemon=True, name="signal-engine")
    se_thread.start()

    url = f"http://localhost:{PORT}"
    if OPEN_BROWSER:
        time.sleep(1.5)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    logger.info(f"Starting FastAPI & Dashboard on {url} (bound to {HOST}) ...")
    uvicorn.run("api.main:app", host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
