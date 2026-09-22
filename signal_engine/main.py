import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from .scheduler import start_scheduler
from .models import Signal
from alerts.telegram_bot import send_telegram_alert
from alerts.email_alerts import send_email_alert

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")


def handle_trade_alert(signal: Signal):
    """Dispatch signal alert via Telegram and Email."""
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            send_telegram_alert(signal, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        except Exception as e:
            logger.error(f"Failed to dispatch Telegram alert: {e}")

    if ALERT_EMAIL_TO and SMTP_USER and SMTP_PASSWORD:
        try:
            send_email_alert(
                signal,
                to_email=ALERT_EMAIL_TO,
                smtp_host=SMTP_HOST,
                smtp_port=SMTP_PORT,
                smtp_user=SMTP_USER,
                smtp_password=SMTP_PASSWORD,
            )
        except Exception as e:
            logger.error(f"Failed to dispatch Email alert: {e}")


def main():
    logger.info("Initializing Trading Signal Engine Service...")
    start_scheduler(alert_callback=handle_trade_alert)


if __name__ == "__main__":
    main()
