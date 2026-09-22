import httpx
from loguru import logger
from signal_engine.models import Signal, SignalType

EMOJI_MAP = {
    SignalType.STRONG_BUY: "?? STRONG BUY",
    SignalType.BUY: "?? BUY",
    SignalType.HOLD: "?? HOLD",
    SignalType.SELL: "?? SELL",
    SignalType.STRONG_SELL: "?? STRONG SELL",
}


def send_telegram_alert(signal: Signal, bot_token: str, chat_id: str) -> bool:
    """Send formatted trade alert via Telegram Bot API."""
    if not bot_token or not chat_id:
        return False

    emoji_header = EMOJI_MAP.get(signal.signal_type, "?? ALERT")
    reasons_text = "\n".join(f"? {r}" for r in signal.reasons)

    win_rate_line = f"?? Historical Win Rate: {signal.win_rate}%\n" if signal.win_rate else ""
    rsi_line = f"?? RSI: {signal.rsi}\n" if signal.rsi else ""

    text = (
        f"{emoji_header} ? {signal.symbol}\n\n"
        f"?? Price: ${signal.current_price:.2f}\n"
        f"{rsi_line}"
        f"?? Confidence: {signal.confidence}%\n"
        f"{win_rate_line}\n"
        f"Key Factors:\n{reasons_text}\n\n"
        f"?? Automated technical analysis for informational purposes only. Not financial advice."
    )

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=payload)
            if resp.is_success:
                logger.info(f"Telegram alert sent for {signal.symbol}")
                return True
            else:
                logger.error(f"Telegram alert failed ({resp.status_code}): {resp.text}")
                return False
    except Exception as e:
        logger.error(f"Failed to post to Telegram API: {e}")
        return False
