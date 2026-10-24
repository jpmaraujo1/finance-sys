import httpx
from loguru import logger
from signal_engine.models import Signal, SignalType

EMOJI_MAP = {
    SignalType.STRONG_BUY: "\U0001F7E2 STRONG BUY",
    SignalType.BUY: "\U0001F7E2 BUY",
    SignalType.HOLD: "\u26AA HOLD",
    SignalType.SELL: "\U0001F534 SELL",
    SignalType.STRONG_SELL: "\U0001F534 STRONG SELL",
}


def send_telegram_alert(signal: Signal, bot_token: str, chat_id: str) -> bool:
    """Send formatted trade alert via Telegram Bot API."""
    if not bot_token or not chat_id:
        return False

    emoji_header = EMOJI_MAP.get(signal.signal_type, "\U0001F514 ALERT")
    reasons_text = "\n".join(f"\u2022 {r}" for r in signal.reasons)

    if signal.win_rate is not None:
        track_line = f"Backtested hit rate: {signal.win_rate}%\n"
    else:
        track_line = "Track record: unproven (not enough history yet)\n"
    rsi_line = f"RSI: {signal.rsi}\n" if signal.rsi is not None else ""

    text = (
        f"{emoji_header} \u2014 {signal.symbol}\n\n"
        f"Price: {signal.current_price:.4f}\n"
        f"{rsi_line}"
        f"Rule score: {signal.confidence} (indicator agreement, not a probability)\n"
        f"{track_line}\n"
        f"Key factors:\n{reasons_text}\n\n"
        f"\u26A0\uFE0F Automated technical analysis for informational purposes only. Not financial advice."
    )

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
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
