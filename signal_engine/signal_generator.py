from datetime import datetime, timezone
from typing import List, Optional

from . import config
from .models import Signal, SignalType, MarketType
from .technical_analysis import Indicators


def generate_signal(
    ind: Indicators,
    market: MarketType,
    win_rate: Optional[float] = None,
    sentiment: Optional[float] = None,
    news_count: int = 0,
) -> Signal:
    """
    Evaluate indicator confluence (plus optional news sentiment) and produce
    STRONG_BUY, BUY, HOLD, SELL or STRONG_SELL.

    `confidence` is a rule score that ranks how many factors agree. It is NOT a
    probability. The measured hit rate (backtest / live tracking) is what tells you
    how often a signal type has actually been right.
    """
    reasons: List[str] = []
    bullish_points = 0
    bearish_points = 0
    strong_bullish_points = 0
    strong_bearish_points = 0

    # 1. RSI
    if ind.rsi is not None:
        if ind.rsi < 30:
            strong_bullish_points += 1
            bullish_points += 2
            reasons.append(f"RSI is oversold at {ind.rsi:.1f} (<30)")
        elif ind.rsi < 42:
            bullish_points += 1
            reasons.append(f"RSI is in lower accumulation zone ({ind.rsi:.1f})")
        elif ind.rsi > 70:
            strong_bearish_points += 1
            bearish_points += 2
            reasons.append(f"RSI is overbought at {ind.rsi:.1f} (>70)")
        elif ind.rsi > 58:
            bearish_points += 1
            reasons.append(f"RSI is elevated ({ind.rsi:.1f})")

    # 2. MACD
    if ind.macd is not None and ind.macd_signal is not None:
        if ind.macd > ind.macd_signal:
            bullish_points += 1
            if ind.macd_hist is not None and ind.macd_hist > 0:
                strong_bullish_points += 1
                reasons.append("MACD above signal line with positive momentum")
            else:
                reasons.append("MACD is above signal line")
        elif ind.macd < ind.macd_signal:
            bearish_points += 1
            if ind.macd_hist is not None and ind.macd_hist < 0:
                strong_bearish_points += 1
                reasons.append("MACD below signal line with negative momentum")
            else:
                reasons.append("MACD is below signal line")

    # 3. Bollinger Bands, measured in standard deviations from the middle band
    if ind.bb_z is not None:
        if ind.bb_z <= -config.BB_Z_THRESHOLD:
            strong_bullish_points += 1
            bullish_points += 1
            reasons.append(f"Price is at/below the lower Bollinger Band (z = {ind.bb_z:.2f})")
        elif ind.bb_z >= config.BB_Z_THRESHOLD:
            strong_bearish_points += 1
            bearish_points += 1
            reasons.append(f"Price is at/above the upper Bollinger Band (z = {ind.bb_z:.2f})")

    # 4. Short-term trend (EMA 9 vs EMA 21)
    if ind.ema9 is not None and ind.ema21 is not None:
        if ind.ema9 > ind.ema21:
            bullish_points += 1
            reasons.append("Short-term EMA9 trending above EMA21")
        elif ind.ema9 < ind.ema21:
            bearish_points += 1
            reasons.append("Short-term EMA9 trending below EMA21")

    # 5. News sentiment (only when enough tagged headlines exist)
    if (
        config.SENTIMENT_WEIGHT > 0
        and sentiment is not None
        and news_count >= config.SENTIMENT_MIN_ARTICLES
    ):
        if sentiment >= config.SENTIMENT_THRESHOLD:
            bullish_points += config.SENTIMENT_WEIGHT
            reasons.append(f"Recent news sentiment is positive ({sentiment:+.2f} over {news_count} headlines)")
        elif sentiment <= -config.SENTIMENT_THRESHOLD:
            bearish_points += config.SENTIMENT_WEIGHT
            reasons.append(f"Recent news sentiment is negative ({sentiment:+.2f} over {news_count} headlines)")

    # 6. Volume confirmation of whichever side is winning
    if ind.volume is not None and ind.volume_sma20 is not None and ind.volume_sma20 > 0:
        vol_ratio = ind.volume / ind.volume_sma20
        if vol_ratio >= 1.4:
            if bullish_points > bearish_points:
                strong_bullish_points += 1
                reasons.append(f"Strong buying volume spike ({vol_ratio:.1f}x 20-period average)")
            elif bearish_points > bullish_points:
                strong_bearish_points += 1
                reasons.append(f"High volume selloff ({vol_ratio:.1f}x 20-period average)")

    # Final decision
    if strong_bullish_points >= 3 and bullish_points >= 4 and bullish_points > bearish_points:
        sig_type = SignalType.STRONG_BUY
        confidence = min(95.0, 75.0 + strong_bullish_points * 6.0)
    elif bullish_points >= 3 and bullish_points > bearish_points + 1:
        sig_type = SignalType.BUY
        confidence = min(78.0, 55.0 + bullish_points * 5.0)
    elif strong_bearish_points >= 3 and bearish_points >= 4 and bearish_points > bullish_points:
        sig_type = SignalType.STRONG_SELL
        confidence = min(95.0, 75.0 + strong_bearish_points * 6.0)
    elif bearish_points >= 3 and bearish_points > bullish_points + 1:
        sig_type = SignalType.SELL
        confidence = min(78.0, 55.0 + bearish_points * 5.0)
    else:
        sig_type = SignalType.HOLD
        confidence = 50.0
        if not reasons:
            reasons.append("Market is consolidating within standard volatility ranges")

    return Signal(
        symbol=ind.symbol,
        market=market,
        timestamp=ind.timestamp or datetime.now(timezone.utc),
        signal_type=sig_type,
        confidence=round(confidence, 1),
        reasons=reasons,
        current_price=ind.close,
        rsi=round(ind.rsi, 2) if ind.rsi is not None else None,
        macd=round(ind.macd, 4) if ind.macd is not None else None,
        win_rate=round(win_rate, 1) if win_rate is not None else None,
        sentiment=round(sentiment, 3) if sentiment is not None else None,
    )
