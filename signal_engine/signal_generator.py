from datetime import datetime, timezone
from typing import List, Optional
from loguru import logger

from .models import Signal, SignalType, MarketType, WatchlistItem
from .technical_analysis import Indicators


def generate_signal(
    ind: Indicators,
    market: MarketType,
    win_rate: Optional[float] = None
) -> Signal:
    """
    Evaluate technical indicator confluence to generate actionable signals:
    STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL.
    """
    reasons: List[str] = []
    bullish_points = 0
    bearish_points = 0
    strong_bullish_points = 0
    strong_bearish_points = 0

    # 1. RSI analysis
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

    # 2. MACD analysis
    if ind.macd is not None and ind.macd_signal is not None:
        if ind.macd > ind.macd_signal:
            bullish_points += 1
            if ind.macd_hist is not None and ind.macd_hist > 0:
                strong_bullish_points += 1
                reasons.append(f"MACD line ({ind.macd:.2f}) above signal ({ind.macd_signal:.2f}) with positive momentum")
            else:
                reasons.append("MACD is above signal line")
        elif ind.macd < ind.macd_signal:
            bearish_points += 1
            if ind.macd_hist is not None and ind.macd_hist < 0:
                strong_bearish_points += 1
                reasons.append(f"MACD line ({ind.macd:.2f}) below signal ({ind.macd_signal:.2f}) with negative momentum")
            else:
                reasons.append("MACD is below signal line")

    # 3. Bollinger Bands analysis
    if ind.bb_lower is not None and ind.bb_upper is not None:
        if ind.close <= ind.bb_lower * 1.01:
            strong_bullish_points += 1
            bullish_points += 1
            reasons.append(f"Price (${ind.close:.2f}) is testing lower Bollinger Band (${ind.bb_lower:.2f})")
        elif ind.close >= ind.bb_upper * 0.99:
            strong_bearish_points += 1
            bearish_points += 1
            reasons.append(f"Price (${ind.close:.2f}) is testing upper Bollinger Band (${ind.bb_upper:.2f})")

    # 4. Moving Average Trend (EMA 9 vs EMA 21 / 50)
    if ind.ema9 is not None and ind.ema21 is not None:
        if ind.ema9 > ind.ema21:
            bullish_points += 1
            reasons.append(f"Short-term EMA9 (${ind.ema9:.2f}) trending above EMA21 (${ind.ema21:.2f})")
        else:
            bearish_points += 1
            reasons.append(f"Short-term EMA9 (${ind.ema9:.2f}) trending below EMA21 (${ind.ema21:.2f})")

    # 5. Volume Confirmation
    if ind.volume is not None and ind.volume_sma20 is not None and ind.volume_sma20 > 0:
        vol_ratio = ind.volume / ind.volume_sma20
        if vol_ratio >= 1.4:
            if bullish_points > bearish_points:
                strong_bullish_points += 1
                reasons.append(f"Strong buying volume spike ({vol_ratio:.1f}x 20-period average)")
            elif bearish_points > bullish_points:
                strong_bearish_points += 1
                reasons.append(f"High volume selloff ({vol_ratio:.1f}x 20-period average)")

    # Decide final SignalType and confidence score
    if strong_bullish_points >= 3 and bullish_points >= 4:
        sig_type = SignalType.STRONG_BUY
        confidence = min(95.0, 75.0 + strong_bullish_points * 6.0)
    elif bullish_points >= 3 and bullish_points > bearish_points + 1:
        sig_type = SignalType.BUY
        confidence = min(78.0, 55.0 + bullish_points * 5.0)
    elif strong_bearish_points >= 3 and bearish_points >= 4:
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
        macd=round(ind.macd, 2) if ind.macd is not None else None,
        win_rate=round(win_rate, 1) if win_rate is not None else None,
    )
