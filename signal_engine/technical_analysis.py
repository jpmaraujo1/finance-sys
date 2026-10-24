from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class Indicators:
    symbol: str
    timestamp: datetime
    close: float
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_mid: Optional[float] = None
    bb_z: Optional[float] = None
    ema9: Optional[float] = None
    ema21: Optional[float] = None
    ema50: Optional[float] = None
    ema200: Optional[float] = None
    sma50: Optional[float] = None
    sma200: Optional[float] = None
    obv: Optional[float] = None
    atr: Optional[float] = None
    volume: Optional[float] = None
    volume_sma20: Optional[float] = None


INDICATOR_COLUMNS = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_mid", "bb_z",
    "ema9", "ema21", "ema50", "ema200", "sma50", "sma200",
    "obv", "atr", "volume", "volume_sma20",
]


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (exponential smoothing with alpha = 1/period)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    # No losses at all -> RSI 100; completely flat -> 50
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi.where(avg_gain.notna())


def compute_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute every indicator for every row.

    All calculations are trailing (rolling windows / EWMs that only look backwards),
    so row i only uses data up to and including candle i. That makes the same frame
    safe to use for live signals (last row) and for walk-forward backtests (every row)
    without look-ahead bias.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    n = len(df)

    close = df["close"].astype(float)
    vol = df["volume"].astype(float) if "volume" in df else pd.Series(0.0, index=df.index)

    out = pd.DataFrame(index=df.index)
    out["close"] = close
    out["rsi"] = compute_rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    bb_mid = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_lower"] = bb_mid - 2 * bb_std
    # Distance from the middle band in standard deviations. Unlike a fixed % tolerance,
    # this means the same thing for a calm FX pair and a volatile altcoin.
    out["bb_z"] = (close - bb_mid) / bb_std.replace(0, np.nan)

    out["ema9"] = close.ewm(span=9, adjust=False).mean()
    out["ema21"] = close.ewm(span=21, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean() if n >= 50 else np.nan
    out["ema200"] = close.ewm(span=200, adjust=False).mean() if n >= 200 else np.nan
    out["sma50"] = close.rolling(window=50).mean()
    out["sma200"] = close.rolling(window=200).mean()

    out["volume"] = vol
    out["volume_sma20"] = vol.rolling(window=20).mean()

    if "high" in df and "low" in df:
        prev_close = close.shift(1)
        tr = pd.concat(
            [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = tr.rolling(window=14).mean()
    else:
        out["atr"] = np.nan

    direction = np.sign(close.diff().fillna(0.0))
    out["obv"] = (vol * direction).cumsum()
    return out


def _clean(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) or np.isinf(f) else f


def indicators_from_row(row: pd.Series, symbol: str, timestamp) -> Indicators:
    ts = timestamp
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if not isinstance(ts, datetime):
        ts = datetime.now(timezone.utc)
    return Indicators(
        symbol=symbol,
        timestamp=ts,
        close=float(row["close"]),
        **{col: _clean(row.get(col)) for col in INDICATOR_COLUMNS},
    )


def compute_indicators(df: pd.DataFrame, symbol: str = "") -> Optional[Indicators]:
    """Indicators for the most recent candle in df (used by the live engine)."""
    if df is None or len(df) < 30:
        return None
    try:
        frame = compute_indicator_frame(df)
        return indicators_from_row(frame.iloc[-1], symbol, frame.index[-1])
    except Exception as e:
        logger.error(f"Error computing indicators for {symbol}: {e}")
        return None
