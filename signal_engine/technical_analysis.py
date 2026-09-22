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


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_indicators(df: pd.DataFrame, symbol: str = "") -> Optional[Indicators]:
    """
    Pure pandas & numpy technical indicators (Zero external C dependencies).
    Computes RSI, MACD, Bollinger Bands, EMAs, SMAs, OBV, ATR.
    """
    if df is None or len(df) < 15:
        return None

    try:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        close = df["close"]
        vol = df["volume"] if "volume" in df else pd.Series(0, index=df.index)

        # RSI (14)
        rsi_series = compute_rsi(close, 14)

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_sig = macd_line.ewm(span=9, adjust=False).mean()
        macd_h = macd_line - macd_sig

        # Bollinger Bands (20, 2)
        bb_mid = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        bb_upper = bb_mid + (bb_std * 2)
        bb_lower = bb_mid - (bb_std * 2)

        # EMAs
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean() if len(df) >= 50 else None
        ema200 = close.ewm(span=200, adjust=False).mean() if len(df) >= 200 else None

        # SMAs
        sma50 = close.rolling(window=50).mean() if len(df) >= 50 else None
        sma200 = close.rolling(window=200).mean() if len(df) >= 200 else None

        # Volume SMA 20
        vol_sma20 = vol.rolling(window=20).mean() if len(df) >= 20 else None

        # ATR (14)
        if "high" in df and "low" in df:
            high = df["high"]
            low = df["low"]
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_series = tr.rolling(window=14).mean()
        else:
            atr_series = None

        # OBV
        direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
        obv_series = (vol * direction).cumsum()

        def safe_val(val):
            if val is None or pd.isna(val) or np.isnan(val):
                return None
            return float(val)

        last_idx = df.index[-1]
        last_ts = last_idx if isinstance(last_idx, datetime) else datetime.now(timezone.utc)

        return Indicators(
            symbol=symbol,
            timestamp=last_ts,
            close=float(close.iloc[-1]),
            rsi=safe_val(rsi_series.iloc[-1]),
            macd=safe_val(macd_line.iloc[-1]),
            macd_signal=safe_val(macd_sig.iloc[-1]),
            macd_hist=safe_val(macd_h.iloc[-1]),
            bb_upper=safe_val(bb_upper.iloc[-1]),
            bb_lower=safe_val(bb_lower.iloc[-1]),
            bb_mid=safe_val(bb_mid.iloc[-1]),
            ema9=safe_val(ema9.iloc[-1]),
            ema21=safe_val(ema21.iloc[-1]),
            ema50=safe_val(ema50.iloc[-1]) if ema50 is not None else None,
            ema200=safe_val(ema200.iloc[-1]) if ema200 is not None else None,
            sma50=safe_val(sma50.iloc[-1]) if sma50 is not None else None,
            sma200=safe_val(sma200.iloc[-1]) if sma200 is not None else None,
            obv=safe_val(obv_series.iloc[-1]),
            atr=safe_val(atr_series.iloc[-1]) if atr_series is not None else None,
            volume=float(vol.iloc[-1]) if len(vol) > 0 else None,
            volume_sma20=safe_val(vol_sma20.iloc[-1]) if vol_sma20 is not None else None,
        )
    except Exception as e:
        logger.error(f"Error computing indicators for {symbol}: {e}")
        return None
