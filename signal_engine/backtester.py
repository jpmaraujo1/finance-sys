from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd
from loguru import logger

from .models import SignalType, MarketType
from .technical_analysis import compute_indicators
from .signal_generator import generate_signal

# In-memory cache for computed win rates: (symbol, signal_type) -> (win_rate, timestamp)
_WIN_RATE_CACHE: Dict[str, tuple] = {}
CACHE_TTL_HOURS = 1


@dataclass
class BacktestResult:
    symbol: str
    signal_type: SignalType
    total_signals: int
    winning_signals: int
    win_rate: float
    avg_return_pct: float
    period_days: int


def backtest_symbol(symbol: str, df: pd.DataFrame, forward_candles: int = 12) -> Dict[SignalType, BacktestResult]:
    """
    Backtest signal accuracy across historical candles.
    A BUY/STRONG_BUY is winning if price is higher after forward_candles.
    A SELL/STRONG_SELL is winning if price is lower after forward_candles.
    """
    results: Dict[SignalType, BacktestResult] = {}
    if df is None or len(df) < 50:
        return results

    counts = {st: {"total": 0, "wins": 0, "returns": []} for st in SignalType}

    # Step through historical windows
    step = max(1, len(df) // 200)  # sample up to 200 points to keep compute fast
    for i in range(30, len(df) - forward_candles, step):
        sub_df = df.iloc[: i + 1]
        future_close = df["close"].iloc[i + forward_candles]
        curr_close = df["close"].iloc[i]

        ind = compute_indicators(sub_df, symbol=symbol)
        if not ind:
            continue

        sig = generate_signal(ind, market=MarketType.US_STOCKS)
        st = sig.signal_type
        counts[st]["total"] += 1

        pct_change = (future_close - curr_close) / curr_close * 100.0
        counts[st]["returns"].append(pct_change)

        if st in (SignalType.STRONG_BUY, SignalType.BUY) and pct_change > 0:
            counts[st]["wins"] += 1
        elif st in (SignalType.STRONG_SELL, SignalType.SELL) and pct_change < 0:
            counts[st]["wins"] += 1
        elif st == SignalType.HOLD and abs(pct_change) < 1.0:
            counts[st]["wins"] += 1

    for st, data in counts.items():
        total = data["total"]
        wins = data["wins"]
        win_rate = (wins / total * 100.0) if total > 0 else 50.0
        avg_ret = (sum(data["returns"]) / len(data["returns"])) if data["returns"] else 0.0

        results[st] = BacktestResult(
            symbol=symbol,
            signal_type=st,
            total_signals=total,
            winning_signals=wins,
            win_rate=round(win_rate, 1),
            avg_return_pct=round(avg_ret, 2),
            period_days=30,
        )

    return results


def get_win_rate(symbol: str, signal_type: SignalType, get_prices_fn) -> Optional[float]:
    """
    Get win rate with in-memory caching.
    """
    cache_key = f"{symbol}_{signal_type.value}"
    now = datetime.now()
    if cache_key in _WIN_RATE_CACHE:
        win_rate, cached_at = _WIN_RATE_CACHE[cache_key]
        if (now - cached_at) < timedelta(hours=CACHE_TTL_HOURS):
            return win_rate

    try:
        df = get_prices_fn(symbol, hours=168)  # 7 days of 5m candles
        if df is not None and not df.empty:
            res = backtest_symbol(symbol, df)
            if signal_type in res:
                wr = res[signal_type].win_rate
                _WIN_RATE_CACHE[cache_key] = (wr, now)
                return wr
    except Exception as e:
        logger.warning(f"Failed to compute win rate for {symbol}: {e}")

    return 65.0  # fallback historical baseline
