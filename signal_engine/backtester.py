"""
Walk-forward backtest of the live signal rules.

How it stays honest:
  * Indicators are trailing-only, and each candle's signal is produced by the exact
    same `generate_signal()` the live engine uses, so there is no look-ahead.
  * Only *entries* are counted (the candle where the signal type changes). Counting
    every 5-minute repeat of the same signal would count one trade dozens of times.
  * Every return is reported net of a realistic round-trip cost per market.
  * Results are compared with an "always long" baseline over the same bars: a BUY
    signal in a rising market is only useful if it beats simply being long.
  * A t-statistic shows whether the edge is distinguishable from noise.

CLI:
    python -m signal_engine.backtester                      # whole watchlist
    python -m signal_engine.backtester --symbols BTCUSDT --horizon 48 --days 60
"""
import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from . import config
from .models import MarketType, SignalType
from .signal_generator import generate_signal
from .technical_analysis import compute_indicator_frame, Indicators, INDICATOR_COLUMNS

WARMUP_BARS = 60
DIRECTION = {
    SignalType.STRONG_BUY.value: 1,
    SignalType.BUY.value: 1,
    SignalType.SELL.value: -1,
    SignalType.STRONG_SELL.value: -1,
}
ORDERED_TYPES = ["STRONG_BUY", "BUY", "SELL", "STRONG_SELL"]


# ---------------------------------------------------------------------------
# Signal generation over history
# ---------------------------------------------------------------------------

def rolling_sentiment(index: pd.DatetimeIndex, news: Optional[pd.DataFrame], hours: float):
    """Mean sentiment and headline count in the window (t - hours, t] for every candle time t."""
    n = len(index)
    if news is None or news.empty:
        return np.full(n, np.nan), np.zeros(n, dtype=int)
    t_news = news.index.values
    scores = news["sentiment_score"].to_numpy(dtype=float)
    csum = np.concatenate([[0.0], np.cumsum(scores)])
    t_bars = index.values
    hi = np.searchsorted(t_news, t_bars, side="right")
    lo = np.searchsorted(t_news, t_bars - np.timedelta64(int(hours * 3600), "s"), side="right")
    counts = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, (csum[hi] - csum[lo]) / np.maximum(counts, 1), np.nan)
    return means, counts


def build_signal_frame(
    df: pd.DataFrame,
    symbol: str,
    market: MarketType,
    news: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Indicator frame plus the signal the live engine would have produced at each candle."""
    frame = compute_indicator_frame(df)
    sent_mean, sent_count = rolling_sentiment(frame.index, news, config.SENTIMENT_WINDOW_HOURS)

    n = len(frame)
    types: List[Optional[str]] = [None] * n
    scores = np.full(n, np.nan)
    close = frame["close"].to_numpy(dtype=float)
    cols = {c: frame[c].to_numpy(dtype=float) for c in INDICATOR_COLUMNS}
    times = frame.index.to_pydatetime()
    for i in range(WARMUP_BARS, n):
        values = {c: (None if not np.isfinite(arr[i]) else float(arr[i])) for c, arr in cols.items()}
        ind = Indicators(symbol=symbol, timestamp=times[i], close=float(close[i]), **values)
        s = None if np.isnan(sent_mean[i]) else float(sent_mean[i])
        sig = generate_signal(ind, market=market, sentiment=s, news_count=int(sent_count[i]))
        types[i] = sig.signal_type.value
        scores[i] = sig.confidence
    frame["signal"] = types
    frame["score"] = scores
    return frame


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def entry_records(frame: pd.DataFrame, symbol: str, market: MarketType, horizon: int, cost: float) -> pd.DataFrame:
    """One row per signal entry, with forward return over `horizon` candles."""
    close = frame["close"]
    fwd = close.shift(-horizon) / close - 1.0
    sig = frame["signal"]
    valid = sig.notna() & fwd.notna()
    baseline = float(fwd[valid].mean()) if valid.any() else 0.0

    is_entry = sig.ne(sig.shift(1)) & valid
    rec = pd.DataFrame({
        "symbol": symbol,
        "market": market.value,
        "time": frame.index[is_entry],
        "signal_type": sig[is_entry].values,
        "score": frame["score"][is_entry].values,
        "fwd_return": fwd[is_entry].values,
    })
    rec = rec[rec["signal_type"].isin(DIRECTION.keys())].copy()
    rec["direction"] = rec["signal_type"].map(DIRECTION)
    rec["gross"] = rec["direction"] * rec["fwd_return"]
    rec["net"] = rec["gross"] - cost
    # Positive excess means the signal beat simply being long (for buys) or beat
    # being short-the-drift (for sells) over the same period.
    rec["excess"] = rec["direction"] * (rec["fwd_return"] - baseline)
    rec.attrs["baseline"] = baseline
    return rec


def summarize(records: pd.DataFrame, cost: Optional[float] = None) -> Dict[str, dict]:
    """Per-signal-type statistics from entry records."""
    out: Dict[str, dict] = {}
    for st in ORDERED_TYPES:
        r = records[records["signal_type"] == st]
        n = len(r)
        if n == 0:
            continue
        excess = r["excess"].to_numpy(dtype=float)
        sd = float(np.std(excess, ddof=1)) if n > 1 else float("nan")
        t_stat = float(np.mean(excess) / (sd / math.sqrt(n))) if n > 1 and sd > 0 else float("nan")
        out[st] = {
            "n_entries": int(n),
            "hit_rate": round(float((r["net"] > 0).mean() * 100), 1),
            "avg_gross_pct": round(float(r["gross"].mean() * 100), 4),
            "avg_net_pct": round(float(r["net"].mean() * 100), 4),
            "edge_vs_long_pct": round(float(np.mean(excess) * 100), 4),
            "t_stat": None if math.isnan(t_stat) else round(t_stat, 2),
            "enough_data": n >= config.MIN_SAMPLES_FOR_STATS,
        }
        if cost is not None:
            out[st]["cost_pct"] = round(cost * 100, 4)
    return out


def simulate_long_only(frame: pd.DataFrame, cost: float) -> dict:
    """
    What following the signals would have done: go long on BUY/STRONG_BUY, go flat
    on SELL/STRONG_SELL, keep the current position on HOLD. Compared with buy & hold.
    """
    f = frame[frame["signal"].notna()]
    if len(f) < 2:
        return {}
    close = f["close"].to_numpy(dtype=float)
    sig = f["signal"].to_numpy()

    pos = np.zeros(len(f))
    current = 0.0
    for i, s in enumerate(sig):
        if s in ("BUY", "STRONG_BUY"):
            current = 1.0
        elif s in ("SELL", "STRONG_SELL"):
            current = 0.0
        pos[i] = current

    bar_ret = np.zeros(len(f))
    bar_ret[1:] = close[1:] / close[:-1] - 1.0
    held = np.concatenate([[0.0], pos[:-1]])           # position decided at the previous close
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    strat = held * bar_ret - turnover * cost / 2.0      # half the round trip on each side

    equity = np.cumprod(1.0 + strat)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity / peak - 1.0))

    bh_equity = close / close[0]
    bh_dd = float(np.min(bh_equity / np.maximum.accumulate(bh_equity) - 1.0))
    return {
        "strategy_return_pct": round(float(equity[-1] - 1.0) * 100, 2),
        "buy_hold_return_pct": round(float(bh_equity[-1] - 1.0) * 100, 2),
        "strategy_max_drawdown_pct": round(max_dd * 100, 2),
        "buy_hold_max_drawdown_pct": round(bh_dd * 100, 2),
        "trades": int((np.diff(np.concatenate([[0.0], pos])) > 0).sum()),
        "time_in_market_pct": round(float(pos.mean() * 100), 1),
        "cost_paid_pct": round(float((turnover * cost / 2.0).sum() * 100), 2),
        "start": str(f.index[0]),
        "end": str(f.index[-1]),
        "candles": int(len(f)),
    }


def backtest_frame(
    df: pd.DataFrame,
    symbol: str,
    market: MarketType,
    horizon: int = config.BACKTEST_HORIZON_BARS,
    news: Optional[pd.DataFrame] = None,
    cost: Optional[float] = None,
):
    """Backtest one symbol. Returns (per-type stats, strategy stats, entry records)."""
    cost = config.cost_for(market) if cost is None else cost
    if df is None or len(df) < WARMUP_BARS + horizon + 10:
        return {}, {}, pd.DataFrame()
    frame = build_signal_frame(df, symbol, market, news)
    records = entry_records(frame, symbol, market, horizon, cost)
    return summarize(records, cost), simulate_long_only(frame, cost), records


# ---------------------------------------------------------------------------
# Database-backed runs
# ---------------------------------------------------------------------------

def run_symbol_from_db(symbol: str, market: MarketType, days: int, horizon: int):
    from .database import get_prices, get_news_frame

    df = get_prices(symbol, hours=days * 24)
    news = get_news_frame(symbol)
    return backtest_frame(df, symbol, market, horizon=horizon, news=news)


def run_all_backtests(days: int = config.BACKTEST_LOOKBACK_DAYS, horizon: int = config.BACKTEST_HORIZON_BARS,
                      symbols: Optional[List[str]] = None, save: bool = True):
    """Backtest every watchlist symbol; optionally store results for the dashboard."""
    from .database import get_watchlist, save_backtest_results

    results: Dict[str, dict] = {}
    all_records = []
    for item in get_watchlist():
        if symbols and item.symbol not in symbols:
            continue
        try:
            stats, strategy, records = run_symbol_from_db(item.symbol, item.market, days, horizon)
        except Exception as e:
            logger.error(f"Backtest failed for {item.symbol}: {e}")
            continue
        if not stats and not strategy:
            continue
        per_symbol = dict(stats)
        if strategy:
            per_symbol["STRATEGY"] = strategy
        results[item.symbol] = per_symbol
        if not records.empty:
            all_records.append(records)

    if save and results:
        save_backtest_results(results)
    pooled = summarize(pd.concat(all_records, ignore_index=True)) if all_records else {}
    return results, pooled, (pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame())


def get_hit_rate(symbol: str, signal_type: SignalType) -> Optional[float]:
    """Backtested hit rate for this symbol and signal type, or None if there isn't enough evidence."""
    if signal_type == SignalType.HOLD:
        return None
    from .database import get_backtest_results

    for row in get_backtest_results(symbol):
        if row.get("signal_type") == signal_type.value and row.get("enough_data"):
            return row.get("hit_rate")
    return None


# ---------------------------------------------------------------------------
# CLI report
# ---------------------------------------------------------------------------

def _fmt(v, suffix=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v}{suffix}"


def print_report(results: Dict[str, dict], pooled: Dict[str, dict], horizon: int):
    hours = horizon * config.CANDLE_MINUTES / 60
    print(f"\nWalk-forward backtest, holding period {horizon} candles ({hours:g}h), net of costs\n")
    header = f"{'symbol':<10} {'signal':<12} {'entries':>7} {'hit%':>6} {'avg net%':>9} {'edge vs long%':>14} {'t':>6}"
    print(header)
    print("-" * len(header))
    for symbol, per_type in results.items():
        for st in ORDERED_TYPES:
            s = per_type.get(st)
            if not s:
                continue
            print(f"{symbol:<10} {st:<12} {s['n_entries']:>7} {s['hit_rate']:>6} {s['avg_net_pct']:>9} "
                  f"{s['edge_vs_long_pct']:>14} {_fmt(s['t_stat']):>6}")
    print("\nAll symbols pooled")
    print("-" * len(header))
    for st in ORDERED_TYPES:
        s = pooled.get(st)
        if s:
            print(f"{'ALL':<10} {st:<12} {s['n_entries']:>7} {s['hit_rate']:>6} {s['avg_net_pct']:>9} "
                  f"{s['edge_vs_long_pct']:>14} {_fmt(s['t_stat']):>6}")

    print("\nFollowing the signals long-only vs buy & hold")
    h2 = f"{'symbol':<10} {'strategy%':>10} {'buy&hold%':>10} {'strat DD%':>10} {'B&H DD%':>9} {'trades':>7} {'in mkt%':>8}"
    print(h2)
    print("-" * len(h2))
    for symbol, per_type in results.items():
        s = per_type.get("STRATEGY")
        if s:
            print(f"{symbol:<10} {s['strategy_return_pct']:>10} {s['buy_hold_return_pct']:>10} "
                  f"{s['strategy_max_drawdown_pct']:>10} {s['buy_hold_max_drawdown_pct']:>9} "
                  f"{s['trades']:>7} {s['time_in_market_pct']:>8}")
    print(
        "\nHow to read this: 'edge vs long' > 0 with t above ~2 suggests a real effect. With 24 symbols x 4 signal\n"
        "types, a few will look significant by pure chance, so trust the pooled rows and results that persist\n"
        "over months of live tracking more than any single cell.\n"
    )


def main():
    from dotenv import load_dotenv
    load_dotenv()
    from .database import init_db, DATA_DIR

    parser = argparse.ArgumentParser(description="Walk-forward backtest of the signal rules")
    parser.add_argument("--days", type=int, default=config.BACKTEST_LOOKBACK_DAYS)
    parser.add_argument("--horizon", type=int, default=config.BACKTEST_HORIZON_BARS,
                        help="holding period in 5-minute candles (12 = 1 hour)")
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--no-save", action="store_true", help="don't store results for the dashboard")
    args = parser.parse_args()

    init_db()
    results, pooled, records = run_all_backtests(args.days, args.horizon, args.symbols, save=not args.no_save)
    if not results:
        print("No results. Run `python -m signal_engine.backfill` first so there is history to test on.")
        return
    print_report(results, pooled, args.horizon)
    if not records.empty:
        out_dir = DATA_DIR / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"backtest_entries_h{args.horizon}.csv"
        records.to_csv(path, index=False)
        print(f"Entry-level results written to {path}")


if __name__ == "__main__":
    main()
