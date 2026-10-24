"""
Live forward tracking ("paper trading" the signals).

Every actionable signal the engine stores is graded later, at several horizons
(1h, 4h, 24h by default): what was the price change after the signal, and what
would the result have been after trading costs? Unlike a backtest, these
signals were produced in real time, before the outcome existed, so the numbers
can't be flattered by hindsight. After a few months this is the most trustworthy
evidence the system has.
"""
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from . import config
from .database import (
    get_signals_awaiting_outcome,
    get_first_price_at_or_after,
    save_outcomes,
    get_outcomes_frame,
    get_prices,
    parse_ts,
    to_utc_iso,
    get_tracking_start,
)
from .models import MarketType

DIRECTION = {"STRONG_BUY": 1, "BUY": 1, "SELL": -1, "STRONG_SELL": -1}
ORDERED_TYPES = ["STRONG_BUY", "BUY", "SELL", "STRONG_SELL"]

# If the first candle after the horizon is much later than planned (market closed,
# data outage), the outcome no longer measures the intended horizon. Allow some slack
# for overnight/weekend gaps, then skip.
MAX_EXIT_DELAY = timedelta(days=4)


def evaluate_pending_outcomes(horizons: Optional[List[int]] = None) -> int:
    """Grade every signal whose horizon has passed. Returns how many outcomes were stored."""
    horizons = horizons or config.TRACK_HORIZONS_MIN
    stored = 0
    for horizon in horizons:
        rows = []
        for s in get_signals_awaiting_outcome(horizon):
            entry_time = parse_ts(s["timestamp"])
            target = entry_time + timedelta(minutes=horizon)
            hit = get_first_price_at_or_after(s["symbol"], target)
            if hit is None:
                continue  # price data not there yet; try again next cycle
            exit_time, exit_price = hit
            if exit_time - target > MAX_EXIT_DELAY:
                logger.debug(f"Skipping outcome for signal {s['id']}: exit candle too far from horizon")
                exit_price = float("nan")
            entry_price = float(s["current_price"])
            if not entry_price or np.isnan(exit_price):
                exit_price, gross, net = None, None, None
            else:
                gross = exit_price / entry_price - 1.0
                net = DIRECTION[s["signal_type"]] * gross - config.cost_for(MarketType(s["market"]))
            rows.append((
                s["id"], horizon, s["symbol"], s["market"], s["signal_type"], int(s["is_entry"]),
                to_utc_iso(entry_time), entry_price, to_utc_iso(exit_time),
                exit_price, gross, net,
            ))
        save_outcomes(rows)
        stored += len(rows)
    if stored:
        logger.info(f"Forward tracking: graded {stored} signal outcomes")
    return stored


def _baseline_returns(symbols: List[str], horizon_min: int, since) -> Dict[str, float]:
    """Average forward return of simply holding each symbol over the same period ('always long')."""
    bars = max(1, horizon_min // config.CANDLE_MINUTES)
    out = {}
    for sym in symbols:
        df = get_prices(sym, since=since)
        if len(df) <= bars:
            continue
        fwd = (df["close"].shift(-bars) / df["close"] - 1.0).dropna()
        if len(fwd):
            out[sym] = float(fwd.mean())
    return out


def performance_summary(days: Optional[float] = None, entries_only: bool = True) -> dict:
    """
    Live performance per horizon and signal type.

    entries_only=True counts only the first signal of each run (when the signal type
    changed). Repeated 5-minute copies of the same signal overlap heavily and would
    inflate the sample size.
    """
    df = get_outcomes_frame(days)
    start = get_tracking_start()
    result = {
        "tracking_since": start.isoformat() if start else None,
        "entries_only": entries_only,
        "min_samples": config.MIN_SAMPLES_FOR_STATS,
        "horizons": [],
    }
    if df.empty:
        return result
    df = df.dropna(subset=["gross_return", "net_return"])
    if entries_only:
        df = df[df["is_entry"] == 1]

    for horizon in sorted(df["horizon_min"].unique()):
        h = df[df["horizon_min"] == horizon].copy()
        since = pd.to_datetime(h["entry_time"], utc=True).min()
        base = _baseline_returns(sorted(h["symbol"].unique()), int(horizon), since)
        h["baseline"] = h["symbol"].map(base).fillna(0.0)
        h["direction"] = h["signal_type"].map(DIRECTION)
        h["excess"] = h["direction"] * (h["gross_return"] - h["baseline"])

        per_type = []
        for st in ORDERED_TYPES:
            r = h[h["signal_type"] == st]
            n = len(r)
            if n == 0:
                continue
            ex = r["excess"].to_numpy(dtype=float)
            sd = float(np.std(ex, ddof=1)) if n > 1 else float("nan")
            t = float(np.mean(ex) / (sd / np.sqrt(n))) if n > 1 and sd > 0 else None
            per_type.append({
                "signal_type": st,
                "n": int(n),
                "enough_data": n >= config.MIN_SAMPLES_FOR_STATS,
                "hit_rate": round(float((r["net_return"] > 0).mean() * 100), 1),
                "avg_gross_pct": round(float(r["gross_return"].mean() * 100), 3),
                "avg_net_pct": round(float(r["net_return"].mean() * 100), 3),
                "edge_vs_long_pct": round(float(np.mean(ex) * 100), 3),
                "t_stat": round(t, 2) if t is not None and np.isfinite(t) else None,
            })

        by_market = []
        for market, r in h.groupby("market"):
            by_market.append({
                "market": market,
                "n": int(len(r)),
                "hit_rate": round(float((r["net_return"] > 0).mean() * 100), 1),
                "avg_net_pct": round(float(r["net_return"].mean() * 100), 3),
                "edge_vs_long_pct": round(float(r["excess"].mean() * 100), 3),
            })

        result["horizons"].append({
            "horizon_min": int(horizon),
            "graded": int(len(h)),
            "by_signal_type": per_type,
            "by_market": by_market,
        })
    return result
